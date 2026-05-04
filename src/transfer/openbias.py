"""
Zero-shot Transfer to OpenBiasBench.

OpenBiasBench는 BBQ의 9개 카테고리를 넘어 31개 새 demographic 카테고리를
포함하는 확장 벤치마크입니다. 학습된 시스템(7-signal + MoE)을
zero-shot으로 적용하여 unseen 카테고리에서의 일반화를 평가합니다.

핵심 분석:
    1. 새 카테고리에서의 metric 평가
    2. Cluster routing accuracy:
       - 새 카테고리가 어느 cluster로 라우팅되는지
       - 의미적으로 가장 가까운 cluster에 라우팅되는지 (수동 정답 라벨 필요)

데이터 위치:
    data/openbias/{category}.jsonl 또는 data/openbias/test.parquet
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from src.evaluation.bbq_evaluator import evaluate_bbq, parse_prediction
from src.evaluation.stacking_ablation import (
    EmbeddingExtractor,
    SignalExtractor,
    stack_baseline_with_pipeline,
)
from src.transfer.implicit_bbq import (
    ClusterRoutingStats,
    TransferEvalResult,
    analyze_cluster_routing,
    save_transfer_result,
    transfer_evaluate,
)

if TYPE_CHECKING:
    from src.models.moe_aggregator import MoEAggregator

logger = logging.getLogger(__name__)


# =============================================================
# 카테고리 → cluster 의미적 매핑 (수동 정의)
# =============================================================
# 31개 카테고리(또는 그 이상)를 4개 cluster로 분류하는 의미적 정답.
# 'manual ground truth' 역할로, routing accuracy 측정에 사용됩니다.
# 사용자는 실제 OpenBiasBench 카테고리 목록에 맞춰 확장하세요.

DEFAULT_CATEGORY_TO_CLUSTER: dict[str, int] = {
    # Cluster 0: Lexically-Substitutable (단어 치환만으로 swap 가능)
    "Gender_identity": 0, "Religion": 0, "Marital_status": 0,
    "Pregnancy": 0, "Language": 0,

    # Cluster 1: Numerically-Verifiable (숫자/명시적 정보)
    "Age": 1, "SES": 1, "Income": 1, "Education": 1, "Wealth": 1,

    # Cluster 2: Cultural-Contextual (문화적 맥락)
    "Race_ethnicity": 2, "Nationality": 2, "Geographic_origin": 2,
    "Caste": 2, "Tribe": 2, "Accent": 2,

    # Cluster 3: Identity-Sensitive (정체성 민감)
    "Disability_status": 3, "Sexual_orientation": 3, "Mental_health": 3,
    "Body_type": 3, "Gender_expression": 3,
}


# =============================================================
# Data loading
# =============================================================
def load_openbias(
    data_dir: str | Path = "data/openbias",
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """
    OpenBiasBench 데이터를 로드합니다.

    Args:
        data_dir: 데이터 디렉토리.
        categories: 로드할 카테고리. None이면 전체.

    Returns:
        instance dict 리스트.

    Raises:
        FileNotFoundError: 데이터 없음.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"OpenBiasBench 디렉토리 없음: {data_path}\n"
            f"데이터를 다운로드하여 {data_path}에 배치하세요."
        )

    items: list[dict] = []

    # parquet 우선
    parquet_path = data_path / "test.parquet"
    if parquet_path.exists():
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        if categories:
            df = df[df["category"].isin(categories)]
        for _, row in df.iterrows():
            rec = row.to_dict()
            for col in ("answer_info", "additional_metadata"):
                if isinstance(rec.get(col), str):
                    try:
                        rec[col] = json.loads(rec[col])
                    except json.JSONDecodeError:
                        pass
            items.append(rec)
        return items

    # JSONL 폴백
    jsonl_files = sorted(data_path.glob("*.jsonl"))
    if categories:
        jsonl_files = [f for f in jsonl_files if f.stem in categories]
    if not jsonl_files:
        raise FileNotFoundError(f"OpenBiasBench JSONL 없음: {data_path}")

    for f in jsonl_files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    rec = json.loads(line)
                    rec.setdefault("category", f.stem)
                    items.append(rec)
    return items


# =============================================================
# Routing accuracy
# =============================================================
@dataclass
class RoutingAccuracyResult:
    """카테고리 → cluster 라우팅 정확도."""

    accuracy: float                                 # 전체 routing 정확도
    accuracy_per_category: dict[str, float]
    confusion_matrix: dict[str, dict[int, int]]    # {category: {cluster_idx: count}}
    n_evaluated: int
    n_unmapped: int                                 # ground truth 매핑 없는 instance 수


def compute_routing_accuracy(
    routing_stats: ClusterRoutingStats,
    category_to_cluster: Optional[dict[str, int]] = None,
) -> RoutingAccuracyResult:
    """
    Cluster routing의 정확도를 측정합니다.

    "정확도" 정의:
        각 instance의 dominant cluster (argmax gating weight) ==
        해당 카테고리의 ground truth cluster.

    Args:
        routing_stats: analyze_cluster_routing 결과.
        category_to_cluster: 카테고리 → cluster 정답 매핑.
            None이면 DEFAULT_CATEGORY_TO_CLUSTER 사용.

    Returns:
        RoutingAccuracyResult.
    """
    if category_to_cluster is None:
        category_to_cluster = DEFAULT_CATEGORY_TO_CLUSTER

    accuracy_per_category: dict[str, float] = {}
    confusion: dict[str, dict[int, int]] = {}
    correct_total = 0
    n_total = 0
    n_unmapped = 0

    for cat, n_instances in routing_stats.n_per_category.items():
        if cat not in category_to_cluster:
            n_unmapped += n_instances
            continue

        gt_cluster = category_to_cluster[cat]
        avg_w = routing_stats.avg_weights_per_category.get(cat, [])
        dom = routing_stats.dominant_cluster_per_category.get(cat, -1)

        is_correct = (dom == gt_cluster)
        accuracy_per_category[cat] = 1.0 if is_correct else 0.0

        confusion[cat] = {dom: n_instances}
        if is_correct:
            correct_total += n_instances
        n_total += n_instances

    accuracy = correct_total / n_total if n_total > 0 else 0.0
    return RoutingAccuracyResult(
        accuracy=accuracy,
        accuracy_per_category=accuracy_per_category,
        confusion_matrix=confusion,
        n_evaluated=n_total,
        n_unmapped=n_unmapped,
    )


# =============================================================
# Transfer evaluation (extends ImplicitBBQ pattern)
# =============================================================
@dataclass
class OpenBiasTransferResult:
    """OpenBiasBench transfer 평가 + routing accuracy."""

    eval_result: TransferEvalResult
    routing_accuracy: RoutingAccuracyResult


def transfer_evaluate_openbias(
    instances: list[dict],
    primary_answers: list[str],
    moe_model: "MoEAggregator",
    signal_extractor: SignalExtractor,
    embedding_extractor: EmbeddingExtractor,
    threshold: float = 0.5,
    category_to_cluster: Optional[dict[str, int]] = None,
    show_progress: bool = True,
) -> OpenBiasTransferResult:
    """
    OpenBiasBench zero-shot transfer + cluster routing accuracy.

    Args:
        instances: OpenBiasBench instance.
        primary_answers: 1차 LLM 답변.
        moe_model: 학습된 MoEAggregator.
        signal_extractor: 신호 추출.
        embedding_extractor: embedding 추출.
        threshold: override 임계값.
        category_to_cluster: 정답 routing 매핑.
        show_progress: tqdm.

    Returns:
        OpenBiasTransferResult.
    """
    eval_result = transfer_evaluate(
        instances=instances,
        primary_answers=primary_answers,
        moe_model=moe_model,
        signal_extractor=signal_extractor,
        embedding_extractor=embedding_extractor,
        threshold=threshold,
        show_progress=show_progress,
    )

    routing_acc = compute_routing_accuracy(
        eval_result.routing_stats,
        category_to_cluster=category_to_cluster,
    )

    return OpenBiasTransferResult(
        eval_result=eval_result,
        routing_accuracy=routing_acc,
    )


def save_openbias_result(
    result: OpenBiasTransferResult,
    path: str | Path,
) -> None:
    """OpenBias 결과를 JSON으로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "overall_metrics": result.eval_result.overall_metrics,
        "metrics_per_category": result.eval_result.metrics_per_category,
        "routing_stats": {
            "avg_weights_per_category": result.eval_result.routing_stats.avg_weights_per_category,
            "dominant_cluster_per_category": result.eval_result.routing_stats.dominant_cluster_per_category,
            "overall_avg_weights": result.eval_result.routing_stats.overall_avg_weights,
            "n_per_category": result.eval_result.routing_stats.n_per_category,
        },
        "routing_accuracy": {
            "accuracy": result.routing_accuracy.accuracy,
            "accuracy_per_category": result.routing_accuracy.accuracy_per_category,
            "confusion_matrix": {
                cat: {str(k): v for k, v in conf.items()}
                for cat, conf in result.routing_accuracy.confusion_matrix.items()
            },
            "n_evaluated": result.routing_accuracy.n_evaluated,
            "n_unmapped": result.routing_accuracy.n_unmapped,
        },
        "n_total": result.eval_result.n_total,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[저장] {path}")
