"""
Zero-shot Transfer to OpenBiasBench.

OpenBiasBench는 BBQ의 9개 카테고리를 넘어 31개 새 demographic 카테고리를
포함하는 확장 벤치마크입니다. 학습된 시스템(7-signal + MoE)을
zero-shot으로 적용하여 unseen 카테고리에서의 일반화를 평가합니다.

핵심 분석:
    1. 새 카테고리에서의 metric 평가
    2. Learned-expert routing diagnostics:
       - 새 카테고리가 어느 learned expert index로 라우팅되는지
       - 과거 수동 category-to-index taxonomy와의 기술적 alignment

Learned experts are exchangeable: expert index 0, for example, has no stable
semantic identity across fits. The legacy alignment is retained for artifact
compatibility and must not be interpreted as semantic routing accuracy.

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
# Legacy category → learned-expert-index alignment map
# =============================================================
# This hand-authored mapping predates the current interpretation policy. It is
# not ground truth: learned expert indices are exchangeable across fits. Keep
# it only to reproduce historical alignment artifacts.

LEGACY_CATEGORY_TO_EXPERT_INDEX: dict[str, int] = {
    # Legacy index 0
    "Gender_identity": 0, "Religion": 0, "Marital_status": 0,
    "Pregnancy": 0, "Language": 0,

    # Legacy index 1
    "Age": 1, "SES": 1, "Income": 1, "Education": 1, "Wealth": 1,

    # Legacy index 2
    "Race_ethnicity": 2, "Nationality": 2, "Geographic_origin": 2,
    "Caste": 2, "Tribe": 2, "Accent": 2,

    # Legacy index 3
    "Disability_status": 3, "Sexual_orientation": 3, "Mental_health": 3,
    "Body_type": 3, "Gender_expression": 3,
}

# Backward-compatible import name. Do not use this alias to claim semantic
# cluster ground truth.
DEFAULT_CATEGORY_TO_CLUSTER = LEGACY_CATEGORY_TO_EXPERT_INDEX


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
# Legacy taxonomy alignment
# =============================================================
@dataclass
class LegacyTaxonomyAlignmentResult:
    """Alignment with a legacy category-to-index map, not ground-truth accuracy."""

    accuracy: float                                 # historical field name
    accuracy_per_category: dict[str, float]
    confusion_matrix: dict[str, dict[int, int]]    # {category: {expert_idx: count}}
    n_evaluated: int
    n_unmapped: int                                 # legacy map에 없는 instance 수

    @property
    def alignment_rate(self) -> float:
        """Canonical name for the historical ``accuracy`` field."""
        return self.accuracy


def compute_legacy_taxonomy_alignment(
    routing_stats: ClusterRoutingStats,
    category_to_cluster: Optional[dict[str, int]] = None,
) -> LegacyTaxonomyAlignmentResult:
    """
    Compare dominant learned-expert indices with the legacy manual map.

    The historical ``accuracy`` field is the sample-weighted agreement rate.
    It is permutation-dependent and is not evidence that an expert learned a
    named semantic role.

    Args:
        routing_stats: analyze_cluster_routing 결과.
        category_to_cluster: legacy 카테고리 → expert-index 매핑.
            None이면 LEGACY_CATEGORY_TO_EXPERT_INDEX 사용.

    Returns:
        LegacyTaxonomyAlignmentResult.
    """
    if category_to_cluster is None:
        category_to_cluster = LEGACY_CATEGORY_TO_EXPERT_INDEX

    accuracy_per_category: dict[str, float] = {}
    confusion: dict[str, dict[int, int]] = {}
    correct_total = 0
    n_total = 0
    n_unmapped = 0

    for cat, n_instances in routing_stats.n_per_category.items():
        if cat not in category_to_cluster:
            n_unmapped += n_instances
            continue

        legacy_index = category_to_cluster[cat]
        dom = routing_stats.dominant_cluster_per_category.get(cat, -1)

        is_correct = (dom == legacy_index)
        accuracy_per_category[cat] = 1.0 if is_correct else 0.0

        confusion[cat] = {dom: n_instances}
        if is_correct:
            correct_total += n_instances
        n_total += n_instances

    accuracy = correct_total / n_total if n_total > 0 else 0.0
    return LegacyTaxonomyAlignmentResult(
        accuracy=accuracy,
        accuracy_per_category=accuracy_per_category,
        confusion_matrix=confusion,
        n_evaluated=n_total,
        n_unmapped=n_unmapped,
    )


# Backward-compatible type and function names for downstream artifacts.
RoutingAccuracyResult = LegacyTaxonomyAlignmentResult


def compute_routing_accuracy(
    routing_stats: ClusterRoutingStats,
    category_to_cluster: Optional[dict[str, int]] = None,
) -> LegacyTaxonomyAlignmentResult:
    """Deprecated alias for :func:`compute_legacy_taxonomy_alignment`."""
    return compute_legacy_taxonomy_alignment(
        routing_stats, category_to_cluster=category_to_cluster
    )


# =============================================================
# Transfer evaluation (extends ImplicitBBQ pattern)
# =============================================================
@dataclass
class OpenBiasTransferResult:
    """OpenBiasBench transfer evaluation plus a legacy alignment diagnostic."""

    eval_result: TransferEvalResult
    routing_accuracy: LegacyTaxonomyAlignmentResult

    @property
    def legacy_taxonomy_alignment(self) -> LegacyTaxonomyAlignmentResult:
        """Canonical name for the backward-compatible ``routing_accuracy`` field."""
        return self.routing_accuracy


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
    OpenBiasBench zero-shot transfer + legacy taxonomy alignment.

    Args:
        instances: OpenBiasBench instance.
        primary_answers: 1차 LLM 답변.
        moe_model: 학습된 MoEAggregator.
        signal_extractor: 신호 추출.
        embedding_extractor: embedding 추출.
        threshold: override 임계값.
        category_to_cluster: legacy category-to-expert-index map.
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

    routing_acc = compute_legacy_taxonomy_alignment(
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

    alignment_payload = {
        "interpretation": "legacy_taxonomy_alignment_only",
        "expert_index_interpretation": "exchangeable_learned_expert_index",
        "alignment_rate": result.routing_accuracy.alignment_rate,
        "accuracy": result.routing_accuracy.accuracy,
        "accuracy_per_category": result.routing_accuracy.accuracy_per_category,
        "confusion_matrix": {
            cat: {str(k): v for k, v in conf.items()}
            for cat, conf in result.routing_accuracy.confusion_matrix.items()
        },
        "n_evaluated": result.routing_accuracy.n_evaluated,
        "n_unmapped": result.routing_accuracy.n_unmapped,
    }
    payload = {
        "overall_metrics": result.eval_result.overall_metrics,
        "metrics_per_category": result.eval_result.metrics_per_category,
        "routing_stats": {
            "avg_weights_per_category": result.eval_result.routing_stats.avg_weights_per_category,
            "dominant_cluster_per_category": result.eval_result.routing_stats.dominant_cluster_per_category,
            "overall_avg_weights": result.eval_result.routing_stats.overall_avg_weights,
            "n_per_category": result.eval_result.routing_stats.n_per_category,
        },
        "legacy_taxonomy_alignment": alignment_payload,
        # Historical key retained for readers of existing result files.
        "routing_accuracy": alignment_payload,
        "routing_accuracy_interpretation": "legacy_taxonomy_alignment_only",
        "n_total": result.eval_result.n_total,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[저장] {path}")
