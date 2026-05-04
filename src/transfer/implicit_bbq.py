"""
Zero-shot Transfer to ImplicitBBQ.

ImplicitBBQ는 BBQ를 명시적 demographic 단어 없이 implicit cue로 재작성한 데이터셋입니다.
학습된 시스템(Stage 1-4)을 zero-shot으로 그대로 적용하여 일반화 능력을 평가합니다.

기능:
    - load_implicit_bbq(): 데이터 로드
    - transfer_evaluate(): 학습된 MoE를 zero-shot 적용 + 평가
    - analyze_cluster_routing(): gating network가 어떤 cluster로 라우팅하는지 분석

데이터 위치 (default):
    data/implicit_bbq/{category}.jsonl
    또는 data/implicit_bbq/test.parquet
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
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
from src.models.moe_aggregator import predict_p, signals_dict_to_tensor

if TYPE_CHECKING:
    from src.models.moe_aggregator import MoEAggregator

logger = logging.getLogger(__name__)


# =============================================================
# Data loading
# =============================================================
def load_implicit_bbq(
    data_dir: str | Path = "data/implicit_bbq",
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """
    ImplicitBBQ 데이터를 로드합니다.

    파일 형식:
        - {data_dir}/{category}.jsonl 형식의 JSONL 파일
        - parquet 파일도 지원: {data_dir}/test.parquet

    Args:
        data_dir: 데이터 디렉토리.
        categories: 로드할 카테고리. None이면 전체 jsonl.

    Returns:
        instance dict 리스트. 'category' 필드가 채워짐.

    Raises:
        FileNotFoundError: data_dir이 존재하지 않거나 파일이 없는 경우.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"ImplicitBBQ 디렉토리 없음: {data_path}\n"
            f"데이터를 다운로드하여 {data_path}에 배치하세요."
        )

    items: list[dict] = []

    # parquet 우선
    parquet_path = data_path / "test.parquet"
    if parquet_path.exists():
        import pandas as pd
        df = pd.read_parquet(parquet_path)
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
        raise FileNotFoundError(f"ImplicitBBQ JSONL 없음: {data_path}")

    for f in jsonl_files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    rec = json.loads(line)
                    rec.setdefault("category", f.stem)
                    items.append(rec)
    return items


# =============================================================
# Cluster routing analysis
# =============================================================
@dataclass
class ClusterRoutingStats:
    """카테고리별 cluster routing 통계."""

    avg_weights_per_category: dict[str, list[float]]   # {category: [w1, w2, w3, w4]}
    dominant_cluster_per_category: dict[str, int]
    overall_avg_weights: list[float]
    n_per_category: dict[str, int]


def analyze_cluster_routing(
    moe_model: "MoEAggregator",
    instances: list[dict],
    embedding_extractor: EmbeddingExtractor,
    device: Optional[torch.device] = None,
) -> ClusterRoutingStats:
    """
    각 instance의 gating weight를 추출하여 카테고리별 라우팅 패턴을 분석합니다.

    Args:
        moe_model: 학습된 MoEAggregator.
        instances: instance 리스트.
        embedding_extractor: instance -> embedding 텐서.
        device: torch device.

    Returns:
        ClusterRoutingStats.
    """
    moe_model.eval()
    if device is None:
        device = next(moe_model.parameters()).device

    weights_by_cat: dict[str, list[np.ndarray]] = defaultdict(list)
    overall: list[np.ndarray] = []

    with torch.inference_mode():
        for item in instances:
            cat = item.get("category", "_unknown")
            embed = embedding_extractor(item).to(device).unsqueeze(0)
            # signals는 routing에 영향 안 줌 (gating은 embedding만 봄) — dummy
            dummy_signals = torch.zeros(1, moe_model.signal_dim, device=device)
            out = moe_model(dummy_signals, embed)
            w = out.gate_w.squeeze(0).cpu().numpy()
            weights_by_cat[cat].append(w)
            overall.append(w)

    avg_weights_per_category = {
        cat: np.mean(weights, axis=0).tolist()
        for cat, weights in weights_by_cat.items()
    }
    dominant_cluster_per_category = {
        cat: int(np.argmax(w))
        for cat, w in avg_weights_per_category.items()
    }
    overall_avg = (
        np.mean(overall, axis=0).tolist()
        if overall else [1.0 / moe_model.num_experts] * moe_model.num_experts
    )
    n_per_category = {cat: len(weights) for cat, weights in weights_by_cat.items()}

    return ClusterRoutingStats(
        avg_weights_per_category=avg_weights_per_category,
        dominant_cluster_per_category=dominant_cluster_per_category,
        overall_avg_weights=overall_avg,
        n_per_category=n_per_category,
    )


# =============================================================
# Transfer evaluation
# =============================================================
@dataclass
class TransferEvalResult:
    """Transfer 평가 결과 (overall + per-category)."""

    overall_metrics: dict[str, float]
    metrics_per_category: dict[str, dict[str, float]]
    routing_stats: ClusterRoutingStats
    n_total: int


def transfer_evaluate(
    instances: list[dict],
    primary_answers: list[str],
    moe_model: "MoEAggregator",
    signal_extractor: SignalExtractor,
    embedding_extractor: EmbeddingExtractor,
    threshold: float = 0.5,
    show_progress: bool = True,
) -> TransferEvalResult:
    """
    학습된 MoE를 zero-shot으로 ImplicitBBQ에 적용하고 카테고리별 평가합니다.

    Args:
        instances: ImplicitBBQ instance.
        primary_answers: 1차 LLM 답변 (Stage 1 vanilla).
        moe_model: Llama에서 학습된 MoEAggregator.
        signal_extractor: 신호 추출 함수.
        embedding_extractor: embedding 함수.
        threshold: override 임계값.
        show_progress: tqdm.

    Returns:
        TransferEvalResult.
    """
    if len(instances) != len(primary_answers):
        raise ValueError(
            f"길이 불일치: instances={len(instances)}, "
            f"primary_answers={len(primary_answers)}"
        )

    # Stacking 적용
    stacking = stack_baseline_with_pipeline(
        instances=instances,
        baseline_answers=primary_answers,
        moe_model=moe_model,
        signal_extractor=signal_extractor,
        embedding_extractor=embedding_extractor,
        threshold=threshold,
        show_progress=show_progress,
    )

    final_preds = [r.final_answer for r in stacking]

    # Overall metrics
    overall_metrics = evaluate_bbq([str(p) for p in final_preds], instances)

    # Per-category
    by_cat: dict[str, list[int]] = defaultdict(list)
    items_by_cat: dict[str, list[dict]] = defaultdict(list)
    for item, pred in zip(instances, final_preds):
        cat = item.get("category", "_unknown")
        by_cat[cat].append(pred)
        items_by_cat[cat].append(item)

    metrics_per_category = {
        cat: evaluate_bbq([str(p) for p in preds], items_by_cat[cat])
        for cat, preds in by_cat.items()
    }

    # Cluster routing
    routing = analyze_cluster_routing(moe_model, instances, embedding_extractor)

    return TransferEvalResult(
        overall_metrics=overall_metrics,
        metrics_per_category=metrics_per_category,
        routing_stats=routing,
        n_total=len(instances),
    )


# =============================================================
# Save/Load helpers
# =============================================================
def save_transfer_result(result: TransferEvalResult, path: str | Path) -> None:
    """결과를 JSON으로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall_metrics": result.overall_metrics,
        "metrics_per_category": result.metrics_per_category,
        "routing_stats": {
            "avg_weights_per_category": result.routing_stats.avg_weights_per_category,
            "dominant_cluster_per_category": result.routing_stats.dominant_cluster_per_category,
            "overall_avg_weights": result.routing_stats.overall_avg_weights,
            "n_per_category": result.routing_stats.n_per_category,
        },
        "n_total": result.n_total,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[저장] {path}")
