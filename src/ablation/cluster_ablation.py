"""
11.3 Cluster Ablation

MoE의 expert 수(K)와 cluster taxonomy를 변경하며 성능을 비교합니다.

세 가지 axis:
    1. K = 1, 2, 4, 8         — expert 수 변경 (4가 default).
    2. Cluster taxonomy        — 카테고리를 expert에 매핑하는 방식 변경.
        (a) "default"          — Lexical / Numerical / Cultural / Identity (4 cluster).
        (b) "by_polarity"       — neg / nonneg 2 cluster.
        (c) "flat_per_category" — 각 카테고리 = 1 expert (7 cluster).
        (d) "single"            — 1 expert (K=1과 동일).

K=1은 Gating이 의미 없어지므로 expert가 직접 출력 → ablation의 lower bound.
K=8은 cluster보다 많은 expert → soft routing의 자유도가 커짐.

사용 예시:
    summary = run_cluster_ablation(
        train_records, val_records, embeddings,
        k_options=(1, 2, 4, 8),
        taxonomies=("default", "by_polarity", "flat_per_category"),
        category_to_expert_default={...},
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from src.models.moe_aggregator import MoEAggregator
from src.models.trainer import SignalsDataset, TrainConfig, train_moe

logger = logging.getLogger(__name__)


# =============================================================
# 1. Cluster taxonomies
# =============================================================
DEFAULT_TAXONOMY: dict[str, int] = {
    # Lexically-Substitutable
    "Gender_identity": 0,
    "Religion": 0,
    # Numerically-Verifiable
    "Age": 1,
    "SES": 1,
    # Cultural-Contextual
    "Race_ethnicity": 2,
    # Identity-Sensitive
    "Disability_status": 3,
    "Sexual_orientation": 3,
}

POLARITY_TAXONOMY: dict[str, int] = {
    # 2-cluster: neg vs nonneg는 record 단위에서 결정되므로 여기서는
    # category 기반의 "static fallback"만 정의 (실제 routing은 polarity 기반).
    "Gender_identity": 0, "Religion": 0, "Age": 0, "SES": 0,
    "Race_ethnicity": 1, "Disability_status": 1, "Sexual_orientation": 1,
}

FLAT_TAXONOMY: dict[str, int] = {
    "Gender_identity": 0,
    "Race_ethnicity": 1,
    "Age": 2,
    "Religion": 3,
    "Disability_status": 4,
    "SES": 5,
    "Sexual_orientation": 6,
}

SINGLE_TAXONOMY: dict[str, int] = {
    cat: 0 for cat in [
        "Gender_identity", "Race_ethnicity", "Age", "Religion",
        "Disability_status", "SES", "Sexual_orientation",
    ]
}


TAXONOMIES: dict[str, dict[str, int]] = {
    "default": DEFAULT_TAXONOMY,
    "by_polarity": POLARITY_TAXONOMY,
    "flat_per_category": FLAT_TAXONOMY,
    "single": SINGLE_TAXONOMY,
}


def num_experts_in_taxonomy(taxonomy: dict[str, int]) -> int:
    """taxonomy에서 사용된 expert 수."""
    if not taxonomy:
        return 1
    return max(taxonomy.values()) + 1


# =============================================================
# 2. Result containers
# =============================================================
@dataclass
class ClusterAblationConfig:
    axis: str               # "k" | "taxonomy"
    value: str              # "4" 또는 "default"
    num_experts: int
    taxonomy_name: str = "default"


@dataclass
class ClusterAblationResult:
    config: ClusterAblationConfig
    best_val_loss: float
    val_acc_amb: Optional[float] = None
    val_acc_dis: Optional[float] = None
    val_bias_amb: Optional[float] = None
    expert_usage: Optional[list[float]] = None


@dataclass
class ClusterAblationSummary:
    by_axis: dict[str, list[ClusterAblationResult]] = field(default_factory=dict)


# =============================================================
# 3. Driver
# =============================================================
def run_cluster_ablation(
    train_records: list[dict],
    val_records: list[dict],
    embeddings: dict[str, torch.Tensor],
    k_options: tuple[int, ...] = (1, 2, 4, 8),
    taxonomies: tuple[str, ...] = ("default", "by_polarity", "flat_per_category"),
    embed_dim: int = 4096,
    train_config: Optional[TrainConfig] = None,
    save_dir: Optional[str] = None,
) -> ClusterAblationSummary:
    """
    K(expert 수)와 taxonomy를 바꿔가며 학습/평가.

    Args:
        k_options: 비교할 K 값들 (taxonomy를 default로 고정한 채 K만 변경).
        taxonomies: 비교할 taxonomy 이름들 (TAXONOMIES 키).
        나머지: 표준.

    Returns:
        ClusterAblationSummary.
    """
    if train_config is None:
        train_config = TrainConfig()

    summary = ClusterAblationSummary()

    # ---- (a) K 변경 (taxonomy=default) ----
    summary.by_axis["k"] = []
    for k in k_options:
        cfg = ClusterAblationConfig(
            axis="k", value=str(k), num_experts=k, taxonomy_name="default",
        )
        res = _run_one(cfg, train_records, val_records, embeddings,
                       embed_dim, train_config)
        summary.by_axis["k"].append(res)

    # ---- (b) Taxonomy 변경 (K = num_experts_in_taxonomy) ----
    summary.by_axis["taxonomy"] = []
    for name in taxonomies:
        if name not in TAXONOMIES:
            logger.warning(f"  [Cluster] 알 수 없는 taxonomy '{name}' skip")
            continue
        k = num_experts_in_taxonomy(TAXONOMIES[name])
        cfg = ClusterAblationConfig(
            axis="taxonomy", value=name, num_experts=k, taxonomy_name=name,
        )
        res = _run_one(cfg, train_records, val_records, embeddings,
                       embed_dim, train_config)
        summary.by_axis["taxonomy"].append(res)

    if save_dir:
        _save_summary(summary, save_dir)

    return summary


def _run_one(
    cfg: ClusterAblationConfig,
    train_records: list[dict],
    val_records: list[dict],
    embeddings: dict[str, torch.Tensor],
    embed_dim: int,
    train_config: TrainConfig,
) -> ClusterAblationResult:
    """단일 cluster 설정으로 학습."""
    logger.info(
        f"[ClusterAblation] axis={cfg.axis} value={cfg.value} "
        f"K={cfg.num_experts} taxonomy={cfg.taxonomy_name}"
    )

    train_ds = SignalsDataset(train_records, embeddings)
    val_ds = SignalsDataset(val_records, embeddings)

    model = MoEAggregator(
        signal_dim=7,
        embed_dim=embed_dim,
        num_experts=cfg.num_experts,
    )
    out = train_moe(train_ds, val_ds, model, train_config)

    history = out["history"]
    best = next(
        (h for h in history if h.get("epoch") == out.get("best_epoch")),
        history[-1] if history else {},
    )

    return ClusterAblationResult(
        config=cfg,
        best_val_loss=float(out.get("best_val_loss", float("inf"))),
        val_acc_amb=best.get("val_acc_amb"),
        val_acc_dis=best.get("val_acc_dis"),
        val_bias_amb=best.get("val_bias_amb"),
        expert_usage=best.get("expert_usage"),
    )


def _save_summary(summary: ClusterAblationSummary, save_dir: str) -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        axis: [
            {
                "config": asdict(r.config),
                "best_val_loss": r.best_val_loss,
                "val_acc_amb": r.val_acc_amb,
                "val_acc_dis": r.val_acc_dis,
                "val_bias_amb": r.val_bias_amb,
                "expert_usage": r.expert_usage,
            }
            for r in results
        ]
        for axis, results in summary.by_axis.items()
    }
    (out_dir / "cluster_ablation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"  [저장] {out_dir / 'cluster_ablation.json'}")
