"""
11.2 SAE Ablation

SAE(Sparse Autoencoder) 관련 하이퍼파라미터를 변경하며 s7_sae_feature
신호의 품질이 어떻게 달라지는지 분석합니다.

세 가지 축의 ablation:
    1. Top-K: 상위 몇 개 feature를 평균할 것인가? (10/50/100/200)
    2. Layer: 어느 layer의 hidden state를 SAE에 통과시킬 것인가? (12/15/20)
    3. Bias feature 식별 방법: 어떤 기준으로 bias-related feature를 고를 것인가?
        (a) "max_activation"      — BBQ ambig 샘플에서 평균 활성도가 높은 feature
        (b) "category_separability" — 카테고리 간 분산이 높은 feature (ANOVA-like)
        (c) "stereotype_correlation" — stereotyped/anti_stereotyped 답변 시 활성도 차이가 큰 feature

각 ablation은 s7 신호값을 재계산한 뒤,
SignalsDataset / MoE 학습을 다시 돌려서 metric을 비교합니다.

사용 예시:
    from src.ablation.sae_ablation import run_sae_ablation
    summary = run_sae_ablation(
        signal_records=records,
        embeddings=embeddings,
        s7_recompute_fn=recompute_fn,
        topk_options=(10, 50, 100, 200),
        layer_options=(12, 15, 20),
        identification_methods=("max_activation", "category_separability"),
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from src.models.moe_aggregator import MoEAggregator
from src.models.trainer import SignalsDataset, TrainConfig, train_moe

logger = logging.getLogger(__name__)


# =============================================================
# 1. Bias feature 식별 방법
# =============================================================
def identify_bias_features_max_activation(
    activations: np.ndarray,
    top_k: int = 50,
) -> list[int]:
    """
    BBQ 샘플 전체에서 평균 활성도가 높은 feature 인덱스를 반환합니다.

    Args:
        activations: (n_samples, n_features) feature activation matrix.
        top_k: 상위 몇 개를 선택할지.

    Returns:
        feature index 리스트 (활성도 내림차순).
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D, got {activations.shape}")
    mean_act = activations.mean(axis=0)
    top_indices = np.argsort(mean_act)[::-1][:top_k]
    return top_indices.tolist()


def identify_bias_features_category_separability(
    activations: np.ndarray,
    categories: list[str],
    top_k: int = 50,
) -> list[int]:
    """
    카테고리 간 분산이 높은 feature를 선택합니다 (ANOVA F-statistic 유사).

    카테고리별 평균 activation의 분산이 클수록 demographic 정보를 더 강하게
    인코딩하는 feature로 간주합니다.

    Args:
        activations: (n_samples, n_features) matrix.
        categories: 각 sample의 BBQ 카테고리 리스트 (길이 n_samples).
        top_k: 선택할 개수.

    Returns:
        feature index 리스트.
    """
    if len(categories) != activations.shape[0]:
        raise ValueError("categories와 activations 길이 불일치")

    cat_means: dict[str, np.ndarray] = {}
    for cat in set(categories):
        mask = np.array([c == cat for c in categories])
        if mask.sum() == 0:
            continue
        cat_means[cat] = activations[mask].mean(axis=0)

    if not cat_means:
        return []

    stacked = np.stack(list(cat_means.values()), axis=0)  # (K, n_features)
    between_var = stacked.var(axis=0)
    return np.argsort(between_var)[::-1][:top_k].tolist()


def identify_bias_features_stereotype_correlation(
    activations: np.ndarray,
    is_stereotyped: list[int],
    top_k: int = 50,
) -> list[int]:
    """
    stereotyped 답변 vs anti_stereotyped 답변 시 평균 활성도 차이가 큰
    feature를 선택합니다 (절대값 기준).

    Args:
        activations: (n_samples, n_features) matrix.
        is_stereotyped: 0/1 리스트 (1=stereotype 방향 답변).
        top_k: 선택 개수.

    Returns:
        feature index 리스트.
    """
    is_st = np.asarray(is_stereotyped, dtype=bool)
    if is_st.sum() == 0 or (~is_st).sum() == 0:
        # 한 쪽이라도 비어 있으면 fallback
        return identify_bias_features_max_activation(activations, top_k=top_k)

    diff = activations[is_st].mean(axis=0) - activations[~is_st].mean(axis=0)
    return np.argsort(np.abs(diff))[::-1][:top_k].tolist()


IDENTIFICATION_FNS: dict[str, Callable] = {
    "max_activation": identify_bias_features_max_activation,
    "category_separability": identify_bias_features_category_separability,
    "stereotype_correlation": identify_bias_features_stereotype_correlation,
}


# =============================================================
# 2. Result containers
# =============================================================
@dataclass
class SAEAblationConfig:
    """단일 ablation 설정 한 줄."""

    axis: str                             # "topk" | "layer" | "identification"
    value: str                            # "50", "12", "max_activation" 등
    top_k: int = 50
    layer: int = 16
    identification: str = "max_activation"


@dataclass
class SAEAblationResult:
    """단일 ablation 결과."""

    config: SAEAblationConfig
    n_bias_features: int
    best_val_loss: float
    val_acc_amb: Optional[float] = None
    val_acc_dis: Optional[float] = None
    val_bias_amb: Optional[float] = None


@dataclass
class SAEAblationSummary:
    """전체 SAE ablation 요약."""

    by_axis: dict[str, list[SAEAblationResult]] = field(default_factory=dict)


# =============================================================
# 3. Driver
# =============================================================
def run_sae_ablation(
    train_records: list[dict],
    val_records: list[dict],
    embeddings: dict[str, torch.Tensor],
    s7_recompute_fn: Callable[[SAEAblationConfig], dict[str, float]],
    topk_options: tuple[int, ...] = (10, 50, 100, 200),
    layer_options: tuple[int, ...] = (12, 15, 20),
    identification_methods: tuple[str, ...] = (
        "max_activation",
        "category_separability",
        "stereotype_correlation",
    ),
    embed_dim: int = 4096,
    train_config: Optional[TrainConfig] = None,
    save_dir: Optional[str] = None,
) -> SAEAblationSummary:
    """
    SAE 하이퍼파라미터별 ablation 실행.

    Args:
        train_records / val_records: signal record 리스트.
        embeddings: {example_id: tensor}.
        s7_recompute_fn: ablation config를 받아 {example_id: new_s7_value}를
                         반환하는 함수. 외부에서 주입 (SAE 호출 비용 큼).
        topk_options / layer_options: 탐색할 값 리스트.
        identification_methods: 비교할 bias feature 식별 방법.
        embed_dim: question embedding 차원.
        train_config: TrainConfig.
        save_dir: 결과 저장 경로.

    Returns:
        SAEAblationSummary.
    """
    if train_config is None:
        train_config = TrainConfig()

    summary = SAEAblationSummary()

    # ---- (a) Top-K 변경 ----
    summary.by_axis["topk"] = []
    for k in topk_options:
        cfg = SAEAblationConfig(axis="topk", value=str(k), top_k=k)
        res = _run_one_ablation(
            cfg, train_records, val_records, embeddings,
            s7_recompute_fn, embed_dim, train_config,
        )
        summary.by_axis["topk"].append(res)

    # ---- (b) Layer 변경 ----
    summary.by_axis["layer"] = []
    for layer in layer_options:
        cfg = SAEAblationConfig(axis="layer", value=str(layer), layer=layer)
        res = _run_one_ablation(
            cfg, train_records, val_records, embeddings,
            s7_recompute_fn, embed_dim, train_config,
        )
        summary.by_axis["layer"].append(res)

    # ---- (c) Identification 방법 변경 ----
    summary.by_axis["identification"] = []
    for method in identification_methods:
        cfg = SAEAblationConfig(axis="identification", value=method, identification=method)
        res = _run_one_ablation(
            cfg, train_records, val_records, embeddings,
            s7_recompute_fn, embed_dim, train_config,
        )
        summary.by_axis["identification"].append(res)

    if save_dir:
        _save_summary(summary, save_dir)

    return summary


# =============================================================
# 4. Internal helpers
# =============================================================
def _run_one_ablation(
    cfg: SAEAblationConfig,
    train_records: list[dict],
    val_records: list[dict],
    embeddings: dict[str, torch.Tensor],
    s7_recompute_fn: Callable[[SAEAblationConfig], dict[str, float]],
    embed_dim: int,
    train_config: TrainConfig,
) -> SAEAblationResult:
    """단일 SAE 설정에 대해 s7 재계산 + 학습."""
    logger.info(f"[SAEAblation] axis={cfg.axis} value={cfg.value}")

    # 1) s7 재계산
    new_s7 = s7_recompute_fn(cfg)
    n_features = sum(1 for v in new_s7.values() if v is not None)

    # 2) records 복제 후 s7 갱신
    train_updated = _update_s7(train_records, new_s7)
    val_updated = _update_s7(val_records, new_s7)

    # 3) MoE 학습
    train_ds = SignalsDataset(train_updated, embeddings)
    val_ds = SignalsDataset(val_updated, embeddings)
    model = MoEAggregator(embed_dim=embed_dim)
    out = train_moe(train_ds, val_ds, model, train_config)

    history = out["history"]
    best = next(
        (h for h in history if h.get("epoch") == out.get("best_epoch")),
        history[-1] if history else {},
    )

    return SAEAblationResult(
        config=cfg,
        n_bias_features=n_features,
        best_val_loss=float(out.get("best_val_loss", float("inf"))),
        val_acc_amb=best.get("val_acc_amb"),
        val_acc_dis=best.get("val_acc_dis"),
        val_bias_amb=best.get("val_bias_amb"),
    )


def _update_s7(
    records: list[dict],
    new_s7: dict[str, float],
) -> list[dict]:
    """records를 deep-copy하지 않고 s7만 교체한 새 리스트 반환."""
    out: list[dict] = []
    for rec in records:
        ex_id = rec["example_id"]
        new_signals = dict(rec["signals"])
        new_signals["s7_sae_feature"] = new_s7.get(ex_id)
        new_rec = dict(rec)
        new_rec["signals"] = new_signals
        out.append(new_rec)
    return out


def _save_summary(summary: SAEAblationSummary, save_dir: str) -> None:
    """JSON으로 저장."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        axis: [
            {
                "config": asdict(r.config),
                "n_bias_features": r.n_bias_features,
                "best_val_loss": r.best_val_loss,
                "val_acc_amb": r.val_acc_amb,
                "val_acc_dis": r.val_acc_dis,
                "val_bias_amb": r.val_bias_amb,
            }
            for r in results
        ]
        for axis, results in summary.by_axis.items()
    }
    (out_dir / "sae_ablation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"  [저장] {out_dir / 'sae_ablation.json'}")
