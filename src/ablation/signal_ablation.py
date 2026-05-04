"""
11.1 Signal Ablation

7개 신호(s1~s7) 중 1개씩 제거하여 MoE Aggregator를 재학습한 뒤,
각 신호의 contribution을 측정합니다.

Contribution 정의:
    delta = metric(full) - metric(without_signal_i)
    delta > 0 → signal i가 metric 향상에 기여

지원 metric:
    - val_loss (낮을수록 좋음, 부호 반대로 해석)
    - bbq_accuracy_amb / accuracy_dis (높을수록 좋음)
    - bbq_bias_score_amb (|값|이 작을수록 좋음)

사용 예시:
    from src.ablation.signal_ablation import run_signal_ablation
    results = run_signal_ablation(
        train_records=train_records,
        val_records=val_records,
        embeddings=embeddings,
        config=cfg,
        save_dir="results/ablation/signals",
    )
    # results["per_signal"]["s5_bias_head"]["delta_acc_amb"]
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from src.models.moe_aggregator import MoEAggregator, signals_dict_to_tensor
from src.models.trainer import SignalsDataset, TrainConfig, train_moe

logger = logging.getLogger(__name__)


SIGNAL_NAMES: tuple[str, ...] = (
    "s1_evidence",
    "s2_counterfactual",
    "s3_confidence",
    "s4_consistency",
    "s5_bias_head",
    "s6_prompt_sensitivity",
    "s7_sae_feature",
)


# =============================================================
# 1. Masked dataset (특정 신호를 0으로 치환)
# =============================================================
class MaskedSignalsDataset(SignalsDataset):
    """
    특정 신호 인덱스를 0으로 치환한 SignalsDataset.

    Note: signal을 0으로 만드는 것은 "정보 없음"으로 모델에 전달됩니다.
    Signal temperature가 학습되므로 완벽한 제거는 아니지만,
    일반적인 leave-one-out ablation 관행과 일치합니다.

    Args:
        signal_records, embeddings: SignalsDataset과 동일.
        mask_index: 0으로 만들 신호 인덱스 (-1이면 마스킹 없음 = full).
    """

    def __init__(
        self,
        signal_records: list[dict],
        embeddings: dict[str, torch.Tensor],
        mask_index: int = -1,
        require_all: bool = False,
    ) -> None:
        super().__init__(signal_records, embeddings, require_all=require_all)
        if mask_index >= 0:
            for rec in self.records:
                rec["signals"][mask_index] = 0.0
        self.mask_index = mask_index


# =============================================================
# 2. Result containers
# =============================================================
@dataclass
class SignalAblationResult:
    """단일 신호 ablation 결과."""

    signal_name: str
    masked_index: int                   # -1 = full, 0..6 = masked
    best_val_loss: float
    best_epoch: int
    final_train_loss: float
    val_acc_amb: Optional[float] = None
    val_acc_dis: Optional[float] = None
    val_bias_amb: Optional[float] = None
    expert_usage: Optional[list[float]] = None


@dataclass
class SignalAblationSummary:
    """전체 ablation 요약."""

    full: SignalAblationResult
    per_signal: dict[str, SignalAblationResult] = field(default_factory=dict)

    def contributions(self) -> dict[str, dict[str, float]]:
        """
        full vs without_signal_i metric 차이 계산.

        Returns:
            {signal_name: {"delta_val_loss": ..., "delta_acc_amb": ..., ...}}
            delta > 0 = signal이 해당 metric 향상에 기여 (단, val_loss는 반대).
        """
        out: dict[str, dict[str, float]] = {}
        for name, res in self.per_signal.items():
            row: dict[str, float] = {}
            row["delta_val_loss"] = res.best_val_loss - self.full.best_val_loss
            for key in ("val_acc_amb", "val_acc_dis"):
                fv = getattr(self.full, key)
                rv = getattr(res, key)
                if fv is not None and rv is not None:
                    row[f"delta_{key}"] = fv - rv
            if self.full.val_bias_amb is not None and res.val_bias_amb is not None:
                # bias는 |값| 감소가 좋음
                row["delta_bias_abs_amb"] = abs(res.val_bias_amb) - abs(self.full.val_bias_amb)
            out[name] = row
        return out


# =============================================================
# 3. Single-run helper
# =============================================================
def _run_one(
    train_records: list[dict],
    val_records: list[dict],
    embeddings: dict[str, torch.Tensor],
    mask_index: int,
    signal_name: str,
    embed_dim: int,
    train_config: TrainConfig,
    seed: int = 42,
) -> SignalAblationResult:
    """한 번의 학습 실행."""
    train_ds = MaskedSignalsDataset(train_records, embeddings, mask_index=mask_index)
    val_ds = MaskedSignalsDataset(val_records, embeddings, mask_index=mask_index)

    torch.manual_seed(seed)
    model = MoEAggregator(
        signal_dim=len(SIGNAL_NAMES),
        embed_dim=embed_dim,
    )

    out = train_moe(train_ds, val_ds, model, train_config)
    history = out["history"]
    last = history[-1] if history else {}
    best = next(
        (h for h in history if h.get("epoch") == out.get("best_epoch")),
        last,
    )

    return SignalAblationResult(
        signal_name=signal_name,
        masked_index=mask_index,
        best_val_loss=float(out.get("best_val_loss", float("inf"))),
        best_epoch=int(out.get("best_epoch", -1)),
        final_train_loss=float(last.get("train_loss", float("nan"))),
        val_acc_amb=best.get("val_acc_amb"),
        val_acc_dis=best.get("val_acc_dis"),
        val_bias_amb=best.get("val_bias_amb"),
        expert_usage=best.get("expert_usage"),
    )


# =============================================================
# 4. Driver
# =============================================================
def run_signal_ablation(
    train_records: list[dict],
    val_records: list[dict],
    embeddings: dict[str, torch.Tensor],
    embed_dim: int = 4096,
    train_config: Optional[TrainConfig] = None,
    save_dir: Optional[str] = None,
    seed: int = 42,
) -> SignalAblationSummary:
    """
    7개 신호별 leave-one-out ablation을 실행합니다.

    총 8회 학습:
        - full: 모든 신호 사용
        - 각 s_i 제거: 7회 (해당 신호 컬럼을 0으로)

    Args:
        train_records: 학습용 신호 record 리스트.
        val_records: 검증용 신호 record 리스트.
        embeddings: {example_id: embedding tensor}.
        embed_dim: question embedding 차원.
        train_config: TrainConfig (None이면 기본).
        save_dir: 결과 저장 경로 (None이면 저장 안 함).
        seed: 재현성용 시드.

    Returns:
        SignalAblationSummary.
    """
    if train_config is None:
        train_config = TrainConfig()

    logger.info(f"[SignalAblation] full + {len(SIGNAL_NAMES)}개 ablation 실행")

    # 1) full
    full_res = _run_one(
        train_records, val_records, embeddings,
        mask_index=-1, signal_name="full",
        embed_dim=embed_dim, train_config=train_config, seed=seed,
    )
    logger.info(f"  [full]   val_loss={full_res.best_val_loss:.4f}")

    # 2) each masked
    summary = SignalAblationSummary(full=full_res)
    for idx, name in enumerate(SIGNAL_NAMES):
        res = _run_one(
            train_records, val_records, embeddings,
            mask_index=idx, signal_name=name,
            embed_dim=embed_dim, train_config=train_config, seed=seed,
        )
        summary.per_signal[name] = res
        logger.info(
            f"  [{name:25s}] val_loss={res.best_val_loss:.4f} "
            f"(Δ={res.best_val_loss - full_res.best_val_loss:+.4f})"
        )

    if save_dir:
        _save_summary(summary, save_dir)

    return summary


def _save_summary(summary: SignalAblationSummary, save_dir: str) -> None:
    """결과를 JSON으로 저장."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "full": asdict(summary.full),
        "per_signal": {k: asdict(v) for k, v in summary.per_signal.items()},
        "contributions": summary.contributions(),
    }
    (out_dir / "signal_ablation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"  [저장] {out_dir / 'signal_ablation.json'}")
