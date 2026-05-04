"""
11.4 Leave-One-Category-Out (LOCO) Ablation

7개 카테고리 중 1개를 held-out으로 두고, 나머지 6개로 학습한 뒤
held-out 카테고리에서 평가합니다. 7-fold cross validation 형태.

목적:
    - MoE Aggregator가 학습되지 않은 카테고리에 일반화되는지 검증.
    - Cluster 정의의 robustness 측정 (held-out 카테고리가 어느 expert에
      라우팅되는지 분석).

각 fold:
    1. 학습 set: held-out 제외 6개 카테고리.
    2. 평가 set: held-out 카테고리 (모두).
    3. metric: BBQ accuracy_amb/dis, bias_score_amb, false_abstention_rate.

사용 예시:
    summary = run_loco_ablation(
        all_records=records,
        embeddings=embeddings,
        instances_by_id=instances_by_id,
        categories=("Gender_identity", "Race_ethnicity", ...),
    )
    # summary.per_fold["Gender_identity"].held_out_acc_amb
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from src.evaluation.bbq_evaluator import evaluate_bbq
from src.models.moe_aggregator import MoEAggregator, predict_p
from src.models.override import apply_threshold_override
from src.models.trainer import SignalsDataset, TrainConfig, train_moe

logger = logging.getLogger(__name__)


# =============================================================
# 1. Result containers
# =============================================================
@dataclass
class LOCOFoldResult:
    """단일 fold 결과."""

    held_out_category: str
    n_train: int
    n_held_out: int
    best_val_loss: float
    held_out_acc_amb: float
    held_out_acc_dis: float
    held_out_bias_amb: Optional[float] = None
    held_out_false_abstention: float = 0.0
    held_out_expert_usage: Optional[list[float]] = None  # held-out에서 사용된 expert 분포


@dataclass
class LOCOAblationSummary:
    per_fold: dict[str, LOCOFoldResult] = field(default_factory=dict)

    def aggregate(self) -> dict[str, float]:
        """7-fold 평균 metric."""
        if not self.per_fold:
            return {}
        n = len(self.per_fold)
        keys = ("held_out_acc_amb", "held_out_acc_dis", "held_out_false_abstention")
        agg = {k: sum(getattr(r, k) for r in self.per_fold.values()) / n for k in keys}
        biases = [r.held_out_bias_amb for r in self.per_fold.values() if r.held_out_bias_amb is not None]
        if biases:
            agg["held_out_bias_amb"] = sum(biases) / len(biases)
        return agg


# =============================================================
# 2. Driver
# =============================================================
def run_loco_ablation(
    all_records: list[dict],
    embeddings: dict[str, torch.Tensor],
    instances_by_id: dict[str, dict],
    categories: tuple[str, ...] = (
        "Gender_identity", "Race_ethnicity", "Age", "Religion",
        "Disability_status", "SES", "Sexual_orientation",
    ),
    embed_dim: int = 4096,
    train_config: Optional[TrainConfig] = None,
    threshold: float = 0.5,
    save_dir: Optional[str] = None,
) -> LOCOAblationSummary:
    """
    Leave-One-Category-Out cross validation.

    Args:
        all_records: 전체 signal records (모든 카테고리 포함).
                     각 record는 'category' 키를 가져야 함.
        embeddings: {example_id: tensor}.
        instances_by_id: {example_id: BBQ instance dict}.
        categories: 7개 카테고리.
        embed_dim: question embedding 차원.
        train_config: TrainConfig.
        threshold: held-out에서 override 적용에 쓸 tau.
        save_dir: 결과 저장 경로.

    Returns:
        LOCOAblationSummary.
    """
    if train_config is None:
        train_config = TrainConfig()

    summary = LOCOAblationSummary()

    for held_out in categories:
        logger.info(f"[LOCO] held-out = {held_out}")

        train_records = [r for r in all_records if r.get("category") != held_out]
        held_records = [r for r in all_records if r.get("category") == held_out]

        if not held_records:
            logger.warning(f"  [LOCO] '{held_out}' 카테고리 record 없음 — fold skip")
            continue

        # 1) train (val_split 내부에서 자동 처리하지 않으므로,
        #    train_records의 일부를 val로 분리)
        split_idx = int(len(train_records) * 0.85)
        train_subset = train_records[:split_idx]
        val_subset = train_records[split_idx:]

        train_ds = SignalsDataset(train_subset, embeddings)
        val_ds = SignalsDataset(val_subset, embeddings)
        held_ds = SignalsDataset(held_records, embeddings)

        # 2) MoE 학습
        model = MoEAggregator(signal_dim=7, embed_dim=embed_dim)
        out = train_moe(train_ds, val_ds, model, train_config)

        # 3) Held-out 평가
        fold_res = _evaluate_held_out(
            model=model,
            held_dataset=held_ds,
            held_records=held_records,
            instances_by_id=instances_by_id,
            threshold=threshold,
            held_out_category=held_out,
            n_train=len(train_subset),
            best_val_loss=float(out.get("best_val_loss", float("inf"))),
        )

        summary.per_fold[held_out] = fold_res
        logger.info(
            f"  [{held_out}] held_acc_amb={fold_res.held_out_acc_amb:.4f} "
            f"acc_dis={fold_res.held_out_acc_dis:.4f}"
        )

    if save_dir:
        _save_summary(summary, save_dir)

    return summary


# =============================================================
# 3. Held-out evaluation
# =============================================================
def _evaluate_held_out(
    model: MoEAggregator,
    held_dataset: SignalsDataset,
    held_records: list[dict],
    instances_by_id: dict[str, dict],
    threshold: float,
    held_out_category: str,
    n_train: int,
    best_val_loss: float,
) -> LOCOFoldResult:
    """Held-out 카테고리에서 BBQ 평가."""
    model.eval()
    device = next(model.parameters()).device

    final_preds: list[int] = []
    final_items: list[dict] = []
    expert_uses: list[list[float]] = []

    rec_by_id = {r["example_id"]: r for r in held_records}
    with torch.inference_mode():
        for ds_item in held_dataset:
            ex_id = ds_item.get("example_id")
            rec = rec_by_id.get(ex_id) if ex_id is not None else None
            if rec is None:
                continue

            sig = ds_item["signals"].to(device)
            emb = ds_item["embedding"].to(device)
            out = model(sig.unsqueeze(0), emb.unsqueeze(0))
            p = float(out.p.item())
            expert_uses.append(out.gate_w[0].cpu().tolist())

            # primary_answer + override
            primary = int(rec["primary_answer"])
            inst = instances_by_id.get(ex_id, {})

            override = apply_threshold_override(
                primary_answer=primary, p_score=p, item=inst, threshold=threshold,
            )
            final_preds.append(override["final_answer"])
            final_items.append(inst)

    # BBQ 평가
    metrics = evaluate_bbq(final_preds, final_items)

    # expert usage 평균
    expert_usage = None
    if expert_uses:
        n_exp = len(expert_uses[0])
        expert_usage = [
            sum(u[k] for u in expert_uses) / len(expert_uses)
            for k in range(n_exp)
        ]

    return LOCOFoldResult(
        held_out_category=held_out_category,
        n_train=n_train,
        n_held_out=len(held_records),
        best_val_loss=best_val_loss,
        held_out_acc_amb=float(metrics.get("accuracy_amb", 0.0)),
        held_out_acc_dis=float(metrics.get("accuracy_dis", 0.0)),
        held_out_bias_amb=metrics.get("bias_score_amb"),
        held_out_false_abstention=float(metrics.get("false_abstention_rate", 0.0)),
        held_out_expert_usage=expert_usage,
    )


def _save_summary(summary: LOCOAblationSummary, save_dir: str) -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "per_fold": {k: asdict(v) for k, v in summary.per_fold.items()},
        "aggregate": summary.aggregate(),
    }
    (out_dir / "loco_ablation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"  [저장] {out_dir / 'loco_ablation.json'}")
