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
from src.models.override import (
    apply_per_condition_override,
    apply_threshold_override,
    search_optimal_threshold_per_condition,
)
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
    thresholds_used: Optional[dict[str, float]] = None   # per-condition τ (val에서 search)


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
    use_per_condition_threshold: bool = True,
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
        threshold: legacy 단일 τ. use_per_condition_threshold=False일 때만 사용.
        use_per_condition_threshold: True면 fold마다 val_subset에서 ambig/disambig
            별 τ를 grid search한 뒤 held-out에 적용. False면 단일 τ 사용 (legacy).
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

        # train_records의 일부를 val로 분리 — stratified by (category, context_condition)
        # 이전 버전: 단순 [:split_idx] 슬라이싱이라 val이 마지막 카테고리에 편중되는
        # 버그가 있었음. seed를 fold마다 달리해서 fold별 noise도 분리.
        train_subset, val_subset = _stratified_split(
            train_records,
            val_ratio=0.15,
            seed=hash(held_out) & 0xFFFFFFFF,
        )

        train_ds = SignalsDataset(train_subset, embeddings)
        val_ds = SignalsDataset(val_subset, embeddings)
        held_ds = SignalsDataset(held_records, embeddings)

        # 2) MoE 학습
        model = MoEAggregator(signal_dim=7, embed_dim=embed_dim)
        out = train_moe(train_ds, val_ds, model, train_config)

        # 3) Per-condition threshold search on val_subset (메인 평가와 일관)
        if use_per_condition_threshold:
            val_predictions = _predict_for_threshold_search(
                model, val_subset, embeddings, instances_by_id,
            )
            if val_predictions:
                pc_search = search_optimal_threshold_per_condition(
                    val_predictions,
                    metric_amb="accuracy_amb",
                    metric_dis="accuracy_dis",
                )
                fold_thresholds = pc_search.thresholds
            else:
                fold_thresholds = {"ambig": threshold, "disambig": threshold}
        else:
            fold_thresholds = {"ambig": threshold, "disambig": threshold}

        # 4) Held-out 평가
        fold_res = _evaluate_held_out(
            model=model,
            held_dataset=held_ds,
            held_records=held_records,
            instances_by_id=instances_by_id,
            thresholds=fold_thresholds,
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
    thresholds: dict[str, float],
    held_out_category: str,
    n_train: int,
    best_val_loss: float,
) -> LOCOFoldResult:
    """Held-out 카테고리에서 BBQ 평가 (per-condition override)."""
    model.eval()
    device = next(model.parameters()).device

    final_preds: list[int] = []
    final_items: list[dict] = []
    expert_uses: list[list[float]] = []

    # held_records의 unique_id를 키로 — example_id 단독은 cross-category 충돌 위험
    def _ukey(r):
        u = r.get("unique_id")
        if u:
            return u
        raw = r.get("example_id")
        c = r.get("category", "_unknown")
        return f"{c}::{raw}" if raw is not None else None

    rec_by_ukey = {_ukey(r): r for r in held_records if _ukey(r) is not None}
    with torch.inference_mode():
        for ds_item in held_dataset:
            ukey = ds_item.get("example_id")  # SignalsDataset은 ex_id 슬롯에 unique_id 저장
            rec = rec_by_ukey.get(ukey) if ukey is not None else None
            if rec is None:
                continue

            sig = ds_item["signals"].to(device)
            emb = ds_item["embedding"].to(device)
            out = model(sig.unsqueeze(0), emb.unsqueeze(0))
            p = float(out.p.item())
            expert_uses.append(out.gate_w[0].cpu().tolist())

            # primary_answer + per-condition override
            primary = int(rec["primary_answer"])
            inst = instances_by_id.get(ukey, {})

            override = apply_per_condition_override(
                primary_answer=primary, p_score=p, item=inst, thresholds=thresholds,
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
        thresholds_used=dict(thresholds),
    )


def _predict_for_threshold_search(
    model: MoEAggregator,
    val_records: list[dict],
    embeddings: dict[str, torch.Tensor],
    instances_by_id: dict[str, dict],
) -> list[dict]:
    """
    val_records에 대해 MoE 추론 → search_optimal_threshold_per_condition 입력 형식.

    embeddings/instances_by_id는 unique_id (composite key) 기반.
    """
    from src.models.moe_aggregator import signals_dict_to_tensor

    model.eval()
    device = next(model.parameters()).device
    out: list[dict] = []
    with torch.inference_mode():
        for rec in val_records:
            ukey = rec.get("unique_id")
            if not ukey:
                raw_id = rec.get("example_id")
                cat = rec.get("category", "_unknown")
                ukey = f"{cat}::{raw_id}" if raw_id is not None else None
            if not ukey or ukey not in embeddings:
                continue
            sig = signals_dict_to_tensor(rec.get("signals", {})).unsqueeze(0).to(device)
            emb = embeddings[ukey].to(torch.float32).unsqueeze(0).to(device)
            res = model(sig, emb)
            p = float(res.p.item())
            primary = int(rec.get("primary_answer", -1))
            inst = instances_by_id.get(ukey, {})
            out.append({"primary_answer": primary, "p_score": p, "item": inst})
    return out


def _stratified_split(
    records: list[dict],
    val_ratio: float,
    seed: int,
    stratify_keys: tuple[str, ...] = ("category", "context_condition"),
) -> tuple[list[dict], list[dict]]:
    """LOCO 내부 train/val 분할 (stratified + shuffled)."""
    import random

    rng = random.Random(seed)
    by_stratum: dict[tuple, list[dict]] = {}
    for rec in records:
        key = tuple(rec.get(k, "_unknown") for k in stratify_keys)
        by_stratum.setdefault(key, []).append(rec)

    train: list[dict] = []
    val: list[dict] = []
    for group in by_stratum.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) >= 2 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


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
