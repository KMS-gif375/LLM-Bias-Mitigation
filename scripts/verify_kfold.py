#!/usr/bin/env python3
"""
5-fold cross-validation: 모든 9000 instances를 한 번씩 test로 사용 → 통계 신뢰도 ↑.

각 fold:
    test  = 1/5 (~1773 instances, 별도 평가)
    train = 4/5 중 90% (~6383, MoE 학습)
    val   = 4/5 중 10% (~709, threshold calibration)

→ 5 fold 합치면 effective test = 9000 (각 인스턴스 1회 평가)
→ bias_amb 분모 = 4432 (전체 ambig)로 충분

3 seeds × 5 folds = 15 MoE 학습 (~30s each = ~8min on Mac MPS).

CLI:
    python scripts/verify_kfold.py --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stratified_kfold_indices(records, k=5, seed=42):
    """category × context_condition stratified k-fold split."""
    from sklearn.model_selection import StratifiedKFold

    strat = [
        f"{r.get('category','_unk')}::{r.get('context_condition','_unk')}"
        for r in records
    ]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    idxs = np.arange(len(records))
    for tr, te in skf.split(idxs, strat):
        folds.append((tr.tolist(), te.tolist()))
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/v2/verify_kfold.json")
    args = parser.parse_args()

    from run_pipeline import _collect_records_and_embeddings, _instances_by_id, _moe_predict_all
    from src.models.moe_aggregator import MoEAggregator
    from src.models.trainer import SignalsDataset, TrainConfig, train_moe
    from src.models.override import (
        apply_threshold_override,
        apply_threshold_override_per_condition,
        search_optimal_threshold,
        search_optimal_threshold_per_condition,
    )
    from src.evaluation.bbq_evaluator import evaluate_bbq
    from sklearn.model_selection import train_test_split
    import yaml

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    from src.utils.data_loader import DEFAULT_CATEGORIES_V2
    config["data"]["sampled_dir"] = "data/sampled_v2"
    config["data"]["samples_per_category"] = 1000
    config["data"]["categories"] = list(DEFAULT_CATEGORIES_V2)
    config["output"]["results_dir"] = "results/v2"

    class _Args:
        model = "main"
        categories = None
    args_ns = _Args()

    print("Loading records + embeddings...")
    records, embeddings = _collect_records_and_embeddings(config, args_ns)
    instances_by_id = _instances_by_id(records, config, args_ns)
    embed_dim = next(iter(embeddings.values())).shape[0]
    print(f"  {len(records)} records, {len(embeddings)} embeddings, dim={embed_dim}")

    all_results = []

    for seed in args.seeds:
        print(f"\n========== seed {seed} ==========")
        folds = stratified_kfold_indices(records, k=args.k, seed=seed)

        # 각 fold별 prediction 수집 → 마지막에 concat 후 metric
        all_test_preds_single = []  # final answer 인덱스
        all_test_preds_pc = []
        all_test_items = []
        per_fold_tau = []

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            train_recs = [records[i] for i in train_idx]
            test_recs = [records[i] for i in test_idx]

            # train 중 10%를 val로 sub-split (threshold tuning)
            tr_strat = [
                f"{r.get('category','_unk')}::{r.get('context_condition','_unk')}"
                for r in train_recs
            ]
            sub_train, sub_val = train_test_split(
                train_recs, test_size=0.1, random_state=seed, stratify=tr_strat,
            )

            train_ds = SignalsDataset(sub_train, embeddings)
            val_ds = SignalsDataset(sub_val, embeddings)

            model = MoEAggregator(signal_dim=7, embed_dim=embed_dim, num_experts=4,
                                  gating_hidden=64, expert_hidden=128)
            train_moe(train_ds, val_ds, model, TrainConfig(
                epochs=30, batch_size=32, lr=1e-3, weight_decay=1e-5,
                val_every=5, seed=seed,
            ))

            val_preds = _moe_predict_all(model, sub_val, embeddings, instances_by_id)
            test_preds = _moe_predict_all(model, test_recs, embeddings, instances_by_id)

            # τ tune on VAL only
            single = search_optimal_threshold(val_preds, threshold_range=(0.05, 0.95), step=0.025)
            per_cond = search_optimal_threshold_per_condition(
                val_preds, threshold_range=(0.05, 0.95), step=0.025,
            )

            # Apply to TEST
            for p in test_preds:
                r_s = apply_threshold_override(
                    p["primary_answer"], p["p_score"], p["item"], single.best_threshold,
                )
                r_p = apply_threshold_override_per_condition(
                    p["primary_answer"], p["p_score"], p["item"],
                    thresholds_by_cond=per_cond.thresholds, default_threshold=single.best_threshold,
                )
                all_test_preds_single.append(r_s["final_answer"])
                all_test_preds_pc.append(r_p["final_answer"])
                all_test_items.append(p["item"])

            per_fold_tau.append({
                "fold": fold_i,
                "single": single.best_threshold,
                "per_cond": per_cond.thresholds,
            })
            print(f"  fold {fold_i}: train={len(sub_train)} val={len(sub_val)} test={len(test_recs)} "
                  f"τ_single={single.best_threshold} τ_pc={per_cond.thresholds}")

        # 5 fold 합쳐서 전체 평가
        m_single = evaluate_bbq(all_test_preds_single, all_test_items)
        m_pc = evaluate_bbq(all_test_preds_pc, all_test_items)

        result = {
            "seed": seed,
            "k": args.k,
            "n_total_evaluated": len(all_test_items),
            "per_fold_tau": per_fold_tau,
            "test_single": m_single,
            "test_per_cond": m_pc,
        }
        all_results.append(result)
        print(f"\n  seed {seed} aggregated over {args.k} folds (n={len(all_test_items)}):")
        print(f"    [single  ] acc_amb={m_single['accuracy_amb']:.4f} acc_dis={m_single['accuracy_dis']:.4f} "
              f"bias_amb={m_single['bias_score_amb']} far={m_single['false_abstention_rate']:.4f}")
        print(f"    [per-cond] acc_amb={m_pc['accuracy_amb']:.4f} acc_dis={m_pc['accuracy_dis']:.4f} "
              f"bias_amb={m_pc['bias_score_amb']} far={m_pc['false_abstention_rate']:.4f}")

    # Aggregate across seeds
    print("\n" + "="*72)
    print(f" Aggregate ({args.k}-fold CV, {len(args.seeds)} seeds, effective test = 9000 per seed)")
    print("="*72)
    for variant in ["test_single", "test_per_cond"]:
        accs_amb = [r[variant]["accuracy_amb"] for r in all_results]
        accs_dis = [r[variant]["accuracy_dis"] for r in all_results]
        fars = [r[variant]["false_abstention_rate"] for r in all_results]
        biases = [
            r[variant]["bias_score_amb"]
            for r in all_results
            if isinstance(r[variant].get("bias_score_amb"), (int, float))
        ]
        label = "single τ" if variant == "test_single" else "per-cond τ"
        print(f"\n  [{label}]")
        print(f"    acc_amb: {np.mean(accs_amb):.4f} ± {np.std(accs_amb):.4f}")
        print(f"    acc_dis: {np.mean(accs_dis):.4f} ± {np.std(accs_dis):.4f}")
        print(f"    far:     {np.mean(fars):.4f} ± {np.std(fars):.4f}")
        if biases:
            print(f"    bias_amb: {np.mean(biases):.4f} ± {np.std(biases):.4f} (n_valid={len(biases)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=float))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
