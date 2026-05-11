#!/usr/bin/env python3
"""
Honest validation: train/val/test split을 통해 threshold가 test set에서도 유효한지 검증.

기존 Stage 7은 같은 9000 인스턴스에서 MoE 학습 + threshold calibration + 평가 → optimistic.
여기선 70/15/15 stratified split → MoE는 train로 학습, threshold는 val에서 search, test에서만 평가.

비교:
    [A] Full data (Stage 7 방식): 9000 instances 전부 사용 (optimistic)
    [B] Clean split: train=6300 (학습) / val=1330 (τ search) / test=1330 (평가)

CLI:
    python scripts/verify_split.py --seeds 42 123 456 789 999
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stratified_split_3way(records, embeddings, train_frac=0.7, val_frac=0.15, seed=42):
    """category × context_condition stratified split."""
    from sklearn.model_selection import train_test_split

    n = len(records)
    # Stratify key: category::context_condition
    strat = [
        f"{r.get('category','_unk')}::{r.get('context_condition','_unk')}"
        for r in records
    ]
    idxs = list(range(n))
    # Step 1: train vs (val+test)
    train_idx, rest_idx = train_test_split(
        idxs, train_size=train_frac, random_state=seed, stratify=strat,
    )
    rest_strat = [strat[i] for i in rest_idx]
    # Step 2: val vs test
    val_size = val_frac / (1 - train_frac)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_size, random_state=seed, stratify=rest_strat,
    )

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]
    return train_records, val_records, test_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--out", type=str, default="results/v2/verify_split.json")
    args = parser.parse_args()

    from run_pipeline import _collect_records_and_embeddings, _instances_by_id, _moe_predict_all, _stage_output_dir
    from src.models.moe_aggregator import MoEAggregator
    from src.models.trainer import SignalsDataset, TrainConfig, train_moe
    from src.models.override import (
        apply_threshold_override,
        apply_threshold_override_per_condition,
        search_optimal_threshold,
        search_optimal_threshold_per_condition,
    )
    from src.evaluation.bbq_evaluator import evaluate_bbq
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
    print(f"  {len(records)} records, {len(embeddings)} embeddings")

    all_results = []
    for seed in args.seeds:
        print(f"\n========== seed {seed} ==========")
        train_recs, val_recs, test_recs = stratified_split_3way(
            records, embeddings, train_frac=0.7, val_frac=0.15, seed=seed,
        )
        print(f"  split: train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

        # MoE 학습 (train만)
        train_ds = SignalsDataset(train_recs, embeddings)
        val_ds = SignalsDataset(val_recs, embeddings)

        embed_dim = next(iter(embeddings.values())).shape[0] if embeddings else 4096
        model = MoEAggregator(signal_dim=7, embed_dim=embed_dim, num_experts=4,
                              gating_hidden=64, expert_hidden=128)
        out = train_moe(train_ds, val_ds, model, TrainConfig(
            epochs=30, batch_size=32, lr=1e-3, weight_decay=1e-5,
            val_every=5, seed=seed,
        ))
        print(f"  best_val_loss={out.get('best_val_loss', 'NA')}")

        # 학습된 MoE로 val + test 추론
        val_preds = _moe_predict_all(model, val_recs, embeddings, instances_by_id)
        test_preds = _moe_predict_all(model, test_recs, embeddings, instances_by_id)

        # τ search on VAL only
        single = search_optimal_threshold(val_preds, threshold_range=(0.05, 0.95), step=0.025)
        per_cond = search_optimal_threshold_per_condition(
            val_preds, threshold_range=(0.05, 0.95), step=0.025,
        )
        tau_pc = per_cond.thresholds  # {"ambig": ..., "disambig": ...}

        # Evaluate on TEST
        test_items = [p["item"] for p in test_preds]

        # single tau
        final_single = []
        for p in test_preds:
            r = apply_threshold_override(p["primary_answer"], p["p_score"], p["item"], single.best_threshold)
            final_single.append(r["final_answer"])
        m_single = evaluate_bbq(final_single, test_items)

        # per-condition
        final_pc = []
        for p in test_preds:
            r = apply_threshold_override_per_condition(
                p["primary_answer"], p["p_score"], p["item"],
                thresholds_by_cond=tau_pc, default_threshold=single.best_threshold,
            )
            final_pc.append(r["final_answer"])
        m_pc = evaluate_bbq(final_pc, test_items)

        result = {
            "seed": seed,
            "n_train": len(train_recs), "n_val": len(val_recs), "n_test": len(test_recs),
            "best_val_loss": out.get("best_val_loss"),
            "tau_single": single.best_threshold,
            "tau_per_cond": tau_pc,
            "test_single": m_single,
            "test_per_cond": m_pc,
        }
        all_results.append(result)
        print(f"  τ single={single.best_threshold} per-cond={tau_pc}")
        print(f"  [single]   acc_amb={m_single['accuracy_amb']:.4f} acc_dis={m_single['accuracy_dis']:.4f} "
              f"bias_amb={m_single['bias_score_amb']} far={m_single['false_abstention_rate']:.4f}")
        print(f"  [per-cond] acc_amb={m_pc['accuracy_amb']:.4f} acc_dis={m_pc['accuracy_dis']:.4f} "
              f"bias_amb={m_pc['bias_score_amb']} far={m_pc['false_abstention_rate']:.4f}")

    # Aggregate
    print("\n" + "="*72)
    print(" Aggregate (test set, 별도 split)")
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
