#!/usr/bin/env python3
"""
실제 데이터 누설 정량화: MoE 학습/평가 데이터 overlap 측정.

검사 대상:
    [A] MoE 학습 데이터 vs Stage 4 평가 데이터
    [B] Threshold τ tuning 데이터 vs 평가 데이터
    [C] Bias-head 식별 샘플 vs 평가 데이터
    [D] SAE feature 식별 데이터 vs 평가 데이터
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.utils.data_loader import DEFAULT_CATEGORIES_V2
    from run_pipeline import _collect_records_and_embeddings
    from sklearn.model_selection import train_test_split

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    config["data"]["sampled_dir"] = "data/sampled_v2"
    config["data"]["samples_per_category"] = 1000
    config["data"]["categories"] = list(DEFAULT_CATEGORIES_V2)
    config["output"]["results_dir"] = "results/v2"

    class _Args:
        model = "main"
        categories = None
    args_ns = _Args()

    records, _ = _collect_records_and_embeddings(config, args_ns)
    n = len(records)
    ex_ids_all = set(f"{r.get('category','_unk')}::{r.get('example_id')}" for r in records)

    print("=" * 72)
    print(f" 데이터 누설 검사 (v2, {n} records)")
    print("=" * 72)

    # [A] Stage 3 MoE train/val split (현재 run_pipeline 방식 시뮬레이션)
    print("\n[A] MoE 학습 → Stage 4 평가 누설")
    cats = [r.get("category", "_unk") for r in records]
    train_idx, val_idx = train_test_split(
        list(range(n)), test_size=0.2, random_state=42, stratify=cats,
    )
    train_ex_ids = set(f"{records[i].get('category','_unk')}::{records[i].get('example_id')}" for i in train_idx)
    print(f"  Stage 3 train: {len(train_idx)} ({len(train_idx)/n*100:.1f}%)")
    print(f"  Stage 3 val:   {len(val_idx)} ({len(val_idx)/n*100:.1f}%)")
    print(f"  Stage 4 eval:  {n} (ALL records — train 포함)")
    overlap = train_ex_ids & ex_ids_all
    print(f"  → Stage 4 eval 중 MoE가 학습한 instance: {len(overlap)} ({len(overlap)/n*100:.1f}%)")
    print(f"  → 진짜 unseen test로 평가된 instance:    {n - len(overlap)} ({(n-len(overlap))/n*100:.1f}%)")
    print(f"  🚨 LEAK: 80% instances이 MoE에 의해 학습된 상태로 metric 계산됨")

    # [B] Threshold τ tuning
    print("\n[B] Threshold τ tuning → 평가 누설")
    # run_pipeline.py:471-481 — search_optimal_threshold(val_predictions, ...) 에서 val_predictions는 _moe_predict_all(model, records, ...)
    # 즉 9000 전부에 대해 추론 → 9000으로 τ search → 9000으로 평가
    print(f"  τ search input: 9000 instances (all)")
    print(f"  Final eval:     9000 instances (same)")
    print(f"  🚨 LEAK: τ는 평가 set에 over-tune됨")

    # [C] Bias-head identification
    print("\n[C] Bias-head 식별 (verify_bias_heads.py)")
    print(f"  Pool: 9 categories × first 50 BBQ items = ~450")
    print(f"  Selected: first 20 items × LLM forward → top-20 (layer, head) 추출")
    print(f"  Output capacity: 20 (layer, head) 페어 = 작은 정보")
    print(f"  → 이 20개 instance가 9000 평가 pool에 포함됨 (0.22% overlap)")
    print(f"  ⚠ MINOR LEAK: 선택된 head는 학습 데이터 영향 받음, 영향 작음 (n=20, 정보=indices)")

    # [D] SAE feature identification
    print("\n[D] SAE bias feature 식별 (sae_ablation.identify_bias_features_*)")
    print(f"  Pool: 모든 9000 instances의 SAE activations")
    print(f"  Selected: top-50 features (out of ~32K = 0.15%)")
    print(f"  → 전체 9000으로 feature 선택")
    print(f"  ⚠ MINOR LEAK: 선택된 feature는 학습+평가 데이터 모두에서 도출, 출력 정보 capacity 작음")

    # [E] 정량 비교 (이미 가지고 있는 결과)
    print("\n[E] 누설 magnitude 정량화 (verify_split 결과 대비)")
    print(f"  Stage 4 (leaked):  acc_amb=0.999  acc_dis=0.875  far=0.075")
    print(f"  Verify split:      acc_amb=0.988  acc_dis=0.869  far=0.081")
    print(f"  진짜 generalization 손실: acc_amb -1.1pp, acc_dis -0.6pp, far +0.6pp")
    print(f"  → 누설 효과 작음 but 존재. 페이퍼엔 split 수치 보고 권장.")


if __name__ == "__main__":
    main()
