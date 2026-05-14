"""
Minimal-Core Signal Ablation (Q2).

Section 8.1 signal_ablation 은 각 신호 leave-one-out (single drop).
본 ablation 은 신호 *부분집합 (subset)* 으로 학습한 MoE 의 성능 비교.
"메소드에 정말 필요한 minimal core 가 무엇인지" 확인.

테스트 variants:
    - core_4: {s1, s3, s4, s6}  ← signal_ablation 에서 contribution 양수
    - +s5:    {s1, s3, s4, s5, s6}
    - +s7:    {s1, s3, s4, s6, s7}
    - +s2:    {s1, s2, s3, s4, s6}
    - +s5+s7: {s1, s3, s4, s5, s6, s7}
    - full_7: 모두

각 variant × 5 seeds = 30 runs. Mac M4 Pro CPU ~ 1시간.

CLI:
    python -m src.analysis.minimal_core_ablation \
        --signals-dir results/v2_runpod/signals/main \
        --seeds 42,123,456,789,999
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("minimal_core_ablation")


SIGNAL_NAMES = [
    "s1_evidence", "s2_counterfactual", "s3_confidence",
    "s4_consistency", "s5_bias_head", "s6_prompt_sensitivity",
    "s7_sae_feature",
]


# Variants: 각 variant 마다 mask 할 신호 인덱스 list (0-based)
VARIANTS = {
    "full_7":           [],                  # 모두 keep
    "core_4_s1346":     [1, 4, 6],           # mask s2, s5, s7
    "core5_plus_s5":    [1, 6],              # mask s2, s7
    "core5_plus_s7":    [1, 4],              # mask s2, s5
    "core5_plus_s2":    [4, 6],              # mask s5, s7
    "core6_plus_s5_s7": [1],                 # mask s2 only
}


def _multi_mask_records(records: list[dict], mask_indices: list[int]) -> list[dict]:
    """signals 의 mask_indices 위치 값을 0으로 치환한 새 record list 반환."""
    if not mask_indices:
        return records
    masked = []
    for rec in records:
        new_rec = dict(rec)
        sig = dict(rec.get("signals", {}))
        for idx in mask_indices:
            sig[SIGNAL_NAMES[idx]] = 0.0
        new_rec["signals"] = sig
        masked.append(new_rec)
    return masked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals-dir", type=str,
                        default="results/v2_runpod/signals/main")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,999")
    parser.add_argument("--out-dir", type=str,
                        default="results/v2_runpod/qualitative/minimal_core")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dotenv import load_dotenv
    load_dotenv()

    # 1) signals + embeddings 로드
    from run_pipeline import (
        _collect_records_and_embeddings,
        _stratified_three_way_split,
    )
    import yaml
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    class _Args:
        model = "main"
        categories = None

    records, embeddings = _collect_records_and_embeddings(config, _Args())
    logger.info(f"Loaded {len(records)} records + {len(embeddings)} embeddings")

    # 2) Train: MoE for each (variant, seed)
    from src.models.trainer import SignalsDataset, TrainConfig, train_moe
    from src.models.moe_aggregator import MoEAggregator

    # Infer embed_dim from first embedding
    sample_emb = next(iter(embeddings.values()))
    embed_dim = sample_emb.shape[-1]
    logger.info(f"embed_dim={embed_dim}")

    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []

    for seed in seeds:
        train_records, val_records, test_records = _stratified_three_way_split(
            records, val_ratio=0.15, test_ratio=0.15, seed=seed,
        )
        logger.info(f"\n=== Seed {seed} | train={len(train_records)} val={len(val_records)} test={len(test_records)} ===")

        for vname, mask_idx in VARIANTS.items():
            try:
                train_m = _multi_mask_records(train_records, mask_idx)
                val_m = _multi_mask_records(val_records, mask_idx)
                test_m = _multi_mask_records(test_records, mask_idx)

                torch.manual_seed(seed)
                np.random.seed(seed)

                train_ds = SignalsDataset(train_m, embeddings, require_all=False)
                val_ds = SignalsDataset(val_m, embeddings, require_all=False)

                cfg = TrainConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=1e-3,
                    weight_decay=0.01,
                    lambda_bias=0.5,
                    lambda_lb=0.1,
                    early_stop_patience=5,
                    seed=seed,
                    device="cpu",   # 작은 MoE — CPU 충분 + device mismatch 방지
                )

                # MoE 직접 생성
                model = MoEAggregator(
                    signal_dim=7,
                    embed_dim=embed_dim,
                    num_experts=4,
                    gating_hidden=64,
                    expert_hidden=128,
                )

                train_log = train_moe(train_ds, val_ds, model, cfg)

                # Test 평가
                test_ds = SignalsDataset(test_m, embeddings, require_all=False)
                test_loss = 0.0
                n_test = len(test_ds)
                model.eval()
                with torch.no_grad():
                    p_all, y_all = [], []
                    from torch.utils.data import DataLoader
                    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
                    for batch in test_loader:
                        s = batch["signals"]
                        e = batch["embedding"]
                        y = batch["label"].float()
                        out = model(s, e)
                        p_all.extend(out.p.cpu().tolist())
                        y_all.extend(y.cpu().tolist())
                # 단순 BCE test loss
                if p_all:
                    p_t = torch.tensor(p_all).clamp(1e-7, 1-1e-7)
                    y_t = torch.tensor(y_all)
                    test_loss = float(torch.nn.functional.binary_cross_entropy(p_t, y_t).item())

                result = {
                    "variant": vname,
                    "mask_indices": mask_idx,
                    "n_signals_kept": 7 - len(mask_idx),
                    "seed": seed,
                    "best_val_loss": float(train_log.get("best_val_loss", -1)),
                    "best_epoch": int(train_log.get("best_epoch", -1)),
                    "test_loss": test_loss,
                    "n_test": n_test,
                }
                all_results.append(result)
                logger.info(
                    f"  [{vname:25s}] val_loss={result['best_val_loss']:.4f} "
                    f"test_loss={test_loss:.4f}"
                )
            except Exception as e:
                logger.warning(f"  [{vname}] seed {seed} FAILED: {e}")

    # 3) Aggregate
    by_variant: dict = {}
    for r in all_results:
        by_variant.setdefault(r["variant"], []).append(r)

    aggregate = {}
    for vname, runs in by_variant.items():
        vlosses = [r["best_val_loss"] for r in runs]
        tlosses = [r["test_loss"] for r in runs]
        aggregate[vname] = {
            "n_seeds": len(runs),
            "n_signals_kept": runs[0]["n_signals_kept"],
            "val_loss_mean": float(np.mean(vlosses)),
            "val_loss_std": float(np.std(vlosses)),
            "test_loss_mean": float(np.mean(tlosses)),
            "test_loss_std": float(np.std(tlosses)),
        }

    # 4) Save JSON
    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps({
        "variants": VARIANTS,
        "signal_names": SIGNAL_NAMES,
        "aggregate": aggregate,
        "per_seed": all_results,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seeds": seeds,
        },
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    logger.info(f"[저장] {out_json}")

    # 5) Markdown
    md = ["# Minimal-Core Signal Ablation (Q2)", ""]
    md.append("MoE 를 다음 신호 subset 으로 학습 → minimal core 검증.")
    md.append("")
    md.append("## Results (5 seeds 평균)")
    md.append("")
    md.append("| Variant | n signals | val_loss | test_loss |")
    md.append("|---|---|---|---|")
    full = aggregate.get("full_7", {})
    for vname in VARIANTS:
        if vname not in aggregate:
            continue
        a = aggregate[vname]
        marker = " ⭐" if vname == "full_7" else ""
        md.append(
            f"| {vname}{marker} | {a['n_signals_kept']} | "
            f"{a['val_loss_mean']:.4f} ± {a['val_loss_std']:.4f} | "
            f"{a['test_loss_mean']:.4f} ± {a['test_loss_std']:.4f} |"
        )
    md.append("")
    md.append("## 해석")
    md.append("")
    if "core_4_s1346" in aggregate and "full_7" in aggregate:
        c4 = aggregate["core_4_s1346"]["val_loss_mean"]
        f7 = aggregate["full_7"]["val_loss_mean"]
        if abs(c4 - f7) < 0.005:
            md.append(f"- **core_4 (s1+s3+s4+s6) val_loss {c4:.4f} ≈ full_7 {f7:.4f}** → s2/s5/s7 정말 redundant.")
        else:
            md.append(f"- core_4 val_loss {c4:.4f} vs full_7 {f7:.4f}, Δ={c4-f7:+.4f}.")
    md.append("- +s5/+s7 추가 시 향상 폭이 작으면 → 해당 신호 marginal.")
    md.append("- minimal core 의 경우 paper 에서 '4-signal core MoE' 로 streamlined version 제시 가능.")
    md.append("")

    out_md = out_dir / "results.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"[저장] {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
