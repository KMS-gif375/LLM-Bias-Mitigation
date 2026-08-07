#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_lowlabel_bootstrap.py — approximate reconstruction sensitivity audit for
the low-label hybrid-vs-condition-only experiment (Table 4).

Reconstruction protocol (CPU only, no LLM):
  * test membership per seed  : results/v2/clean_experiments/seed_*/test_predictions.jsonl
  * MoE retention score p     : saved p_score in the same file (identical model)
  * gating thresholds         : the AUDITED per-seed (conf_threshold, tau_risk)
                                recorded in low_label_metrics.csv (no re-tuning)
  * low-label condition clf   : balanced logistic regression on MiniLM embeddings,
                                retrained on a stratified label fraction of the
                                non-test pool (random_state=seed)
Fidelity anchor: per-seed reconstructed metrics are compared with the original
low_label_metrics.csv rows. Because the original low-label run used a different
feature/training-pool construction and did not save exact per-example decisions,
this reconstruction must not be presented as an exact inferential test of the
reported Table 4 means.
Sensitivity summary: per seed and label fraction, paired example-level
bootstrap (10,000 resamples) of hybrid-minus-condition-only deltas on Acc_amb /
Acc_dis / FAR. ``p_two_sided`` is a descriptive two-sided bootstrap tail mass
with an add-one Monte Carlo correction; it is not a sign-flipping test or an
exact p-value for Table 4. The report uses 0.05/9 only as a family-wise
reference threshold across the fraction-metric combinations.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.utils.data_loader import load_split  # noqa: E402

SEEDS = [42, 123, 456, 789, 999]
FRACS = [0.01, 0.05, 0.10]
NBOOT = 10000
OUT = REPO / "results/v2/reviewer_audits/r2_audits"
ARTIFACT_SCHEMA_VERSION = 2
P_VALUE_METHOD = "paired_bootstrap_tail_mass_add_one"
INFERENCE_SCOPE = "approximate_reconstruction_sensitivity_only"


def cond01(c):
    c = str(c).lower()
    return 0 if c.startswith("ambig") else 1


def unknown_idx(item):
    info = item.get("answer_info", {}) or {}
    for i in range(3):
        a = info.get(f"ans{i}", [])
        if len(a) >= 2 and a[1] == "unknown":
            return i
    return -1


def descriptive_bootstrap_tail_pvalue(
    bootstrap_deltas: np.ndarray,
    null_value: float = 0.0,
) -> float:
    """Return a finite-sample-corrected two-sided bootstrap tail mass.

    This descriptive quantity is deliberately not labeled as an exact
    randomization/sign-flipping p-value.
    """
    deltas = np.asarray(bootstrap_deltas, dtype=float).reshape(-1)
    if deltas.size == 0:
        raise ValueError("bootstrap_deltas must not be empty")
    if not np.isfinite(deltas).all():
        raise ValueError("bootstrap_deltas must contain only finite values")
    denominator = deltas.size + 1
    lower = (np.count_nonzero(deltas <= null_value) + 1) / denominator
    upper = (np.count_nonzero(deltas >= null_value) + 1) / denominator
    return float(min(1.0, 2.0 * min(lower, upper)))


def main():
    pool = {}
    for s in ("train", "val", "test"):
        for _, row in load_split(REPO / "data/sampled_v2", s).iterrows():
            d = row.to_dict()
            pool[f"{d['category']}::{int(d['example_id'])}"] = d
    emb = {}
    for f in glob.glob(str(REPO / "results/v2/signals/main/*_embeddings.pt")):
        cat = Path(f).name.replace("_embeddings.pt", "")
        for k, v in torch.load(f, map_location="cpu", weights_only=True).items():
            emb[f"{cat}::{int(k)}"] = v.numpy().astype(np.float32)
    prim = {}
    for f in glob.glob(str(REPO / "results/v2/signals/main/*_signals.jsonl")):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                prim[f"{r['category']}::{int(r['example_id'])}"] = int(r["primary_answer"])

    thr = pd.read_csv(REPO / "results/v2/hybrid_abstention_audits/low_label_metrics.csv")
    thr = thr[(thr.scope == "clean_low_label")]

    rows_out, boot_out = [], []
    for seed in SEEDS:
        tp = [json.loads(l) for l in (REPO / f"results/v2/clean_experiments/seed_{seed}/test_predictions.jsonl").read_text().splitlines() if l.strip()]
        test_uids = [r["uid"] for r in tp]
        p_moe = {r["uid"]: float(r["p_score"]) for r in tp}
        rest = [u for u in pool if u not in set(test_uids)]
        ys = {u: cond01(pool[u]["context_condition"]) for u in pool}
        strat = [f"{pool[u]['category']}|{ys[u]}" for u in rest]

        for frac in FRACS:
            # stratified label subset of the non-test pool (proxy for original draw)
            n_lab = max(20, int(round(frac * 6208)))
            lab_uids, _ = train_test_split(rest, train_size=n_lab, random_state=seed, stratify=strat)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
            clf.fit(np.stack([emb[u] for u in lab_uids]), np.array([ys[u] for u in lab_uids]))
            P = clf.predict_proba(np.stack([emb[u] for u in test_uids]))  # [:,1]=disambig
            row = thr[(thr.seed == seed) & (np.isclose(thr.label_frac, frac)) & (thr.system == "hybrid_uncertain_signal_fallback")]
            conf_t = float(row.conf_threshold.iloc[0]) if len(row) else 0.95
            tau_r = float(row.tau_risk.iloc[0]) if len(row) else 0.3

            yv = np.array([ys[u] for u in test_uids])
            lab = np.array([int(pool[u]["label"]) for u in test_uids])
            unk = np.array([unknown_idx(pool[u]) for u in test_uids])
            pri = np.array([prim.get(u, -1) for u in test_uids])
            pm = np.array([p_moe[u] for u in test_uids])
            pdis = (P[:, 1] >= 0.5)
            conf = P.max(axis=1)

            cond_final = np.where(pdis, pri, unk)
            hyb_final = np.where(conf >= conf_t, cond_final, np.where(pm >= tau_r, pri, unk))

            def metr(final):
                amb = yv == 0; dis = yv == 1
                return (float((final[amb] == lab[amb]).mean()),
                        float((final[dis] == lab[dis]).mean()),
                        float((final[dis] == unk[dis]).mean()))

            mc, mh = metr(cond_final), metr(hyb_final)
            rows_out.append({"artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                             "inference_scope": INFERENCE_SCOPE,
                             "seed": seed, "frac": frac,
                             "cond_amb": mc[0], "cond_dis": mc[1], "cond_far": mc[2],
                             "hyb_amb": mh[0], "hyb_dis": mh[1], "hyb_far": mh[2],
                             "conf_t": conf_t, "tau_r": tau_r, "n_lab": n_lab})
            print(f"[seed {seed} frac {frac:.2f}] cond {mc[0]:.4f}/{mc[1]:.4f}/{mc[2]:.4f}  "
                  f"hyb {mh[0]:.4f}/{mh[1]:.4f}/{mh[2]:.4f}  (conf={conf_t}, tau={tau_r})")

            # paired bootstrap over examples
            rng = np.random.default_rng(seed * 1000 + int(frac * 100))
            n = len(test_uids)
            deltas = np.zeros((NBOOT, 3))
            amb = yv == 0; dis = yv == 1
            for b in range(NBOOT):
                idx = rng.integers(0, n, n)
                fA, fC = hyb_final[idx], cond_final[idx]
                yb, lb, ub = yv[idx], lab[idx], unk[idx]
                a, d = yb == 0, yb == 1
                deltas[b, 0] = (fA[a] == lb[a]).mean() - (fC[a] == lb[a]).mean()
                deltas[b, 1] = (fA[d] == lb[d]).mean() - (fC[d] == lb[d]).mean()
                deltas[b, 2] = (fA[d] == ub[d]).mean() - (fC[d] == ub[d]).mean()
            for mi, mname in enumerate(["acc_amb", "acc_dis", "far"]):
                dm = deltas[:, mi]
                obs = {"acc_amb": mh[0] - mc[0], "acc_dis": mh[1] - mc[1], "far": mh[2] - mc[2]}[mname]
                lo, hi2 = np.percentile(dm, [2.5, 97.5])
                p = descriptive_bootstrap_tail_pvalue(dm)
                boot_out.append({
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "inference_scope": INFERENCE_SCOPE,
                    "p_value_method": P_VALUE_METHOD,
                    "seed": seed,
                    "frac": frac,
                    "metric": mname,
                    "delta": obs,
                    "ci_lo": float(lo),
                    "ci_hi": float(hi2),
                    "p_two_sided": p,
                })

    df = pd.DataFrame(rows_out); bf = pd.DataFrame(boot_out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "lowlabel_reconstructed_metrics.csv", index=False)
    bf.to_csv(OUT / "lowlabel_bootstrap.csv", index=False)

    print("\n=== fidelity anchor vs original low_label_metrics.csv (per-frac means) ===")
    for frac in FRACS:
        sub = df[df.frac == frac]
        o_h = thr[(np.isclose(thr.label_frac, frac)) & (thr.system == "hybrid_uncertain_signal_fallback")]
        o_c = thr[(np.isclose(thr.label_frac, frac)) & (thr.system == "simple_condition_only")]
        print(f"frac {frac:.2f}: recon hyb {sub.hyb_amb.mean():.4f}/{sub.hyb_dis.mean():.4f}/{sub.hyb_far.mean():.4f}"
              f"  orig hyb {o_h.accuracy_amb.mean():.4f}/{o_h.accuracy_dis.mean():.4f}/{o_h.false_abstention_rate.mean():.4f}")
        print(f"          recon cond {sub.cond_amb.mean():.4f}/{sub.cond_dis.mean():.4f}/{sub.cond_far.mean():.4f}"
              f"  orig cond {o_c.accuracy_amb.mean():.4f}/{o_c.accuracy_dis.mean():.4f}/{o_c.false_abstention_rate.mean():.4f}")

    max_drift = 0.0
    for frac in FRACS:
        sub = df[df.frac == frac]
        for sys, prefix in (("hybrid_uncertain_signal_fallback", "hyb"),
                            ("simple_condition_only", "cond")):
            orig = thr[(np.isclose(thr.label_frac, frac)) & (thr.system == sys)]
            for metric, col in (("accuracy_amb", f"{prefix}_amb"),
                                ("accuracy_dis", f"{prefix}_dis"),
                                ("false_abstention_rate", f"{prefix}_far")):
                max_drift = max(
                    max_drift,
                    abs(float(sub[col].mean()) - float(orig[metric].mean())),
                )
    validity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "exact_reconstruction": False,
        "max_mean_metric_drift": max_drift,
        "valid_for_exact_inference": False,
        "inference_scope": INFERENCE_SCOPE,
        "bootstrap_unit": "paired_test_example_within_seed",
        "bootstrap_resamples": NBOOT,
        "p_value_field": "p_two_sided",
        "p_value_method": P_VALUE_METHOD,
        "p_value_interpretation": (
            "Descriptive bootstrap tail mass with add-one correction; not a "
            "sign-flipping/randomization p-value and not an exact test of Table 4."
        ),
        "bonferroni_reference_alpha": 0.05 / 9,
        "note": (
            "Approximate sensitivity analysis only; original per-example "
            "low-label decisions were not retained."
        ),
    }
    (OUT / "lowlabel_bootstrap_validity.json").write_text(
        json.dumps(validity, indent=2)
    )

    print("\n=== approximate bootstrap sensitivity (not an exact test of Table 4) ===")
    for frac in FRACS:
        for m in ["acc_amb", "acc_dis", "far"]:
            sub = bf[(bf.frac == frac) & (bf.metric == m)]
            print(f"frac {frac:.2f} {m:8s}: mean Δ={sub.delta.mean():+.4f}  "
                  f"per-seed p_max={sub.p_two_sided.max():.4g}  all_seeds_sig(α=0.0056)={bool((sub.p_two_sided < 0.0056).all())}")
    print(f"[fidelity] max mean-metric drift={max_drift:.4f}; exact inference disabled")
    print(f"\n[done] wrote {OUT}/lowlabel_reconstructed_metrics.csv, "
          "lowlabel_bootstrap.csv, and validity metadata")


if __name__ == "__main__":
    main()
