#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_lowlabel_bootstrap.py — paired example-level bootstrap for the low-label
hybrid-vs-condition-only claim (Table 4), reviewer requirement R1/#2.

Reconstruction protocol (CPU only, no LLM):
  * test membership per seed  : results/v2/clean_experiments/seed_*/test_predictions.jsonl
  * MoE retention score p     : saved p_score in the same file (identical model)
  * gating thresholds         : the AUDITED per-seed (conf_threshold, tau_risk)
                                recorded in low_label_metrics.csv (no re-tuning)
  * low-label condition clf   : balanced logistic regression on MiniLM embeddings,
                                retrained on a stratified label fraction of the
                                non-test pool (random_state=seed)
Fidelity anchor: per-seed reconstructed metrics are compared with the original
low_label_metrics.csv rows before any inference is drawn.
Test: per seed and label fraction, paired example-level bootstrap (10,000
resamples) of hybrid-minus-condition-only deltas on Acc_amb / Acc_dis / FAR;
two-sided p via sign flipping, Bonferroni over the 9 fraction-metric tests.
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
            rows_out.append({"seed": seed, "frac": frac,
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
                p = 2 * min((dm <= 0).mean(), (dm >= 0).mean())
                boot_out.append({"seed": seed, "frac": frac, "metric": mname, "delta": obs,
                                 "ci_lo": float(lo), "ci_hi": float(hi2), "p_two_sided": float(min(1.0, p))})

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

    print("\n=== bootstrap summary (Bonferroni alpha = 0.05/9 = 0.0056) ===")
    for frac in FRACS:
        for m in ["acc_amb", "acc_dis", "far"]:
            sub = bf[(bf.frac == frac) & (bf.metric == m)]
            print(f"frac {frac:.2f} {m:8s}: mean Δ={sub.delta.mean():+.4f}  "
                  f"per-seed p_max={sub.p_two_sided.max():.4g}  all_seeds_sig(α=0.0056)={bool((sub.p_two_sided < 0.0056).all())}")
    print(f"\n[done] wrote {OUT}/lowlabel_reconstructed_metrics.csv and lowlabel_bootstrap.csv")


if __name__ == "__main__":
    main()
