#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_aspire_selfeval.py  —  ASPIRE-style LEARNED SELF-EVALUATION baseline (Reviewer Q2 / W3).

ASPIRE (Chen et al., Findings of EMNLP 2023) learns a self-evaluation signal to improve
selective-prediction ranking. This script provides a FROZEN-MODEL analogue: instead of
fine-tuning a self-evaluation head, we train a small classifier on the model's own saved
inference-time signals (s1..s7) to predict per-example correctness, and compare its
risk-coverage ranking (AURC / E-AURC / AUROC) against the existing MoE retention score and
the chosen-softmax confidence on the SAME clean test split used for the MoE.

No LLM is needed — it reuses saved signals + MoE predictions. Runs on CPU.

Inputs (already in the repo):
  results/v2/signals/main/<Category>_signals.jsonl        # s1..s7, label, primary_answer
  results/v2/clean_experiments/seed_<S>/test_predictions.jsonl  # uid, p_score (MoE retention)

USAGE:
  ./venv/bin/python scripts/run_aspire_selfeval.py \
      --out results/v2/reviewer_audits/aspire_selfeval
Writes a per-seed CSV + an aggregated summary (mean +/- std across seeds) and prints a table.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
SIGNAL_KEYS = ["s1_evidence", "s2_counterfactual", "s3_confidence", "s4_consistency",
               "s5_bias_head", "s6_prompt_sensitivity", "s7_sae_feature"]
SEEDS = [42, 123, 456, 789, 999]


# ---- AURC / E-AURC (matches scripts/run_minor_revision_audits.py:aurc_and_eaurc) ----
def aurc_and_eaurc(correct, scores):
    """correct: 1=correct,0=wrong. scores: higher = more confident kept. Lower AURC better."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    c = np.asarray(correct, dtype=float)[order]
    n = len(c)
    cum_err = np.cumsum(1.0 - c)
    risk = cum_err / np.arange(1, n + 1)
    aurc = float(risk.mean())
    # oracle: rank all correct first
    c_oracle = np.sort(c)[::-1]
    cum_err_o = np.cumsum(1.0 - c_oracle)
    risk_o = cum_err_o / np.arange(1, n + 1)
    eaurc = float(aurc - risk_o.mean())
    return aurc, eaurc


def load_signals():
    """example_id(str) -> {signals[7], label, primary, correct, cond}."""
    idx = {}
    for f in glob.glob(str(REPO / "results/v2/signals/main/*_signals.jsonl")):
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sig = r.get("signals", {})
            vec = [float(sig.get(k, 0.0) or 0.0) for k in SIGNAL_KEYS]
            try:
                prim = int(r.get("primary_answer"))
                lab = int(r.get("label"))
            except (TypeError, ValueError):
                continue
            key = f"{r.get('category')}::{r.get('example_id')}"
            idx[key] = {
                "vec": vec, "primary": prim, "label": lab,
                "correct": int(prim == lab), "cond": r.get("context_condition"),
            }
    return idx


def load_moe_test(seed):
    """uid(str) -> p_score, for the clean test split of this seed."""
    f = REPO / f"results/v2/clean_experiments/seed_{seed}/test_predictions.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        uid = str(r.get("uid", r.get("example_id", "")))
        if "p_score" in r:
            out[uid] = {"p": float(r["p_score"]), "primary": r.get("primary_answer")}
    return out


def match_uid(uid, sig_idx):
    """test_predictions uid is 'Category::example_id', matching the signals key."""
    return uid if uid in sig_idx else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/v2/reviewer_audits/aspire_selfeval"))
    ap.add_argument("--features", default="signals",
                    help="signals | signals_primary (adds primary-answer one-hot)")
    args = ap.parse_args()

    sig_idx = load_signals()
    print(f"[load] {len(sig_idx)} signal records")
    if not sig_idx:
        raise SystemExit("no signals found under results/v2/signals/main/")

    def feats(rec):
        x = list(rec["vec"])
        if args.features == "signals_primary":
            oh = [0.0, 0.0, 0.0, 0.0]
            p = rec["primary"]
            oh[p + 1 if p in (-1, 0, 1, 2) else 0] = 1.0
            x = x + oh
        return x

    rows = []
    for seed in SEEDS:
        moe = load_moe_test(seed)
        if not moe:
            print(f"[seed {seed}] no test_predictions — skipped")
            continue
        # resolve test uids -> signal keys
        test_keys, miss = {}, 0
        for uid, m in moe.items():
            k = match_uid(uid, sig_idx)
            if k is None:
                miss += 1
            else:
                test_keys[k] = m
        test_set = set(test_keys)
        train_keys = [k for k in sig_idx if k not in test_set]
        if miss:
            print(f"[seed {seed}] {miss}/{len(moe)} MoE uids unmatched to signals")
        if len(test_keys) < 20 or len(train_keys) < 50:
            print(f"[seed {seed}] too few matched (test={len(test_keys)}) — skipped")
            continue

        Xtr = np.array([feats(sig_idx[k]) for k in train_keys], dtype=float)
        ytr = np.array([sig_idx[k]["correct"] for k in train_keys])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=seed))
        # guard: need both classes
        if len(set(ytr)) < 2:
            print(f"[seed {seed}] train has one class — skipped")
            continue
        clf.fit(Xtr, ytr)

        tk = list(test_keys)
        Xte = np.array([feats(sig_idx[k]) for k in tk], dtype=float)
        yte = np.array([sig_idx[k]["correct"] for k in tk])
        learned = clf.predict_proba(Xte)[:, 1]
        moe_score = np.array([test_keys[k]["p"] for k in tk], dtype=float)
        s3 = np.array([sig_idx[k]["vec"][2] for k in tk], dtype=float)  # s3_confidence

        for name, sc in [("learned_self_eval(ASPIRE-style)", learned),
                         ("moe_retention", moe_score),
                         ("chosen_softmax(s3)", s3)]:
            aurc, eaurc = aurc_and_eaurc(yte, sc)
            try:
                auroc = float(roc_auc_score(yte, sc)) if len(set(yte)) > 1 else float("nan")
            except ValueError:
                auroc = float("nan")
            rows.append({"seed": seed, "scorer": name, "n_test": len(tk),
                         "test_acc": float(yte.mean()), "AURC": aurc,
                         "E_AURC": eaurc, "AUROC": auroc})
        print(f"[seed {seed}] n_test={len(tk)} acc={yte.mean():.4f}  done")

    if not rows:
        raise SystemExit("no seeds produced results (check uid matching / paths)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # aggregate
    scorers = ["learned_self_eval(ASPIRE-style)", "moe_retention", "chosen_softmax(s3)"]
    print("\n=== AURC / E-AURC / AUROC (mean +/- std over seeds; lower AURC, higher AUROC better) ===")
    summ = []
    for sc in scorers:
        sub = [r for r in rows if r["scorer"] == sc]
        def ms(key):
            vals = [r[key] for r in sub if r[key] == r[key]]
            return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        a = ms("AURC"); e = ms("E_AURC"); u = ms("AUROC")
        summ.append({"scorer": sc, "AURC_mean": a[0], "AURC_std": a[1],
                     "E_AURC_mean": e[0], "E_AURC_std": e[1],
                     "AUROC_mean": u[0], "AUROC_std": u[1], "n_seeds": len(sub)})
        print(f"  {sc:34s}  AURC {a[0]:.4f}+/-{a[1]:.4f}   E-AURC {e[0]:.4f}+/-{e[1]:.4f}   "
              f"AUROC {u[0]:.4f}+/-{u[1]:.4f}")
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    print(f"\n[done] wrote {out}/per_seed.csv and summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
