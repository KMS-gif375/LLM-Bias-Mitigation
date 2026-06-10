#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_review_hardening.py — two light audits requested in pre-submission review hardening.

(A) LENGTH-ONLY condition classifier (R2's by-construction confound control):
    BBQ disambiguated contexts are the ambiguous context plus an appended
    disambiguating sentence, so context LENGTH alone may recover the condition
    boundary. We train a logistic regression on ONLY [char count, whitespace-token
    count, sentence count] of the context and report 5-seed test condition accuracy,
    using the SAME per-seed test membership as the clean experiments
    (results/v2/clean_experiments/seed_*/test_predictions.jsonl).

(B) PER-CATEGORY FAR under category-aware condition thresholds (R4):
    Retrain the embedding-only condition classifier per seed, tune a per-category
    disambiguated-probability threshold on validation (uniform 0.5 vs per-category),
    and report per-category FAR of the condition-only retention rule on test,
    to see whether the Physical appearance / Religion FAR gap narrows.

CPU-only; reuses saved embeddings + signals + per-seed test memberships.
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.utils.data_loader import load_split  # noqa: E402

SEEDS = [42, 123, 456, 789, 999]
OUT = REPO / "results/v2/reviewer_audits/review_hardening"


def cond01(c):  # 0=ambig 1=disambig
    c = str(c).lower()
    return 0 if c.startswith("ambig") else (1 if c.startswith("disambig") else -1)


def load_pool():
    dfs = [load_split(REPO / "data/sampled_v2", s) for s in ("train", "val", "test")]
    pool = {}
    for df in dfs:
        for _, row in df.iterrows():
            d = row.to_dict()
            pool[f"{d['category']}::{int(d['example_id'])}"] = d
    return pool


def unknown_idx(item):
    info = item.get("answer_info", {}) or {}
    for i in range(3):
        a = info.get(f"ans{i}", [])
        if len(a) >= 2 and a[1] == "unknown":
            return i
    return -1


def load_primary():
    out = {}
    for f in glob.glob(str(REPO / "results/v2/signals/main/*_signals.jsonl")):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                try:
                    out[f"{r['category']}::{int(r['example_id'])}"] = int(r["primary_answer"])
                except (TypeError, ValueError, KeyError):
                    pass
    return out


def load_embeddings():
    emb = {}
    for f in glob.glob(str(REPO / "results/v2/signals/main/*_embeddings.pt")):
        cat = Path(f).name.replace("_embeddings.pt", "")
        d = torch.load(f, map_location="cpu", weights_only=True)
        for k, v in d.items():
            emb[f"{cat}::{int(k)}"] = v.numpy().astype(np.float32)
    return emb


SENT_PAT = re.compile(r"[.!?]+(?:\s|$)")


def length_feats(ctx: str):
    return [len(ctx), len(ctx.split()), max(1, len(SENT_PAT.findall(ctx)))]


def fit(X, y, seed):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    clf.fit(X, y)
    return clf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    prim = load_primary()
    emb = load_embeddings()
    print(f"[load] pool={len(pool)}  primary={len(prim)}  emb={len(emb)}")

    len_rows, far_rows = [], []
    for seed in SEEDS:
        tp = REPO / f"results/v2/clean_experiments/seed_{seed}/test_predictions.jsonl"
        test_uids = [json.loads(l)["uid"] for l in tp.read_text().splitlines() if l.strip()]
        test_uids = [u for u in test_uids if u in pool]
        rest = [u for u in pool if u not in set(test_uids)]
        strat = [f"{pool[u]['category']}|{cond01(pool[u]['context_condition'])}" for u in rest]
        tr_uids, va_uids = train_test_split(rest, test_size=1328, random_state=seed, stratify=strat)

        y = {u: cond01(pool[u]["context_condition"]) for u in pool}
        ytr = np.array([y[u] for u in tr_uids]); yva = np.array([y[u] for u in va_uids]); yte = np.array([y[u] for u in test_uids])

        # ---------- (A) length-only ----------
        Xl = {u: length_feats(str(pool[u]["context"])) for u in pool}
        clfL = fit(np.array([Xl[u] for u in tr_uids]), ytr, seed)
        accL = float((clfL.predict(np.array([Xl[u] for u in test_uids])) == yte).mean())
        # sentence-count-only
        clfS = fit(np.array([[Xl[u][2]] for u in tr_uids]), ytr, seed)
        accS = float((clfS.predict(np.array([[Xl[u][2]] for u in test_uids])) == yte).mean())
        len_rows.append({"seed": seed, "len3_acc": accL, "sent_only_acc": accS, "n_test": len(test_uids)})
        print(f"[seed {seed}] length-3feat acc={accL:.4f}  sentence-count-only acc={accS:.4f}")

        # ---------- (B) embedding-only + per-category thresholds ----------
        ok = [u for u in pool if u in emb]
        tr2 = [u for u in tr_uids if u in emb]; va2 = [u for u in va_uids if u in emb]; te2 = [u for u in test_uids if u in emb]
        clfE = fit(np.stack([emb[u] for u in tr2]), np.array([y[u] for u in tr2]), seed)
        pva = clfE.predict_proba(np.stack([emb[u] for u in va2]))[:, 1]
        pte = clfE.predict_proba(np.stack([emb[u] for u in te2]))[:, 1]
        accE = float(((pte >= 0.5).astype(int) == np.array([y[u] for u in te2])).mean())

        # per-category threshold tuned on val (maximize condition accuracy)
        cats = sorted({pool[u]["category"] for u in pool})
        thr = {}
        for c in cats:
            idx = [i for i, u in enumerate(va2) if pool[u]["category"] == c]
            if not idx:
                thr[c] = 0.5; continue
            yv = np.array([y[va2[i]] for i in idx]); pv = pva[idx]
            best, bacc = 0.5, -1
            for t in np.arange(0.05, 0.96, 0.01):
                a = float(((pv >= t).astype(int) == yv).mean())
                if a > bacc: bacc, best = a, float(t)
            thr[c] = best

        def far_by_cat(use_per_cat):
            num = defaultdict(int); den = defaultdict(int)
            for i, u in enumerate(te2):
                if y[u] != 1:  # disambig only
                    continue
                c = pool[u]["category"]; den[c] += 1
                t = thr[c] if use_per_cat else 0.5
                pred_dis = pte[i] >= t
                abstain = (not pred_dis) or (prim.get(u, -1) == unknown_idx(pool[u]))
                num[c] += int(abstain)
            return {c: num[c] / den[c] for c in den}

        f_uni = far_by_cat(False); f_cat = far_by_cat(True)
        for c in sorted(f_uni):
            far_rows.append({"seed": seed, "category": c, "far_uniform": f_uni[c], "far_cataware": f_cat[c]})
        print(f"[seed {seed}] emb-only acc={accE:.4f}  FAR(Phys uni/cat)="
              f"{f_uni.get('Physical_appearance', float('nan')):.3f}/{f_cat.get('Physical_appearance', float('nan')):.3f}")

    # ---------- aggregate ----------
    def agg(rows, key):
        v = [r[key] for r in rows]
        return float(np.mean(v)), float(np.std(v))

    mL, sL = agg(len_rows, "len3_acc"); mS, sS = agg(len_rows, "sent_only_acc")
    print("\n=== (A) LENGTH-ONLY condition accuracy (5 seeds) ===")
    print(f"  3 length features : {mL:.4f} ± {sL:.4f}")
    print(f"  sentence-count only: {mS:.4f} ± {sS:.4f}")

    print("\n=== (B) per-category FAR: uniform 0.5 vs category-aware (5-seed mean) ===")
    cats = sorted({r["category"] for r in far_rows})
    summary = []
    for c in cats:
        rs = [r for r in far_rows if r["category"] == c]
        mu, _ = agg(rs, "far_uniform"); mc, _ = agg(rs, "far_cataware")
        summary.append({"category": c, "far_uniform": mu, "far_cataware": mc})
        print(f"  {c:22s} {mu:.4f} -> {mc:.4f}")
    spread_u = max(s["far_uniform"] for s in summary) - min(s["far_uniform"] for s in summary)
    spread_c = max(s["far_cataware"] for s in summary) - min(s["far_cataware"] for s in summary)
    print(f"  spread(max-min): uniform {spread_u:.4f} -> cat-aware {spread_c:.4f}")

    with open(OUT / "length_only.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(len_rows[0])); w.writeheader(); w.writerows(len_rows)
    with open(OUT / "per_category_far_calibration.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "category", "far_uniform", "far_cataware"]); w.writeheader(); w.writerows(far_rows)
    print(f"\n[done] wrote {OUT}/length_only.csv and per_category_far_calibration.csv")


if __name__ == "__main__":
    main()
