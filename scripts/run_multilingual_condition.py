#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_multilingual_condition.py  —  multilingual sentence-encoder condition transfer (Reviewer Q7).

Tests whether MULTILINGUAL sentence encoders recover the ambiguous/disambiguated condition
boundary on KoBBQ better than the English-centric all-MiniLM-L6-v2 used in the main paper.
For each encoder we embed context+question text, train a balanced logistic-regression condition
classifier on English BBQ (embedding-only), and report:
  (a) English held-out condition accuracy,
  (b) English -> KoBBQ zero-shot transfer condition accuracy,
  (c) within-KoBBQ condition accuracy (train/test on KoBBQ).

No LLM needed; just sentence encoders + logistic regression. Runs on CPU/MPS (downloads the
encoders + KoBBQ on first run).

USAGE:
  ./venv/bin/python scripts/run_multilingual_condition.py \
      --encoders sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/LaBSE intfloat/multilingual-e5-base \
      --ko-per-cat 150 --out results/v2/reviewer_audits/multilingual_condition
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def cond_label(rec):
    c = str(rec.get("context_condition", "")).lower()
    if c.startswith("ambig"):
        return 0
    if c.startswith("disambig"):
        return 1
    return -1


def text_of(rec):
    return f"{rec.get('context','')} {rec.get('question','')}".strip()


def load_english(n_per_cat):
    from src.utils.data_loader import load_split
    df = load_split(REPO / "data" / "sampled_v2", "test")
    recs = []
    for _, row in df.iterrows():
        d = row.to_dict()
        if cond_label(d) in (0, 1):
            recs.append(d)
    if n_per_cat:
        # cap per category-condition for speed
        from collections import defaultdict
        cnt = defaultdict(int); out = []
        for d in recs:
            k = (d.get("category"), cond_label(d))
            if cnt[k] < n_per_cat:
                cnt[k] += 1; out.append(d)
        recs = out
    return recs


def load_kobbq(n_per_cat):
    try:
        from src.transfer.run_kobbq import load_kobbq_as_bbq
        recs = load_kobbq_as_bbq(max_samples_per_category=n_per_cat)
        return [r for r in recs if cond_label(r) in (0, 1)]
    except Exception as e:
        print(f"[kobbq] could not load via load_kobbq_as_bbq ({e}); skipping KoBBQ rows.")
        return []


def embed(texts, model_name, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    # e5 models expect a 'query:'/'passage:' prefix
    if "e5" in model_name.lower():
        texts = [f"query: {t}" for t in texts]
    return np.asarray(model.encode(texts, batch_size=64, show_progress_bar=False,
                                   normalize_embeddings=True), dtype=np.float32)


def fit_eval(Xtr, ytr, Xte, yte, seed=42):
    clf = make_pipeline(StandardScaler(with_mean=True),
                        LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=seed))
    clf.fit(Xtr, ytr)
    return float(accuracy_score(yte, clf.predict(Xte)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=[
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/LaBSE",
        "intfloat/multilingual-e5-base",
    ])
    ap.add_argument("--en-per-cat", type=int, default=300)
    ap.add_argument("--ko-per-cat", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(REPO / "results/v2/reviewer_audits/multilingual_condition"))
    args = ap.parse_args()

    en = load_english(args.en_per_cat)
    ko = load_kobbq(args.ko_per_cat)
    print(f"[data] English={len(en)}  KoBBQ={len(ko)}")
    if len(en) < 100:
        raise SystemExit("not enough English records")

    en_text = [text_of(r) for r in en]; en_y = np.array([cond_label(r) for r in en])
    ko_text = [text_of(r) for r in ko]; ko_y = np.array([cond_label(r) for r in ko]) if ko else None

    rows = []
    for enc in args.encoders:
        print(f"\n[encode] {enc}")
        try:
            Xen = embed(en_text, enc, args.device)
            Xko = embed(ko_text, enc, args.device) if ko else None
        except Exception as e:
            print(f"  [skip] {enc}: {e}")
            rows.append({"encoder": enc, "en_dim": -1, "en_heldout": float("nan"),
                         "en_to_kobbq": float("nan"), "within_kobbq": float("nan"), "note": f"load_fail:{e}"})
            continue

        # (a) English held-out
        Xtr, Xva, ytr, yva = train_test_split(Xen, en_y, test_size=0.2, random_state=42, stratify=en_y)
        en_held = fit_eval(Xtr, ytr, Xva, yva)
        # (b) English -> KoBBQ transfer (train on ALL English)
        en_ko = fit_eval(Xen, en_y, Xko, ko_y) if ko else float("nan")
        # (c) within-KoBBQ
        if ko and len(set(ko_y)) > 1 and len(ko_y) > 40:
            Kxtr, Kxte, Kytr, Kyte = train_test_split(Xko, ko_y, test_size=0.3, random_state=42, stratify=ko_y)
            within_ko = fit_eval(Kxtr, Kytr, Kxte, Kyte)
        else:
            within_ko = float("nan")

        rows.append({"encoder": enc, "en_dim": int(Xen.shape[1]), "en_heldout": en_held,
                     "en_to_kobbq": en_ko, "within_kobbq": within_ko, "note": ""})
        print(f"  en_heldout={en_held:.4f}  en->KoBBQ={en_ko:.4f}  within_KoBBQ={within_ko:.4f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n=== Condition accuracy by encoder (higher better) ===")
    print(f"  {'encoder':40s} {'EN held':>9s} {'EN->Ko':>9s} {'within-Ko':>10s}")
    for r in rows:
        print(f"  {r['encoder']:40s} {r['en_heldout']:9.4f} {r['en_to_kobbq']:9.4f} {r['within_kobbq']:10.4f}")
    print(f"\n[done] wrote {out}/summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
