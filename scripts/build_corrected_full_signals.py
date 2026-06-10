#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_corrected_full_signals.py — produce results/v2/signals/corrected_full:
the fully corrected training inputs behind the Limitations item-(i) retrain
(0.9937 / 0.8753 / 0.0822).

Starting from the corrected s5/s7 signals (results/v2/signals/corrected_s5s7,
see scripts/extract_corrected_s5s7.py), this rebuilds the Eq. 7 bias-penalty
target `is_stereotype` with the corrected normalized group matcher
(bbq_evaluator.is_stereotyped_answer), changing 994/8,864 = 11.2% of labels,
and copies the cached MiniLM embeddings so the directory is a drop-in
`--model corrected_full` source for scripts/run_clean_experiments.py.

CPU-only, deterministic. Follow with:
  ./venv/bin/python scripts/run_clean_experiments.py --model corrected_full \
      --seeds 42 123 456 789 999 \
      --out-dir results/v2/clean_experiments_corrected_full
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.utils.data_loader import load_split  # noqa: E402
from src.evaluation.bbq_evaluator import is_stereotyped_answer  # noqa: E402

SRC = REPO / "results/v2/signals/corrected_s5s7"
DST = REPO / "results/v2/signals/corrected_full"


def main():
    pool = {}
    for s in ("train", "val", "test"):
        for _, r in load_split(REPO / "data/sampled_v2", s).iterrows():
            d = r.to_dict()
            pool[f"{d['category']}::{int(d['example_id'])}"] = d

    DST.mkdir(parents=True, exist_ok=True)
    n = ch = 0
    for f in sorted(SRC.glob("*_signals.jsonl")):
        out = []
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            it = pool[f"{r['category']}::{int(r['example_id'])}"]
            p = int(r["primary_answer"])
            d = is_stereotyped_answer(it, p) if p in (0, 1, 2) else None
            new = 1.0 if d == "stereotyped" else 0.0
            if float(r.get("is_stereotype", 0.0)) != new:
                ch += 1
            r = dict(r)
            r["is_stereotype"] = new
            out.append(json.dumps(r))
            n += 1
        (DST / f.name).write_text("\n".join(out) + "\n")
    for f in glob.glob(str(REPO / "results/v2/signals/main/*_embeddings.pt")):
        shutil.copy(f, DST)
    print(f"[corrected_full] {n} rows, is_stereotype changed on {ch} ({ch / n:.1%})")


if __name__ == "__main__":
    main()
