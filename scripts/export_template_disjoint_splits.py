#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_template_disjoint_splits.py — release artifact for independent replication
(reviewer request: a fully disjoint split that does not share templates between
train and test).

Reproduces EXACTLY the template-disjoint control of the paper (Section VI):
instances are grouped by (category, question_index) into templates and split
70/30 with GroupShuffleSplit over five group-split seeds, guaranteeing that
train and test share no BBQ template.

Output: data/splits/template_disjoint/seed_{s}.json with
  {"seed", "n_templates", "train_uids", "test_uids"}  (uid = "Category::example_id")
plus a README.md describing the protocol. Deterministic — safe to re-run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.utils.data_loader import load_split  # noqa: E402

OUT = REPO / "data/splits/template_disjoint"
OUT.mkdir(parents=True, exist_ok=True)
GROUP_SEEDS = [0, 1, 2, 3, 4]


def main():
    uids, groups = [], []
    for s in ("train", "val", "test"):
        for _, row in load_split(REPO / "data/sampled_v2", s).iterrows():
            d = row.to_dict()
            uids.append(f"{d['category']}::{int(d['example_id'])}")
            groups.append(f"{d['category']}|{int(d.get('question_index', -1))}")
    uids = np.array(uids)
    groups = np.array(groups)
    n_templates = len(set(groups.tolist()))
    print(f"[splits] {len(uids)} instances, {n_templates} templates")

    for seed in GROUP_SEEDS:
        gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
        tr_idx, te_idx = next(gss.split(uids, groups=groups))
        tr_g = set(groups[tr_idx].tolist())
        te_g = set(groups[te_idx].tolist())
        assert not (tr_g & te_g), "template leakage!"
        payload = {
            "seed": seed,
            "protocol": "GroupShuffleSplit(train_size=0.7) grouped by (category, question_index)",
            "n_templates": n_templates,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "train_uids": sorted(uids[tr_idx].tolist()),
            "test_uids": sorted(uids[te_idx].tolist()),
        }
        (OUT / f"seed_{seed}.json").write_text(json.dumps(payload, indent=1))
        print(f"[splits] seed {seed}: train {len(tr_idx)} / test {len(te_idx)} "
              f"({len(tr_g)}/{len(te_g)} templates) -> {OUT}/seed_{seed}.json")

    (OUT / "README.md").write_text(
        "# Template-disjoint splits (replication artifact)\n\n"
        "Five 70/30 splits of the 8,864-instance pool grouped by BBQ template\n"
        "`(category, question_index)` via `GroupShuffleSplit(random_state=seed)`,\n"
        "so train and test share **no template**. These are the exact splits behind\n"
        "the paper's template-disjoint control (embedding-only condition accuracy\n"
        "0.8874 +- 0.0369; condition-only retention 0.9319 / 0.7993 / 0.1480).\n\n"
        "`uid = \"<Category>::<example_id>\"` indexes `data/sampled_v2`.\n"
        "Regenerate deterministically with `scripts/export_template_disjoint_splits.py`.\n"
    )
    print(f"[done] artifact in {OUT}")


if __name__ == "__main__":
    main()
