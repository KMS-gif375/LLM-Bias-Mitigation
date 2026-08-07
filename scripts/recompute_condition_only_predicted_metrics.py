#!/usr/bin/env python
"""Rebuild the corrected predicted-condition-only metrics from saved artifacts.

The replay uses no MoE score or threshold.  For each saved test prediction, a
predicted ambiguous condition maps to that item's released unknown option;
otherwise the saved primary answer is retained.  The script is intentionally
artifact-only so the headline condition-only row can be regenerated without
LLM inference or model retraining.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.bbq_evaluator import evaluate_bbq  # noqa: E402
from src.models.override import find_unknown_index  # noqa: E402
from src.utils.data_loader import DEFAULT_CATEGORIES_V2, load_bbq_category  # noqa: E402


DEFAULT_ROOT = REPO / "results/v2/clean_experiments_corrected_full"
DEFAULT_OUTPUT = DEFAULT_ROOT / "condition_only_predicted_metrics.csv"
DEFAULT_SEEDS = (42, 123, 456, 789, 999)


def _uid(row: dict) -> str:
    return f"{row.get('category', '_unknown')}::{row.get('example_id')}"


def _load_items(data_dir: Path) -> dict[str, dict]:
    # The saved prediction files contain the exact evaluated UID set.  Load the
    # released raw BBQ pool here, then select only those UIDs, rather than
    # assuming that a later regenerated sampled_v2/test.parquet has identical
    # membership.
    items = [
        item
        for category in DEFAULT_CATEGORIES_V2
        for item in load_bbq_category(data_dir, category)
    ]
    by_uid = {_uid(item): item for item in items}
    if len(by_uid) != len(items):
        raise ValueError("Duplicate category::example_id keys in the raw BBQ pool")
    return by_uid


def replay_seed(path: Path, items_by_uid: dict[str, dict]) -> dict[str, float]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No predictions in {path}")

    final_answers: list[int] = []
    items: list[dict] = []
    seen: set[str] = set()
    for record in records:
        uid = str(record["uid"])
        if uid in seen:
            raise ValueError(f"Duplicate prediction UID {uid} in {path}")
        seen.add(uid)
        if uid not in items_by_uid:
            raise KeyError(f"Prediction UID {uid} is absent from the released BBQ pool")
        condition = record.get("predicted_condition")
        if condition not in {"ambig", "disambig"}:
            raise ValueError(f"Invalid predicted_condition={condition!r} for {uid}")
        item = items_by_uid[uid]
        answer = (
            find_unknown_index(item)
            if condition == "ambig"
            else int(record["primary_answer"])
        )
        final_answers.append(answer)
        items.append(item)

    metrics = evaluate_bbq(final_answers, items)
    return {
        "n_test": int(metrics["n_total"]),
        "accuracy_amb": float(metrics["accuracy_amb"]),
        "accuracy_dis": float(metrics["accuracy_dis"]),
        "false_abstention_rate": float(metrics["false_abstention_rate"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data/bbq")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    items_by_uid = _load_items(args.data_dir)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        metrics = replay_seed(
            args.artifact_root / f"seed_{seed}" / "test_predictions.jsonl",
            items_by_uid,
        )
        rows.append({"seed": seed, "system": "condition_only_predicted", **metrics})

    metric_names = ("accuracy_amb", "accuracy_dis", "false_abstention_rate")
    summary_base = {
        "system": "condition_only_predicted",
        "n_test": rows[0]["n_test"],
    }
    means = {name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names}
    stds = {
        name: float(np.std([float(row[name]) for row in rows], ddof=1))
        for name in metric_names
    }
    rows.append({"seed": "mean", **summary_base, **means})
    rows.append({"seed": "sample_std", **summary_base, **stds})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "system", "n_test", *metric_names]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
