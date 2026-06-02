#!/usr/bin/env python3
"""Generate rule-based explanation artifacts for BBQ predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.rule_explanations import (  # noqa: E402
    SIGNAL_NAMES,
    answer_group,
    answer_text,
    explain_decision,
)
from src.models.override import find_unknown_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic explanation CSV/JSONL/Markdown artifacts.")
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=sorted(str(p) for p in Path("results/v2/clean_experiments").glob("seed_*/test_predictions.jsonl")),
        help="Prediction JSONL files. Defaults to clean_experiments seed_*/test_predictions.jsonl.",
    )
    parser.add_argument("--system", default="ours_predicted_condition", help="Prediction column to explain.")
    parser.add_argument("--sampled-dir", default="data/sampled_v2", help="Directory with train/val/test parquet files.")
    parser.add_argument("--signals-dir", default="results/v2/signals/main", help="Directory with *_signals.jsonl files.")
    parser.add_argument("--out-dir", default="results/v2/rule_explanations")
    parser.add_argument("--max-md-cases", type=int, default=18)
    return parser.parse_args()


def uid_for(row: dict[str, Any]) -> str:
    if row.get("uid"):
        return str(row["uid"])
    return f"{row.get('category', '_unknown')}::{row.get('example_id')}"


def load_items(sampled_dir: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = sampled_dir / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for row in df.to_dict(orient="records"):
            uid = f"{row.get('category', '_unknown')}::{row.get('example_id')}"
            row["split"] = split
            items[uid] = row
    return items


def load_signals(signals_dir: Path) -> dict[str, dict[str, Any]]:
    signals: dict[str, dict[str, Any]] = {}
    for path in sorted(signals_dir.glob("*_signals.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                uid = uid_for(row)
                signals[uid] = row.get("signals", {})
    return signals


def load_prediction_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str)
        seed = None
        for part in path.parts:
            if part.startswith("seed_"):
                try:
                    seed = int(part.split("_", 1)[1])
                except ValueError:
                    seed = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["prediction_file"] = str(path)
                row["seed"] = seed
                rows.append(row)
    return rows


def as_int(value: Any) -> int:
    if value is None or value == "":
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_output_row(
    pred_row: dict[str, Any],
    item: dict[str, Any],
    signals: dict[str, Any],
    system: str,
) -> dict[str, Any]:
    primary_answer = as_int(pred_row.get("primary_answer"))
    final_answer = as_int(pred_row.get(system))
    explanation = explain_decision(
        item,
        signals,
        primary_answer=primary_answer,
        final_answer=final_answer,
        predicted_condition=pred_row.get("predicted_condition"),
    )
    label = as_int(item.get("label"))
    return {
        "seed": pred_row.get("seed"),
        "uid": uid_for(pred_row),
        "example_id": item.get("example_id"),
        "category": item.get("category"),
        "context_condition": item.get("context_condition"),
        "predicted_condition": pred_row.get("predicted_condition"),
        "condition_match": pred_row.get("predicted_condition") == item.get("context_condition"),
        "question_polarity": item.get("question_polarity"),
        "label": label,
        "label_text": answer_text(item, label),
        "primary_answer": primary_answer,
        "primary_text": answer_text(item, primary_answer),
        "primary_group": answer_group(item, primary_answer),
        "final_answer": final_answer,
        "final_text": answer_text(item, final_answer),
        "final_group": answer_group(item, final_answer),
        "unknown_index": find_unknown_index(item),
        "decision_type": explanation.decision_type,
        "raw_bias_direction": explanation.raw_bias_direction,
        "final_bias_direction": explanation.final_bias_direction,
        "signal_flags": ";".join(explanation.signal_flags),
        "bias_risk_level": explanation.bias_risk_level,
        "bias_risk_reasons": " | ".join(explanation.bias_risk_reasons),
        "bias_risk_explanation": explanation.bias_risk_explanation,
        "runtime_explanation": explanation.runtime_explanation,
        "audit_explanation": explanation.audit_explanation,
        **{name: signals.get(name) for name in SIGNAL_NAMES},
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], max_cases: int) -> None:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["decision_type"])].append(row)
    risk_counts = Counter(str(row.get("bias_risk_level", "unclassified")) for row in rows)
    risk_total = sum(risk_counts.values())

    priority = [
        "stereotyped_answer_blocked",
        "bias_slip",
        "anti_stereotype_slip",
        "false_abstention",
        "wrong_stereotyped_keep",
        "wrong_anti_stereotyped_keep",
        "utility_preserved",
    ]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for dtype in priority:
        for row in by_type.get(dtype, [])[: max(1, max_cases // len(priority))]:
            key = (row.get("seed"), row.get("uid"), row.get("decision_type"))
            if key not in seen:
                selected.append(row)
                seen.add(key)
    if len(selected) < max_cases:
        for row in rows:
            key = (row.get("seed"), row.get("uid"), row.get("decision_type"))
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(selected) >= max_cases:
                break
    selected = selected[:max_cases]

    lines = [
        "# Rule-Based Explanation Audit",
        "",
        "Runtime explanations use predicted condition and extracted signals. Benchmark audit labels additionally use gold BBQ metadata after evaluation to categorize residual errors.",
        "",
        "## Summary",
        "",
        "| Decision type | Count | Share |",
        "|---|---:|---:|",
    ]
    total = sum(int(row["count"]) for row in summary_rows)
    for row in summary_rows:
        share = float(row["share"])
        lines.append(f"| {row['decision_type']} | {row['count']} | {share:.4f} |")

    lines += [
        "",
        "## Bias-Risk Summary",
        "",
        "| Bias-risk level | Count | Share |",
        "|---|---:|---:|",
    ]
    for key, value in sorted(risk_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        share = value / risk_total if risk_total else 0.0
        lines.append(f"| {key} | {value} | {share:.4f} |")

    lines += ["", "## Representative Cases", ""]
    for idx, row in enumerate(selected, 1):
        lines += [
            f"### Case {idx}: {row['decision_type']} ({row['category']}::{row['example_id']}, seed={row['seed']})",
            "",
            f"- Condition: gold={row['context_condition']}, predicted={row['predicted_condition']}",
            f"- Label: {row['label_text']}",
            f"- Raw answer: {row['primary_text']} ({row['raw_bias_direction']})",
            f"- Final answer: {row['final_text']} ({row['final_bias_direction']})",
            f"- Signal flags: {row['signal_flags'] or 'none'}",
            f"- Bias-risk level: {row['bias_risk_level']}",
            f"- Bias-risk explanation: {row['bias_risk_explanation']}",
            f"- Runtime explanation: {row['runtime_explanation']}",
            f"- Audit explanation: {row['audit_explanation']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) / args.system
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(Path(args.sampled_dir))
    signals = load_signals(Path(args.signals_dir))
    prediction_rows = load_prediction_rows(args.predictions)

    output_rows: list[dict[str, Any]] = []
    skipped = Counter()
    for pred_row in prediction_rows:
        uid = uid_for(pred_row)
        item = items.get(uid)
        if item is None:
            skipped["missing_item"] += 1
            continue
        if args.system not in pred_row:
            skipped["missing_system_prediction"] += 1
            continue
        output_rows.append(make_output_row(pred_row, item, signals.get(uid, {}), args.system))

    counts = Counter(row["decision_type"] for row in output_rows)
    risk_counts = Counter(row["bias_risk_level"] for row in output_rows)
    total = sum(counts.values())
    summary_rows = [
        {"decision_type": key, "count": value, "share": value / total if total else 0.0}
        for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    summary_rows.append({"decision_type": "__total__", "count": total, "share": 1.0 if total else 0.0})
    for key, value in sorted(skipped.items()):
        summary_rows.append({"decision_type": f"__skipped_{key}__", "count": value, "share": 0.0})
    risk_summary_rows = [
        {"bias_risk_level": key, "count": value, "share": value / total if total else 0.0}
        for key, value in sorted(risk_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    write_csv(out_dir / "explanations.csv", output_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "bias_risk_summary.csv", risk_summary_rows)
    with (out_dir / "explanations.jsonl").open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_markdown(out_dir / "explanation_audit.md", output_rows, summary_rows[:-len(skipped) or None], args.max_md_cases)

    print(f"Wrote {len(output_rows)} explanations to {out_dir}")
    if skipped:
        print(f"Skipped: {dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
