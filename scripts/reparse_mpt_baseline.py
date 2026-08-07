#!/usr/bin/env python3
"""Strictly reparse the saved MPT baseline without rerunning the LLM.

The released MPT directory currently stores raw generations in the
``prediction_text`` field of ``predictions.jsonl``. Some environments may
instead name that file ``raw_outputs.jsonl``; this script accepts either
schema. It recognizes only a complete, explicit ``Answer: (X)`` marker and
uses the last such marker in a response. There is deliberately no generic
first-letter fallback for chain-of-thought text.

The source JSONL and legacy ``final.json`` are read-only. Strict metrics and
aligned parsed predictions are written to new, clearly named artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.bbq_evaluator import evaluate_bbq  # noqa: E402
from src.utils.data_loader import load_split  # noqa: E402


MPT_DIR = REPO / "results/v2_runpod/baselines/mpt"
SOURCE_CANDIDATES = ("raw_outputs.jsonl", "predictions.jsonl")
RAW_TEXT_FIELDS = ("raw_output", "prediction_text", "response", "text")
STRICT_FINAL_ANSWER = re.compile(r"\bAnswer\s*:\s*\(([ABC])\)", re.IGNORECASE)
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly reparse saved MPT generations using the last Answer: (X)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Raw MPT JSONL. If omitted, raw_outputs.jsonl is preferred and the "
            "legacy predictions.jsonl is used when it contains prediction_text."
        ),
    )
    parser.add_argument("--sampled-dir", type=Path, default=REPO / "data/sampled_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--out",
        type=Path,
        default=MPT_DIR / "strict_reparse_metrics.json",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=MPT_DIR / "strict_reparsed_predictions.jsonl",
    )
    return parser.parse_args()


def canonical_id(value: Any) -> str:
    """Normalize parquet/JSON numeric IDs without changing their identity."""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def uid(record: dict[str, Any]) -> str:
    return f"{record.get('category')}::{canonical_id(record.get('example_id'))}"


def resolve_source(explicit: Path | None) -> Path:
    if explicit is not None:
        source = explicit.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"MPT input does not exist: {source}")
        return source
    for name in SOURCE_CANDIDATES:
        candidate = MPT_DIR / name
        if candidate.exists():
            return candidate
    tried = ", ".join(str(MPT_DIR / name) for name in SOURCE_CANDIDATES)
    raise FileNotFoundError(f"No MPT raw-output JSONL found; tried: {tried}")


def raw_text(record: dict[str, Any]) -> tuple[str, str]:
    for field in RAW_TEXT_FIELDS:
        if field in record:
            value = record[field]
            return ("" if value is None else str(value), field)
    raise KeyError(
        f"No raw text field found for {uid(record)}; expected one of {RAW_TEXT_FIELDS}"
    )


def strict_last_answer(text: str) -> tuple[str | None, int]:
    matches = list(STRICT_FINAL_ANSWER.finditer(text))
    if not matches:
        return None, 0
    return matches[-1].group(1).upper(), len(matches)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def main() -> int:
    args = parse_args()
    source = resolve_source(args.input)
    raw_records = load_jsonl(source)

    raw_by_uid: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        key = uid(record)
        if key in raw_by_uid:
            raise ValueError(f"Duplicate MPT record: {key}")
        raw_by_uid[key] = record

    frame = load_split(args.sampled_dir, args.split)
    instances = [row.to_dict() for _, row in frame.iterrows()]
    instance_uids = [uid(item) for item in instances]
    missing = [key for key in instance_uids if key not in raw_by_uid]
    extra = sorted(set(raw_by_uid) - set(instance_uids))
    if missing or extra:
        raise ValueError(
            "MPT/data coverage mismatch: "
            f"missing={len(missing)} (sample={missing[:3]}), "
            f"extra={len(extra)} (sample={extra[:3]})"
        )

    parsed_indices: list[int] = []
    parsed_rows: list[dict[str, Any]] = []
    text_fields: set[str] = set()
    multiple_explicit_answers = 0

    for item, key in zip(instances, instance_uids):
        record = raw_by_uid[key]
        text, field = raw_text(record)
        text_fields.add(field)
        letter, n_matches = strict_last_answer(text)
        if n_matches > 1:
            multiple_explicit_answers += 1
        prediction_index = -1 if letter is None else LETTER_TO_INDEX[letter]
        parsed_indices.append(prediction_index)
        parsed_rows.append(
            {
                "uid": key,
                "example_id": item.get("example_id"),
                "category": item.get("category"),
                "context_condition": item.get("context_condition"),
                "gold_label": item.get("label"),
                "strict_answer": None if letter is None else f"({letter})",
                "prediction_index": prediction_index,
                "explicit_answer_matches": n_matches,
                "parse_failed": letter is None,
            }
        )

    overall = evaluate_bbq(parsed_indices, instances)
    per_category: dict[str, dict[str, float]] = {}
    category_pairs: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for prediction, item in zip(parsed_indices, instances):
        category_pairs[str(item.get("category"))].append((prediction, item))
    for category in sorted(category_pairs):
        pairs = category_pairs[category]
        per_category[category] = evaluate_bbq(
            [prediction for prediction, _ in pairs],
            [item for _, item in pairs],
        )

    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    parse_failures = sum(row["parse_failed"] for row in parsed_rows)
    payload = {
        "method": "mpt_strict_reparse",
        "source_jsonl": str(source.relative_to(REPO) if source.is_relative_to(REPO) else source),
        "source_sha256": source_sha256,
        "source_text_fields": sorted(text_fields),
        "parser": {
            "policy": "last complete explicit Answer: (X); no fallback",
            "regex": STRICT_FINAL_ANSWER.pattern,
            "case_insensitive": True,
        },
        "n_source_records": len(raw_records),
        "n_instances": len(instances),
        "n_strictly_parsed": len(instances) - parse_failures,
        "n_parse_failures": parse_failures,
        "n_responses_with_multiple_explicit_answers": multiple_explicit_answers,
        "overall": overall,
        "per_category": per_category,
        "raw_data_modified": False,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.predictions_out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in parsed_rows) + "\n",
        encoding="utf-8",
    )

    print(f"[source] {source} ({len(raw_records)} records, sha256={source_sha256[:12]}...)")
    print(
        "[strict parse] "
        f"parsed={len(instances) - parse_failures}/{len(instances)} "
        f"failures={parse_failures} multiple={multiple_explicit_answers}"
    )
    print(
        "[metrics] "
        f"Acc_amb={overall['accuracy_amb']:.4f} "
        f"Acc_dis={overall['accuracy_dis']:.4f} "
        f"FAR={overall['false_abstention_rate']:.4f} "
        f"parse_fail={overall['parse_fail_rate']:.4f}"
    )
    print(f"[write] {args.out}")
    print(f"[write] {args.predictions_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
