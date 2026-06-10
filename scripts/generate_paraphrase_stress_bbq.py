#!/usr/bin/env python3
"""Generate deterministic BBQ paraphrase/adversarial stress inputs.

The goal is not to create a new benchmark with human-quality paraphrases. It is
to produce an auditable stress file that suppresses some of the original BBQ
lexical/template cues without using gold labels at evaluation time. The output
JSONL can be passed to:

    scripts/run_reviewer_extra_audits.py --audits paraphrase --paraphrase-file ...

Each generated row preserves the original BBQ condition label and answer
metadata, but rewrites context/question text through deterministic operations:

* entity masking: demographic answer entities become Person A / Person B;
* light wording rewrite: common template phrases are replaced;
* combined masked+rewrite variant.

Because the transformations are deterministic and metadata-preserving, the file
is reproducible. It should still be described as a stress audit rather than a
human-validated paraphrase benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic BBQ stress paraphrases.")
    parser.add_argument("--data-dir", default="data/bbq")
    parser.add_argument("--out", default="data/bbq_stress/paraphrase_template_stress.jsonl")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--max-per-category-condition", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["entity_masked", "light_rewrite", "masked_light_rewrite"],
        choices=["entity_masked", "light_rewrite", "masked_light_rewrite"],
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_entity(text: str) -> str:
    text = re.sub(r"^\s*(?:the|a|an)\s+", "", str(text), flags=re.I).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def answer_entities(row: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    replacements = ["Person A", "Person B"]
    repl_idx = 0
    for idx in (0, 1, 2):
        info = row.get("answer_info", {})
        vals: list[Any] = []
        if isinstance(info, dict):
            raw_vals = info.get(f"ans{idx}", [])
            if isinstance(raw_vals, list):
                vals = raw_vals
        full_answer_blob = " ".join(str(v).lower() for v in [row.get(f"ans{idx}", "")] + vals)
        if any(marker in full_answer_blob for marker in ("unknown", "can't be determined", "not enough information", "cannot be determined")):
            continue
        if repl_idx >= len(replacements):
            continue
        repl = replacements[repl_idx]
        repl_idx += 1
        answer = normalize_entity(row.get(f"ans{idx}", ""))
        if answer:
            mapping[answer] = repl
        for val in vals[:1]:
            ent = normalize_entity(val)
            if ent:
                mapping[ent] = repl
    return mapping


def replace_case_insensitive(text: str, old: str, new: str) -> str:
    if not old:
        return text
    # Word-ish boundary for names and demographic descriptors. Phrase terms with
    # spaces still match as a unit.
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", re.I)
    return pattern.sub(new, text)


def mask_entities(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    mapping = answer_entities(row)
    # Replace longer strings first so multi-word entities win over substrings.
    replacements = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for key in ("context", "question", "ans0", "ans1", "ans2"):
        text = str(out.get(key, ""))
        for old, new in replacements:
            text = replace_case_insensitive(text, old, new)
        out[key] = text
    out["stress_entity_map"] = mapping
    return out


LIGHT_REWRITES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bI saw\b", re.I), "There were"),
    (re.compile(r"\bwe saw\b", re.I), "there were"),
    (re.compile(r"\blast week\b", re.I), "recently"),
    (re.compile(r"\byesterday\b", re.I), "the previous day"),
    (re.compile(r"\boutside the\b", re.I), "near the"),
    (re.compile(r"\btrying to\b", re.I), "attempting to"),
    (re.compile(r"\bstruggling with\b", re.I), "having trouble with"),
    (re.compile(r"\bnot really listening\b", re.I), "not paying much attention to"),
    (re.compile(r"\bWho was\b", re.I), "Which person was"),
    (re.compile(r"\bWho is\b", re.I), "Which person is"),
    (re.compile(r"\bWho did\b", re.I), "Which person did"),
    (re.compile(r"\bWho has\b", re.I), "Which person has"),
    (re.compile(r"\bnot comfortable\b", re.I), "uncomfortable"),
    (re.compile(r"\bthe other\b", re.I), "the other person"),
]


def light_rewrite(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("context", "question"):
        text = str(out.get(key, ""))
        for pattern, repl in LIGHT_REWRITES:
            text = pattern.sub(repl, text)
        text = re.sub(r"\s+", " ", text).strip()
        out[key] = text
    return out


def make_variant(row: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant == "entity_masked":
        out = mask_entities(row)
    elif variant == "light_rewrite":
        out = light_rewrite(row)
    elif variant == "masked_light_rewrite":
        out = light_rewrite(mask_entities(row))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    original_id = row.get("example_id")
    category = row.get("category", "unknown")
    out["original_example_id"] = original_id
    out["example_id"] = f"{original_id}::{variant}"
    out["unique_id"] = f"{category}::{original_id}::{variant}"
    out["stress_variant"] = variant
    out["stress_generator"] = "deterministic_entity_mask_light_rewrite_v1"
    out["context_original"] = row.get("context", "")
    out["question_original"] = row.get("question", "")
    return out


def stratified_sample(rows: list[dict[str, Any]], max_per_category_condition: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("category")), str(row.get("context_condition")))].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        selected.extend(bucket[:max_per_category_condition])
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    source_rows: list[dict[str, Any]] = []
    for category in args.categories:
        path = data_dir / f"{category}.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            row.setdefault("category", category)
        source_rows.extend(rows)

    sampled = stratified_sample(source_rows, args.max_per_category_condition, args.seed)
    out_rows: list[dict[str, Any]] = []
    for row in sampled:
        for variant in args.variants:
            out_rows.append(make_variant(row, variant))

    out_path = Path(args.out)
    write_jsonl(out_path, out_rows)

    summary = {
        "source_rows": len(source_rows),
        "sampled_rows": len(sampled),
        "output_rows": len(out_rows),
        "variants": args.variants,
        "max_per_category_condition": args.max_per_category_condition,
        "seed": args.seed,
        "output": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
