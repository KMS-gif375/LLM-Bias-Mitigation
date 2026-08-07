#!/usr/bin/env python3
"""Condition-prediction transfer audits for manuscript revision.

The clean BBQ result shows that ambiguous/disambiguated condition prediction is
nearly solved by embeddings. This script checks whether that separability
survives on transfer data such as KoBBQ, using saved signal artifacts and
freshly cached sentence-transformer embeddings only. It does not call an LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_clean_experiments as clean  # noqa: E402


SEEDS = [42, 123, 456, 789, 999]
TRANSFER_MODES = [
    ("signals+embedding+primary", "signals,embedding,primary"),
    ("signals+embedding", "signals,embedding"),
    ("embedding_only", "embedding"),
    ("signals_only", "signals"),
    ("primary_only", "primary"),
    ("category_only", "category"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run condition-prediction transfer audits without LLM inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="main")
    parser.add_argument("--results-dir", default="results/v2")
    parser.add_argument("--sampled-dir", default="data/sampled_v2")
    parser.add_argument("--transfer-name", default="kobbq", choices=["kobbq"])
    parser.add_argument("--transfer-dir", default="results/v2_runpod/transfer/kobbq")
    parser.add_argument("--out-dir", default="results/v2/reviewer_audits")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--samples-per-category", type=int, default=1000)
    parser.add_argument("--transfer-max-samples", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def base_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        model=args.model,
        results_dir=args.results_dir,
        sampled_dir=args.sampled_dir,
        categories=None,
        no_discover_categories=False,
        samples_per_category=args.samples_per_category,
        val_split=args.val_split,
        test_split=args.test_split,
        current_seed=args.seeds[0],
        device=args.device,
        tau_min=0.0,
        tau_max=1.0,
        tau_step=0.025,
        low_tau_min=0.0,
        low_tau_max=0.10,
        low_tau_step=0.01,
        skip_low_threshold_audit=True,
        bootstrap_iters=200,
        bootstrap_seed=42,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-5,
        val_every=5,
        lambda_bias=0.5,
        lambda_lb=0.1,
    )


def uid_for(record: dict[str, Any]) -> str:
    if record.get("unique_id"):
        return str(record["unique_id"])
    return f"{record.get('category', '_unknown')}::{record.get('example_id')}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_transfer_records(path: Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    for rec in records:
        rec["unique_id"] = uid_for(rec)
    return records


def deduplicate_transfer_records_first(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the first archived signal row for each composite KoBBQ ID."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique: list[dict[str, Any]] = []
    for record in records:
        key = uid_for(record)
        if not groups[key]:
            unique.append(record)
        groups[key].append(record)

    duplicates = [rows for rows in groups.values() if len(rows) > 1]
    varied_fields: Counter[str] = Counter()
    n_varied_groups = 0
    for rows in duplicates:
        first = rows[0].get("signals", {})
        varied = False
        for field in sorted(set().union(*(row.get("signals", {}) for row in rows))):
            if any(row.get("signals", {}).get(field) != first.get(field) for row in rows[1:]):
                varied_fields[field] += 1
                varied = True
        n_varied_groups += int(varied)

    return unique, {
        "key": "category::example_id",
        "retention": "first archived occurrence",
        "input_records": len(records),
        "unique_records": len(unique),
        "removed_records": len(records) - len(unique),
        "duplicate_keys": len(duplicates),
        "duplicate_multiplicity": {
            str(size): count
            for size, count in sorted(Counter(map(len, duplicates)).items())
        },
        "duplicate_groups_with_signal_variation": n_varied_groups,
        "varied_signal_fields": dict(sorted(varied_fields.items())),
    }


def load_kobbq_items(max_samples: int, categories: list[str]) -> dict[str, dict[str, Any]]:
    from src.transfer.run_kobbq import load_kobbq_as_bbq

    items = load_kobbq_as_bbq(max_samples_per_category=max_samples, categories=categories)
    return {f"{item.get('category')}::{item.get('example_id')}": item for item in items}


def transfer_embeddings(
    transfer_name: str,
    transfer_records: list[dict[str, Any]],
    categories: list[str],
    cache_path: Path,
    max_samples: int,
    model_name: str,
) -> dict[str, Any]:
    if transfer_name != "kobbq":
        return {}

    from src.models.embedding import EmbeddingExtractor, cache_embeddings

    item_by_uid = load_kobbq_items(max_samples=max_samples, categories=categories)
    items = [item_by_uid[uid_for(rec)] for rec in transfer_records if uid_for(rec) in item_by_uid]
    missing = len(transfer_records) - len(items)
    if missing:
        print(f"[warn] missing KoBBQ items for {missing}/{len(transfer_records)} transfer records")

    extractor = EmbeddingExtractor(model_name=model_name, device="cpu")
    raw = cache_embeddings(items, extractor, cache_path)
    out: dict[str, Any] = {}
    for item in items:
        raw_key = item.get("example_id")
        if raw_key in raw:
            out[f"{item.get('category')}::{raw_key}"] = raw[raw_key]
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


_KOBBQ_COMPANION_SUFFIX = re.compile(r"-(?:amb|dis)-(?:bsd|cnt)$")


def kobbq_companion_group(record: dict[str, Any]) -> str:
    """Return the shared stem for the four KoBBQ condition/polarity companions."""
    raw_id = str(record.get("example_id", ""))
    stem = _KOBBQ_COMPANION_SUFFIX.sub("", raw_id)
    if not stem or stem == raw_id:
        raise ValueError(f"Unexpected KoBBQ example_id format: {raw_id!r}")
    return f"{record.get('category', '_unknown')}::{stem}"


def split_kobbq_companion_disjoint(
    records: list[dict[str, Any]],
    seed: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split whole KoBBQ companion groups, preventing paired-stem leakage."""
    from sklearn.model_selection import train_test_split

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[kobbq_companion_group(record)].append(record)
    if any(len(rows) != 4 for rows in grouped.values()):
        sizes = Counter(len(rows) for rows in grouped.values())
        raise ValueError(f"Expected four rows per KoBBQ companion group, got {sizes}")

    group_keys = sorted(grouped)
    strata = [str(grouped[key][0].get("category", "_unknown")) for key in group_keys]
    train_ratio = 1.0 - val_ratio - test_ratio
    train_keys, rest_keys = train_test_split(
        group_keys,
        train_size=train_ratio,
        random_state=seed,
        stratify=strata,
    )
    rest_strata = [str(grouped[key][0].get("category", "_unknown")) for key in rest_keys]
    val_fraction_of_rest = val_ratio / (val_ratio + test_ratio)
    val_keys, test_keys = train_test_split(
        rest_keys,
        train_size=val_fraction_of_rest,
        random_state=seed,
        stratify=rest_strata,
    )

    def expand(keys: list[str]) -> list[dict[str, Any]]:
        return [record for key in keys for record in grouped[key]]

    return expand(train_keys), expand(val_keys), expand(test_keys)


def add_rows_for_scope(
    scope: str,
    seeds: list[int],
    train_val_test_builder,
    transfer_records: list[dict[str, Any]],
    embeddings: dict[str, Any],
    categories: list[str],
    detail_rows: list[dict[str, Any]],
) -> None:
    for seed in seeds:
        train, val, test = train_val_test_builder(seed)
        for label, mode in TRANSFER_MODES:
            payload = clean.fit_condition_classifier(train, val, test, embeddings, categories, mode, seed)
            detail_rows.append(
                {
                    "scope": scope,
                    "transfer": "kobbq",
                    "seed": seed,
                    "feature_set": label,
                    "modes": mode,
                    "accuracy": payload["test"]["accuracy"],
                    "n_train": payload["train_n"],
                    "n_val": payload["val"]["n"],
                    "n_test": payload["test"]["n"],
                    "confusion_matrix": json.dumps(payload["test"]["confusion_matrix"]),
                }
            )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_args = base_args(args)
    config = clean.load_experiment_config(clean_args)
    clean_records, clean_embeddings, _ = clean.load_records_embeddings_instances(config, clean_args)
    categories = list(config["data"]["categories"])

    transfer_path = Path(args.transfer_dir) / "_signals.jsonl"
    weighted_transfer_records = load_transfer_records(transfer_path)
    transfer_records, deduplication = deduplicate_transfer_records_first(
        weighted_transfer_records
    )
    embedding_model = config.get("moe", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    transfer_emb = transfer_embeddings(
        args.transfer_name,
        transfer_records,
        categories,
        Path(args.transfer_dir) / "_embeddings.pt",
        args.transfer_max_samples,
        embedding_model,
    )
    embeddings = {**clean_embeddings, **transfer_emb}

    detail_rows: list[dict[str, Any]] = []

    def english_to_transfer(seed: int):
        clean_args.current_seed = seed
        train, val, _ = clean.split_records(clean_records, clean_args)
        return train, val, transfer_records

    add_rows_for_scope(
        "english_bbq_to_kobbq",
        args.seeds,
        english_to_transfer,
        transfer_records,
        embeddings,
        categories,
        detail_rows,
    )

    def within_transfer(seed: int):
        clean_args.current_seed = seed
        return clean.split_records(transfer_records, clean_args)

    add_rows_for_scope(
        "within_kobbq",
        args.seeds,
        within_transfer,
        transfer_records,
        embeddings,
        categories,
        detail_rows,
    )

    def within_transfer_companion_disjoint(seed: int):
        return split_kobbq_companion_disjoint(
            transfer_records,
            seed,
            val_ratio=args.val_split,
            test_ratio=args.test_split,
        )

    add_rows_for_scope(
        "within_kobbq_companion_disjoint",
        args.seeds,
        within_transfer_companion_disjoint,
        transfer_records,
        embeddings,
        categories,
        detail_rows,
    )

    summary_rows: list[dict[str, Any]] = []
    scopes = sorted({str(row["scope"]) for row in detail_rows})
    for scope in scopes:
        for label, mode in TRANSFER_MODES:
            vals = [
                float(row["accuracy"])
                for row in detail_rows
                if row["scope"] == scope and row["feature_set"] == label
            ]
            m, s = mean_std(vals)
            summary_rows.append(
                {
                    "scope": scope,
                    "transfer": args.transfer_name,
                    "feature_set": label,
                    "modes": mode,
                    "accuracy_mean": m,
                    "accuracy_std": s,
                    "n_train": next(
                        (
                            row["n_train"]
                            for row in detail_rows
                            if row["scope"] == scope and row["feature_set"] == label
                        ),
                        0,
                    ),
                    "n_val": next(
                        (
                            row["n_val"]
                            for row in detail_rows
                            if row["scope"] == scope and row["feature_set"] == label
                        ),
                        0,
                    ),
                    "n_test": next(
                        (
                            row["n_test"]
                            for row in detail_rows
                            if row["scope"] == scope and row["feature_set"] == label
                        ),
                        0,
                    ),
                    "values": " ".join(f"{v:.4f}" for v in vals),
                }
            )

    write_csv(out_dir / f"{args.transfer_name}_condition_transfer_audit.csv", detail_rows)
    write_csv(out_dir / f"{args.transfer_name}_condition_transfer_audit_summary.csv", summary_rows)
    report = {
        "audit": "kobbq_deduplicated_condition_transfer",
        "no_llm_calls": True,
        "deduplication": deduplication,
        "protocol": {
            "seeds": args.seeds,
            "row_split": "stratified 70/15/15 by category::context_condition",
            "companion_disjoint_split": (
                "70/15/15 over 644 category-stratified base stems; all four "
                "ambiguous/disambiguated x polarity companions remain together"
            ),
            "standard_deviation": "sample (n-1 denominator)",
            "embeddings": "reuse archived MiniLM embeddings",
        },
        "summary": summary_rows,
        "limitations": [
            "This audit reuses archived signals and embeddings rather than rerunning extraction.",
            "Thirty duplicate groups have different saved s4_consistency values; the first "
            "occurrence is an explicit deterministic archival convention.",
            "The row-level within-KoBBQ split is unique-ID-disjoint but leaks paired "
            "companion stems across partitions; the companion-disjoint scope removes "
            "that leakage but is not a broader semantic-template-disjoint benchmark.",
        ],
    }
    (out_dir / f"{args.transfer_name}_condition_transfer_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
