#!/usr/bin/env python
"""Recompute KoBBQ routing after removing repeated archived example IDs.

This is an artifact-only audit.  It uses the saved KoBBQ signals and MiniLM
embeddings, the saved English-BBQ MoE checkpoint, and the cached English-BBQ
embeddings used to fit the transfer condition classifier.  It makes no LLM or
HuggingFace dataset calls and keeps the first archived occurrence of each
``category::example_id``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.moe_aggregator import (  # noqa: E402
    MoEAggregator,
    signals_dict_to_tensor,
)
from src.utils.data_loader import load_split  # noqa: E402


def record_key(record: dict) -> str:
    return f"{record.get('category', '_unknown')}::{record.get('example_id')}"


def deduplicate_first(records: list[dict]) -> tuple[list[dict], dict]:
    """Keep the first occurrence of each composite ID and describe repeats."""
    groups: dict[str, list[dict]] = defaultdict(list)
    unique: list[dict] = []
    for record in records:
        key = record_key(record)
        if not groups[key]:
            unique.append(record)
        groups[key].append(record)

    duplicate_groups = {key: rows for key, rows in groups.items() if len(rows) > 1}
    varied_signal_fields: Counter[str] = Counter()
    n_varied_groups = 0
    for rows in duplicate_groups.values():
        first = rows[0].get("signals", {})
        varied = False
        for field in sorted(set().union(*(row.get("signals", {}) for row in rows))):
            if any(row.get("signals", {}).get(field) != first.get(field) for row in rows[1:]):
                varied_signal_fields[field] += 1
                varied = True
        n_varied_groups += int(varied)

    audit = {
        "input_records": len(records),
        "unique_records": len(unique),
        "removed_records": len(records) - len(unique),
        "duplicate_keys": len(duplicate_groups),
        "duplicate_multiplicity": {
            str(size): count
            for size, count in sorted(Counter(map(len, duplicate_groups.values())).items())
        },
        "duplicate_groups_with_signal_variation": n_varied_groups,
        "varied_signal_fields": dict(sorted(varied_signal_fields.items())),
    }
    return unique, audit


def infer_unknown_indices(records: list[dict]) -> dict[str, int]:
    """Infer the unknown option from each paired ambiguous KoBBQ record."""
    lookup = {record_key(record): record for record in records}
    unknown: dict[str, int] = {}
    pattern = re.compile(r"-dis-(bsd|cnt)$")

    for record in records:
        key = record_key(record)
        condition = str(record.get("context_condition", ""))
        if condition == "ambig":
            unknown[key] = int(record["label"])
            continue
        if condition != "disambig":
            raise ValueError(f"Unexpected KoBBQ condition for {key}: {condition!r}")

        example_id = str(record.get("example_id"))
        ambig_id, replacements = pattern.subn(r"-amb-\1", example_id)
        companion_key = f"{record.get('category', '_unknown')}::{ambig_id}"
        if replacements != 1 or companion_key not in lookup:
            raise ValueError(f"No ambiguous companion found for {key}")
        unknown[key] = int(lookup[companion_key]["label"])

    return unknown


def condition_code(condition: str) -> int:
    value = str(condition).lower()
    if value.startswith("ambig"):
        return 0
    if value.startswith("disambig"):
        return 1
    raise ValueError(f"Unexpected condition: {condition!r}")


def load_condition_classifier():
    pool: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        for _, row in load_split(REPO / "data/sampled_v2", split).iterrows():
            item = row.to_dict()
            pool[f"{item['category']}::{int(item['example_id'])}"] = item

    embeddings: dict[str, np.ndarray] = {}
    pattern = str(REPO / "results/v2/signals/main/*_embeddings.pt")
    for filename in sorted(glob.glob(pattern)):
        category = Path(filename).name.removesuffix("_embeddings.pt")
        cached = torch.load(filename, map_location="cpu", weights_only=True)
        for example_id, embedding in cached.items():
            embeddings[f"{category}::{int(example_id)}"] = (
                embedding.numpy().astype(np.float32)
            )

    ids = list(pool)
    missing = [key for key in ids if key not in embeddings]
    if missing:
        raise KeyError(f"Missing {len(missing)} English condition embeddings")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )
    classifier.fit(
        np.stack([embeddings[key] for key in ids]),
        np.asarray([condition_code(pool[key]["context_condition"]) for key in ids]),
    )
    return classifier, len(ids)


def load_moe(checkpoint_path: Path) -> MoEAggregator:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    model = MoEAggregator(
        signal_dim=int(config.get("signal_dim", 7)),
        embed_dim=int(config.get("embed_dim", 384)),
        num_experts=int(config.get("num_experts", 4)),
        gating_hidden=int(config.get("gating_hidden", 64)),
        expert_hidden=int(config.get("expert_hidden", 128)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()
    return model


def score_records(
    records: list[dict],
    embeddings: dict,
    model: MoEAggregator,
    classifier,
    unknown_indices: dict[str, int],
    thresholds: dict[str, float],
) -> list[dict]:
    outcomes: list[dict] = []
    with torch.inference_mode():
        for record in records:
            key = record_key(record)
            raw_id = str(record.get("example_id"))
            if raw_id not in embeddings:
                raise KeyError(f"Missing archived KoBBQ embedding for {key}")

            embedding = embeddings[raw_id].to(torch.float32)
            signals = signals_dict_to_tensor(record.get("signals", {}))
            p_score = float(model(signals.unsqueeze(0), embedding.unsqueeze(0)).p.item())
            gold_condition = str(record["context_condition"])
            predicted_condition = (
                "disambig"
                if int(classifier.predict(embedding.numpy()[None, :])[0]) == 1
                else "ambig"
            )
            primary = int(record.get("primary_answer", -1))
            unknown = unknown_indices[key]

            def route(condition: str) -> int:
                threshold = thresholds[condition]
                return primary if primary == -1 or p_score >= threshold else unknown

            outcomes.append(
                {
                    "key": key,
                    "category": str(record.get("category", "_unknown")),
                    "gold_condition": gold_condition,
                    "predicted_condition": predicted_condition,
                    "label": int(record["label"]),
                    "unknown": unknown,
                    "oracle_final": route(gold_condition),
                    "predicted_final": route(predicted_condition),
                }
            )
    return outcomes


def aggregate(outcomes: list[dict], final_field: str) -> dict[str, float | int]:
    ambig = [row for row in outcomes if row["gold_condition"] == "ambig"]
    disambig = [row for row in outcomes if row["gold_condition"] == "disambig"]

    def accuracy(rows: list[dict]) -> float:
        return sum(row[final_field] == row["label"] for row in rows) / max(len(rows), 1)

    return {
        "n": len(outcomes),
        "n_ambig": len(ambig),
        "n_disambig": len(disambig),
        "acc_amb": accuracy(ambig),
        "acc_dis": accuracy(disambig),
        "far": sum(row[final_field] == row["unknown"] for row in disambig)
        / max(len(disambig), 1),
    }


def summarize(outcomes: list[dict]) -> dict:
    categories = sorted({row["category"] for row in outcomes})
    predicted_counts = Counter(row["predicted_condition"] for row in outcomes)
    return {
        "condition_prediction": {
            "agreement": sum(
                row["gold_condition"] == row["predicted_condition"] for row in outcomes
            )
            / max(len(outcomes), 1),
            "counts": dict(sorted(predicted_counts.items())),
        },
        "oracle": aggregate(outcomes, "oracle_final"),
        "predicted": aggregate(outcomes, "predicted_final"),
        "per_category": {
            category: {
                "oracle": aggregate(
                    [row for row in outcomes if row["category"] == category],
                    "oracle_final",
                ),
                "predicted": aggregate(
                    [row for row in outcomes if row["category"] == category],
                    "predicted_final",
                ),
            }
            for category in categories
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_metric_delta(actual: dict, expected: dict) -> float:
    pairs = (
        ("oracle", "acc_amb"),
        ("oracle", "acc_dis"),
        ("oracle", "far"),
        ("predicted", "acc_amb"),
        ("predicted", "acc_dis"),
        ("predicted", "far"),
    )
    return max(abs(actual[route][metric] - expected[route][metric]) for route, metric in pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signals",
        type=Path,
        default=REPO / "results/v2_runpod/transfer/kobbq/_signals.jsonl",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=REPO / "results/v2_runpod/transfer/kobbq/_embeddings.pt",
    )
    parser.add_argument(
        "--moe-checkpoint",
        type=Path,
        default=REPO / "results/v2_runpod/moe/main/moe_best.pt",
    )
    parser.add_argument(
        "--routing-reference",
        type=Path,
        default=(
            REPO
            / "results/v2/reviewer_audits/routing_unify_published/report.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO
            / "results/v2/reviewer_audits/kobbq_deduplicated_routing_published/report.json"
        ),
    )
    parser.add_argument("--threshold-amb", type=float, default=0.95)
    parser.add_argument("--threshold-dis", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.signals = args.signals.resolve()
    args.embeddings = args.embeddings.resolve()
    args.moe_checkpoint = args.moe_checkpoint.resolve()
    args.routing_reference = args.routing_reference.resolve()
    args.output = args.output.resolve()
    records = [
        json.loads(line)
        for line in args.signals.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique_records, duplicate_audit = deduplicate_first(records)
    unknown_indices = infer_unknown_indices(unique_records)
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    classifier, classifier_training_n = load_condition_classifier()
    model = load_moe(args.moe_checkpoint)
    thresholds = {
        "ambig": float(args.threshold_amb),
        "disambig": float(args.threshold_dis),
    }

    original = summarize(
        score_records(
            records, embeddings, model, classifier, unknown_indices, thresholds
        )
    )
    deduplicated = summarize(
        score_records(
            unique_records, embeddings, model, classifier, unknown_indices, thresholds
        )
    )

    reference = json.loads(args.routing_reference.read_text(encoding="utf-8"))[
        "kobbq_tau9505"
    ]
    anchor_delta = max_metric_delta(original, reference)
    if anchor_delta > 1e-12:
        raise RuntimeError(
            "Archived routing fidelity check failed: "
            f"max metric delta is {anchor_delta:.3g}"
        )

    report = {
        "audit": "kobbq_deduplicated_archival_routing",
        "no_llm_calls": True,
        "deduplication": {
            "key": "category::example_id",
            "retention": "first archived occurrence",
            **duplicate_audit,
        },
        "protocol": {
            "thresholds": thresholds,
            "condition_classifier": (
                "balanced logistic regression fit on cached English-BBQ MiniLM "
                "embeddings; no KoBBQ condition labels"
            ),
            "condition_classifier_training_n": classifier_training_n,
            "unknown_index": (
                "inferred from the archived paired ambiguous record with the "
                "same category, item stem, and polarity"
            ),
            "moe_checkpoint": str(args.moe_checkpoint.relative_to(REPO)),
        },
        "sources": {
            "signals": {
                "path": str(args.signals.relative_to(REPO)),
                "sha256": sha256(args.signals),
            },
            "embeddings": {
                "path": str(args.embeddings.relative_to(REPO)),
                "sha256": sha256(args.embeddings),
            },
            "moe_checkpoint": {
                "path": str(args.moe_checkpoint.relative_to(REPO)),
                "sha256": sha256(args.moe_checkpoint),
            },
        },
        "fidelity_anchor": {
            "reference": str(args.routing_reference.relative_to(REPO)),
            "max_abs_metric_delta": anchor_delta,
            "passed": True,
        },
        "original_weighted_archive": original,
        "deduplicated_archive": deduplicated,
        "metric_delta_deduplicated_minus_original": {
            route: {
                metric: deduplicated[route][metric] - original[route][metric]
                for metric in ("acc_amb", "acc_dis", "far")
            }
            for route in ("oracle", "predicted")
        },
        "limitations": [
            "This audit reweights the saved archive to one row per composite ID; "
            "it does not add later unique KoBBQ rows to restore each category cap.",
            "The duplicated underlying examples are identical, but 30 duplicate "
            "groups have different saved s4_consistency values. First occurrence "
            "is therefore an explicit deterministic archival convention.",
            "The corrected result reuses archived signals, embeddings, and the "
            "saved MoE checkpoint; it is not a fresh end-to-end KoBBQ run.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        "deduplicated predicted: "
        f"n={deduplicated['predicted']['n']} "
        f"acc_amb={deduplicated['predicted']['acc_amb']:.10f} "
        f"acc_dis={deduplicated['predicted']['acc_dis']:.10f} "
        f"far={deduplicated['predicted']['far']:.10f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
