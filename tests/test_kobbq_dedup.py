"""Regression tests for KoBBQ duplicate handling and archival deduplication."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scripts.recompute_kobbq_deduplicated_routing import (
    deduplicate_first,
    infer_unknown_indices,
)
from scripts.run_transfer_condition_audits import (
    deduplicate_transfer_records_first,
    kobbq_companion_group,
    split_kobbq_companion_disjoint,
)
from src.transfer.run_kobbq import (
    _deduplicate_raw_by_sample_id,
    load_kobbq_as_bbq,
)


def raw_row(sample_id: str, condition: str = "amb", answer: str = "unknown") -> dict:
    choices = ["group A", "group B", "unknown"]
    return {
        "sample_id": f"{sample_id}-{condition}-bsd",
        "label_annotation": "ST",
        "context": f"context {sample_id}",
        "question": "question",
        "choices": choices,
        "biased_answer": "group A",
        "answer": answer,
        "bbq_id": 1.0,
        "bbq_category": "Age",
        "prediction": None,
    }


def signal_record(sample_id: str, condition: str, label: int, s4: float) -> dict:
    return {
        "example_id": f"{sample_id}-{condition}-bsd",
        "category": "Age",
        "context_condition": "ambig" if condition == "amb" else "disambig",
        "label": label,
        "primary_answer": 0,
        "signals": {"s4_consistency": s4},
    }


def test_raw_exact_duplicates_keep_first():
    first = raw_row("age-001")
    unique, removed = _deduplicate_raw_by_sample_id([first, dict(first), raw_row("age-002")])
    assert [row["sample_id"] for row in unique] == [
        "age-001-amb-bsd",
        "age-002-amb-bsd",
    ]
    assert removed == 1


def test_raw_conflicting_duplicate_raises():
    first = raw_row("age-001")
    conflict = {**first, "question": "different question"}
    with pytest.raises(ValueError, match="conflicting rows"):
        _deduplicate_raw_by_sample_id([first, conflict])


def test_loader_deduplicates_before_category_cap(monkeypatch):
    first = raw_row("age-001")
    rows = [first, dict(first), raw_row("age-002")]
    fake_datasets = SimpleNamespace(load_dataset=lambda *args, **kwargs: rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    loaded = load_kobbq_as_bbq(max_samples_per_category=2)

    assert [row["example_id"] for row in loaded] == [
        "age-001-amb-bsd",
        "age-002-amb-bsd",
    ]


def test_archival_dedup_keeps_first_signal_and_audits_variation():
    first = signal_record("age-001", "amb", 2, 0.2)
    records = [first, signal_record("age-001", "amb", 2, 0.8)]

    unique, audit = deduplicate_first(records)

    assert unique == [first]
    assert audit["removed_records"] == 1
    assert audit["duplicate_keys"] == 1
    assert audit["duplicate_groups_with_signal_variation"] == 1
    assert audit["varied_signal_fields"] == {"s4_consistency": 1}


def test_condition_audit_uses_same_first_unique_convention():
    first = signal_record("age-001", "amb", 2, 0.2)
    records = [first, signal_record("age-001", "amb", 2, 0.8)]
    unique, audit = deduplicate_transfer_records_first(records)
    assert unique == [first]
    assert audit["retention"] == "first archived occurrence"
    assert audit["removed_records"] == 1


def test_unknown_index_is_inferred_from_ambiguous_companion():
    records = [
        signal_record("age-001", "amb", 2, 0.5),
        signal_record("age-001", "dis", 0, 0.5),
    ]
    unknown = infer_unknown_indices(records)
    assert unknown["Age::age-001-amb-bsd"] == 2
    assert unknown["Age::age-001-dis-bsd"] == 2


def test_companion_disjoint_split_keeps_all_four_rows_together():
    records = []
    for category in ("Age", "Religion"):
        prefix = category.lower()
        for idx in range(10):
            stem = f"{prefix}-{idx:03d}"
            for condition in ("amb", "dis"):
                for polarity in ("bsd", "cnt"):
                    records.append({
                        "example_id": f"{stem}-{condition}-{polarity}",
                        "category": category,
                    })

    train, val, test = split_kobbq_companion_disjoint(records, seed=42)
    partitions = [
        {kobbq_companion_group(row) for row in part}
        for part in (train, val, test)
    ]

    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])
    for part in (train, val, test):
        counts = {}
        for row in part:
            group = kobbq_companion_group(row)
            counts[group] = counts.get(group, 0) + 1
        assert set(counts.values()) == {4}
