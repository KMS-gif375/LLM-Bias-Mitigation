#!/usr/bin/env python3
"""
Run clean leave-one-category-out experiments from saved signal files.

This is the reviewer-defense LOCO suite:
  - no LLM inference
  - one held-out category at a time
  - train/val split only inside the non-held-out categories
  - optional no-oracle condition classifier evaluated on the held-out category
  - repeated over multiple seeds
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_clean_experiments as clean  # noqa: E402

LOGGER = logging.getLogger("clean_loco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-seed clean LOCO generalization experiments without LLM inference."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="main")
    parser.add_argument("--results-dir", default="results/v2")
    parser.add_argument("--sampled-dir", default="data/sampled_v2")
    parser.add_argument("--out-dir", default="results/v2/acceptance_package/loco")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--no-discover-categories", action="store_true")
    parser.add_argument("--samples-per-category", type=int, default=1000)

    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 999])
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--lambda-bias", type=float, default=0.5)
    parser.add_argument("--lambda-lb", type=float, default=0.1)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--tau-min", type=float, default=0.05)
    parser.add_argument("--tau-max", type=float, default=0.95)
    parser.add_argument("--tau-step", type=float, default=0.025)
    parser.add_argument("--low-tau-min", type=float, default=0.0)
    parser.add_argument("--low-tau-max", type=float, default=0.10)
    parser.add_argument("--low-tau-step", type=float, default=0.01)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    parser.add_argument(
        "--condition-features",
        default="signals,embedding,primary",
        help=(
            "Comma-separated features for held-out no-oracle condition classifier. "
            "Default excludes category to avoid unseen-category one-hot artifacts."
        ),
    )
    parser.add_argument("--skip-condition-classifier", action="store_true")
    parser.add_argument(
        "--run-low-threshold-audit",
        action="store_true",
        help="Also run low-threshold audit inside every LOCO fold. Slower and usually not needed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def stable_train_val_split(
    records: list[dict],
    val_ratio: float,
    seed: int,
    stratify_keys: tuple[str, ...] = ("category", "context_condition"),
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_stratum: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    for rec in records:
        key = tuple(rec.get(k, "_unknown") for k in stratify_keys)
        by_stratum[key].append(rec)

    train: list[dict] = []
    val: list[dict] = []
    for group in by_stratum.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) >= 2 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def residual_bias_counts(predictions_by_uid: dict[str, int], ordered_predictions: list[dict]) -> dict[str, Any]:
    from src.evaluation.bbq_evaluator import is_stereotyped_answer

    n_ambig = 0
    unknown_correct = 0
    unknown_pred = 0
    stereo = 0
    anti = 0
    invalid = 0
    missing = 0

    for pred in ordered_predictions:
        item = pred["item"]
        if item.get("context_condition") != "ambig":
            continue
        n_ambig += 1
        uid = pred["uid"]
        if uid not in predictions_by_uid:
            missing += 1
            continue
        final = int(predictions_by_uid[uid])
        if final == item.get("label"):
            unknown_correct += 1
        kind = is_stereotyped_answer(item, final)
        if kind == "unknown":
            unknown_pred += 1
        elif kind == "stereotyped":
            stereo += 1
        elif kind == "anti_stereotyped":
            anti += 1
        else:
            invalid += 1

    non_unknown = stereo + anti
    bias = None if non_unknown == 0 else 2 * (stereo / non_unknown) - 1
    return {
        "n_ambig": n_ambig,
        "unknown_correct": unknown_correct,
        "unknown_pred": unknown_pred,
        "non_unknown_residual": non_unknown,
        "stereo": stereo,
        "anti": anti,
        "invalid": invalid,
        "missing": missing,
        "bias_score_amb": bias,
        "bias_abs_amb": None if bias is None else abs(float(bias)),
    }


def save_fold_predictions(
    path: Path,
    held_predictions: list[dict],
    variants: dict[str, Any],
    predicted_condition: Optional[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pred in held_predictions:
            uid = pred["uid"]
            item = pred["item"]
            row = {
                "uid": uid,
                "example_id": item.get("example_id"),
                "category": item.get("category", pred.get("category")),
                "context_condition": item.get("context_condition", pred.get("context_condition")),
                "label": item.get("label"),
                "primary_answer": pred["primary_answer"],
                "p_score": pred["p_score"],
                "predicted_condition": None if predicted_condition is None else predicted_condition.get(uid),
            }
            for system_name, payload in variants.items():
                row[system_name] = payload["predictions_by_uid"].get(uid)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def metric_row(
    seed: int,
    held_out: str,
    variant: str,
    payload: dict[str, Any],
    n_train: int,
    n_val: int,
    n_held_out: int,
    condition_accuracy: Optional[float],
) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    thresholds = payload.get("thresholds", {})
    return {
        "seed": seed,
        "held_out_category": held_out,
        "variant": variant,
        "n_train": n_train,
        "n_val": n_val,
        "n_held_out": n_held_out,
        "condition_accuracy": condition_accuracy,
        "tau_amb": thresholds.get("ambig"),
        "tau_dis": thresholds.get("disambig"),
        "tau_default": thresholds.get("default"),
        "n_total": metrics.get("n_total"),
        "n_ambig": metrics.get("n_ambig"),
        "n_disambig": metrics.get("n_disambig"),
        "accuracy_amb": metrics.get("accuracy_amb"),
        "accuracy_dis": metrics.get("accuracy_dis"),
        "bias_score_amb": metrics.get("bias_score_amb"),
        "bias_score_dis": metrics.get("bias_score_dis"),
        "false_abstention_rate": metrics.get("false_abstention_rate"),
        "parse_fail_rate": metrics.get("parse_fail_rate"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def write_aggregate(metrics_rows: list[dict[str, Any]], out_dir: Path) -> None:
    wanted = (
        "accuracy_amb",
        "accuracy_dis",
        "bias_score_amb",
        "false_abstention_rate",
        "condition_accuracy",
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        groups[str(row["variant"])].append(row)

    rows: list[dict[str, Any]] = []
    for variant, vals in sorted(groups.items()):
        out: dict[str, Any] = {
            "variant": variant,
            "n_folds": len(vals),
            "n_seeds": len({v["seed"] for v in vals}),
            "n_categories": len({v["held_out_category"] for v in vals}),
        }
        for key in wanted:
            nums = [numeric(v.get(key)) for v in vals]
            nums = [v for v in nums if v is not None]
            out[f"{key}_mean"] = mean(nums) if nums else None
            out[f"{key}_std"] = stdev(nums) if len(nums) > 1 else 0.0 if nums else None
        rows.append(out)
    write_csv(out_dir / "loco_aggregate.csv", rows)


def write_report(out_dir: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Clean LOCO Report",
        "",
        "Leave-one-category-out generalization using saved signals only.",
        "",
        "## Aggregate",
        "",
        "| Variant | folds | acc_amb | acc_dis | FAR | condition_acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        def fmt(key: str) -> str:
            m = numeric(row.get(f"{key}_mean"))
            s = numeric(row.get(f"{key}_std")) or 0.0
            if m is None:
                return ""
            return f"{m:.4f}+/-{s:.4f}"

        lines.append(
            "| {variant} | {n_folds} | {acc_amb} | {acc_dis} | {far} | {cond} |".format(
                variant=row["variant"],
                n_folds=row["n_folds"],
                acc_amb=fmt("accuracy_amb"),
                acc_dis=fmt("accuracy_dis"),
                far=fmt("false_abstention_rate"),
                cond=fmt("condition_accuracy"),
            )
        )
    lines += [
        "",
        "## Files",
        "",
        "- `loco_metrics.csv`: per-seed, per-held-out-category metrics.",
        "- `loco_aggregate.csv`: aggregate mean/std across folds.",
        "- `loco_residual_bias_counts.csv`: ambiguous residual bias denominator audit.",
        "- `seed_*/folds/*/predictions.jsonl`: held-out predictions for auditability.",
    ]
    (out_dir / "loco_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    setup_logging()
    args = parse_args()
    args.skip_low_threshold_audit = not args.run_low_threshold_audit
    args.baselines = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = clean.load_experiment_config(args)
    categories = list(config["data"]["categories"])
    LOGGER.info("Categories: %s", ", ".join(categories))
    records, embeddings, instances_by_id = clean.load_records_embeddings_instances(config, args)
    LOGGER.info("Loaded records=%d embeddings=%d instances=%d", len(records), len(embeddings), len(instances_by_id))

    if args.dry_run:
        LOGGER.info("[dry-run] seeds=%s folds=%d", args.seeds, len(categories))
        for seed in args.seeds:
            for held_out in categories:
                pool = [r for r in records if r.get("category") != held_out]
                held = [r for r in records if r.get("category") == held_out]
                train, val = stable_train_val_split(pool, args.val_split, seed + 1009 * categories.index(held_out))
                LOGGER.info("[dry-run] seed=%s held=%s train=%d val=%d held=%d", seed, held_out, len(train), len(val), len(held))
        return 0

    all_results: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        LOGGER.info("=" * 72)
        LOGGER.info("Seed %d", seed)
        seed_result: dict[str, Any] = {"seed": seed, "folds": {}}
        for fold_index, held_out in enumerate(categories):
            fold_seed = seed + 1009 * (fold_index + 1)
            LOGGER.info("-" * 72)
            LOGGER.info("[seed=%d] held_out=%s", seed, held_out)
            train_pool = [r for r in records if r.get("category") != held_out]
            held_records = [r for r in records if r.get("category") == held_out]
            train_records, val_records = stable_train_val_split(train_pool, args.val_split, fold_seed)
            LOGGER.info("Split: train=%d val=%d held_out=%d", len(train_records), len(val_records), len(held_records))

            fold_dir = out_dir / f"seed_{seed}" / "folds" / safe_name(held_out)
            fold_dir.mkdir(parents=True, exist_ok=True)

            predicted_val_condition = None
            predicted_held_condition = None
            condition_payload = None
            condition_accuracy = None
            if not args.skip_condition_classifier:
                condition_payload = clean.fit_condition_classifier(
                    train_records,
                    val_records,
                    held_records,
                    embeddings,
                    categories,
                    args.condition_features,
                    seed=fold_seed,
                )
                predicted_val_condition = condition_payload["val"]["condition_by_uid"]
                predicted_held_condition = condition_payload["test"]["condition_by_uid"]
                condition_accuracy = condition_payload["test"]["accuracy"]
                LOGGER.info("Condition classifier held-out acc=%.4f", condition_accuracy)

            model, train_out = clean.train_moe(
                train_records,
                val_records,
                embeddings,
                config,
                args,
                seed=fold_seed,
                save_dir=fold_dir / "checkpoints" / "main",
            )
            val_predictions = clean.predict_records(model, val_records, embeddings, instances_by_id)
            held_predictions = clean.predict_records(model, held_records, embeddings, instances_by_id)
            moe_eval = clean.evaluate_moe_variants(
                val_predictions,
                held_predictions,
                predicted_val_condition=predicted_val_condition,
                predicted_test_condition=predicted_held_condition,
                args=args,
            )
            save_fold_predictions(
                fold_dir / "predictions.jsonl",
                held_predictions,
                moe_eval["variants"],
                predicted_held_condition,
            )

            fold_payload = {
                "held_out_category": held_out,
                "fold_seed": fold_seed,
                "split": {
                    "n_train": len(train_records),
                    "n_val": len(val_records),
                    "n_held_out": len(held_records),
                },
                "train": {
                    "best_val_loss": train_out.get("best_val_loss"),
                    "best_epoch": train_out.get("best_epoch"),
                    "checkpoint_path": train_out.get("checkpoint_path"),
                },
                "condition_classifier": condition_payload,
                "moe_eval": moe_eval,
            }

            for variant, payload in moe_eval["variants"].items():
                m = payload["metrics"]
                LOGGER.info(
                    "%s: acc_amb=%.4f acc_dis=%.4f FAR=%.4f bias_amb=%s",
                    variant,
                    m.get("accuracy_amb", 0.0),
                    m.get("accuracy_dis", 0.0),
                    m.get("false_abstention_rate", 0.0),
                    m.get("bias_score_amb"),
                )
                metrics_rows.append(
                    metric_row(
                        seed,
                        held_out,
                        variant,
                        payload,
                        len(train_records),
                        len(val_records),
                        len(held_records),
                        condition_accuracy,
                    )
                )
                counts = residual_bias_counts(payload["predictions_by_uid"], held_predictions)
                residual_rows.append(
                    {
                        "seed": seed,
                        "held_out_category": held_out,
                        "variant": variant,
                        **counts,
                    }
                )

            compact = clean.strip_large_maps(fold_payload)
            (fold_dir / "result.json").write_text(
                json.dumps(compact, indent=2, ensure_ascii=False, default=float),
                encoding="utf-8",
            )
            seed_result["folds"][held_out] = compact

        (out_dir / f"seed_{seed}_summary.json").write_text(
            json.dumps(seed_result, indent=2, ensure_ascii=False, default=float),
            encoding="utf-8",
        )
        all_results.append(seed_result)

    write_csv(out_dir / "loco_metrics.csv", metrics_rows)
    write_csv(out_dir / "loco_residual_bias_counts.csv", residual_rows)
    write_aggregate(metrics_rows, out_dir)

    aggregate_rows = []
    with (out_dir / "loco_aggregate.csv").open("r", encoding="utf-8") as f:
        aggregate_rows = list(csv.DictReader(f))
    write_report(out_dir, aggregate_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(clean.strip_large_maps(all_results), indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    LOGGER.info("Saved LOCO package: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
