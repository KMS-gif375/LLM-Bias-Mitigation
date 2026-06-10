#!/usr/bin/env python3
"""Reviewer-risk audits for the IEEE Access manuscript.

This script uses saved signals, embeddings, predictions, and clean split logic.
It does not call an LLM. Outputs are intended to make reviewer-facing claims
auditable rather than hand-copied from ad hoc notebooks.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_clean_experiments as clean  # noqa: E402
import run_loco_clean as loco  # noqa: E402
from src.evaluation.bbq_evaluator import evaluate_bbq  # noqa: E402
from src.models.override import find_unknown_index  # noqa: E402


SEEDS = [42, 123, 456, 789, 999]
CONDITION_MODES = [
    ("signals+embedding+category+primary", "signals,embedding,category,primary"),
    ("minus_primary", "signals,embedding,category"),
    ("signals+embedding", "signals,embedding"),
    ("embedding_only", "embedding"),
    ("signals_only", "signals"),
    ("primary_only", "primary"),
    ("category_only", "category"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manuscript reviewer-risk audits without LLM inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="main")
    parser.add_argument("--results-dir", default="results/v2")
    parser.add_argument("--sampled-dir", default="data/sampled_v2")
    parser.add_argument("--out-dir", default="results/v2/reviewer_audits")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--samples-per-category", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau-min", type=float, default=0.0)
    parser.add_argument("--tau-max", type=float, default=1.0)
    parser.add_argument("--tau-step", type=float, default=0.025)
    parser.add_argument("--low-tau-min", type=float, default=0.0)
    parser.add_argument("--low-tau-max", type=float, default=0.10)
    parser.add_argument("--low-tau-step", type=float, default=0.01)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--lambda-bias", type=float, default=0.5)
    parser.add_argument("--lambda-lb", type=float, default=0.1)
    parser.add_argument(
        "--run-moe-subsets",
        action="store_true",
        help="Retrain all-zero/s3-only/core MoE variants. Slower, but still no LLM inference.",
    )
    return parser.parse_args()


def mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def fmt(values: list[float]) -> str:
    m, s = mean_std(values)
    return f"{m:.4f} +/- {s:.4f}"


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
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_step=args.tau_step,
        low_tau_min=args.low_tau_min,
        low_tau_max=args.low_tau_max,
        low_tau_step=args.low_tau_step,
        skip_low_threshold_audit=True,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_every=args.val_every,
        lambda_bias=args.lambda_bias,
        lambda_lb=args.lambda_lb,
    )


def split_for_seed(records: list[dict], clean_args: SimpleNamespace, seed: int):
    clean_args.current_seed = seed
    return clean.split_records(records, clean_args)


def condition_classifier_ablation(
    records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    clean_args: SimpleNamespace,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for label, mode in CONDITION_MODES:
        accs: list[float] = []
        for seed in seeds:
            train, val, test = split_for_seed(records, clean_args, seed)
            payload = clean.fit_condition_classifier(train, val, test, embeddings, categories, mode, seed)
            acc = float(payload["test"]["accuracy"])
            accs.append(acc)
            detail.append({"seed": seed, "feature_set": label, "mode": mode, "test_accuracy": acc})
        m, s = mean_std(accs)
        summary.append(
            {
                "feature_set": label,
                "mode": mode,
                "test_accuracy_mean": m,
                "test_accuracy_std": s,
                "seed_accuracies": " ".join(f"{a:.4f}" for a in accs),
            }
        )
    return detail, summary


def condition_only_predictions(records: list[dict], instances_by_id: dict[str, dict], condition_by_uid: dict[str, str]):
    preds: list[int] = []
    items: list[dict] = []
    for rec in records:
        uid = clean.uid_for(rec)
        item = instances_by_id.get(uid, rec)
        if condition_by_uid[uid] == "ambig":
            pred = find_unknown_index(item)
        else:
            pred = int(rec.get("primary_answer", -1))
        preds.append(pred)
        items.append(item)
    return preds, items


def s3_prediction_payload(records: list[dict], instances_by_id: dict[str, dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        uid = clean.uid_for(rec)
        item = instances_by_id.get(uid, rec)
        rows.append(
            {
                "uid": uid,
                "primary_answer": int(rec.get("primary_answer", -1)),
                "p_score": float((rec.get("signals") or {}).get("s3_confidence") or 0.0),
                "item": item,
                "category": rec.get("category"),
                "context_condition": rec.get("context_condition"),
            }
        )
    return rows


def simple_policy_baselines(
    records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    instances_by_id: dict[str, dict],
    clean_args: SimpleNamespace,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = split_for_seed(records, clean_args, seed)
        items = [instances_by_id[clean.uid_for(r)] for r in test]
        primary_preds = [int(r.get("primary_answer", -1)) for r in test]
        rows.append({"seed": seed, "system": "primary_answer_only", **evaluate_bbq(primary_preds, items)})

        for label, mode in [
            ("condition_only_full_features", "signals,embedding,category,primary"),
            ("condition_only_signals_embedding", "signals,embedding"),
            ("condition_only_embedding", "embedding"),
        ]:
            payload = clean.fit_condition_classifier(train, val, test, embeddings, categories, mode, seed)
            preds, pred_items = condition_only_predictions(test, instances_by_id, payload["test"]["condition_by_uid"])
            rows.append({"seed": seed, "system": label, **evaluate_bbq(preds, pred_items)})

        clf = clean.fit_condition_classifier(train, val, test, embeddings, categories, "signals,embedding,category,primary", seed)
        eval_payload = clean.evaluate_moe_variants(
            s3_prediction_payload(val, instances_by_id),
            s3_prediction_payload(test, instances_by_id),
            predicted_val_condition=clf["val"]["condition_by_uid"],
            predicted_test_condition=clf["test"]["condition_by_uid"],
            args=clean_args,
        )
        s3 = eval_payload["variants"]["ours_predicted_condition"]
        rows.append(
            {
                "seed": seed,
                "system": "s3_only_predicted_condition",
                **s3["metrics"],
                "thresholds": json.dumps(s3["thresholds"], sort_keys=True),
            }
        )
    summary: list[dict[str, Any]] = []
    for system in sorted({r["system"] for r in rows}):
        sys_rows = [r for r in rows if r["system"] == system]
        out = {"system": system, "n_seeds": len(sys_rows)}
        for metric in ("accuracy_amb", "accuracy_dis", "false_abstention_rate"):
            vals = [float(r[metric]) for r in sys_rows]
            m, s = mean_std(vals)
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s
        summary.append(out)
    return rows, summary


def loco_condition_ablation(
    records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    clean_args: SimpleNamespace,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    modes = [
        ("signals+embedding+primary", "signals,embedding,primary"),
        ("signals+embedding", "signals,embedding"),
        ("embedding_only", "embedding"),
        ("signals_only", "signals"),
    ]
    for seed in seeds:
        for fold_index, held_out in enumerate(categories):
            fold_seed = seed + 1009 * (fold_index + 1)
            pool = [r for r in records if r.get("category") != held_out]
            held = [r for r in records if r.get("category") == held_out]
            train, val = loco.stable_train_val_split(pool, clean_args.val_split, fold_seed)
            for label, mode in modes:
                payload = clean.fit_condition_classifier(train, val, held, embeddings, categories, mode, fold_seed)
                detail.append(
                    {
                        "seed": seed,
                        "held_out_category": held_out,
                        "feature_set": label,
                        "mode": mode,
                        "heldout_accuracy": float(payload["test"]["accuracy"]),
                    }
                )
    summary: list[dict[str, Any]] = []
    for label in sorted({r["feature_set"] for r in detail}):
        vals = [float(r["heldout_accuracy"]) for r in detail if r["feature_set"] == label]
        m, s = mean_std(vals)
        summary.append({"feature_set": label, "n_folds": len(vals), "heldout_accuracy_mean": m, "heldout_accuracy_std": s})
    return detail, summary


def threshold_plateau(clean_dir: Path) -> list[dict[str, Any]]:
    path = clean_dir / "low_threshold_audit.csv"
    wanted = {0.00, 0.01, 0.02, 0.03, 0.05, 0.10}
    grouped: dict[float, list[dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "test":
                continue
            tau = round(float(row["tau_dis"]), 2)
            if tau not in wanted:
                continue
            grouped.setdefault(tau, []).append(
                {
                    "accuracy_amb": float(row["accuracy_amb"]),
                    "accuracy_dis": float(row["accuracy_dis"]),
                    "false_abstention_rate": float(row["false_abstention_rate"]),
                    "score": float(row["score"]),
                }
            )
    rows: list[dict[str, Any]] = []
    for tau in sorted(grouped):
        vals = grouped[tau]
        out = {"tau_dis": tau, "n_seeds": len(vals)}
        for metric in ("accuracy_amb", "accuracy_dis", "false_abstention_rate", "score"):
            xs = [v[metric] for v in vals]
            m, s = mean_std(xs)
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s
        rows.append(out)
    return rows


def self_debiasing_full_coverage(
    records: list[dict],
    instances_by_id: dict[str, dict],
    clean_args: SimpleNamespace,
    seeds: list[int],
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = clean.load_baseline_predictions(path)["predictions"]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        _, _, test = split_for_seed(records, clean_args, seed)
        preds: list[Any] = []
        fallback = 0
        items: list[dict] = []
        for rec in test:
            uid = clean.uid_for(rec)
            if uid in base:
                preds.append(base[uid])
            else:
                preds.append(int(rec.get("primary_answer", -1)))
                fallback += 1
            items.append(instances_by_id[uid])
        rows.append({"seed": seed, "system": "self_debiasing_full_or_primary_fallback", "fallback_n": fallback, **evaluate_bbq(preds, items)})
    summary: list[dict[str, Any]] = []
    out = {"system": "self_debiasing_full_or_primary_fallback", "n_seeds": len(rows), "fallback_n_max": max(r["fallback_n"] for r in rows)}
    for metric in ("accuracy_amb", "accuracy_dis", "false_abstention_rate"):
        vals = [float(r[metric]) for r in rows]
        m, s = mean_std(vals)
        out[f"{metric}_mean"] = m
        out[f"{metric}_std"] = s
    summary.append(out)
    return rows, summary


def mask_records(records: list[dict], keep: set[str]) -> list[dict]:
    out: list[dict] = []
    for rec in records:
        new = dict(rec)
        sig = dict(rec.get("signals") or {})
        for name in clean.SIGNAL_NAMES:
            if name not in keep:
                sig[name] = 0.0
        new["signals"] = sig
        out.append(new)
    return out


def moe_subset_ablation(
    records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    instances_by_id: dict[str, dict],
    config: dict,
    clean_args: SimpleNamespace,
    seeds: list[int],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = {
        "all_zero_moe": set(),
        "s3_only_moe": {"s3_confidence"},
        "core4_s1346_moe": {"s1_evidence", "s3_confidence", "s4_consistency", "s6_prompt_sensitivity"},
    }
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = split_for_seed(records, clean_args, seed)
        clf = clean.fit_condition_classifier(train, val, test, embeddings, categories, "signals,embedding,category,primary", seed)
        for name, keep in variants.items():
            model, train_out = clean.train_moe(
                mask_records(train, keep),
                mask_records(val, keep),
                embeddings,
                config,
                clean_args,
                seed,
                save_dir=out_dir / "checkpoints" / f"seed_{seed}" / name,
            )
            eval_payload = clean.evaluate_moe_variants(
                clean.predict_records(model, mask_records(val, keep), embeddings, instances_by_id),
                clean.predict_records(model, mask_records(test, keep), embeddings, instances_by_id),
                predicted_val_condition=clf["val"]["condition_by_uid"],
                predicted_test_condition=clf["test"]["condition_by_uid"],
                args=clean_args,
            )
            payload = eval_payload["variants"]["ours_predicted_condition"]
            rows.append(
                {
                    "seed": seed,
                    "variant": name,
                    "signals_kept": ",".join(sorted(keep)) or "<none>",
                    "best_val_loss": train_out.get("best_val_loss"),
                    "best_epoch": train_out.get("best_epoch"),
                    **payload["metrics"],
                    "thresholds": json.dumps(payload["thresholds"], sort_keys=True),
                }
            )
    summary: list[dict[str, Any]] = []
    for variant in sorted(variants):
        vals = [r for r in rows if r["variant"] == variant]
        out = {"variant": variant, "n_seeds": len(vals), "signals_kept": ",".join(sorted(variants[variant])) or "<none>"}
        for metric in ("accuracy_amb", "accuracy_dis", "false_abstention_rate"):
            xs = [float(r[metric]) for r in vals]
            m, s = mean_std(xs)
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s
        summary.append(out)
    return rows, summary


def write_report(out_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Reviewer Audit Report", ""]
    lines.append("All results are computed from saved signals, embeddings, and predictions; no LLM inference is run.")
    lines.append("")
    lines.append("## Condition Classifier Feature Ablation")
    lines.append("| Feature set | Test acc. |")
    lines.append("|---|---:|")
    for row in tables["condition_summary"]:
        lines.append(f"| {row['feature_set']} | {row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f} |")
    lines.append("")
    lines.append("## Simple Policy Baselines")
    lines.append("| System | Acc_amb | Acc_dis | FAR |")
    lines.append("|---|---:|---:|---:|")
    for row in tables["simple_summary"]:
        lines.append(
            f"| {row['system']} | {row['accuracy_amb_mean']:.4f} +/- {row['accuracy_amb_std']:.4f} | "
            f"{row['accuracy_dis_mean']:.4f} +/- {row['accuracy_dis_std']:.4f} | "
            f"{row['false_abstention_rate_mean']:.4f} +/- {row['false_abstention_rate_std']:.4f} |"
        )
    lines.append("")
    lines.append("## LOCO Condition Prediction")
    lines.append("| Feature set | Held-out acc. |")
    lines.append("|---|---:|")
    for row in tables["loco_condition_summary"]:
        lines.append(f"| {row['feature_set']} | {row['heldout_accuracy_mean']:.4f} +/- {row['heldout_accuracy_std']:.4f} |")
    lines.append("")
    lines.append("## Low-threshold Plateau")
    lines.append("| tau_dis | Acc_amb | Acc_dis | FAR |")
    lines.append("|---:|---:|---:|---:|")
    for row in tables["threshold_plateau"]:
        lines.append(
            f"| {row['tau_dis']:.2f} | {row['accuracy_amb_mean']:.4f} +/- {row['accuracy_amb_std']:.4f} | "
            f"{row['accuracy_dis_mean']:.4f} +/- {row['accuracy_dis_std']:.4f} | "
            f"{row['false_abstention_rate_mean']:.4f} +/- {row['false_abstention_rate_std']:.4f} |"
        )
    if tables.get("moe_subset_summary"):
        lines.append("")
        lines.append("## MoE Signal Subsets")
        lines.append("| Variant | Signals kept | Acc_amb | Acc_dis | FAR |")
        lines.append("|---|---|---:|---:|---:|")
        for row in tables["moe_subset_summary"]:
            lines.append(
                f"| {row['variant']} | {row['signals_kept']} | "
                f"{row['accuracy_amb_mean']:.4f} +/- {row['accuracy_amb_std']:.4f} | "
                f"{row['accuracy_dis_mean']:.4f} +/- {row['accuracy_dis_std']:.4f} | "
                f"{row['false_abstention_rate_mean']:.4f} +/- {row['false_abstention_rate_std']:.4f} |"
            )
    (out_dir / "reviewer_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_args = base_args(args)
    config = clean.load_experiment_config(clean_args)
    records, embeddings, instances_by_id = clean.load_records_embeddings_instances(config, clean_args)
    categories = list(config["data"]["categories"])

    condition_detail, condition_summary = condition_classifier_ablation(records, embeddings, categories, clean_args, args.seeds)
    simple_detail, simple_summary = simple_policy_baselines(records, embeddings, categories, instances_by_id, clean_args, args.seeds)
    loco_detail, loco_summary = loco_condition_ablation(records, embeddings, categories, clean_args, args.seeds)
    threshold_rows = threshold_plateau(Path(args.results_dir) / "clean_experiments")
    self_detail, self_summary = self_debiasing_full_coverage(
        records,
        instances_by_id,
        clean_args,
        args.seeds,
        Path(args.results_dir) / "baselines" / "self_debiasing" / "predictions.jsonl",
    )

    write_csv(out_dir / "condition_classifier_ablation.csv", condition_detail)
    write_csv(out_dir / "condition_classifier_ablation_summary.csv", condition_summary)
    write_csv(out_dir / "simple_policy_baselines.csv", simple_detail)
    write_csv(out_dir / "simple_policy_baselines_summary.csv", simple_summary)
    write_csv(out_dir / "loco_condition_ablation.csv", loco_detail)
    write_csv(out_dir / "loco_condition_ablation_summary.csv", loco_summary)
    write_csv(out_dir / "threshold_plateau_summary.csv", threshold_rows)
    write_csv(out_dir / "self_debiasing_full_coverage.csv", self_detail)
    write_csv(out_dir / "self_debiasing_full_coverage_summary.csv", self_summary)

    tables = {
        "condition_summary": condition_summary,
        "simple_summary": simple_summary,
        "loco_condition_summary": loco_summary,
        "threshold_plateau": threshold_rows,
    }
    if args.run_moe_subsets:
        subset_detail, subset_summary = moe_subset_ablation(
            records,
            embeddings,
            categories,
            instances_by_id,
            config,
            clean_args,
            args.seeds,
            out_dir,
        )
        write_csv(out_dir / "moe_signal_subset_ablation.csv", subset_detail)
        write_csv(out_dir / "moe_signal_subset_ablation_summary.csv", subset_summary)
        tables["moe_subset_summary"] = subset_summary

    write_report(out_dir, tables)
    print(f"Wrote reviewer audits to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
