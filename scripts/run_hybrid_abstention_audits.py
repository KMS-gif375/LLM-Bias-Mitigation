#!/usr/bin/env python3
"""Hybrid abstention audits for reviewer-facing robustness claims.

This script answers a narrow question:

    Does the seven-signal layer help when the no-oracle condition classifier is
    uncertain or distribution-shifted?

It uses saved BBQ signals/embeddings only. No LLM inference is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
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


LOGGER = logging.getLogger("hybrid_audits")
DEFAULT_SEEDS = [42, 123, 456, 789, 999]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple-vs-hybrid abstention audits from saved signals.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="main")
    parser.add_argument("--results-dir", default="results/v2")
    parser.add_argument("--sampled-dir", default="data/sampled_v2")
    parser.add_argument("--out-dir", default="results/v2/hybrid_abstention_audits")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--no-discover-categories", action="store_true")
    parser.add_argument("--samples-per-category", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
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

    parser.add_argument("--tau-min", type=float, default=0.0)
    parser.add_argument("--tau-max", type=float, default=1.0)
    parser.add_argument("--tau-step", type=float, default=0.025)
    parser.add_argument("--low-tau-min", type=float, default=0.0)
    parser.add_argument("--low-tau-max", type=float, default=0.10)
    parser.add_argument("--low-tau-step", type=float, default=0.01)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    parser.add_argument("--condition-features", default="signals,embedding,category,primary")
    parser.add_argument(
        "--loco-condition-features",
        default="signals,embedding,primary",
        help="LOCO condition classifier features. Category is excluded by default for unseen held-out categories.",
    )
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--run-loco", action="store_true")
    parser.add_argument("--max-loco-categories", type=int, default=None)
    parser.add_argument("--skip-low-label", action="store_true")
    parser.add_argument("--label-fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0])
    parser.add_argument("--confidence-min", type=float, default=0.50)
    parser.add_argument("--confidence-max", type=float, default=0.95)
    parser.add_argument("--confidence-step", type=float, default=0.05)
    parser.add_argument("--uncertainty-cuts", type=float, nargs="+", default=[0.55, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def base_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        model=args.model,
        results_dir=args.results_dir,
        sampled_dir=args.sampled_dir,
        categories=args.categories,
        no_discover_categories=args.no_discover_categories,
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
        baselines=[],
        condition_features=args.condition_features,
        skip_condition_classifier=False,
        run_signal_ablation=False,
        dry_run=args.dry_run,
    )


def threshold_values(args: argparse.Namespace) -> list[float]:
    vals = np.arange(args.tau_min, args.tau_max + args.tau_step / 2, args.tau_step)
    return [round(float(v), 4) for v in vals]


def confidence_values(args: argparse.Namespace) -> list[float]:
    vals = np.arange(args.confidence_min, args.confidence_max + args.confidence_step / 2, args.confidence_step)
    return [round(float(v), 4) for v in vals]


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_for_seed(records: list[dict], clean_args: SimpleNamespace, seed: int):
    clean_args.current_seed = seed
    return clean.split_records(records, clean_args)


def uid_for(record: dict) -> str:
    return clean.uid_for(record)


def simple_predictions(
    records: list[dict],
    instances_by_id: dict[str, dict],
    condition_by_uid: dict[str, str],
) -> tuple[list[int], list[dict], dict[str, int]]:
    preds: list[int] = []
    items: list[dict] = []
    by_uid: dict[str, int] = {}
    for rec in records:
        uid = uid_for(rec)
        item = instances_by_id.get(uid, rec)
        cond = condition_by_uid.get(uid, "disambig")
        pred = find_unknown_index(item) if cond == "ambig" else int(rec.get("primary_answer", -1))
        preds.append(pred)
        items.append(item)
        by_uid[uid] = pred
    return preds, items, by_uid


def primary_predictions(records: list[dict], instances_by_id: dict[str, dict]) -> tuple[list[int], list[dict], dict[str, int]]:
    preds: list[int] = []
    items: list[dict] = []
    by_uid: dict[str, int] = {}
    for rec in records:
        uid = uid_for(rec)
        item = instances_by_id.get(uid, rec)
        pred = int(rec.get("primary_answer", -1))
        preds.append(pred)
        items.append(item)
        by_uid[uid] = pred
    return preds, items, by_uid


def hybrid_predictions(
    moe_predictions: list[dict],
    condition_by_uid: dict[str, str],
    probability_by_uid: dict[str, dict[str, float]],
    conf_threshold: float,
    tau_risk: float,
) -> tuple[list[int], list[dict], dict[str, int]]:
    preds: list[int] = []
    items: list[dict] = []
    by_uid: dict[str, int] = {}
    for pred in moe_predictions:
        uid = pred["uid"]
        item = pred["item"]
        cond = condition_by_uid.get(uid, "disambig")
        probs = probability_by_uid.get(uid, {})
        confidence = float(probs.get("confidence", 1.0))
        if confidence >= conf_threshold:
            final = find_unknown_index(item) if cond == "ambig" else int(pred["primary_answer"])
        else:
            final = find_unknown_index(item) if float(pred["p_score"]) < tau_risk else int(pred["primary_answer"])
        preds.append(final)
        items.append(item)
        by_uid[uid] = final
    return preds, items, by_uid


def objective(metrics: dict[str, Any]) -> tuple[float, float, float]:
    macro = (float(metrics.get("accuracy_amb", 0.0)) + float(metrics.get("accuracy_dis", 0.0))) / 2.0
    far = float(metrics.get("false_abstention_rate", 1.0))
    acc_amb = float(metrics.get("accuracy_amb", 0.0))
    return macro, -far, acc_amb


def tune_hybrid(
    val_predictions: list[dict],
    condition_by_uid: dict[str, str],
    probability_by_uid: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for conf_threshold in confidence_values(args):
        for tau_risk in threshold_values(args):
            preds, items, _ = hybrid_predictions(
                val_predictions,
                condition_by_uid,
                probability_by_uid,
                conf_threshold,
                tau_risk,
            )
            metrics = evaluate_bbq(preds, items)
            candidate = {
                "conf_threshold": conf_threshold,
                "tau_risk": tau_risk,
                "metrics": metrics,
                "score": objective(metrics),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    assert best is not None
    return best


def payload_from_predictions(name: str, preds: list[int], items: list[dict], by_uid: dict[str, int], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "system": name,
        "metrics": evaluate_bbq(preds, items),
        "predictions_by_uid": by_uid,
    }
    if extra:
        payload.update(extra)
    return payload


def metric_row(scope: str, seed: int, system: str, payload: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = payload["metrics"]
    row = {
        "scope": scope,
        "seed": seed,
        "system": system,
        "n_total": metrics.get("n_total"),
        "n_ambig": metrics.get("n_ambig"),
        "n_disambig": metrics.get("n_disambig"),
        "accuracy_amb": metrics.get("accuracy_amb"),
        "accuracy_dis": metrics.get("accuracy_dis"),
        "bias_score_amb": metrics.get("bias_score_amb"),
        "false_abstention_rate": metrics.get("false_abstention_rate"),
    }
    if extra:
        row.update(extra)
    return row


def ordered_subset_from_map(
    records: list[dict],
    instances_by_id: dict[str, dict],
    predictions_by_uid: dict[str, int],
    allowed_uids: set[str],
) -> tuple[list[int], list[dict]]:
    preds: list[int] = []
    items: list[dict] = []
    for rec in records:
        uid = uid_for(rec)
        if uid not in allowed_uids or uid not in predictions_by_uid:
            continue
        preds.append(predictions_by_uid[uid])
        items.append(instances_by_id.get(uid, rec))
    return preds, items


def uncertainty_rows(
    scope: str,
    seed: int,
    records: list[dict],
    instances_by_id: dict[str, dict],
    probability_by_uid: dict[str, dict[str, float]],
    systems: dict[str, dict[str, Any]],
    cuts: list[float],
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cut in cuts:
        allowed = {
            uid_for(rec)
            for rec in records
            if float(probability_by_uid.get(uid_for(rec), {}).get("confidence", 1.0)) < cut
        }
        for system, payload in systems.items():
            preds, items = ordered_subset_from_map(records, instances_by_id, payload["predictions_by_uid"], allowed)
            metrics = evaluate_bbq(preds, items) if items else {
                "n_total": 0,
                "n_ambig": 0,
                "n_disambig": 0,
                "accuracy_amb": None,
                "accuracy_dis": None,
                "bias_score_amb": None,
                "false_abstention_rate": None,
            }
            row = {
                "scope": scope,
                "seed": seed,
                "confidence_cut": cut,
                "system": system,
                "n_total": metrics.get("n_total"),
                "n_ambig": metrics.get("n_ambig"),
                "n_disambig": metrics.get("n_disambig"),
                "accuracy_amb": metrics.get("accuracy_amb"),
                "accuracy_dis": metrics.get("accuracy_dis"),
                "bias_score_amb": metrics.get("bias_score_amb"),
                "false_abstention_rate": metrics.get("false_abstention_rate"),
            }
            if extra:
                row.update(extra)
            rows.append(row)
    return rows


def sample_labeled_records(records: list[dict], frac: float, seed: int) -> list[dict]:
    if frac >= 0.999:
        return records
    rng = random.Random(seed)
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_condition[str(rec.get("context_condition", "_unknown"))].append(rec)
    sampled: list[dict] = []
    for group in by_condition.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n = max(2, int(round(len(shuffled) * frac))) if len(shuffled) >= 2 else len(shuffled)
        sampled.extend(shuffled[:n])
    rng.shuffle(sampled)
    return sampled


def run_clean_seed(
    seed: int,
    records: list[dict],
    embeddings: dict[str, Any],
    instances_by_id: dict[str, dict],
    categories: list[str],
    config: dict,
    clean_args: SimpleNamespace,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_records, val_records, test_records = split_for_seed(records, clean_args, seed)
    LOGGER.info("[clean seed=%d] train=%d val=%d test=%d", seed, len(train_records), len(val_records), len(test_records))

    condition_payload = clean.fit_condition_classifier(
        train_records,
        val_records,
        test_records,
        embeddings,
        categories,
        args.condition_features,
        seed,
    )
    LOGGER.info(
        "[clean seed=%d] condition val=%.4f test=%.4f",
        seed,
        condition_payload["val"]["accuracy"],
        condition_payload["test"]["accuracy"],
    )

    model, train_out = clean.train_moe(
        train_records,
        val_records,
        embeddings,
        config,
        clean_args,
        seed,
        save_dir=out_dir / "checkpoints" / "clean" / f"seed_{seed}" / "main",
    )
    LOGGER.info("[clean seed=%d] MoE best_val_loss=%s epoch=%s", seed, train_out.get("best_val_loss"), train_out.get("best_epoch"))
    val_predictions = clean.predict_records(model, val_records, embeddings, instances_by_id)
    test_predictions = clean.predict_records(model, test_records, embeddings, instances_by_id)

    simple_pred, simple_items, simple_by_uid = simple_predictions(test_records, instances_by_id, condition_payload["test"]["condition_by_uid"])
    primary_pred, primary_items, primary_by_uid = primary_predictions(test_records, instances_by_id)
    moe_eval = clean.evaluate_moe_variants(
        val_predictions,
        test_predictions,
        predicted_val_condition=condition_payload["val"]["condition_by_uid"],
        predicted_test_condition=condition_payload["test"]["condition_by_uid"],
        args=clean_args,
    )
    hybrid_search = tune_hybrid(
        val_predictions,
        condition_payload["val"]["condition_by_uid"],
        condition_payload["val"]["probability_by_uid"],
        args,
    )
    hybrid_pred, hybrid_items, hybrid_by_uid = hybrid_predictions(
        test_predictions,
        condition_payload["test"]["condition_by_uid"],
        condition_payload["test"]["probability_by_uid"],
        hybrid_search["conf_threshold"],
        hybrid_search["tau_risk"],
    )

    systems = {
        "primary_answer": payload_from_predictions("primary_answer", primary_pred, primary_items, primary_by_uid),
        "simple_condition_only": payload_from_predictions(
            "simple_condition_only",
            simple_pred,
            simple_items,
            simple_by_uid,
            {"condition_accuracy": condition_payload["test"]["accuracy"]},
        ),
        "hybrid_uncertain_signal_fallback": payload_from_predictions(
            "hybrid_uncertain_signal_fallback",
            hybrid_pred,
            hybrid_items,
            hybrid_by_uid,
            {
                "condition_accuracy": condition_payload["test"]["accuracy"],
                "conf_threshold": hybrid_search["conf_threshold"],
                "tau_risk": hybrid_search["tau_risk"],
                "val_metrics": hybrid_search["metrics"],
            },
        ),
    }
    if "ours_predicted_condition" in moe_eval["variants"]:
        systems["seven_signal_moe_predicted"] = {
            "metrics": moe_eval["variants"]["ours_predicted_condition"]["metrics"],
            "predictions_by_uid": moe_eval["variants"]["ours_predicted_condition"]["predictions_by_uid"],
            "thresholds": moe_eval["variants"]["ours_predicted_condition"]["thresholds"],
        }

    metrics_rows = [
        metric_row(
            "clean",
            seed,
            name,
            payload,
            {
                "condition_accuracy": payload.get("condition_accuracy"),
                "conf_threshold": payload.get("conf_threshold"),
                "tau_risk": payload.get("tau_risk"),
            },
        )
        for name, payload in systems.items()
    ]
    uncertain = uncertainty_rows(
        "clean_uncertain_subset",
        seed,
        test_records,
        instances_by_id,
        condition_payload["test"]["probability_by_uid"],
        systems,
        args.uncertainty_cuts,
    )

    low_label_rows: list[dict[str, Any]] = []
    if not args.skip_low_label:
        for frac in args.label_fracs:
            labeled_train = sample_labeled_records(train_records, frac, seed + int(frac * 10000))
            low_payload = clean.fit_condition_classifier(
                labeled_train,
                val_records,
                test_records,
                embeddings,
                categories,
                args.condition_features,
                seed + int(frac * 10000),
            )
            low_simple_pred, low_simple_items, low_simple_by_uid = simple_predictions(
                test_records,
                instances_by_id,
                low_payload["test"]["condition_by_uid"],
            )
            low_hybrid_search = tune_hybrid(
                val_predictions,
                low_payload["val"]["condition_by_uid"],
                low_payload["val"]["probability_by_uid"],
                args,
            )
            low_hybrid_pred, low_hybrid_items, low_hybrid_by_uid = hybrid_predictions(
                test_predictions,
                low_payload["test"]["condition_by_uid"],
                low_payload["test"]["probability_by_uid"],
                low_hybrid_search["conf_threshold"],
                low_hybrid_search["tau_risk"],
            )
            for name, preds, items, by_uid, search in [
                ("simple_condition_only", low_simple_pred, low_simple_items, low_simple_by_uid, None),
                ("hybrid_uncertain_signal_fallback", low_hybrid_pred, low_hybrid_items, low_hybrid_by_uid, low_hybrid_search),
            ]:
                payload = payload_from_predictions(name, preds, items, by_uid)
                low_label_rows.append(
                    metric_row(
                        "clean_low_label",
                        seed,
                        name,
                        payload,
                        {
                            "label_frac": frac,
                            "condition_train_n": len(labeled_train),
                            "condition_accuracy": low_payload["test"]["accuracy"],
                            "conf_threshold": None if search is None else search["conf_threshold"],
                            "tau_risk": None if search is None else search["tau_risk"],
                        },
                    )
                )

    save_prediction_rows(out_dir / "predictions" / "clean" / f"seed_{seed}.jsonl", test_predictions, systems, condition_payload["test"])
    return metrics_rows, uncertain, low_label_rows


def run_loco_fold(
    seed: int,
    fold_index: int,
    held_out: str,
    records: list[dict],
    embeddings: dict[str, Any],
    instances_by_id: dict[str, dict],
    categories: list[str],
    config: dict,
    clean_args: SimpleNamespace,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_seed = seed + 1009 * (fold_index + 1)
    train_pool = [r for r in records if r.get("category") != held_out]
    held_records = [r for r in records if r.get("category") == held_out]
    train_records, val_records = loco.stable_train_val_split(train_pool, args.val_split, fold_seed)
    LOGGER.info(
        "[loco seed=%d held=%s] train=%d val=%d held=%d",
        seed,
        held_out,
        len(train_records),
        len(val_records),
        len(held_records),
    )

    condition_payload = clean.fit_condition_classifier(
        train_records,
        val_records,
        held_records,
        embeddings,
        categories,
        args.loco_condition_features,
        fold_seed,
    )
    LOGGER.info("[loco seed=%d held=%s] condition held=%.4f", seed, held_out, condition_payload["test"]["accuracy"])
    fold_dir = out_dir / "checkpoints" / "loco" / f"seed_{seed}" / clean_safe_name(held_out)
    model, train_out = clean.train_moe(
        train_records,
        val_records,
        embeddings,
        config,
        clean_args,
        fold_seed,
        save_dir=fold_dir / "main",
    )
    LOGGER.info("[loco seed=%d held=%s] MoE best_val_loss=%s epoch=%s", seed, held_out, train_out.get("best_val_loss"), train_out.get("best_epoch"))
    val_predictions = clean.predict_records(model, val_records, embeddings, instances_by_id)
    held_predictions = clean.predict_records(model, held_records, embeddings, instances_by_id)

    simple_pred, simple_items, simple_by_uid = simple_predictions(held_records, instances_by_id, condition_payload["test"]["condition_by_uid"])
    moe_eval = clean.evaluate_moe_variants(
        val_predictions,
        held_predictions,
        predicted_val_condition=condition_payload["val"]["condition_by_uid"],
        predicted_test_condition=condition_payload["test"]["condition_by_uid"],
        args=clean_args,
    )
    hybrid_search = tune_hybrid(
        val_predictions,
        condition_payload["val"]["condition_by_uid"],
        condition_payload["val"]["probability_by_uid"],
        args,
    )
    hybrid_pred, hybrid_items, hybrid_by_uid = hybrid_predictions(
        held_predictions,
        condition_payload["test"]["condition_by_uid"],
        condition_payload["test"]["probability_by_uid"],
        hybrid_search["conf_threshold"],
        hybrid_search["tau_risk"],
    )
    systems = {
        "simple_condition_only": payload_from_predictions(
            "simple_condition_only",
            simple_pred,
            simple_items,
            simple_by_uid,
            {"condition_accuracy": condition_payload["test"]["accuracy"]},
        ),
        "hybrid_uncertain_signal_fallback": payload_from_predictions(
            "hybrid_uncertain_signal_fallback",
            hybrid_pred,
            hybrid_items,
            hybrid_by_uid,
            {
                "condition_accuracy": condition_payload["test"]["accuracy"],
                "conf_threshold": hybrid_search["conf_threshold"],
                "tau_risk": hybrid_search["tau_risk"],
                "val_metrics": hybrid_search["metrics"],
            },
        ),
    }
    if "ours_predicted_condition" in moe_eval["variants"]:
        systems["seven_signal_moe_predicted"] = {
            "metrics": moe_eval["variants"]["ours_predicted_condition"]["metrics"],
            "predictions_by_uid": moe_eval["variants"]["ours_predicted_condition"]["predictions_by_uid"],
            "thresholds": moe_eval["variants"]["ours_predicted_condition"]["thresholds"],
        }

    extra = {"held_out_category": held_out, "fold_seed": fold_seed}
    metrics_rows = [
        metric_row(
            "loco",
            seed,
            name,
            payload,
            {
                **extra,
                "condition_accuracy": payload.get("condition_accuracy"),
                "conf_threshold": payload.get("conf_threshold"),
                "tau_risk": payload.get("tau_risk"),
            },
        )
        for name, payload in systems.items()
    ]
    uncertain = uncertainty_rows(
        "loco_uncertain_subset",
        seed,
        held_records,
        instances_by_id,
        condition_payload["test"]["probability_by_uid"],
        systems,
        args.uncertainty_cuts,
        extra=extra,
    )
    save_prediction_rows(
        out_dir / "predictions" / "loco" / f"seed_{seed}" / f"{clean_safe_name(held_out)}.jsonl",
        held_predictions,
        systems,
        condition_payload["test"],
    )
    return metrics_rows, uncertain


def clean_safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def save_prediction_rows(path: Path, base_predictions: list[dict], systems: dict[str, dict[str, Any]], condition_split: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cond_by_uid = condition_split.get("condition_by_uid", {})
    proba_by_uid = condition_split.get("probability_by_uid", {})
    with path.open("w", encoding="utf-8") as f:
        for pred in base_predictions:
            uid = pred["uid"]
            row = {
                "uid": uid,
                "example_id": pred["item"].get("example_id"),
                "category": pred["item"].get("category", pred.get("category")),
                "context_condition": pred["item"].get("context_condition", pred.get("context_condition")),
                "label": pred["item"].get("label"),
                "primary_answer": pred["primary_answer"],
                "moe_p_score": pred["p_score"],
                "predicted_condition": cond_by_uid.get(uid),
                "condition_confidence": proba_by_uid.get(uid, {}).get("confidence"),
            }
            for system_name, payload in systems.items():
                row[system_name] = payload["predictions_by_uid"].get(uid)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    metrics = ["accuracy_amb", "accuracy_dis", "false_abstention_rate"]
    counts = ["n_total", "n_ambig", "n_disambig"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in group_keys)].append(row)
    out_rows: list[dict[str, Any]] = []
    for key, vals in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out = {k: v for k, v in zip(group_keys, key)}
        out["n_runs"] = len(vals)
        for count in counts:
            nums = [float(v[count]) for v in vals if v.get(count) not in ("", None)]
            m, s = mean_std(nums)
            out[f"{count}_mean"] = m
            out[f"{count}_std"] = s
        for metric in metrics:
            nums = [float(v[metric]) for v in vals if v.get(metric) not in ("", None)]
            m, s = mean_std(nums)
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s
        cond_nums = [float(v["condition_accuracy"]) for v in vals if v.get("condition_accuracy") not in ("", None)]
        if cond_nums:
            m, s = mean_std(cond_nums)
            out["condition_accuracy_mean"] = m
            out["condition_accuracy_std"] = s
        out_rows.append(out)
    return out_rows


def write_report(out_dir: Path, clean_summary: list[dict[str, Any]], loco_summary: list[dict[str, Any]], low_summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Hybrid Abstention Audit Report",
        "",
        "All runs use saved signals/embeddings only; no LLM inference is run.",
        "",
    ]
    if clean_summary:
        lines += ["## Clean BBQ", "", "| System | runs | Acc_amb | Acc_dis | FAR | cond. acc. |", "|---|---:|---:|---:|---:|---:|"]
        for row in clean_summary:
            lines.append(summary_line(row, "system"))
        lines.append("")
    if loco_summary:
        lines += ["## LOCO", "", "| System | runs | Acc_amb | Acc_dis | FAR | cond. acc. |", "|---|---:|---:|---:|---:|---:|"]
        for row in loco_summary:
            lines.append(summary_line(row, "system"))
        lines.append("")
    if low_summary:
        lines += ["## Low-label Condition Classifier", "", "| frac | system | runs | Acc_amb | Acc_dis | FAR | cond. acc. |", "|---:|---|---:|---:|---:|---:|---:|"]
        for row in low_summary:
            lines.append(
                "| {frac} | {system} | {n_runs} | {amb} | {dis} | {far} | {cond} |".format(
                    frac=row.get("label_frac"),
                    system=row.get("system"),
                    n_runs=row.get("n_runs"),
                    amb=fmt_ms(row, "accuracy_amb"),
                    dis=fmt_ms(row, "accuracy_dis"),
                    far=fmt_ms(row, "false_abstention_rate"),
                    cond=fmt_ms(row, "condition_accuracy"),
                )
            )
    (out_dir / "hybrid_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_ms(row: dict[str, Any], metric: str) -> str:
    m = row.get(f"{metric}_mean")
    s = row.get(f"{metric}_std")
    if m is None:
        return "-"
    return f"{float(m):.4f}+/-{float(s or 0.0):.4f}"


def summary_line(row: dict[str, Any], label_key: str) -> str:
    return "| {label} | {n_runs} | {amb} | {dis} | {far} | {cond} |".format(
        label=row.get(label_key),
        n_runs=row.get("n_runs"),
        amb=fmt_ms(row, "accuracy_amb"),
        dis=fmt_ms(row, "accuracy_dis"),
        far=fmt_ms(row, "false_abstention_rate"),
        cond=fmt_ms(row, "condition_accuracy"),
    )


def main() -> int:
    setup_logging()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_args = base_args(args)
    config = clean.load_experiment_config(clean_args)
    categories = list(config["data"]["categories"])
    if args.max_loco_categories is not None:
        loco_categories = categories[: args.max_loco_categories]
    else:
        loco_categories = categories
    records, embeddings, instances_by_id = clean.load_records_embeddings_instances(config, clean_args)
    LOGGER.info("Loaded records=%d embeddings=%d categories=%d", len(records), len(embeddings), len(categories))

    if args.dry_run:
        LOGGER.info("[dry-run] seeds=%s categories=%s", args.seeds, ",".join(loco_categories))
        return 0

    clean_rows: list[dict[str, Any]] = []
    uncertain_rows_all: list[dict[str, Any]] = []
    low_label_rows: list[dict[str, Any]] = []
    loco_rows: list[dict[str, Any]] = []

    if not args.skip_clean:
        for seed in args.seeds:
            rows, uncertain, low_rows = run_clean_seed(
                seed,
                records,
                embeddings,
                instances_by_id,
                categories,
                config,
                clean_args,
                args,
                out_dir,
            )
            clean_rows.extend(rows)
            uncertain_rows_all.extend(uncertain)
            low_label_rows.extend(low_rows)

    if args.run_loco:
        for seed in args.seeds:
            for fold_index, held_out in enumerate(loco_categories):
                rows, uncertain = run_loco_fold(
                    seed,
                    fold_index,
                    held_out,
                    records,
                    embeddings,
                    instances_by_id,
                    categories,
                    config,
                    clean_args,
                    args,
                    out_dir,
                )
                loco_rows.extend(rows)
                uncertain_rows_all.extend(uncertain)

    write_csv(out_dir / "clean_system_metrics.csv", clean_rows)
    write_csv(out_dir / "loco_system_metrics.csv", loco_rows)
    write_csv(out_dir / "uncertain_subset_metrics.csv", uncertain_rows_all)
    write_csv(out_dir / "low_label_metrics.csv", low_label_rows)

    clean_summary = aggregate_rows(clean_rows, ["scope", "system"]) if clean_rows else []
    loco_summary = aggregate_rows(loco_rows, ["scope", "system"]) if loco_rows else []
    low_summary = aggregate_rows(low_label_rows, ["scope", "label_frac", "system"]) if low_label_rows else []
    uncertain_summary = aggregate_rows(uncertain_rows_all, ["scope", "confidence_cut", "system"]) if uncertain_rows_all else []
    write_csv(out_dir / "clean_system_summary.csv", clean_summary)
    write_csv(out_dir / "loco_system_summary.csv", loco_summary)
    write_csv(out_dir / "low_label_summary.csv", low_summary)
    write_csv(out_dir / "uncertain_subset_summary.csv", uncertain_summary)
    write_report(out_dir, clean_summary, loco_summary, low_summary)
    LOGGER.info("Wrote hybrid audit outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
