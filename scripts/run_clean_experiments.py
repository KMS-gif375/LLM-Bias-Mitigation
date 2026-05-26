#!/usr/bin/env python3
"""
Run the clean follow-up experiments without LLM inference.

This script reuses saved Stage 2 signal JSONL files and cached embeddings. If
embedding caches are missing, run_pipeline._collect_records_and_embeddings may
create sentence-transformer embeddings, but it does not call the LLM.

Default suite:
  1. 70/15/15 train/val/test split from saved signal records.
  2. Train MoE confidence estimator on train only.
  3. Tune thresholds on val only.
  4. Evaluate on held-out test:
       - single tau
       - oracle per-condition tau
       - predicted-condition tau (no oracle condition label at test time)
  5. Re-evaluate saved baselines on the exact same test IDs.
  6. Compute compact bootstrap CIs and paired bootstrap tests on matched IDs.

Optional:
  --run-signal-ablation retrains full + leave-one-signal-out MoE models and
  evaluates each on the same held-out test split. This is slower but still
  uses no LLM inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("clean_experiments")

SIGNAL_NAMES: tuple[str, ...] = (
    "s1_evidence",
    "s2_counterfactual",
    "s3_confidence",
    "s4_consistency",
    "s5_bias_head",
    "s6_prompt_sensitivity",
    "s7_sae_feature",
)

DEFAULT_BOOTSTRAP_METRICS: tuple[str, ...] = (
    "accuracy_amb",
    "accuracy_dis",
    "false_abstention_rate",
    "bias_abs_amb",
)


@dataclass
class PipelineArgs:
    model: str
    categories: Optional[list[str]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean MoE/baseline/statistical follow-up experiments without LLM inference."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="main")
    parser.add_argument("--results-dir", default="results/v2")
    parser.add_argument("--sampled-dir", default="data/sampled_v2")
    parser.add_argument("--out-dir", default="results/v2/clean_experiments")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument(
        "--no-discover-categories",
        action="store_true",
        help="Use categories from config/--categories instead of *_signals.jsonl discovery.",
    )
    parser.add_argument("--samples-per-category", type=int, default=1000)

    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
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
        "--baselines",
        nargs="*",
        default=None,
        help="Prediction JSONL files. If omitted, searches results-dir, results/v2_runpod, and results/baselines.",
    )
    parser.add_argument(
        "--condition-features",
        default="signals,embedding,category,primary",
        help="Comma-separated features for no-oracle condition classifier.",
    )
    parser.add_argument(
        "--skip-condition-classifier",
        action="store_true",
        help="Skip predicted-condition/no-oracle experiment.",
    )
    parser.add_argument(
        "--skip-low-threshold-audit",
        action="store_true",
        help="Skip sub-0.05 tau_dis audit for the lower-bound reviewer check.",
    )
    parser.add_argument(
        "--run-signal-ablation",
        action="store_true",
        help="Retrain full + leave-one-signal-out models on the clean split. Slower.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load records/splits and print planned work without training.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def discover_signal_categories(results_dir: str, model: str) -> list[str]:
    signals_dir = Path(results_dir) / "signals" / model
    cats = sorted(
        p.name[: -len("_signals.jsonl")]
        for p in signals_dir.glob("*_signals.jsonl")
        if p.name.endswith("_signals.jsonl")
    )
    if not cats:
        raise FileNotFoundError(f"No *_signals.jsonl files found under {signals_dir}")
    return cats


def load_experiment_config(args: argparse.Namespace) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyYAML is required. Run `pip install -r requirements.txt` or `pip install pyyaml` "
            "in the environment used for this experiment."
        ) from exc

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("data", {})
    config.setdefault("output", {})
    config.setdefault("moe", {})
    config.setdefault("models", {})

    config["output"]["results_dir"] = args.results_dir
    config["data"]["sampled_dir"] = args.sampled_dir
    config["data"]["samples_per_category"] = args.samples_per_category

    if args.categories:
        categories = args.categories
    elif args.no_discover_categories:
        categories = list(config["data"].get("categories", []))
    else:
        categories = discover_signal_categories(args.results_dir, args.model)
    config["data"]["categories"] = categories

    config.setdefault("moe", {}).setdefault("training", {})
    config["moe"]["training"]["val_split"] = args.val_split
    config["moe"]["training"]["test_split"] = args.test_split

    config.setdefault("models", {}).setdefault(args.model, {})
    config["models"][args.model]["device"] = args.device
    return config


def uid_for(record: dict) -> str:
    if record.get("unique_id"):
        return str(record["unique_id"])
    return f"{record.get('category', '_unknown')}::{record.get('example_id')}"


def load_records_embeddings_instances(config: dict, args: argparse.Namespace):
    from run_pipeline import _collect_records_and_embeddings, _instances_by_id

    pipe_args = PipelineArgs(model=args.model, categories=None)
    records, embeddings = _collect_records_and_embeddings(config, pipe_args)
    instances_by_id = _instances_by_id(records, config, pipe_args)
    return records, embeddings, instances_by_id


def split_records(records: list[dict], args: argparse.Namespace):
    from run_pipeline import _stratified_three_way_split

    return _stratified_three_way_split(
        records,
        val_ratio=args.val_split,
        test_ratio=args.test_split,
        seed=int(args.current_seed),
    )


def make_train_config(args: argparse.Namespace, save_dir: Path, seed: int):
    from src.models.trainer import TrainConfig

    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_every=args.val_every,
        lambda_bias=args.lambda_bias,
        lambda_lb=args.lambda_lb,
        device=args.device,
        seed=seed,
        save_dir=str(save_dir),
    )


def infer_embed_dim(embeddings: dict[str, Any], default: int = 4096) -> int:
    if not embeddings:
        return default
    sample = next(iter(embeddings.values()))
    try:
        return int(sample.shape[-1])
    except Exception:
        return default


def copy_records_with_signal_mask(records: list[dict], mask_index: int) -> list[dict]:
    if mask_index < 0:
        return records
    signal_name = SIGNAL_NAMES[mask_index]
    masked: list[dict] = []
    for rec in records:
        new_rec = dict(rec)
        new_signals = dict(rec.get("signals", {}))
        new_signals[signal_name] = 0.0
        new_rec["signals"] = new_signals
        masked.append(new_rec)
    return masked


def train_moe(
    train_records: list[dict],
    val_records: list[dict],
    embeddings: dict[str, Any],
    config: dict,
    args: argparse.Namespace,
    seed: int,
    save_dir: Path,
    mask_index: int = -1,
):
    import torch

    from src.models.moe_aggregator import MoEAggregator
    from src.models.trainer import SignalsDataset, train_moe as train_moe_impl

    train_used = copy_records_with_signal_mask(train_records, mask_index)
    val_used = copy_records_with_signal_mask(val_records, mask_index)

    train_ds = SignalsDataset(train_used, embeddings)
    val_ds = SignalsDataset(val_used, embeddings)
    embed_dim = infer_embed_dim(embeddings)
    moe_cfg = config.get("moe", {})
    model = MoEAggregator(
        signal_dim=len(SIGNAL_NAMES),
        embed_dim=embed_dim,
        num_experts=int(moe_cfg.get("num_experts", 4)),
        gating_hidden=int(moe_cfg.get("gating_hidden_dim", 64)),
        expert_hidden=int(moe_cfg.get("expert_hidden_dim", 128)),
    )
    train_config = make_train_config(args, save_dir, seed)
    out = train_moe_impl(train_ds, val_ds, model, train_config)

    ckpt_path = out.get("checkpoint_path")
    if ckpt_path and Path(ckpt_path).exists():
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location="cpu")
        model_state = state.get("model_state_dict", state)
        model.load_state_dict(model_state, strict=False)
    return model, out


def predict_records(
    model: Any,
    records: list[dict],
    embeddings: dict[str, Any],
    instances_by_id: dict[str, dict],
    mask_index: int = -1,
) -> list[dict]:
    import torch

    from src.models.moe_aggregator import signals_dict_to_tensor

    device = next(model.parameters()).device
    model.eval()
    out: list[dict] = []
    with torch.inference_mode():
        for rec in copy_records_with_signal_mask(records, mask_index):
            uid = uid_for(rec)
            if uid not in embeddings:
                continue
            item = instances_by_id.get(uid, rec)
            sig = signals_dict_to_tensor(rec.get("signals", {})).unsqueeze(0).to(device)
            emb = embeddings[uid].to(torch.float32).unsqueeze(0).to(device)
            result = model(sig, emb)
            out.append(
                {
                    "uid": uid,
                    "primary_answer": int(rec.get("primary_answer", -1)),
                    "p_score": float(result.p.item()),
                    "item": item,
                    "category": rec.get("category", item.get("category")),
                    "context_condition": rec.get(
                        "context_condition", item.get("context_condition")
                    ),
                }
            )
    return out


def threshold_values(args: argparse.Namespace) -> list[float]:
    vals = np.arange(args.tau_min, args.tau_max + args.tau_step / 2, args.tau_step)
    return [round(float(v), 4) for v in vals]


def low_threshold_values(args: argparse.Namespace) -> list[float]:
    vals = np.arange(args.low_tau_min, args.low_tau_max + args.low_tau_step / 2, args.low_tau_step)
    return [round(float(v), 4) for v in vals]


def apply_threshold_policy(
    predictions: list[dict],
    thresholds: dict[str, float],
    condition_by_uid: Optional[dict[str, str]] = None,
    use_oracle_condition: bool = False,
    default_threshold: float = 0.5,
) -> tuple[list[int], list[dict], dict[str, int]]:
    from src.models.override import apply_threshold_override

    final_preds: list[int] = []
    items: list[dict] = []
    by_uid: dict[str, int] = {}
    for pred in predictions:
        if condition_by_uid is not None:
            cond = condition_by_uid.get(pred["uid"], "")
        elif use_oracle_condition:
            cond = pred["item"].get("context_condition", pred.get("context_condition", ""))
        else:
            cond = "default"
        tau = thresholds.get(cond, thresholds.get("default", default_threshold))
        result = apply_threshold_override(
            primary_answer=pred["primary_answer"],
            p_score=pred["p_score"],
            item=pred["item"],
            threshold=tau,
        )
        final_answer = int(result["final_answer"])
        final_preds.append(final_answer)
        items.append(pred["item"])
        by_uid[pred["uid"]] = final_answer
    return final_preds, items, by_uid


def objective_score(metrics: dict[str, Any]) -> float:
    return float(metrics.get("accuracy_amb", 0.0) + metrics.get("accuracy_dis", 0.0)) / 2.0


def run_low_threshold_audit(
    val_predictions: list[dict],
    test_predictions: list[dict],
    tau_amb: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Sweep tau_dis below/around 0.05 while fixing oracle tau_amb.

    This directly addresses the reviewer concern that tau_dis=0.05 is just the
    lower edge of the canonical grid. It still uses oracle condition labels, so
    it is a threshold-shape audit, not a deployable-setting result.
    """
    from src.evaluation.bbq_evaluator import evaluate_bbq

    rows: list[dict[str, Any]] = []
    best_by_val: Optional[dict[str, Any]] = None
    at_canonical: Optional[dict[str, Any]] = None

    for tau_dis in low_threshold_values(args):
        thresholds = {"ambig": tau_amb, "disambig": tau_dis, "default": tau_dis}

        val_final, val_items, _ = apply_threshold_policy(
            val_predictions,
            thresholds=thresholds,
            use_oracle_condition=True,
            default_threshold=tau_dis,
        )
        test_final, test_items, _ = apply_threshold_policy(
            test_predictions,
            thresholds=thresholds,
            use_oracle_condition=True,
            default_threshold=tau_dis,
        )
        val_metrics = evaluate_bbq(val_final, val_items)
        test_metrics = evaluate_bbq(test_final, test_items)
        row = {
            "tau_amb": tau_amb,
            "tau_dis": tau_dis,
            "val_score": objective_score(val_metrics),
            "test_score": objective_score(test_metrics),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        rows.append(row)
        if best_by_val is None or row["val_score"] > best_by_val["val_score"]:
            best_by_val = row
        if abs(tau_dis - 0.05) < (args.low_tau_step / 2 + 1e-9):
            at_canonical = row

    return {
        "description": "oracle-condition tau_dis sub-grid audit; fixes tau_amb from canonical oracle search",
        "grid": {
            "tau_dis_min": args.low_tau_min,
            "tau_dis_max": args.low_tau_max,
            "tau_dis_step": args.low_tau_step,
        },
        "tau_amb_fixed": tau_amb,
        "best_by_val": best_by_val,
        "at_tau_dis_0_05": at_canonical,
        "rows": rows,
    }


def search_thresholds_for_predicted_condition(
    val_predictions: list[dict],
    condition_by_uid: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from src.evaluation.bbq_evaluator import evaluate_bbq

    vals = threshold_values(args)
    best: dict[str, Any] = {
        "thresholds": {"ambig": 0.5, "disambig": 0.5},
        "score": -1.0,
        "metrics": {},
    }
    for tau_amb in vals:
        for tau_dis in vals:
            thresholds = {"ambig": tau_amb, "disambig": tau_dis, "default": 0.5}
            final, items, _ = apply_threshold_policy(
                val_predictions,
                thresholds=thresholds,
                condition_by_uid=condition_by_uid,
                default_threshold=0.5,
            )
            metrics = evaluate_bbq(final, items)
            score = objective_score(metrics)
            tie_break = -float(metrics.get("false_abstention_rate", 0.0))
            best_tie = -float(best.get("metrics", {}).get("false_abstention_rate", 1.0))
            if score > best["score"] or (score == best["score"] and tie_break > best_tie):
                best = {"thresholds": thresholds, "score": score, "metrics": metrics}
    return best


def evaluate_moe_variants(
    val_predictions: list[dict],
    test_predictions: list[dict],
    predicted_val_condition: Optional[dict[str, str]],
    predicted_test_condition: Optional[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from src.evaluation.bbq_evaluator import evaluate_bbq
    from src.models.override import (
        search_optimal_threshold,
        search_optimal_threshold_per_condition,
    )

    single = search_optimal_threshold(
        val_predictions,
        threshold_range=(args.tau_min, args.tau_max),
        step=args.tau_step,
    )
    oracle = search_optimal_threshold_per_condition(
        val_predictions,
        metric_amb="accuracy_amb",
        metric_dis="accuracy_dis",
        threshold_range=(args.tau_min, args.tau_max),
        step=args.tau_step,
    )

    variants: dict[str, Any] = {}

    final, items, by_uid = apply_threshold_policy(
        test_predictions,
        thresholds={"default": single.best_threshold},
        default_threshold=single.best_threshold,
    )
    variants["ours_single_tau"] = {
        "thresholds": {"default": single.best_threshold},
        "metrics": evaluate_bbq(final, items),
        "predictions_by_uid": by_uid,
    }

    final, items, by_uid = apply_threshold_policy(
        test_predictions,
        thresholds={**oracle.thresholds, "default": single.best_threshold},
        use_oracle_condition=True,
        default_threshold=single.best_threshold,
    )
    variants["ours_per_condition_oracle"] = {
        "thresholds": oracle.thresholds,
        "metrics": evaluate_bbq(final, items),
        "predictions_by_uid": by_uid,
        "search_score": oracle.combined_score,
    }

    if predicted_val_condition is not None and predicted_test_condition is not None:
        pred_search = search_thresholds_for_predicted_condition(
            val_predictions,
            predicted_val_condition,
            args,
        )
        final, items, by_uid = apply_threshold_policy(
            test_predictions,
            thresholds=pred_search["thresholds"],
            condition_by_uid=predicted_test_condition,
            default_threshold=single.best_threshold,
        )
        variants["ours_predicted_condition"] = {
            "thresholds": pred_search["thresholds"],
            "metrics": evaluate_bbq(final, items),
            "predictions_by_uid": by_uid,
            "val_search_score": pred_search["score"],
            "val_search_metrics": pred_search["metrics"],
        }

    low_threshold_audit = None
    if not args.skip_low_threshold_audit:
        low_threshold_audit = run_low_threshold_audit(
            val_predictions,
            test_predictions,
            tau_amb=float(oracle.thresholds.get("ambig", single.best_threshold)),
            args=args,
        )

    return {
        "single_search": {
            "threshold": single.best_threshold,
            "score": single.best_score,
        },
        "oracle_per_condition_search": {
            "thresholds": oracle.thresholds,
            "score": oracle.combined_score,
        },
        "variants": variants,
        "low_threshold_audit": low_threshold_audit,
    }


def signal_vector(record: dict) -> list[float]:
    signals = record.get("signals", {})
    return [0.0 if signals.get(k) is None else float(signals.get(k, 0.0)) for k in SIGNAL_NAMES]


def build_condition_features(
    records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    feature_modes: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    rows: list[list[float]] = []
    labels: list[int] = []
    uids: list[str] = []
    for rec in records:
        uid = uid_for(rec)
        if "embedding" in feature_modes and uid not in embeddings:
            continue
        feats: list[float] = []
        if "signals" in feature_modes:
            feats.extend(signal_vector(rec))
        if "primary" in feature_modes:
            primary = int(rec.get("primary_answer", -1))
            feats.extend([1.0 if primary == v else 0.0 for v in (-1, 0, 1, 2)])
        if "category" in feature_modes:
            one_hot = [0.0] * len(categories)
            if rec.get("category") in cat_to_idx:
                one_hot[cat_to_idx[rec.get("category")]] = 1.0
            feats.extend(one_hot)
        if "embedding" in feature_modes:
            emb = embeddings[uid]
            if hasattr(emb, "detach"):
                emb_arr = emb.detach().cpu().numpy()
            else:
                emb_arr = np.asarray(emb)
            feats.extend([float(x) for x in emb_arr.reshape(-1)])
        cond = rec.get("context_condition")
        if cond not in ("ambig", "disambig"):
            continue
        rows.append(feats)
        labels.append(0 if cond == "ambig" else 1)
        uids.append(uid)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), uids


def fit_condition_classifier(
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    modes_csv: str,
    seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    modes = {m.strip() for m in modes_csv.split(",") if m.strip()}
    x_train, y_train, train_uids = build_condition_features(train_records, embeddings, categories, modes)
    x_val, y_val, val_uids = build_condition_features(val_records, embeddings, categories, modes)
    x_test, y_test, test_uids = build_condition_features(test_records, embeddings, categories, modes)

    if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
        raise ValueError("Condition classifier has an empty split; check features/embeddings.")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    clf.fit(x_train, y_train)

    def predict_split(x: np.ndarray, y: np.ndarray, uids: list[str]) -> dict[str, Any]:
        pred = clf.predict(x)
        cond_by_uid = {
            uid: ("ambig" if int(label) == 0 else "disambig")
            for uid, label in zip(uids, pred)
        }
        cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "confusion_matrix_labels": ["ambig", "disambig"],
            "confusion_matrix": cm,
            "condition_by_uid": cond_by_uid,
            "n": int(len(y)),
        }

    return {
        "features": sorted(modes),
        "train_n": int(len(y_train)),
        "val": predict_split(x_val, y_val, val_uids),
        "test": predict_split(x_test, y_test, test_uids),
    }


def load_baseline_paths(args: argparse.Namespace) -> list[Path]:
    if args.baselines:
        return [Path(p) for p in args.baselines]
    search_roots = [
        Path(args.results_dir) / "baselines",
        Path("results/v2_runpod/baselines"),
        Path("results/baselines"),
    ]
    paths: list[Path] = []
    seen: set[str] = set()
    for root in search_roots:
        for path in sorted(root.glob("*/predictions.jsonl")):
            key = str(path.resolve())
            if key not in seen:
                paths.append(path)
                seen.add(key)
    return paths


def load_baseline_predictions(path: Path) -> dict[str, Any]:
    pred_by_uid: dict[str, Any] = {}
    n_rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = f"{row.get('category', '_unknown')}::{row.get('example_id')}"
            pred = (
                row.get("prediction_text")
                if "prediction_text" in row
                else row.get("final_answer", row.get("prediction"))
            )
            pred_by_uid[uid] = pred
            n_rows += 1
    return {"name": path.parent.name, "path": str(path), "n_rows": n_rows, "predictions": pred_by_uid}


def parse_prediction_list(predictions: Iterable[Any]) -> list[int]:
    from src.evaluation.bbq_evaluator import parse_prediction

    return [parse_prediction(p) for p in predictions]


def metric_value(metric: str, items: list[dict], predictions: list[Any]) -> Optional[float]:
    from src.evaluation.bbq_evaluator import (
        compute_accuracy,
        compute_bias_score,
        compute_false_abstention_rate,
        parse_prediction,
    )

    pred_indices = [parse_prediction(p) for p in predictions]

    def split(condition: str) -> tuple[list[dict], list[int]]:
        pairs = [
            (item, pred)
            for item, pred in zip(items, pred_indices)
            if item.get("context_condition") == condition
        ]
        if not pairs:
            return [], []
        split_items, split_preds = zip(*pairs)
        return list(split_items), list(split_preds)

    if metric == "accuracy_amb":
        split_items, split_preds = split("ambig")
        return compute_accuracy(split_items, split_preds)
    if metric == "accuracy_dis":
        split_items, split_preds = split("disambig")
        return compute_accuracy(split_items, split_preds)
    if metric == "false_abstention_rate":
        split_items, split_preds = split("disambig")
        return compute_false_abstention_rate(split_items, split_preds)
    if metric == "bias_score_amb":
        split_items, split_preds = split("ambig")
        return compute_bias_score(split_items, split_preds)
    if metric == "bias_abs_amb":
        split_items, split_preds = split("ambig")
        score = compute_bias_score(split_items, split_preds)
        return None if score is None else abs(float(score))
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_ci_compact(
    predictions: list[Any],
    items: list[dict],
    metric: str,
    n_iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    n = len(items)
    point = metric_value(metric, items, predictions)
    if n == 0 or point is None:
        return {"point": point, "lower": None, "upper": None, "mean": None, "n": n}

    rng = np.random.RandomState(seed)
    samples: list[float] = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        boot_items = [items[i] for i in idx]
        boot_preds = [predictions[i] for i in idx]
        val = metric_value(metric, boot_items, boot_preds)
        if val is not None:
            samples.append(float(val))
    if not samples:
        return {"point": point, "lower": None, "upper": None, "mean": None, "n": n}

    arr = np.asarray(samples)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "point": float(point),
        "mean": float(arr.mean()),
        "lower": float(np.quantile(arr, alpha)),
        "upper": float(np.quantile(arr, 1.0 - alpha)),
        "n": n,
        "n_valid_bootstrap": len(samples),
    }


def better_direction(metric: str) -> str:
    if metric in ("accuracy_amb", "accuracy_dis"):
        return "greater"
    if metric in ("false_abstention_rate", "bias_abs_amb"):
        return "less"
    return "two_sided"


def paired_bootstrap_compact(
    predictions_a: list[Any],
    predictions_b: list[Any],
    items: list[dict],
    metric: str,
    n_iterations: int,
    seed: int,
) -> dict[str, Any]:
    n = len(items)
    score_a = metric_value(metric, items, predictions_a)
    score_b = metric_value(metric, items, predictions_b)
    if n == 0 or score_a is None or score_b is None:
        return {"diff": None, "p_value": None, "n": n}

    diff = float(score_a - score_b)
    rng = np.random.RandomState(seed)
    diffs: list[float] = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        boot_items = [items[i] for i in idx]
        boot_a = [predictions_a[i] for i in idx]
        boot_b = [predictions_b[i] for i in idx]
        a = metric_value(metric, boot_items, boot_a)
        b = metric_value(metric, boot_items, boot_b)
        if a is not None and b is not None:
            diffs.append(float(a - b))
    if not diffs:
        return {"diff": diff, "p_value": None, "n": n}

    arr = np.asarray(diffs)
    direction = better_direction(metric)
    if direction == "greater":
        p_value = float((arr <= 0).mean())
    elif direction == "less":
        p_value = float((arr >= 0).mean())
    else:
        p_value = float((np.abs(arr) >= abs(diff)).mean())
    return {
        "diff": diff,
        "p_value": p_value,
        "direction": direction,
        "n": n,
        "diff_ci_lower": float(np.quantile(arr, 0.025)),
        "diff_ci_upper": float(np.quantile(arr, 0.975)),
    }


def predictions_by_uid_to_ordered(
    pred_by_uid: dict[str, Any],
    uids: list[str],
) -> list[Any]:
    return [pred_by_uid[uid] for uid in uids]


def item_map_from_predictions(test_predictions: list[dict]) -> dict[str, dict]:
    return {p["uid"]: p["item"] for p in test_predictions}


def evaluate_baselines_and_stats(
    test_predictions: list[dict],
    variants: dict[str, Any],
    baseline_paths: list[Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from src.evaluation.bbq_evaluator import evaluate_bbq

    item_by_uid = item_map_from_predictions(test_predictions)
    test_uids = [p["uid"] for p in test_predictions]
    results: dict[str, Any] = {
        "baselines": {},
        "ci": {},
        "paired": {},
    }

    # CIs for our variants on the full test split.
    for system_name, payload in variants.items():
        pred_map = payload["predictions_by_uid"]
        uids = [uid for uid in test_uids if uid in pred_map]
        items = [item_by_uid[uid] for uid in uids]
        preds = predictions_by_uid_to_ordered(pred_map, uids)
        results["ci"][system_name] = {
            metric: bootstrap_ci_compact(
                preds,
                items,
                metric,
                n_iterations=args.bootstrap_iters,
                seed=args.bootstrap_seed,
            )
            for metric in DEFAULT_BOOTSTRAP_METRICS
        }

    for path in baseline_paths:
        if not path.exists():
            LOGGER.warning("Baseline prediction file missing: %s", path)
            continue
        base = load_baseline_predictions(path)
        base_name = base["name"]
        base_map = base["predictions"]
        shared = [uid for uid in test_uids if uid in base_map]
        missing = len(test_uids) - len(shared)
        items = [item_by_uid[uid] for uid in shared]
        base_preds = predictions_by_uid_to_ordered(base_map, shared)
        base_metrics = evaluate_bbq(base_preds, items)
        results["baselines"][base_name] = {
            "path": base["path"],
            "n_rows": base["n_rows"],
            "n_shared_with_test": len(shared),
            "n_missing_test_ids": missing,
            "metrics_on_shared_test": base_metrics,
        }
        results["ci"][base_name] = {
            metric: bootstrap_ci_compact(
                base_preds,
                items,
                metric,
                n_iterations=args.bootstrap_iters,
                seed=args.bootstrap_seed,
            )
            for metric in DEFAULT_BOOTSTRAP_METRICS
        }

        for system_name, payload in variants.items():
            sys_map = payload["predictions_by_uid"]
            paired_uids = [uid for uid in shared if uid in sys_map]
            paired_items = [item_by_uid[uid] for uid in paired_uids]
            sys_preds = predictions_by_uid_to_ordered(sys_map, paired_uids)
            paired_base_preds = predictions_by_uid_to_ordered(base_map, paired_uids)
            key = f"{system_name}_vs_{base_name}"
            results["paired"][key] = {
                metric: paired_bootstrap_compact(
                    sys_preds,
                    paired_base_preds,
                    paired_items,
                    metric,
                    n_iterations=args.bootstrap_iters,
                    seed=args.bootstrap_seed,
                )
                for metric in DEFAULT_BOOTSTRAP_METRICS
            }
    return results


def run_signal_ablation_suite(
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    embeddings: dict[str, Any],
    instances_by_id: dict[str, dict],
    config: dict,
    args: argparse.Namespace,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    LOGGER.info("Running signal ablation: full + %d masked runs", len(SIGNAL_NAMES))
    results: dict[str, Any] = {}
    full_metrics: Optional[dict[str, Any]] = None
    for mask_index, name in [(-1, "full")] + list(enumerate(SIGNAL_NAMES)):
        run_name = f"mask_{name}"
        model, train_out = train_moe(
            train_records,
            val_records,
            embeddings,
            config,
            args,
            seed,
            save_dir=out_dir / "checkpoints" / f"seed_{seed}" / "signal_ablation" / run_name,
            mask_index=mask_index,
        )
        val_preds = predict_records(model, val_records, embeddings, instances_by_id, mask_index=mask_index)
        test_preds = predict_records(model, test_records, embeddings, instances_by_id, mask_index=mask_index)
        eval_payload = evaluate_moe_variants(
            val_preds,
            test_preds,
            predicted_val_condition=None,
            predicted_test_condition=None,
            args=args,
        )
        metrics = eval_payload["variants"]["ours_per_condition_oracle"]["metrics"]
        if name == "full":
            full_metrics = metrics
        deltas = {}
        if full_metrics is not None and name != "full":
            for metric in ("accuracy_amb", "accuracy_dis", "false_abstention_rate"):
                deltas[f"delta_{metric}"] = float(full_metrics.get(metric, 0.0) - metrics.get(metric, 0.0))
            if full_metrics.get("bias_score_amb") is not None and metrics.get("bias_score_amb") is not None:
                deltas["delta_bias_abs_amb"] = (
                    abs(float(metrics["bias_score_amb"])) - abs(float(full_metrics["bias_score_amb"]))
                )
        results[name] = {
            "mask_index": mask_index,
            "train": {
                "best_val_loss": train_out.get("best_val_loss"),
                "best_epoch": train_out.get("best_epoch"),
            },
            "thresholds": eval_payload["variants"]["ours_per_condition_oracle"]["thresholds"],
            "metrics_per_condition_oracle": metrics,
            "deltas_vs_full": deltas,
        }
    return results


def save_seed_predictions(
    path: Path,
    test_predictions: list[dict],
    variants: dict[str, Any],
    predicted_condition: Optional[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pred in test_predictions:
            uid = pred["uid"]
            row = {
                "uid": uid,
                "example_id": pred["item"].get("example_id"),
                "category": pred["item"].get("category", pred.get("category")),
                "context_condition": pred["item"].get("context_condition", pred.get("context_condition")),
                "label": pred["item"].get("label"),
                "primary_answer": pred["primary_answer"],
                "p_score": pred["p_score"],
                "predicted_condition": None if predicted_condition is None else predicted_condition.get(uid),
            }
            for system_name, payload in variants.items():
                row[system_name] = payload["predictions_by_uid"].get(uid)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


SUMMARY_METRICS: tuple[str, ...] = (
    "accuracy_amb",
    "accuracy_dis",
    "bias_abs_amb",
    "false_abstention_rate",
)


def _metric_for_summary(metrics: dict[str, Any], metric: str) -> Optional[float]:
    if metric == "bias_abs_amb":
        value = metrics.get("bias_score_amb")
        return None if value is None else abs(float(value))
    value = metrics.get(metric)
    return None if value is None else float(value)


def _mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _format_mean_std(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return "-"
    return f"{mean:.4f}±{std:.4f}"


def _aggregate_system_records(all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in all_results:
        seed = result["seed"]
        for system, payload in result["moe_eval"]["variants"].items():
            grouped.setdefault((system, "full_test"), []).append(
                {"seed": seed, "metrics": payload["metrics"]}
            )
        for system, payload in result["stats"]["baselines"].items():
            grouped.setdefault((system, "shared_test"), []).append(
                {
                    "seed": seed,
                    "metrics": payload["metrics_on_shared_test"],
                    "n_shared": payload.get("n_shared_with_test"),
                    "n_missing": payload.get("n_missing_test_ids"),
                }
            )

    rows: list[dict[str, Any]] = []
    for (system, subset), records in sorted(grouped.items()):
        row: dict[str, Any] = {
            "system": system,
            "subset": subset,
            "n_seeds": len(records),
        }
        n_totals = [
            float(r["metrics"].get("n_total", 0))
            for r in records
            if r["metrics"].get("n_total") is not None
        ]
        n_mean, n_std = _mean_std(n_totals)
        row["n_total_mean"] = n_mean
        row["n_total_std"] = n_std
        n_shared_vals = [float(r["n_shared"]) for r in records if r.get("n_shared") is not None]
        n_missing_vals = [float(r["n_missing"]) for r in records if r.get("n_missing") is not None]
        row["n_shared_min"] = min(n_shared_vals) if n_shared_vals else None
        row["n_missing_max"] = max(n_missing_vals) if n_missing_vals else None
        for metric in SUMMARY_METRICS:
            vals = [
                v for v in (_metric_for_summary(r["metrics"], metric) for r in records)
                if v is not None
            ]
            mean, std = _mean_std(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return rows


def _write_aggregate_metrics_csv(rows: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "aggregate_metrics.csv"
    fieldnames = [
        "system",
        "subset",
        "n_seeds",
        "n_total_mean",
        "n_total_std",
        "n_shared_min",
        "n_missing_max",
    ]
    for metric in SUMMARY_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_low_threshold_audit_csv(all_results: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "low_threshold_audit.csv"
    fieldnames = [
        "seed",
        "tau_amb",
        "tau_dis",
        "split",
        "score",
        "n_total",
        "accuracy_amb",
        "accuracy_dis",
        "bias_score_amb",
        "bias_abs_amb",
        "false_abstention_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            audit = result.get("moe_eval", {}).get("low_threshold_audit")
            if not audit:
                continue
            for row in audit.get("rows", []):
                for split in ("val", "test"):
                    metrics = row.get(f"{split}_metrics", {})
                    bias = metrics.get("bias_score_amb")
                    writer.writerow(
                        {
                            "seed": result["seed"],
                            "tau_amb": row.get("tau_amb"),
                            "tau_dis": row.get("tau_dis"),
                            "split": split,
                            "score": row.get(f"{split}_score"),
                            "n_total": metrics.get("n_total"),
                            "accuracy_amb": metrics.get("accuracy_amb"),
                            "accuracy_dis": metrics.get("accuracy_dis"),
                            "bias_score_amb": bias,
                            "bias_abs_amb": None if bias is None else abs(float(bias)),
                            "false_abstention_rate": metrics.get("false_abstention_rate"),
                        }
                    )


def write_defense_report(all_results: list[dict[str, Any]], out_dir: Path) -> None:
    rows = _aggregate_system_records(all_results)
    _write_aggregate_metrics_csv(rows, out_dir)
    _write_low_threshold_audit_csv(all_results, out_dir)

    lines: list[str] = [
        "# Clean Defense Suite Report",
        "",
        "This report is generated without LLM inference from saved signals/predictions.",
        "",
        "## Same-Test-ID Aggregate Metrics",
        "",
        "| System | Subset | Seeds | n | acc_amb | acc_dis | abs_bias_amb | FAR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        n_mean = row.get("n_total_mean")
        n_text = "-" if n_mean is None else f"{n_mean:.0f}"
        lines.append(
            "| {system} | {subset} | {n_seeds} | {n_text} | {acc_amb} | {acc_dis} | {bias} | {far} |".format(
                system=row["system"],
                subset=row["subset"],
                n_seeds=row["n_seeds"],
                n_text=n_text,
                acc_amb=_format_mean_std(row, "accuracy_amb"),
                acc_dis=_format_mean_std(row, "accuracy_dis"),
                bias=_format_mean_std(row, "bias_abs_amb"),
                far=_format_mean_std(row, "false_abstention_rate"),
            )
        )

    cond_acc = [
        float(r["condition_classifier"]["test"]["accuracy"])
        for r in all_results
        if r.get("condition_classifier")
    ]
    cond_mean, cond_std = _mean_std(cond_acc)
    lines.extend(["", "## Reviewer-Risk Checks", ""])
    if cond_mean is not None:
        lines.append(f"- Predicted-condition classifier test accuracy: {cond_mean:.4f}±{cond_std:.4f}.")
    else:
        lines.append("- Predicted-condition classifier was skipped.")

    best_taus = []
    canonical_scores = []
    for result in all_results:
        audit = result.get("moe_eval", {}).get("low_threshold_audit")
        if not audit:
            continue
        best = audit.get("best_by_val") or {}
        canonical = audit.get("at_tau_dis_0_05") or {}
        if best.get("tau_dis") is not None:
            best_taus.append(float(best["tau_dis"]))
        if canonical.get("test_score") is not None:
            canonical_scores.append(float(canonical["test_score"]))
    if best_taus:
        tau_mean, tau_std = _mean_std(best_taus)
        score_mean, score_std = _mean_std(canonical_scores)
        lines.append(f"- Sub-0.05 tau_dis audit best-by-val tau: {tau_mean:.4f}±{tau_std:.4f}.")
        if score_mean is not None:
            lines.append(f"- tau_dis=0.05 test macro score in low-grid audit: {score_mean:.4f}±{score_std:.4f}.")
    else:
        lines.append("- Sub-0.05 tau_dis audit was skipped.")

    baseline_rows = [r for r in rows if r["subset"] == "shared_test"]
    if baseline_rows:
        worst_missing = max(
            (r.get("n_missing_max") or 0.0 for r in baseline_rows),
            default=0.0,
        )
        lines.append(f"- Baseline same-ID matching included {len(baseline_rows)} baseline systems; worst missing test IDs: {worst_missing:.0f}.")
    else:
        lines.append("- No baseline prediction files were matched.")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `metrics_summary.csv`: per-seed metrics for ours and baselines.",
            "- `aggregate_metrics.csv`: mean/std table for paper drafting.",
            "- `paired_tests.csv`: paired bootstrap differences and p-values on matched IDs.",
            "- `low_threshold_audit.csv`: sub-0.05 tau_dis sweep for the lower-bound concern.",
            "- `seed_*/test_predictions.jsonl`: same-ID predictions for auditability.",
        ]
    )
    (out_dir / "defense_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_summaries(all_results: list[dict[str, Any]], out_dir: Path) -> None:
    metrics_path = out_dir / "metrics_summary.csv"
    paired_path = out_dir / "paired_tests.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metric_fields = [
        "n_total",
        "n_ambig",
        "n_disambig",
        "accuracy_amb",
        "accuracy_dis",
        "bias_score_amb",
        "bias_score_dis",
        "false_abstention_rate",
        "parse_fail_rate",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "system", "subset"] + metric_fields)
        writer.writeheader()
        for result in all_results:
            seed = result["seed"]
            for system, payload in result["moe_eval"]["variants"].items():
                row = {"seed": seed, "system": system, "subset": "full_test"}
                row.update({k: payload["metrics"].get(k) for k in metric_fields})
                writer.writerow(row)
            for system, payload in result["stats"]["baselines"].items():
                row = {"seed": seed, "system": system, "subset": "shared_test"}
                row.update({k: payload["metrics_on_shared_test"].get(k) for k in metric_fields})
                writer.writerow(row)

    with paired_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "seed",
            "comparison",
            "metric",
            "diff",
            "p_value",
            "direction",
            "n",
            "diff_ci_lower",
            "diff_ci_upper",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            seed = result["seed"]
            for comparison, metric_payload in result["stats"]["paired"].items():
                for metric, values in metric_payload.items():
                    writer.writerow(
                        {
                            "seed": seed,
                            "comparison": comparison,
                            "metric": metric,
                            **{k: values.get(k) for k in fieldnames if k not in ("seed", "comparison", "metric")},
                        }
                    )


def strip_large_maps(payload: Any) -> Any:
    """Remove per-UID maps from the main JSON; predictions are saved as JSONL."""
    if isinstance(payload, dict):
        out = {}
        for key, value in payload.items():
            if key in {"predictions_by_uid", "condition_by_uid"}:
                out[key] = f"<omitted {len(value)} entries>"
            else:
                out[key] = strip_large_maps(value)
        return out
    if isinstance(payload, list):
        return [strip_large_maps(x) for x in payload]
    return payload


def main() -> None:
    setup_logging()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_experiment_config(args)
    categories = list(config["data"]["categories"])
    LOGGER.info("Categories: %s", ", ".join(categories))

    records, embeddings, instances_by_id = load_records_embeddings_instances(config, args)
    LOGGER.info("Loaded records=%d embeddings=%d instances=%d", len(records), len(embeddings), len(instances_by_id))

    baseline_paths = load_baseline_paths(args)
    LOGGER.info("Baseline files: %s", ", ".join(str(p) for p in baseline_paths) or "<none>")

    if args.dry_run:
        for seed in args.seeds:
            args.current_seed = seed
            train_records, val_records, test_records = split_records(records, args)
            LOGGER.info(
                "[dry-run seed=%s] train=%d val=%d test=%d",
                seed,
                len(train_records),
                len(val_records),
                len(test_records),
            )
        return

    all_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        args.current_seed = seed
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("=" * 72)
        LOGGER.info("Seed %d", seed)

        train_records, val_records, test_records = split_records(records, args)
        LOGGER.info("Split: train=%d val=%d test=%d", len(train_records), len(val_records), len(test_records))

        condition_payload = None
        predicted_val_condition = None
        predicted_test_condition = None
        if not args.skip_condition_classifier:
            LOGGER.info("Training no-oracle condition classifier")
            condition_payload = fit_condition_classifier(
                train_records,
                val_records,
                test_records,
                embeddings,
                categories,
                args.condition_features,
                seed,
            )
            predicted_val_condition = condition_payload["val"]["condition_by_uid"]
            predicted_test_condition = condition_payload["test"]["condition_by_uid"]
            LOGGER.info(
                "Condition classifier: val_acc=%.4f test_acc=%.4f",
                condition_payload["val"]["accuracy"],
                condition_payload["test"]["accuracy"],
            )

        LOGGER.info("Training main MoE")
        model, train_out = train_moe(
            train_records,
            val_records,
            embeddings,
            config,
            args,
            seed,
            save_dir=seed_dir / "checkpoints" / "main",
        )
        LOGGER.info("MoE best_val_loss=%s best_epoch=%s", train_out.get("best_val_loss"), train_out.get("best_epoch"))

        val_predictions = predict_records(model, val_records, embeddings, instances_by_id)
        test_predictions = predict_records(model, test_records, embeddings, instances_by_id)
        moe_eval = evaluate_moe_variants(
            val_predictions,
            test_predictions,
            predicted_val_condition=predicted_val_condition,
            predicted_test_condition=predicted_test_condition,
            args=args,
        )
        for system_name, payload in moe_eval["variants"].items():
            m = payload["metrics"]
            LOGGER.info(
                "%s: acc_amb=%.4f acc_dis=%.4f bias_amb=%s FAR=%.4f thresholds=%s",
                system_name,
                m.get("accuracy_amb", 0.0),
                m.get("accuracy_dis", 0.0),
                m.get("bias_score_amb"),
                m.get("false_abstention_rate", 0.0),
                payload.get("thresholds"),
            )

        save_seed_predictions(
            seed_dir / "test_predictions.jsonl",
            test_predictions,
            moe_eval["variants"],
            predicted_test_condition,
        )

        stats = evaluate_baselines_and_stats(
            test_predictions,
            moe_eval["variants"],
            baseline_paths,
            args,
        )

        signal_ablation = None
        if args.run_signal_ablation:
            signal_ablation = run_signal_ablation_suite(
                train_records,
                val_records,
                test_records,
                embeddings,
                instances_by_id,
                config,
                args,
                seed,
                out_dir,
            )

        result = {
            "seed": seed,
            "config": {
                "results_dir": args.results_dir,
                "sampled_dir": args.sampled_dir,
                "val_split": args.val_split,
                "test_split": args.test_split,
                "tau_grid": [args.tau_min, args.tau_max, args.tau_step],
                "low_tau_dis_grid": [args.low_tau_min, args.low_tau_max, args.low_tau_step],
                "categories": categories,
                "epochs": args.epochs,
                "bootstrap_iters": args.bootstrap_iters,
            },
            "split": {
                "n_train": len(train_records),
                "n_val": len(val_records),
                "n_test": len(test_records),
            },
            "train": {
                "best_val_loss": train_out.get("best_val_loss"),
                "best_epoch": train_out.get("best_epoch"),
                "checkpoint_path": train_out.get("checkpoint_path"),
            },
            "condition_classifier": condition_payload,
            "moe_eval": moe_eval,
            "stats": stats,
            "signal_ablation": signal_ablation,
        }
        compact = strip_large_maps(result)
        (seed_dir / "result.json").write_text(
            json.dumps(compact, indent=2, ensure_ascii=False, default=float),
            encoding="utf-8",
        )
        all_results.append(compact)

    write_csv_summaries(all_results, out_dir)
    write_defense_report(all_results, out_dir)
    (out_dir / "summary.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    LOGGER.info("Saved summary: %s", out_dir / "summary.json")
    LOGGER.info("Saved CSVs: %s, %s", out_dir / "metrics_summary.csv", out_dir / "paired_tests.csv")
    LOGGER.info("Saved defense report: %s", out_dir / "defense_report.md")


if __name__ == "__main__":
    main()
