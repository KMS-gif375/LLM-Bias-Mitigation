#!/usr/bin/env python3
"""Re-evaluate saved cross-LLM validation-best MoE checkpoints.

The original cross-LLM multi-seed runner saved a validation-best checkpoint but
evaluated the in-memory final-epoch model. This audit reconstructs the exact
seed split, restores each saved ``moe_best.pt``, selects the two condition
thresholds on validation, and reports metrics on the untouched test split.
It performs no LLM calls and no model training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_pipeline import (  # noqa: E402
    _instances_by_id,
    _make_unique_id,
    _moe_predict_all,
)
from src.evaluation.bbq_evaluator import evaluate_bbq  # noqa: E402
from src.models.moe_aggregator import MoEAggregator  # noqa: E402
from src.models.override import (  # noqa: E402
    apply_per_condition_override,
    search_optimal_threshold_per_condition,
)
from src.utils.data_loader import DEFAULT_CATEGORIES_V2  # noqa: E402


LOGGER = logging.getLogger("cross_llm_best_checkpoint")
DEFAULT_SEEDS = [42, 123, 456, 789, 999]
METRICS = ("accuracy_amb", "accuracy_dis", "false_abstention_rate")


def split_like_original_multi_seed(
    records: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce ``src.analysis.multi_seed``'s exact 70/15/15 split."""
    stratification = [
        f"{row.get('category', '_unk')}::{row.get('context_condition', '_unk')}"
        for row in records
    ]
    indices = list(range(len(records)))
    train_indices, rest_indices = train_test_split(
        indices,
        train_size=0.70,
        random_state=seed,
        stratify=stratification,
    )
    rest_stratification = [stratification[index] for index in rest_indices]
    val_indices, test_indices = train_test_split(
        rest_indices,
        train_size=0.50,
        random_state=seed,
        stratify=rest_stratification,
    )
    return (
        [records[index] for index in train_indices],
        [records[index] for index in val_indices],
        [records[index] for index in test_indices],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved cross-LLM validation-best checkpoints."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--results-root", default="results/v2")
    parser.add_argument("--sampled-dir", default="data/sampled_v2")
    parser.add_argument("--models", nargs="+", default=["qwen", "mistral"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--embedding-model", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir",
        default="results/v2/reviewer_audits/cross_llm_best_checkpoint",
    )
    return parser.parse_args()


def load_records(signals_dir: Path, categories: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for category in categories:
        path = signals_dir / f"{category}_signals.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing cross-LLM signal file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record.setdefault("category", category)
                record["unique_id"] = _make_unique_id(category, record["example_id"])
                records.append(record)
    return records


def load_shared_embeddings(
    results_root: Path,
    embedding_model: str,
    categories: list[str],
) -> dict[str, torch.Tensor]:
    """Load text-only MiniLM embeddings shared by all backbone audits."""
    embeddings: dict[str, torch.Tensor] = {}
    embedding_dir = results_root / "signals" / embedding_model
    for category in categories:
        path = embedding_dir / f"{category}_embeddings.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing shared text embedding file: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected an embedding dictionary in {path}")
        for raw_key, vector in payload.items():
            embeddings[_make_unique_id(category, raw_key)] = vector
    return embeddings


def load_model(checkpoint_path: Path, device: torch.device) -> MoEAggregator:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing validation-best checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise KeyError(f"Checkpoint lacks model_config: {checkpoint_path}")
    model = MoEAggregator(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in (*METRICS, "threshold_amb", "threshold_dis"):
        values = [float(row[key]) for row in rows]
        summary[key] = {
            "mean": float(mean(values)),
            "std": float(stdev(values)) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return summary


def evaluate_model(model_key: str, args: argparse.Namespace) -> dict[str, Any]:
    results_root = Path(args.results_root)
    cross_root = results_root / "cross_llm" / model_key
    signals_dir = cross_root / "signals" / model_key
    run_dir = cross_root / "multi_seed_5seed"
    categories = list(DEFAULT_CATEGORIES_V2)

    records = load_records(signals_dir, categories)
    embeddings = load_shared_embeddings(results_root, args.embedding_model, categories)
    missing = sorted({str(record["unique_id"]) for record in records} - set(embeddings))
    if missing:
        raise KeyError(f"Missing {len(missing)} shared embeddings; first={missing[0]}")

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["data"]["sampled_dir"] = args.sampled_dir
    config["data"]["samples_per_category"] = 1000
    config["data"]["categories"] = categories
    config["output"]["results_dir"] = str(cross_root)
    instance_args = SimpleNamespace(model=model_key, categories=None)
    instances_by_id = _instances_by_id(records, config, instance_args)

    threshold_cfg = config.get("override", {}).get("threshold_search", {})
    threshold_range = tuple(threshold_cfg.get("per_condition_range", [0.05, 0.95]))
    threshold_step = float(threshold_cfg.get("per_condition_step", 0.025))
    device = torch.device(args.device)

    per_seed: list[dict[str, Any]] = []
    for seed in args.seeds:
        train_records, val_records, test_records = split_like_original_multi_seed(
            records, seed
        )
        checkpoint_path = run_dir / f"seed_{seed}" / "moe_best.pt"
        model = load_model(checkpoint_path, device)

        val_predictions = _moe_predict_all(
            model, val_records, embeddings, instances_by_id
        )
        search = search_optimal_threshold_per_condition(
            val_predictions,
            metric_amb="accuracy_amb",
            metric_dis="accuracy_dis",
            threshold_range=threshold_range,
            step=threshold_step,
        )
        test_predictions = _moe_predict_all(
            model, test_records, embeddings, instances_by_id
        )
        predictions: list[int] = []
        items: list[dict[str, Any]] = []
        for row in test_predictions:
            result = apply_per_condition_override(
                primary_answer=int(row["primary_answer"]),
                p_score=float(row["p_score"]),
                item=row["item"],
                thresholds=search.thresholds,
            )
            predictions.append(int(result["final_answer"]))
            items.append(row["item"])
        metrics = evaluate_bbq(predictions, items)

        reported_path = run_dir / f"seed_{seed}_results.json"
        reported = json.loads(reported_path.read_text(encoding="utf-8"))
        row = {
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
            "best_epoch": int(reported.get("best_epoch", -1)),
            "best_val_loss": float(reported.get("best_val_loss", float("nan"))),
            "n_train": len(train_records),
            "n_val": len(val_records),
            "n_test": len(items),
            "accuracy_amb": float(metrics["accuracy_amb"]),
            "accuracy_dis": float(metrics["accuracy_dis"]),
            "false_abstention_rate": float(metrics["false_abstention_rate"]),
            "threshold_amb": float(search.thresholds["ambig"]),
            "threshold_dis": float(search.thresholds["disambig"]),
        }
        per_seed.append(row)
        LOGGER.info(
            "%s seed=%d best_epoch=%d: %.4f / %.4f / %.4f",
            model_key,
            seed,
            row["best_epoch"],
            row["accuracy_amb"],
            row["accuracy_dis"],
            row["false_abstention_rate"],
        )

    return {
        "model": model_key,
        "signals_dir": str(signals_dir),
        "shared_embedding_dir": str(results_root / "signals" / args.embedding_model),
        "checkpoint_dir": str(run_dir),
        "split": {
            "records": len(records),
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "stratification": "category::context_condition",
            "implementation": "two sklearn train_test_split calls, matching src.analysis.multi_seed",
        },
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    payload = {
        "audit": "cross_llm_validation_best_checkpoint_recomputation",
        "selection": "validation thresholds; held-out test metrics",
        "models": [evaluate_model(model, args) for model in args.models],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "summary.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
