#!/usr/bin/env python3
"""Reviewer-requested extra audits for BBQ condition separability.

This script is designed for the H100/RunPod environment but keeps the default
path no-LLM: it reuses saved BBQ signal records, cached predictions, and cached
or recomputed sentence embeddings. The main reviewer concerns covered here are:

1. Lexical/template baselines for condition prediction.
2. Sensitivity to the sentence-embedding encoder.
3. Post-hoc conformal/risk-controlled abstention baselines.
4. Single-prompt low-label condition-only fallback.
5. Optional NLI audit for the s1 evidence signal.
6. Optional paraphrase/adversarial condition-transfer audit from a JSONL file.
7. Condition-aware risk-control/conformal-style baselines.
8. Category-aware condition-calibration audit.
9. Cross-LLM condition-classifier transfer audit.

Outputs are written under results/v2/reviewer_extra_audits by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SEEDS = [42, 123, 456, 789, 999]
DEFAULT_ENCODERS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reviewer-requested extra BBQ audits from saved artifacts."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="main")
    parser.add_argument("--results-dir", default="results/v2")
    parser.add_argument("--sampled-dir", default="data/sampled_v2")
    parser.add_argument("--out-dir", default="results/v2/reviewer_extra_audits")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--samples-per-category", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--audits",
        nargs="+",
        default=["lexical", "embedding", "conformal"],
        choices=[
            "all",
            "lexical",
            "embedding",
            "conformal",
            "condition_conformal",
            "category_calibration",
            "cross_llm",
            "single_prompt",
            "nli",
            "paraphrase",
        ],
        help=(
            "Audit groups to run. Default covers the three fastest reviewer asks: "
            "lexical, embedding, and conformal."
        ),
    )
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=DEFAULT_ENCODERS,
        help="Sentence-transformer encoders for the encoder-sensitivity audit.",
    )
    parser.add_argument(
        "--condition-feature-modes",
        nargs="+",
        default=["embedding"],
        help="Feature modes for embedding audit. Keep default for raw-text embedding only.",
    )
    parser.add_argument(
        "--target-risks",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20],
        help="Validation selective-risk targets for conformal/risk-control baselines.",
    )
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument(
        "--low-label-fractions",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 1.0],
        help="Train-label fractions for single-prompt low-label condition classifiers.",
    )
    parser.add_argument(
        "--nli-model",
        default="facebook/bart-large-mnli",
        help="NLI model used only when the nli audit is enabled.",
    )
    parser.add_argument(
        "--max-nli-examples",
        type=int,
        default=0,
        help="Limit for NLI audit. 0 means use all test records; ignored unless nli audit runs.",
    )
    parser.add_argument(
        "--paraphrase-file",
        default=None,
        help=(
            "Optional JSONL file with paraphrased/adversarial BBQ-like examples. "
            "Expected fields include category/example_id or unique_id, context, question, "
            "and context_condition."
        ),
    )
    parser.add_argument(
        "--cross-llm-targets",
        nargs="+",
        default=[
            "qwen:results/v2/cross_llm/qwen:qwen",
            "mistral:results/v2/cross_llm/mistral:mistral",
        ],
        help=(
            "Cross-LLM condition-transfer targets in name:results_dir:model format. "
            "Records are read from results_dir/signals/model and evaluated with the "
            "same BBQ item IDs."
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for sentence-transformer embedding recomputation.",
    )
    return parser.parse_args()


def audit_set(args: argparse.Namespace) -> set[str]:
    selected = set(args.audits)
    if "all" in selected:
        return {
            "lexical",
            "embedding",
            "conformal",
            "condition_conformal",
            "category_calibration",
            "cross_llm",
            "single_prompt",
            "nli",
            "paraphrase",
        }
    return selected


def import_clean():
    import run_clean_experiments as clean

    return clean


def import_eval_helpers():
    from src.evaluation.bbq_evaluator import evaluate_bbq
    from src.models.override import find_unknown_index

    return evaluate_bbq, find_unknown_index


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def base_clean_args(args: argparse.Namespace) -> SimpleNamespace:
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
        tau_step=args.tau_step,
        low_tau_min=0.0,
        low_tau_max=0.1,
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


def load_artifacts(args: argparse.Namespace):
    clean = import_clean()
    clean_args = base_clean_args(args)
    config = clean.load_experiment_config(clean_args)
    records, embeddings, instances_by_id = clean.load_records_embeddings_instances(config, clean_args)
    categories = list(config["data"]["categories"])
    return clean, clean_args, config, records, embeddings, instances_by_id, categories


def split_for_seed(clean: Any, records: list[dict], clean_args: SimpleNamespace, seed: int):
    clean_args.current_seed = seed
    return clean.split_records(records, clean_args)


def record_uid(clean: Any, record: dict[str, Any]) -> str:
    return clean.uid_for(record)


def record_text(clean: Any, record: dict[str, Any], instances_by_id: dict[str, dict]) -> str:
    uid = record_uid(clean, record)
    item = instances_by_id.get(uid, record)
    context = str(item.get("context", record.get("context", "")) or "")
    question = str(item.get("question", record.get("question", "")) or "")
    return f"{context} {question}".strip()


def condition_label(record: dict[str, Any]) -> int:
    return 0 if record.get("context_condition") == "ambig" else 1


def label_name(value: int) -> str:
    return "ambig" if int(value) == 0 else "disambig"


def condition_only_predictions(
    clean: Any,
    records: list[dict],
    instances_by_id: dict[str, dict],
    condition_by_uid: dict[str, str],
) -> tuple[list[int], list[dict]]:
    _evaluate_bbq, find_unknown_index = import_eval_helpers()
    preds: list[int] = []
    items: list[dict] = []
    for record in records:
        uid = record_uid(clean, record)
        item = instances_by_id.get(uid, record)
        pred_cond = condition_by_uid.get(uid, "disambig")
        if pred_cond == "ambig":
            pred = find_unknown_index(item)
        else:
            pred = int(record.get("primary_answer", -1))
        preds.append(pred)
        items.append(item)
    return preds, items


def condition_accuracy(records: list[dict], condition_by_uid: dict[str, str], clean: Any) -> float:
    if not records:
        return 0.0
    correct = 0
    for record in records:
        uid = record_uid(clean, record)
        correct += int(condition_by_uid.get(uid) == record.get("context_condition"))
    return correct / len(records)


def summarize_by(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in sorted({str(r[group_key]) for r in rows}):
        group_rows = [r for r in rows if str(r[group_key]) == group]
        row: dict[str, Any] = {group_key: group, "n_seeds": len(group_rows)}
        for metric in (
            "condition_accuracy",
            "accuracy_amb",
            "accuracy_dis",
            "false_abstention_rate",
            "selective_risk",
            "coverage",
            "tau",
            "auc",
            "spearman_r",
            "pearson_r",
        ):
            vals = [float(r[metric]) for r in group_rows if metric in r and r[metric] not in ("", None)]
            if vals:
                m, s = mean_std(vals)
                row[f"{metric}_mean"] = m
                row[f"{metric}_std"] = s
        out.append(row)
    return out


def lexical_condition_baselines(
    clean: Any,
    records: list[dict],
    instances_by_id: dict[str, dict],
    clean_args: SimpleNamespace,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    evaluate_bbq, _find_unknown_index = import_eval_helpers()
    variants = [
        ("bow_unigram", {"analyzer": "word", "ngram_range": (1, 1), "min_df": 1, "max_features": 50000}),
        ("word_1_2gram", {"analyzer": "word", "ngram_range": (1, 2), "min_df": 1, "max_features": 100000}),
        ("char_3_5gram", {"analyzer": "char", "ngram_range": (3, 5), "min_df": 1, "max_features": 150000}),
        ("char_wb_3_5gram", {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 1, "max_features": 150000}),
    ]
    detail: list[dict[str, Any]] = []
    for seed in seeds:
        train, _val, test = split_for_seed(clean, records, clean_args, seed)
        train_texts = [record_text(clean, r, instances_by_id) for r in train]
        train_y = [condition_label(r) for r in train]
        test_texts = [record_text(clean, r, instances_by_id) for r in test]
        for system, vec_kwargs in variants:
            clf = make_pipeline(
                TfidfVectorizer(lowercase=True, strip_accents="unicode", **vec_kwargs),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
            clf.fit(train_texts, train_y)
            pred_y = clf.predict(test_texts)
            condition_by_uid = {
                record_uid(clean, record): label_name(int(pred))
                for record, pred in zip(test, pred_y)
            }
            preds, items = condition_only_predictions(clean, test, instances_by_id, condition_by_uid)
            metrics = evaluate_bbq(preds, items)
            detail.append(
                {
                    "seed": seed,
                    "system": system,
                    "text_input": "context_plus_question_only",
                    "condition_accuracy": condition_accuracy(test, condition_by_uid, clean),
                    **metrics,
                }
            )
    return detail, summarize_by(detail, "system")


def safe_model_name(model_name: str) -> str:
    return (
        model_name.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def encode_records(
    clean: Any,
    records: list[dict],
    instances_by_id: dict[str, dict],
    model_name: str,
    device: str,
    batch_size: int,
    cache_dir: Path,
) -> dict[str, Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{safe_model_name(model_name)}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    model = SentenceTransformer(model_name, device=device)
    uids = [record_uid(clean, r) for r in records]
    texts = [record_text(clean, r, instances_by_id) for r in records]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=False,
    ).detach().cpu()
    embeddings = {uid: vectors[i] for i, uid in enumerate(uids)}
    torch.save(embeddings, cache_path)
    return embeddings


def embedding_encoder_sensitivity(
    clean: Any,
    records: list[dict],
    instances_by_id: dict[str, dict],
    categories: list[str],
    clean_args: SimpleNamespace,
    seeds: list[int],
    encoder_names: list[str],
    device: str,
    batch_size: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluate_bbq, _find_unknown_index = import_eval_helpers()
    detail: list[dict[str, Any]] = []
    cache_dir = out_dir / "embedding_cache"
    for encoder in encoder_names:
        embeddings = encode_records(clean, records, instances_by_id, encoder, device, batch_size, cache_dir)
        for seed in seeds:
            train, val, test = split_for_seed(clean, records, clean_args, seed)
            clf = clean.fit_condition_classifier(train, val, test, embeddings, categories, "embedding", seed)
            condition_by_uid = clf["test"]["condition_by_uid"]
            preds, items = condition_only_predictions(clean, test, instances_by_id, condition_by_uid)
            metrics = evaluate_bbq(preds, items)
            detail.append(
                {
                    "seed": seed,
                    "encoder": encoder,
                    "feature_mode": "raw_text_embedding_only",
                    "condition_accuracy": float(clf["test"]["accuracy"]),
                    **metrics,
                }
            )
    return detail, summarize_by(detail, "encoder")


def softmax_from_logprobs(logprobs: dict[str, float], temperature: float = 1.0) -> list[float]:
    vals = [float(logprobs.get(k, float("-inf"))) / max(temperature, 1e-8) for k in ("A", "B", "C")]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return [1.0 / 3.0] * 3
    max_v = max(finite)
    exps = [math.exp(v - max_v) if math.isfinite(v) else 0.0 for v in vals]
    total = sum(exps)
    if total <= 0:
        return [1.0 / 3.0] * 3
    return [e / total for e in exps]


def load_stage1_logprobs(results_dir: Path, model: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    stage_dir = results_dir / "signals" / model
    for path in stage_dir.glob("*_stage1.jsonl"):
        category = path.name[: -len("_stage1.jsonl")]
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                uid = f"{category}::{row.get('example_id')}"
                logprobs = (((row.get("responses") or {}).get("vanilla") or {}).get("logprobs") or {})
                if logprobs:
                    out[uid] = {k: float(v) for k, v in logprobs.items()}
    return out


def make_score_rows(
    clean: Any,
    records: list[dict],
    instances_by_id: dict[str, dict],
    stage1_logprobs: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    _evaluate_bbq, find_unknown_index = import_eval_helpers()
    rows: list[dict[str, Any]] = []
    for record in records:
        uid = record_uid(clean, record)
        item = instances_by_id.get(uid, record)
        primary = int(record.get("primary_answer", -1))
        probs = softmax_from_logprobs(stage1_logprobs.get(uid, {}), temperature=1.0)
        unknown_idx = find_unknown_index(item)
        unknown_prob = float(probs[unknown_idx]) if unknown_idx in (0, 1, 2) else 0.0
        chosen_prob = float(probs[primary]) if primary in (0, 1, 2) else 0.0
        rows.append(
            {
                "uid": uid,
                "record": record,
                "item": item,
                "primary_answer": primary,
                "is_primary_correct": int(primary == int(item.get("label", -999))),
                "chosen_softmax": chosen_prob,
                "max_softmax": float(max(probs)),
                "unknown_inverse": 1.0 - unknown_prob,
            }
        )
    return rows


def selective_risk(rows: list[dict[str, Any]], score_key: str, tau: float) -> tuple[float, float]:
    retained = [r for r in rows if float(r[score_key]) >= tau]
    coverage = len(retained) / max(len(rows), 1)
    if not retained:
        return 0.0, 0.0
    errors = [1 - int(r["is_primary_correct"]) for r in retained]
    return coverage, sum(errors) / len(errors)


def apply_conformal_threshold(
    rows: list[dict[str, Any]],
    score_key: str,
    tau: float,
) -> tuple[list[int], list[dict]]:
    _evaluate_bbq, find_unknown_index = import_eval_helpers()
    preds: list[int] = []
    items: list[dict] = []
    for row in rows:
        item = row["item"]
        if float(row[score_key]) >= tau:
            pred = int(row["primary_answer"])
        else:
            pred = find_unknown_index(item)
        preds.append(pred)
        items.append(item)
    return preds, items


def tune_risk_threshold(rows: list[dict[str, Any]], score_key: str, target_risk: float, step: float) -> dict[str, float]:
    # Conservative risk-control style: among validation thresholds satisfying
    # empirical selective risk <= target, choose the largest coverage.
    best = {"tau": 1.0, "coverage": 0.0, "selective_risk": 0.0}
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        tau = round(i * step, 6)
        coverage, risk = selective_risk(rows, score_key, tau)
        if risk <= target_risk and coverage >= best["coverage"]:
            best = {"tau": tau, "coverage": coverage, "selective_risk": risk}
    return best


def conformal_abstention_baselines(
    clean: Any,
    records: list[dict],
    instances_by_id: dict[str, dict],
    clean_args: SimpleNamespace,
    seeds: list[int],
    results_dir: Path,
    model: str,
    target_risks: list[float],
    step: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluate_bbq, _find_unknown_index = import_eval_helpers()
    stage1_logprobs = load_stage1_logprobs(results_dir, model)
    score_keys = ["chosen_softmax", "max_softmax", "unknown_inverse"]
    detail: list[dict[str, Any]] = []
    for seed in seeds:
        _train, val, test = split_for_seed(clean, records, clean_args, seed)
        val_rows = make_score_rows(clean, val, instances_by_id, stage1_logprobs)
        test_rows = make_score_rows(clean, test, instances_by_id, stage1_logprobs)
        for score_key in score_keys:
            for target in target_risks:
                tuned = tune_risk_threshold(val_rows, score_key, target, step)
                preds, items = apply_conformal_threshold(test_rows, score_key, tuned["tau"])
                metrics = evaluate_bbq(preds, items)
                coverage, risk = selective_risk(test_rows, score_key, tuned["tau"])
                detail.append(
                    {
                        "seed": seed,
                        "system": f"{score_key}_risk_control",
                        "score_key": score_key,
                        "target_risk": target,
                        "tau": tuned["tau"],
                        "validation_coverage": tuned["coverage"],
                        "validation_selective_risk": tuned["selective_risk"],
                        "coverage": coverage,
                        "selective_risk": risk,
                        **metrics,
                    }
                )
    grouped = []
    for row in detail:
        row = dict(row)
        row["system_target"] = f"{row['system']}@{row['target_risk']:.2f}"
        grouped.append(row)
    return detail, summarize_by(grouped, "system_target")


def tune_grouped_risk_thresholds(
    rows: list[dict[str, Any]],
    score_key: str,
    target_risk: float,
    step: float,
    group_key: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "_all"))].append(row)
    all_tuned = tune_risk_threshold(rows, score_key, target_risk, step)
    thresholds: dict[str, dict[str, float]] = {"_fallback": all_tuned}
    for group, group_rows in grouped.items():
        if len(group_rows) < 5:
            thresholds[group] = all_tuned
        else:
            thresholds[group] = tune_risk_threshold(group_rows, score_key, target_risk, step)
    return thresholds


def apply_grouped_conformal_threshold(
    rows: list[dict[str, Any]],
    score_key: str,
    thresholds: dict[str, dict[str, float]],
    group_key: str,
) -> tuple[list[int], list[dict]]:
    _evaluate_bbq, find_unknown_index = import_eval_helpers()
    fallback = thresholds.get("_fallback", {"tau": 1.0})
    preds: list[int] = []
    items: list[dict] = []
    for row in rows:
        group = str(row.get(group_key, "_all"))
        tau = thresholds.get(group, fallback).get("tau", fallback.get("tau", 1.0))
        item = row["item"]
        if float(row[score_key]) >= tau:
            pred = int(row["primary_answer"])
        else:
            pred = find_unknown_index(item)
        preds.append(pred)
        items.append(item)
    return preds, items


def condition_aware_conformal_baselines(
    clean: Any,
    records: list[dict],
    embeddings: dict[str, Any],
    instances_by_id: dict[str, dict],
    categories: list[str],
    clean_args: SimpleNamespace,
    seeds: list[int],
    results_dir: Path,
    model: str,
    target_risks: list[float],
    step: float,
    feature_mode: str = "signals,embedding,category,primary",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Risk-control baselines with separate thresholds by predicted condition.

    This is a stronger reviewer-requested comparator than the global
    confidence-only risk-control audit because it gives the conformal-style
    wrapper access to the same predicted ambiguous/disambiguated partition used
    by CASA, but still makes the keep/abstain decision from generic confidence
    scores rather than from the condition-only rule.
    """

    evaluate_bbq, _find_unknown_index = import_eval_helpers()
    stage1_logprobs = load_stage1_logprobs(results_dir, model)
    score_keys = ["chosen_softmax", "max_softmax", "unknown_inverse"]
    detail: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = split_for_seed(clean, records, clean_args, seed)
        clf = clean.fit_condition_classifier(train, val, test, embeddings, categories, feature_mode, seed)
        val_conditions = clf["val"]["condition_by_uid"]
        test_conditions = clf["test"]["condition_by_uid"]
        val_rows = make_score_rows(clean, val, instances_by_id, stage1_logprobs)
        test_rows = make_score_rows(clean, test, instances_by_id, stage1_logprobs)
        for row in val_rows:
            row["pred_condition"] = val_conditions.get(row["uid"], "_missing")
        for row in test_rows:
            row["pred_condition"] = test_conditions.get(row["uid"], "_missing")
        for score_key in score_keys:
            for target in target_risks:
                thresholds = tune_grouped_risk_thresholds(
                    val_rows,
                    score_key=score_key,
                    target_risk=target,
                    step=step,
                    group_key="pred_condition",
                )
                preds, items = apply_grouped_conformal_threshold(
                    test_rows,
                    score_key=score_key,
                    thresholds=thresholds,
                    group_key="pred_condition",
                )
                metrics = evaluate_bbq(preds, items)
                coverage, risk = selective_risk(test_rows, score_key, thresholds["_fallback"]["tau"])
                detail.append(
                    {
                        "seed": seed,
                        "system": f"condition_aware_{score_key}_risk_control",
                        "score_key": score_key,
                        "target_risk": target,
                        "feature_mode": feature_mode,
                        "ambig_tau": thresholds.get("ambig", thresholds["_fallback"])["tau"],
                        "disambig_tau": thresholds.get("disambig", thresholds["_fallback"])["tau"],
                        "global_tau": thresholds["_fallback"]["tau"],
                        "condition_accuracy": float(clf["test"]["accuracy"]),
                        "global_coverage_at_fallback_tau": coverage,
                        "global_selective_risk_at_fallback_tau": risk,
                        **metrics,
                    }
                )
    grouped = []
    for row in detail:
        row = dict(row)
        row["system_target"] = f"{row['system']}@{row['target_risk']:.2f}"
        grouped.append(row)
    return detail, summarize_by(grouped, "system_target")


def condition_by_probability_threshold(
    records: list[dict],
    proba_by_uid: dict[str, dict[str, float]],
    clean: Any,
    thresholds_by_category: dict[str, float],
    default_tau: float = 0.5,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for record in records:
        uid = record_uid(clean, record)
        cat = str(record.get("category", "_unknown"))
        tau = thresholds_by_category.get(cat, default_tau)
        p_dis = float((proba_by_uid.get(uid) or {}).get("disambig", 0.0))
        out[uid] = "disambig" if p_dis >= tau else "ambig"
    return out


def tune_category_condition_thresholds(
    clean: Any,
    val_records: list[dict],
    instances_by_id: dict[str, dict],
    proba_by_uid: dict[str, dict[str, float]],
    categories: list[str],
    grid: Iterable[float],
) -> dict[str, float]:
    evaluate_bbq, _find_unknown_index = import_eval_helpers()
    thresholds: dict[str, float] = {}
    for category in categories:
        cat_records = [r for r in val_records if str(r.get("category")) == category]
        if not cat_records:
            thresholds[category] = 0.5
            continue
        best_tau = 0.5
        best_tuple = (-1.0, -1.0, 1.0)
        for tau in grid:
            cond = condition_by_probability_threshold(
                cat_records,
                proba_by_uid,
                clean,
                {category: float(tau)},
                default_tau=float(tau),
            )
            preds, items = condition_only_predictions(clean, cat_records, instances_by_id, cond)
            metrics = evaluate_bbq(preds, items)
            score = (float(metrics.get("accuracy_amb", 0.0)) + float(metrics.get("accuracy_dis", 0.0))) / 2.0
            tie = (score, float(metrics.get("accuracy_dis", 0.0)), -float(metrics.get("false_abstention_rate", 1.0)))
            if tie > best_tuple:
                best_tuple = tie
                best_tau = float(tau)
        thresholds[category] = best_tau
    return thresholds


def category_aware_condition_calibration(
    clean: Any,
    records: list[dict],
    embeddings: dict[str, Any],
    instances_by_id: dict[str, dict],
    categories: list[str],
    clean_args: SimpleNamespace,
    seeds: list[int],
    feature_mode: str = "signals,embedding,category,primary",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluate_bbq, _find_unknown_index = import_eval_helpers()
    grid = [round(x, 2) for x in [i / 100 for i in range(5, 96, 5)]]
    overall_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = split_for_seed(clean, records, clean_args, seed)
        clf = clean.fit_condition_classifier(train, val, test, embeddings, categories, feature_mode, seed)
        uniform_cond = clf["test"]["condition_by_uid"]
        uniform_preds, uniform_items = condition_only_predictions(clean, test, instances_by_id, uniform_cond)
        uniform_metrics = evaluate_bbq(uniform_preds, uniform_items)
        cat_taus = tune_category_condition_thresholds(
            clean,
            val,
            instances_by_id,
            clf["val"]["probability_by_uid"],
            categories,
            grid,
        )
        cat_cond = condition_by_probability_threshold(
            test,
            clf["test"]["probability_by_uid"],
            clean,
            cat_taus,
            default_tau=0.5,
        )
        cat_preds, cat_items = condition_only_predictions(clean, test, instances_by_id, cat_cond)
        cat_metrics = evaluate_bbq(cat_preds, cat_items)
        overall_rows.append(
            {
                "seed": seed,
                "system": "uniform_threshold_0_50",
                "feature_mode": feature_mode,
                "condition_accuracy": float(clf["test"]["accuracy"]),
                "tau_mean": 0.5,
                "tau_min": 0.5,
                "tau_max": 0.5,
                **uniform_metrics,
            }
        )
        overall_rows.append(
            {
                "seed": seed,
                "system": "category_aware_condition_threshold",
                "feature_mode": feature_mode,
                "condition_accuracy": sum(
                    int(cat_cond.get(record_uid(clean, r)) == r.get("context_condition")) for r in test
                )
                / max(len(test), 1),
                "tau_mean": mean(list(cat_taus.values())) if cat_taus else 0.5,
                "tau_min": min(cat_taus.values()) if cat_taus else 0.5,
                "tau_max": max(cat_taus.values()) if cat_taus else 0.5,
                **cat_metrics,
            }
        )
        for category in categories:
            cat_test = [r for r in test if str(r.get("category")) == category]
            if not cat_test:
                continue
            cat_only_cond = {record_uid(clean, r): cat_cond[record_uid(clean, r)] for r in cat_test}
            cat_only_preds, cat_only_items = condition_only_predictions(clean, cat_test, instances_by_id, cat_only_cond)
            metrics = evaluate_bbq(cat_only_preds, cat_only_items)
            category_rows.append(
                {
                    "seed": seed,
                    "category": category,
                    "tau": cat_taus.get(category, 0.5),
                    **metrics,
                }
            )
    return overall_rows, summarize_by(overall_rows, "system"), category_rows


def stratified_fraction(records: list[dict], fraction: float, seed: int) -> list[dict]:
    if fraction >= 0.999:
        return list(records)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        buckets[(str(record.get("category", "")), str(record.get("context_condition", "")))].append(record)
    selected: list[dict] = []
    for bucket in buckets.values():
        bucket = list(bucket)
        rng.shuffle(bucket)
        n = max(1, int(round(len(bucket) * fraction)))
        selected.extend(bucket[:n])
    rng.shuffle(selected)
    return selected


def single_prompt_low_label_audit(
    clean: Any,
    records: list[dict],
    embeddings: dict[str, Any],
    instances_by_id: dict[str, dict],
    categories: list[str],
    clean_args: SimpleNamespace,
    seeds: list[int],
    fractions: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluate_bbq, _find_unknown_index = import_eval_helpers()
    detail: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = split_for_seed(clean, records, clean_args, seed)
        for fraction in fractions:
            train_sub = stratified_fraction(train, fraction, seed)
            clf = clean.fit_condition_classifier(train_sub, val, test, embeddings, categories, "embedding", seed)
            condition_by_uid = clf["test"]["condition_by_uid"]
            preds, items = condition_only_predictions(clean, test, instances_by_id, condition_by_uid)
            metrics = evaluate_bbq(preds, items)
            detail.append(
                {
                    "seed": seed,
                    "system": "single_prompt_embedding_condition_only",
                    "label_fraction": fraction,
                    "train_n": len(train_sub),
                    "condition_accuracy": float(clf["test"]["accuracy"]),
                    **metrics,
                }
            )
    grouped = []
    for row in detail:
        row = dict(row)
        row["system_fraction"] = f"{row['system']}@{row['label_fraction']:.2f}"
        grouped.append(row)
    return detail, summarize_by(grouped, "system_fraction")


def answer_text(item: dict[str, Any], answer_idx: int) -> str:
    if answer_idx not in (0, 1, 2):
        return ""
    for key in (f"ans{answer_idx}", f"answer_{answer_idx}", f"answer{answer_idx}"):
        if key in item:
            return str(item[key])
    answer_info = item.get("answer_info", {})
    if isinstance(answer_info, str):
        try:
            answer_info = json.loads(answer_info)
        except json.JSONDecodeError:
            answer_info = {}
    if isinstance(answer_info, dict) and f"ans{answer_idx}" in answer_info:
        val = answer_info[f"ans{answer_idx}"]
        if isinstance(val, list) and val:
            return str(val[0])
    return str(item.get(f"ans{answer_idx}", ""))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def nli_evidence_audit(
    clean: Any,
    records: list[dict],
    instances_by_id: dict[str, dict],
    clean_args: SimpleNamespace,
    seeds: list[int],
    model_name: str,
    max_examples: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    detail: list[dict[str, Any]] = []
    # Use the first test split for the expensive NLI audit; the signal itself is
    # deterministic over examples, so seed repetition is not useful here.
    _train, _val, test = split_for_seed(clean, records, clean_args, seeds[0])
    if max_examples > 0:
        test = test[:max_examples]
    torch_device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch_device)
    model.eval()
    label2id = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}
    entailment_id = label2id.get("entailment", label2id.get("entailed", None))
    if entailment_id is None:
        # Fall back to the common MNLI order: contradiction, neutral, entailment.
        entailment_id = int(model.config.num_labels) - 1
    for record in test:
        uid = record_uid(clean, record)
        item = instances_by_id.get(uid, record)
        primary = int(record.get("primary_answer", -1))
        hypothesis = f"The answer to the question is {answer_text(item, primary)}."
        premise = str(item.get("context", record.get("context", "")))
        try:
            inputs = tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(torch_device)
            with torch.inference_mode():
                logits = model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            entailment = float(probs[entailment_id])
            result = {"entailment_id": entailment_id, "probs": probs}
        except Exception as exc:
            entailment = float("nan")
            result = {"error": str(exc)}
        signals = record.get("signals") or {}
        detail.append(
            {
                "uid": uid,
                "category": record.get("category", item.get("category")),
                "context_condition": record.get("context_condition", item.get("context_condition")),
                "primary_answer": primary,
                "s1_evidence": float(signals.get("s1_evidence", 0.0)),
                "nli_entailment": entailment,
                "nli_result": json.dumps(result)[:500],
            }
        )
    valid = [r for r in detail if math.isfinite(float(r["nli_entailment"]))]
    s1_vals = [float(r["s1_evidence"]) for r in valid]
    nli_vals = [float(r["nli_entailment"]) for r in valid]
    summary = [
        {
            "system": "s1_vs_nli_entailment",
            "n": len(valid),
            "pearson_r": pearson(s1_vals, nli_vals),
            "spearman_r": pearson(rank_values(s1_vals), rank_values(nli_vals)),
            "s1_mean": mean(s1_vals) if s1_vals else 0.0,
            "nli_mean": mean(nli_vals) if nli_vals else 0.0,
        }
    ]
    return detail, summary


def load_paraphrase_records(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        if not record.get("unique_id"):
            category = record.get("category", "paraphrase")
            example_id = record.get("example_id", len(out))
            record["unique_id"] = f"{category}::{example_id}"
        if "context_condition" not in record:
            raise ValueError(f"Missing context_condition in paraphrase row: {record.get('unique_id')}")
        out.append(record)
    return out


def paraphrase_condition_stress(
    clean: Any,
    train_records: list[dict],
    paraphrase_records: list[dict],
    original_instances_by_id: dict[str, dict],
    categories: list[str],
    clean_args: SimpleNamespace,
    encoder: str,
    device: str,
    batch_size: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Train on original English BBQ train split and evaluate on user-supplied
    # paraphrased/adversarial examples. This directly measures whether the
    # condition boundary survives superficial cue changes.
    clean_args.current_seed = SEEDS[0]
    train, val, _test = split_for_seed(clean, train_records, clean_args, SEEDS[0])
    all_for_embedding = list(train) + list(val) + list(paraphrase_records)
    instances_by_id = dict(original_instances_by_id)
    for record in paraphrase_records:
        instances_by_id[record_uid(clean, record)] = record
    embeddings = encode_records(
        clean,
        all_for_embedding,
        instances_by_id,
        encoder,
        device,
        batch_size,
        out_dir / "embedding_cache_paraphrase",
    )
    clf = clean.fit_condition_classifier(train, val, paraphrase_records, embeddings, categories, "embedding", SEEDS[0])
    condition_by_uid = clf["test"]["condition_by_uid"]
    detail = []
    for record in paraphrase_records:
        uid = record_uid(clean, record)
        detail.append(
            {
                "uid": uid,
                "category": record.get("category", ""),
                "gold_condition": record.get("context_condition", ""),
                "pred_condition": condition_by_uid.get(uid, ""),
                "is_correct": int(condition_by_uid.get(uid, "") == record.get("context_condition", "")),
            }
        )
    summary = [
        {
            "system": "english_train_to_paraphrase_embedding_only",
            "encoder": encoder,
            "n": len(paraphrase_records),
            "condition_accuracy": sum(int(r["is_correct"]) for r in detail) / max(len(detail), 1),
        }
    ]
    return detail, summary


def read_signal_records(results_dir: Path, model: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    signals_dir = results_dir / "signals" / model
    for path in sorted(signals_dir.glob("*_signals.jsonl")):
        category = path.name[: -len("_signals.jsonl")]
        for row in read_jsonl(path):
            row.setdefault("category", category)
            row.setdefault("unique_id", f"{row.get('category')}::{row.get('example_id')}")
            records.append(row)
    if not records:
        raise FileNotFoundError(f"No signal records found under {signals_dir}")
    return records


def parse_cross_llm_target(spec: str) -> tuple[str, Path, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid cross-LLM target {spec!r}; expected name:results_dir:model"
        )
    name, results_dir, model = parts
    return name, Path(results_dir), model


def cross_llm_condition_classifier_transfer(
    clean: Any,
    source_records: list[dict],
    embeddings: dict[str, Any],
    instances_by_id: dict[str, dict],
    categories: list[str],
    clean_args: SimpleNamespace,
    seeds: list[int],
    target_specs: list[str],
    feature_modes: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train the condition classifier on one backbone's saved records and test
    on another backbone's saved records with the same BBQ item IDs.

    The text embedding is item-level and therefore shared; signal and primary
    answer features come from the source/target backbone records. This audit is
    intended to answer whether the condition classifier depends on backbone-
    specific output/signal distributions.
    """

    evaluate_bbq, _find_unknown_index = import_eval_helpers()
    if feature_modes is None:
        feature_modes = ["embedding", "signals,primary", "signals,embedding,category,primary"]
    detail: list[dict[str, Any]] = []
    for target_name, target_results_dir, target_model in map(parse_cross_llm_target, target_specs):
        target_records = read_signal_records(target_results_dir, target_model)
        for seed in seeds:
            src_train, src_val, _src_test = split_for_seed(clean, source_records, clean_args, seed)
            _tgt_train, _tgt_val, tgt_test = split_for_seed(clean, target_records, clean_args, seed)
            for feature_mode in feature_modes:
                clf = clean.fit_condition_classifier(
                    src_train,
                    src_val,
                    tgt_test,
                    embeddings,
                    categories,
                    feature_mode,
                    seed,
                )
                condition_by_uid = clf["test"]["condition_by_uid"]
                preds, items = condition_only_predictions(clean, tgt_test, instances_by_id, condition_by_uid)
                metrics = evaluate_bbq(preds, items)
                detail.append(
                    {
                        "seed": seed,
                        "source_model": clean_args.model,
                        "target": target_name,
                        "target_model": target_model,
                        "feature_mode": feature_mode,
                        "condition_accuracy": float(clf["test"]["accuracy"]),
                        **metrics,
                    }
                )
    grouped = []
    for row in detail:
        row = dict(row)
        row["transfer_setting"] = f"{row['source_model']}_to_{row['target']}::{row['feature_mode']}"
        grouped.append(row)
    return detail, summarize_by(grouped, "transfer_setting")


def write_report(out_dir: Path, summaries: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Reviewer Extra Audits", ""]
    for name, rows in summaries.items():
        lines.append(f"## {name}")
        if not rows:
            lines.append("No rows generated.")
            lines.append("")
            continue
        for row in rows[:20]:
            compact = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
            )
            lines.append(f"- {compact}")
        if len(rows) > 20:
            lines.append(f"- ... {len(rows) - 20} more rows")
        lines.append("")
    (out_dir / "reviewer_extra_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected = audit_set(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean, clean_args, _config, records, embeddings, instances_by_id, categories = load_artifacts(args)
    summaries: dict[str, list[dict[str, Any]]] = {}

    if "lexical" in selected:
        detail, summary = lexical_condition_baselines(clean, records, instances_by_id, clean_args, args.seeds)
        write_csv(out_dir / "lexical_condition_baselines.csv", detail)
        write_csv(out_dir / "lexical_condition_baselines_summary.csv", summary)
        summaries["lexical_condition_baselines"] = summary

    if "embedding" in selected:
        detail, summary = embedding_encoder_sensitivity(
            clean,
            records,
            instances_by_id,
            categories,
            clean_args,
            args.seeds,
            args.embedding_models,
            args.device,
            args.embedding_batch_size,
            out_dir,
        )
        write_csv(out_dir / "embedding_encoder_sensitivity.csv", detail)
        write_csv(out_dir / "embedding_encoder_sensitivity_summary.csv", summary)
        summaries["embedding_encoder_sensitivity"] = summary

    if "conformal" in selected:
        detail, summary = conformal_abstention_baselines(
            clean,
            records,
            instances_by_id,
            clean_args,
            args.seeds,
            Path(args.results_dir),
            args.model,
            args.target_risks,
            args.tau_step,
        )
        write_csv(out_dir / "conformal_abstention_baselines.csv", detail)
        write_csv(out_dir / "conformal_abstention_baselines_summary.csv", summary)
        summaries["conformal_abstention_baselines"] = summary

    if "condition_conformal" in selected:
        detail, summary = condition_aware_conformal_baselines(
            clean,
            records,
            embeddings,
            instances_by_id,
            categories,
            clean_args,
            args.seeds,
            Path(args.results_dir),
            args.model,
            args.target_risks,
            args.tau_step,
        )
        write_csv(out_dir / "condition_aware_conformal_baselines.csv", detail)
        write_csv(out_dir / "condition_aware_conformal_baselines_summary.csv", summary)
        summaries["condition_aware_conformal_baselines"] = summary

    if "category_calibration" in selected:
        detail, summary, category_detail = category_aware_condition_calibration(
            clean,
            records,
            embeddings,
            instances_by_id,
            categories,
            clean_args,
            args.seeds,
        )
        write_csv(out_dir / "category_aware_condition_calibration.csv", detail)
        write_csv(out_dir / "category_aware_condition_calibration_summary.csv", summary)
        write_csv(out_dir / "category_aware_condition_calibration_by_category.csv", category_detail)
        summaries["category_aware_condition_calibration"] = summary

    if "cross_llm" in selected:
        detail, summary = cross_llm_condition_classifier_transfer(
            clean,
            records,
            embeddings,
            instances_by_id,
            categories,
            clean_args,
            args.seeds,
            args.cross_llm_targets,
        )
        write_csv(out_dir / "cross_llm_condition_classifier_transfer.csv", detail)
        write_csv(out_dir / "cross_llm_condition_classifier_transfer_summary.csv", summary)
        summaries["cross_llm_condition_classifier_transfer"] = summary

    if "single_prompt" in selected:
        detail, summary = single_prompt_low_label_audit(
            clean,
            records,
            embeddings,
            instances_by_id,
            categories,
            clean_args,
            args.seeds,
            args.low_label_fractions,
        )
        write_csv(out_dir / "single_prompt_low_label.csv", detail)
        write_csv(out_dir / "single_prompt_low_label_summary.csv", summary)
        summaries["single_prompt_low_label"] = summary

    if "nli" in selected:
        detail, summary = nli_evidence_audit(
            clean,
            records,
            instances_by_id,
            clean_args,
            args.seeds,
            args.nli_model,
            args.max_nli_examples,
            args.device,
        )
        write_csv(out_dir / "nli_evidence_audit.csv", detail)
        write_csv(out_dir / "nli_evidence_audit_summary.csv", summary)
        summaries["nli_evidence_audit"] = summary

    if "paraphrase" in selected:
        if not args.paraphrase_file:
            print("[paraphrase] skipped: provide --paraphrase-file to run this audit.", file=sys.stderr)
        else:
            paraphrase_records = load_paraphrase_records(Path(args.paraphrase_file))
            detail, summary = paraphrase_condition_stress(
                clean,
                records,
                paraphrase_records,
                instances_by_id,
                categories,
                clean_args,
                args.embedding_models[0],
                args.device,
                args.embedding_batch_size,
                out_dir,
            )
            write_csv(out_dir / "paraphrase_condition_stress.csv", detail)
            write_csv(out_dir / "paraphrase_condition_stress_summary.csv", summary)
            summaries["paraphrase_condition_stress"] = summary

    write_report(out_dir, summaries)
    print(f"[done] wrote reviewer extra audits to {out_dir}")


if __name__ == "__main__":
    main()
