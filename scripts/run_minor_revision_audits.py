#!/usr/bin/env python3
"""Minor-revision audits requested by selective-prediction reviewers.

The script reuses saved BBQ signals, cached embeddings, and saved stage-1
log-probabilities. It does not call an LLM. It adds three reviewer-facing
checks:

1. confidence-only abstention baselines without condition prediction;
2. selective-prediction AURC / E-AURC curves;
3. per-category coverage and false-abstention audits for condition-only
   retention.
4. simple learned rejectors over the same saved features, without the MoE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from src.evaluation.bbq_evaluator import evaluate_bbq  # noqa: E402
from src.models.override import find_unknown_index  # noqa: E402


SEEDS = [42, 123, 456, 789, 999]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minor-revision audits without LLM inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="main")
    parser.add_argument("--results-dir", default="results/v2")
    parser.add_argument("--sampled-dir", default="data/sampled_v2")
    parser.add_argument("--out-dir", default="results/v2/minor_revision_audits")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--samples-per-category", type=int, default=1000)
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument("--temperature-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    return parser.parse_args()


def mean_std(values: list[float]) -> tuple[float, float]:
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
        device="cpu",
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


def softmax_from_logprobs(logprobs: dict[str, float], temperature: float = 1.0) -> list[float]:
    vals = [float(logprobs.get(k, float("-inf"))) / max(temperature, 1e-8) for k in ("A", "B", "C")]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return [1 / 3, 1 / 3, 1 / 3]
    max_v = max(finite)
    exps = [math.exp(v - max_v) if math.isfinite(v) else 0.0 for v in vals]
    total = sum(exps)
    if total <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
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


def split_for_seed(records: list[dict], clean_args: SimpleNamespace, seed: int):
    clean_args.current_seed = seed
    return clean.split_records(records, clean_args)


def make_confidence_rows(
    records: list[dict],
    instances_by_id: dict[str, dict],
    stage1_logprobs: dict[str, dict[str, float]],
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        uid = clean.uid_for(rec)
        item = instances_by_id.get(uid, rec)
        primary = int(rec.get("primary_answer", -1))
        signals = rec.get("signals") or {}
        logprobs = stage1_logprobs.get(uid, {})
        probs = softmax_from_logprobs(logprobs, temperature=temperature)
        unknown_idx = find_unknown_index(item)
        chosen_prob = float(signals.get("s3_confidence") or (probs[primary] if primary in (0, 1, 2) else 0.0))
        unknown_prob = float(probs[unknown_idx]) if unknown_idx in (0, 1, 2) else 0.0
        rows.append(
            {
                "uid": uid,
                "item": item,
                "category": rec.get("category", item.get("category")),
                "primary_answer": primary,
                "chosen_prob": chosen_prob,
                "max_softmax": float(max(probs)),
                "unknown_prob": unknown_prob,
                "unknown_retention_score": 1.0 - unknown_prob,
            }
        )
    return rows


def apply_retention_threshold(rows: list[dict[str, Any]], score_key: str, tau: float) -> tuple[list[int], list[dict]]:
    preds: list[int] = []
    items: list[dict] = []
    for row in rows:
        item = row["item"]
        if float(row[score_key]) >= tau or int(row["primary_answer"]) == -1:
            pred = int(row["primary_answer"])
        else:
            pred = find_unknown_index(item)
        preds.append(pred)
        items.append(item)
    return preds, items


def objective(metrics: dict[str, Any]) -> tuple[float, float, float]:
    score = (float(metrics.get("accuracy_amb", 0.0)) + float(metrics.get("accuracy_dis", 0.0))) / 2.0
    return score, -float(metrics.get("false_abstention_rate", 1.0)), float(metrics.get("accuracy_amb", 0.0))


def tune_confidence_policy(
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    score_key: str,
    step: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for tau in np.arange(0.0, 1.0 + step / 2, step):
        tau = round(float(tau), 4)
        val_preds, val_items = apply_retention_threshold(val_rows, score_key, tau)
        val_metrics = evaluate_bbq(val_preds, val_items)
        candidate = {"tau": tau, "val_metrics": val_metrics}
        if best is None or objective(val_metrics) > objective(best["val_metrics"]):
            best = candidate
    assert best is not None
    test_preds, test_items = apply_retention_threshold(test_rows, score_key, float(best["tau"]))
    return {"tau": best["tau"], "val_metrics": best["val_metrics"], "test_metrics": evaluate_bbq(test_preds, test_items)}


def summarize_seed_metrics(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for group in sorted({str(r[group_key]) for r in rows}):
        group_rows = [r for r in rows if str(r[group_key]) == group]
        out: dict[str, Any] = {group_key: group, "n": len(group_rows)}
        for metric in ("accuracy_amb", "accuracy_dis", "false_abstention_rate", "coverage", "ambig_abstention", "disambig_coverage"):
            vals = [float(r[metric]) for r in group_rows if metric in r and r[metric] != ""]
            if vals:
                m, s = mean_std(vals)
                out[f"{metric}_mean"] = m
                out[f"{metric}_std"] = s
        summary.append(out)
    return summary


def confidence_baselines(
    records: list[dict],
    instances_by_id: dict[str, dict],
    stage1_logprobs: dict[str, dict[str, float]],
    clean_args: SimpleNamespace,
    seeds: list[int],
    tau_step: float,
    temperature_grid: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    for seed in seeds:
        _train, val, test = split_for_seed(records, clean_args, seed)
        val_rows = make_confidence_rows(val, instances_by_id, stage1_logprobs, temperature=1.0)
        test_rows = make_confidence_rows(test, instances_by_id, stage1_logprobs, temperature=1.0)
        specs = [
            ("chosen_softmax_threshold", "chosen_prob", 1.0),
            ("max_softmax_threshold", "max_softmax", 1.0),
            ("unknown_probability_threshold", "unknown_retention_score", 1.0),
        ]
        for system, score_key, temp in specs:
            result = tune_confidence_policy(val_rows, test_rows, score_key, tau_step)
            detail.append({"seed": seed, "system": system, "temperature": temp, "tau": result["tau"], **result["test_metrics"]})

        best_temp_result: dict[str, Any] | None = None
        best_temp = 1.0
        for temp in temperature_grid:
            val_t = make_confidence_rows(val, instances_by_id, stage1_logprobs, temperature=temp)
            test_t = make_confidence_rows(test, instances_by_id, stage1_logprobs, temperature=temp)
            result = tune_confidence_policy(val_t, test_t, "unknown_retention_score", tau_step)
            if best_temp_result is None or objective(result["val_metrics"]) > objective(best_temp_result["val_metrics"]):
                best_temp_result = result
                best_temp = temp
        assert best_temp_result is not None
        detail.append(
            {
                "seed": seed,
                "system": "unknown_probability_temp_scaled",
                "temperature": best_temp,
                "tau": best_temp_result["tau"],
                **best_temp_result["test_metrics"],
            }
        )
    return detail, summarize_seed_metrics(detail, "system")


def aurc_and_eaurc(correct: list[int], scores: list[float]) -> tuple[float, float]:
    if not correct:
        return 0.0, 0.0
    order = sorted(range(len(correct)), key=lambda i: scores[i], reverse=True)
    errors = [1 - int(correct[i]) for i in order]
    cum_errors = np.cumsum(errors)
    ks = np.arange(1, len(errors) + 1)
    risks = cum_errors / ks
    aurc = float(np.mean(risks))
    n_correct = sum(int(v) for v in correct)
    oracle_errors = [0] * n_correct + [1] * (len(correct) - n_correct)
    oracle_risks = np.cumsum(oracle_errors) / ks
    oracle_aurc = float(np.mean(oracle_risks))
    return aurc, aurc - oracle_aurc


def risk_at_coverages(correct: list[int], scores: list[float], coverages: list[float]) -> dict[float, float]:
    order = sorted(range(len(correct)), key=lambda i: scores[i], reverse=True)
    errors = [1 - int(correct[i]) for i in order]
    out: dict[float, float] = {}
    for cov in coverages:
        k = max(1, int(round(cov * len(errors))))
        out[cov] = float(sum(errors[:k]) / k)
    return out


def load_moe_scores(seed: int) -> dict[str, float]:
    path = ROOT / "results" / "v2" / "clean_experiments" / f"seed_{seed}" / "test_predictions.jsonl"
    scores: dict[str, float] = {}
    if not path.exists():
        return scores
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row.get("uid") or f"{row.get('category')}::{row.get('example_id')}")
            scores[uid] = float(row.get("p_score", 0.0))
    return scores


def selective_risk_audits(
    records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    instances_by_id: dict[str, dict],
    stage1_logprobs: dict[str, dict[str, float]],
    clean_args: SimpleNamespace,
    seeds: list[int],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    detail: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    curve_accumulator: dict[tuple[str, float], list[float]] = defaultdict(list)
    for seed in seeds:
        train, val, test = split_for_seed(records, clean_args, seed)
        clf = clean.fit_condition_classifier(
            train,
            val,
            test,
            embeddings,
            categories,
            "signals,embedding,category,primary",
            seed,
        )
        rows = make_confidence_rows(test, instances_by_id, stage1_logprobs, temperature=1.0)
        moe_scores = load_moe_scores(seed)
        correct = [1 if int(r["primary_answer"]) == int(r["item"].get("label", -999)) else 0 for r in rows]
        score_map = {
            "chosen_softmax": [float(r["chosen_prob"]) for r in rows],
            "unknown_inverse": [float(r["unknown_retention_score"]) for r in rows],
            "condition_disambig_prob": [
                float(clf["test"]["probability_by_uid"].get(r["uid"], {}).get("disambig", 0.0)) for r in rows
            ],
            "moe_retention": [float(moe_scores.get(r["uid"], 0.0)) for r in rows],
        }
        for scorer, scores in score_map.items():
            aurc, eaurc = aurc_and_eaurc(correct, scores)
            detail.append({"seed": seed, "scorer": scorer, "aurc": aurc, "eaurc": eaurc})
            risks = risk_at_coverages(correct, scores, coverages)
            for cov, risk in risks.items():
                curve_rows.append({"seed": seed, "scorer": scorer, "coverage": cov, "risk": risk, "accuracy": 1 - risk})
                curve_accumulator[(scorer, cov)].append(risk)
    summary = summarize_seed_metrics(
        [
            {
                "scorer": r["scorer"],
                "accuracy_amb": r["aurc"],
                "accuracy_dis": r["eaurc"],
                "false_abstention_rate": 0.0,
            }
            for r in detail
        ],
        "scorer",
    )
    for row in summary:
        row["aurc_mean"] = row.pop("accuracy_amb_mean")
        row["aurc_std"] = row.pop("accuracy_amb_std")
        row["eaurc_mean"] = row.pop("accuracy_dis_mean")
        row["eaurc_std"] = row.pop("accuracy_dis_std")
        row.pop("false_abstention_rate_mean", None)
        row.pop("false_abstention_rate_std", None)

    curve_summary: list[dict[str, Any]] = []
    for (scorer, cov), risks in sorted(curve_accumulator.items()):
        m, s = mean_std(risks)
        curve_summary.append({"scorer": scorer, "coverage": cov, "risk_mean": m, "risk_std": s, "accuracy_mean": 1 - m})
    write_csv(out_dir / "risk_coverage_curve.csv", curve_rows)
    write_csv(out_dir / "risk_coverage_curve_summary.csv", curve_summary)
    plot_risk_coverage(curve_summary, out_dir / "risk_coverage_selective_prediction.pdf")
    return detail, summary


def plot_risk_coverage(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except Exception:
        return
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = {
        "chosen_softmax": "chosen softmax",
        "unknown_inverse": "unknown-prob inverse",
        "condition_disambig_prob": "condition prob.",
        "moe_retention": "MoE retention",
    }
    colors = {
        "chosen_softmax": "#777777",
        "unknown_inverse": "#b95f02",
        "condition_disambig_prob": "#1b9e77",
        "moe_retention": "#2b6cb0",
    }
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for scorer in ["chosen_softmax", "unknown_inverse", "condition_disambig_prob", "moe_retention"]:
        pts = [r for r in rows if r["scorer"] == scorer]
        if not pts:
            continue
        xs = [float(r["coverage"]) for r in pts]
        ys = [float(r["risk_mean"]) for r in pts]
        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.0, label=labels.get(scorer, scorer), color=colors.get(scorer))
    ax.set_xlabel("Coverage (retained primary answers)")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.8)
    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", format="png", dpi=300)
    plt.close(fig)


def per_category_condition_only(
    records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    instances_by_id: dict[str, dict],
    clean_args: SimpleNamespace,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = split_for_seed(records, clean_args, seed)
        clf = clean.fit_condition_classifier(train, val, test, embeddings, categories, "signals,embedding,category,primary", seed)
        by_cat: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for rec in test:
            uid = clean.uid_for(rec)
            item = instances_by_id.get(uid, rec)
            cond = clf["test"]["condition_by_uid"].get(uid, "")
            pred = find_unknown_index(item) if cond == "ambig" else int(rec.get("primary_answer", -1))
            by_cat[str(rec.get("category", item.get("category")))].append((pred, item))
        for cat, pairs in by_cat.items():
            preds = [p for p, _ in pairs]
            items = [it for _, it in pairs]
            metrics = evaluate_bbq(preds, items)
            unknown_flags = [1 if p == find_unknown_index(it) else 0 for p, it in pairs]
            dis_pairs = [(p, it) for p, it in pairs if it.get("context_condition") == "disambig"]
            amb_pairs = [(p, it) for p, it in pairs if it.get("context_condition") == "ambig"]
            detail.append(
                {
                    "seed": seed,
                    "category": cat,
                    **metrics,
                    "coverage": 1.0 - (sum(unknown_flags) / max(len(unknown_flags), 1)),
                    "ambig_abstention": sum(1 for p, it in amb_pairs if p == find_unknown_index(it)) / max(len(amb_pairs), 1),
                    "disambig_coverage": sum(1 for p, it in dis_pairs if p != find_unknown_index(it)) / max(len(dis_pairs), 1),
                }
            )
    return detail, summarize_seed_metrics(detail, "category")


def build_rejector_features(
    records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    modes_csv: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    modes = {m.strip() for m in modes_csv.split(",") if m.strip()}
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    rows: list[list[float]] = []
    labels: list[int] = []
    uids: list[str] = []
    for rec in records:
        uid = clean.uid_for(rec)
        if "embedding" in modes and uid not in embeddings:
            continue
        feats: list[float] = []
        if "signals" in modes:
            feats.extend(clean.signal_vector(rec))
        if "primary" in modes:
            primary = int(rec.get("primary_answer", -1))
            feats.extend([1.0 if primary == v else 0.0 for v in (-1, 0, 1, 2)])
        if "category" in modes:
            one_hot = [0.0] * len(categories)
            if rec.get("category") in cat_to_idx:
                one_hot[cat_to_idx[rec.get("category")]] = 1.0
            feats.extend(one_hot)
        if "embedding" in modes:
            emb = embeddings[uid]
            emb_arr = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
            feats.extend([float(x) for x in emb_arr.reshape(-1)])
        label = int(rec.get("label", -999))
        primary = int(rec.get("primary_answer", -1))
        rows.append(feats)
        labels.append(1 if primary == label else 0)
        uids.append(uid)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), uids


def tune_learned_rejector(
    val_scores: dict[str, float],
    test_scores: dict[str, float],
    val_records: list[dict],
    test_records: list[dict],
    instances_by_id: dict[str, dict],
    tau_step: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None

    def apply(records: list[dict], scores: dict[str, float], tau: float) -> tuple[list[int], list[dict]]:
        preds: list[int] = []
        items: list[dict] = []
        for rec in records:
            uid = clean.uid_for(rec)
            item = instances_by_id.get(uid, rec)
            if float(scores.get(uid, 0.0)) >= tau:
                pred = int(rec.get("primary_answer", -1))
            else:
                pred = find_unknown_index(item)
            preds.append(pred)
            items.append(item)
        return preds, items

    for tau in np.arange(0.0, 1.0 + tau_step / 2, tau_step):
        tau = round(float(tau), 4)
        val_preds, val_items = apply(val_records, val_scores, tau)
        val_metrics = evaluate_bbq(val_preds, val_items)
        candidate = {"tau": tau, "val_metrics": val_metrics}
        if best is None or objective(val_metrics) > objective(best["val_metrics"]):
            best = candidate
    assert best is not None
    test_preds, test_items = apply(test_records, test_scores, float(best["tau"]))
    return {"tau": best["tau"], "val_metrics": best["val_metrics"], "test_metrics": evaluate_bbq(test_preds, test_items)}


def learned_rejector_baselines(
    records: list[dict],
    embeddings: dict[str, Any],
    categories: list[str],
    instances_by_id: dict[str, dict],
    clean_args: SimpleNamespace,
    seeds: list[int],
    tau_step: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    variants = [
        ("logistic_rejector_signals", "signals"),
        ("logistic_rejector_raw_text_embedding", "embedding"),
        ("logistic_rejector_signals_embedding", "signals,embedding"),
    ]
    detail: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = split_for_seed(records, clean_args, seed)
        for system, modes in variants:
            x_train, y_train, train_uids = build_rejector_features(train, embeddings, categories, modes)
            x_val, _y_val, val_uids = build_rejector_features(val, embeddings, categories, modes)
            x_test, _y_test, test_uids = build_rejector_features(test, embeddings, categories, modes)
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
            clf.fit(x_train, y_train)
            classes = list(clf.named_steps["logisticregression"].classes_)
            keep_idx = classes.index(1)
            val_scores = {uid: float(p[keep_idx]) for uid, p in zip(val_uids, clf.predict_proba(x_val))}
            test_scores = {uid: float(p[keep_idx]) for uid, p in zip(test_uids, clf.predict_proba(x_test))}
            result = tune_learned_rejector(val_scores, test_scores, val, test, instances_by_id, tau_step)
            detail.append({"seed": seed, "system": system, "feature_modes": modes, "tau": result["tau"], **result["test_metrics"]})
    return detail, summarize_seed_metrics(detail, "system")


def write_latex_snippets(
    out_dir: Path,
    conf_summary: list[dict[str, Any]],
    rc_summary: list[dict[str, Any]],
    cat_summary: list[dict[str, Any]],
    rejector_summary: list[dict[str, Any]],
) -> None:
    def ms(row: dict[str, Any], key: str) -> str:
        return f"${float(row[f'{key}_mean']):.4f}\\pm{float(row[f'{key}_std']):.4f}$"

    lines: list[str] = []
    lines.append("% Confidence-only baseline table rows")
    for row in conf_summary:
        lines.append(
            f"{row['system']} & {ms(row, 'accuracy_amb')} & {ms(row, 'accuracy_dis')} & "
            f"{ms(row, 'false_abstention_rate')} \\\\"
        )
    lines.append("")
    lines.append("% Selective-prediction AURC/E-AURC rows")
    for row in rc_summary:
        lines.append(
            f"{row['scorer']} & ${float(row['aurc_mean']):.4f}\\pm{float(row['aurc_std']):.4f}$ & "
            f"${float(row['eaurc_mean']):.4f}\\pm{float(row['eaurc_std']):.4f}$ \\\\"
        )
    lines.append("")
    lines.append("% Learned rejector baseline rows")
    for row in rejector_summary:
        lines.append(
            f"{row['system']} & {ms(row, 'accuracy_amb')} & {ms(row, 'accuracy_dis')} & "
            f"{ms(row, 'false_abstention_rate')} \\\\"
        )
    lines.append("")
    lines.append("% Per-category condition-only rows")
    for row in cat_summary:
        lines.append(
            f"{row['category']} & {ms(row, 'coverage')} & {ms(row, 'disambig_coverage')} & "
            f"{ms(row, 'false_abstention_rate')} \\\\"
        )
    (out_dir / "latex_snippets.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    clean_args = base_args(args)
    config = clean.load_experiment_config(clean_args)
    categories = list(config["data"]["categories"])
    records, embeddings, instances_by_id = clean.load_records_embeddings_instances(config, clean_args)
    stage1 = load_stage1_logprobs(Path(args.results_dir), args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_detail, conf_summary = confidence_baselines(
        records,
        instances_by_id,
        stage1,
        clean_args,
        args.seeds,
        args.tau_step,
        args.temperature_grid,
    )
    rc_detail, rc_summary = selective_risk_audits(
        records,
        embeddings,
        categories,
        instances_by_id,
        stage1,
        clean_args,
        args.seeds,
        out_dir,
    )
    cat_detail, cat_summary = per_category_condition_only(
        records,
        embeddings,
        categories,
        instances_by_id,
        clean_args,
        args.seeds,
    )
    rejector_detail, rejector_summary = learned_rejector_baselines(
        records,
        embeddings,
        categories,
        instances_by_id,
        clean_args,
        args.seeds,
        args.tau_step,
    )
    write_csv(out_dir / "confidence_only_baselines.csv", conf_detail)
    write_csv(out_dir / "confidence_only_baselines_summary.csv", conf_summary)
    write_csv(out_dir / "selective_risk_detail.csv", rc_detail)
    write_csv(out_dir / "selective_risk_summary.csv", rc_summary)
    write_csv(out_dir / "per_category_condition_only_coverage.csv", cat_detail)
    write_csv(out_dir / "per_category_condition_only_coverage_summary.csv", cat_summary)
    write_csv(out_dir / "learned_rejector_baselines.csv", rejector_detail)
    write_csv(out_dir / "learned_rejector_baselines_summary.csv", rejector_summary)
    write_latex_snippets(out_dir, conf_summary, rc_summary, cat_summary, rejector_summary)
    print(f"Wrote minor-revision audits to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
