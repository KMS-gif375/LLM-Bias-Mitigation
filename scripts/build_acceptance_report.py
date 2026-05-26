#!/usr/bin/env python3
"""
Build reviewer-defense tables and reproducibility notes for the acceptance package.

The script is intentionally read-only with respect to experiment outputs: it only
collects existing JSON/CSV/JSONL artifacts and writes derived paper-ready tables.
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
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build acceptance-package report tables.")
    parser.add_argument("--clean-dir", default="results/v2/clean_experiments")
    parser.add_argument("--loco-dir", default="results/v2/acceptance_package/loco")
    parser.add_argument("--openbbq-dir", default="results/v2/acceptance_package/open_bbq")
    parser.add_argument("--qwen-dir", default="results/v2/cross_llm/qwen/multi_seed_5seed")
    parser.add_argument("--mistral-dir", default="results/v2/cross_llm/mistral/multi_seed_5seed")
    parser.add_argument("--sampled-dir", default="data/sampled_v2")
    parser.add_argument("--out-dir", default="results/v2/acceptance_package/report")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def aggregate(values: Iterable[Any]) -> dict[str, Any]:
    nums = [as_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": mean(nums),
        "std": stdev(nums) if len(nums) > 1 else 0.0,
        "n": len(nums),
    }


def load_sampled_items(sampled_dir: Path) -> dict[str, dict]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("pandas is required to build residual bias tables.") from exc

    items: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        path = sampled_dir / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for row in df.itertuples(index=False):
            rec = row._asdict()
            uid = f"{rec.get('category', '_unknown')}::{rec.get('example_id')}"
            items[uid] = rec
    return items


def residual_bias_counts(clean_dir: Path, sampled_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    from src.evaluation.bbq_evaluator import is_stereotyped_answer

    items = load_sampled_items(sampled_dir)
    fixed_fields = {
        "uid",
        "example_id",
        "category",
        "context_condition",
        "label",
        "primary_answer",
        "p_score",
        "predicted_condition",
    }
    rows: list[dict[str, Any]] = []
    for pred_path in sorted(clean_dir.glob("seed_*/test_predictions.jsonl")):
        seed = pred_path.parent.name.replace("seed_", "")
        data = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not data:
            continue
        variants = [k for k in data[0].keys() if k not in fixed_fields]
        for variant in variants:
            n_ambig = unknown_correct = unknown_pred = stereo = anti = invalid = missing = 0
            for row in data:
                if row.get("context_condition") != "ambig":
                    continue
                n_ambig += 1
                uid = row.get("uid")
                item = items.get(uid or "", row)
                pred = row.get(variant)
                if pred is None:
                    missing += 1
                    continue
                pred = int(pred)
                if pred == item.get("label"):
                    unknown_correct += 1
                kind = is_stereotyped_answer(item, pred)
                if kind == "unknown":
                    unknown_pred += 1
                elif kind == "stereotyped":
                    stereo += 1
                elif kind == "anti_stereotyped":
                    anti += 1
                else:
                    invalid += 1
            denom = stereo + anti
            bias = None if denom == 0 else 2 * (stereo / denom) - 1
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "n_ambig": n_ambig,
                    "unknown_correct": unknown_correct,
                    "unknown_pred": unknown_pred,
                    "non_unknown_residual": denom,
                    "stereo": stereo,
                    "anti": anti,
                    "invalid": invalid,
                    "missing": missing,
                    "bias_score_amb": bias,
                    "bias_abs_amb": None if bias is None else abs(float(bias)),
                }
            )
    write_csv(out_dir / "residual_bias_counts.csv", rows)
    return rows


def paired_tests_summary(clean_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(clean_dir / "paired_tests.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row.get("comparison", "").startswith("ours_predicted_condition_vs_"):
            continue
        if row.get("diff") in ("", None):
            continue
        grouped[(row["comparison"], row["metric"])].append(row)

    out: list[dict[str, Any]] = []
    for (comparison, metric), vals in sorted(grouped.items()):
        diffs = [as_float(v.get("diff")) for v in vals]
        ps = [as_float(v.get("p_value")) for v in vals]
        diffs = [v for v in diffs if v is not None]
        ps = [v for v in ps if v is not None]
        out.append(
            {
                "comparison": comparison,
                "metric": metric,
                "n_seeds": len(vals),
                "mean_diff": mean(diffs) if diffs else None,
                "max_p_value": max(ps) if ps else None,
                "min_n": min(int(float(v["n"])) for v in vals if v.get("n")),
                "max_n": max(int(float(v["n"])) for v in vals if v.get("n")),
            }
        )
    write_csv(out_dir / "paired_tests_summary.csv", out)
    return out


def threshold_audit_summary(clean_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(clean_dir / "low_threshold_audit.csv")
    by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)

    out: list[dict[str, Any]] = []
    for seed, vals in sorted(by_seed.items(), key=lambda kv: int(kv[0])):
        val_rows = [r for r in vals if r.get("split") == "val"]
        test_rows = [r for r in vals if r.get("split") == "test"]
        best_val = max(val_rows, key=lambda r: as_float(r.get("score")) or -1.0) if val_rows else None
        test_at_005 = next((r for r in test_rows if abs((as_float(r.get("tau_dis")) or 999) - 0.05) < 1e-9), None)
        if best_val:
            out.append(
                {
                    "seed": seed,
                    "best_val_tau_dis": best_val.get("tau_dis"),
                    "best_val_score": best_val.get("score"),
                    "test_tau_0_05_score": None if test_at_005 is None else test_at_005.get("score"),
                    "test_tau_0_05_acc_amb": None if test_at_005 is None else test_at_005.get("accuracy_amb"),
                    "test_tau_0_05_acc_dis": None if test_at_005 is None else test_at_005.get("accuracy_dis"),
                    "test_tau_0_05_far": None if test_at_005 is None else test_at_005.get("false_abstention_rate"),
                }
            )
    write_csv(out_dir / "threshold_audit_summary.csv", out)
    return out


def signal_ablation_summary(clean_dir: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = load_json(clean_dir / "summary.json") or []
    rows: list[dict[str, Any]] = []
    for seed_payload in summary:
        seed = seed_payload.get("seed")
        ablation = seed_payload.get("signal_ablation") or {}
        for signal, payload in ablation.items():
            if signal == "full":
                continue
            deltas = payload.get("deltas_vs_full") or {}
            metrics = payload.get("metrics_per_condition_oracle") or {}
            rows.append(
                {
                    "seed": seed,
                    "signal": signal,
                    "delta_accuracy_amb": deltas.get("delta_accuracy_amb"),
                    "delta_accuracy_dis": deltas.get("delta_accuracy_dis"),
                    "delta_false_abstention_rate": deltas.get("delta_false_abstention_rate"),
                    "delta_bias_abs_amb": deltas.get("delta_bias_abs_amb"),
                    "masked_accuracy_amb": metrics.get("accuracy_amb"),
                    "masked_accuracy_dis": metrics.get("accuracy_dis"),
                    "masked_false_abstention_rate": metrics.get("false_abstention_rate"),
                }
            )
    write_csv(out_dir / "signal_ablation_deltas.csv", rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["signal"]].append(row)
    agg_rows: list[dict[str, Any]] = []
    for signal, vals in sorted(grouped.items()):
        out = {"signal": signal, "n_seeds": len(vals)}
        for key in (
            "delta_accuracy_amb",
            "delta_accuracy_dis",
            "delta_false_abstention_rate",
            "delta_bias_abs_amb",
        ):
            stats = aggregate(v.get(key) for v in vals)
            out[f"{key}_mean"] = stats["mean"]
            out[f"{key}_std"] = stats["std"]
        agg_rows.append(out)
    write_csv(out_dir / "signal_ablation_summary.csv", agg_rows)
    return rows, agg_rows


def baseline_coverage(clean_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(clean_dir / "aggregate_metrics.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "system": row.get("system"),
                "subset": row.get("subset"),
                "n_seeds": row.get("n_seeds"),
                "n_total_mean": row.get("n_total_mean"),
                "n_shared_min": row.get("n_shared_min"),
                "n_missing_max": row.get("n_missing_max"),
                "accuracy_amb": f"{fmt(row.get('accuracy_amb_mean'))}+/-{fmt(row.get('accuracy_amb_std'))}",
                "accuracy_dis": f"{fmt(row.get('accuracy_dis_mean'))}+/-{fmt(row.get('accuracy_dis_std'))}",
                "FAR": f"{fmt(row.get('false_abstention_rate_mean'))}+/-{fmt(row.get('false_abstention_rate_std'))}",
                "bias_abs_amb": f"{fmt(row.get('bias_abs_amb_mean'))}+/-{fmt(row.get('bias_abs_amb_std'))}",
            }
        )
    write_csv(out_dir / "main_and_baseline_metrics.csv", out)
    return out


def loco_summary(loco_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(loco_dir / "loco_aggregate.csv")
    write_csv(out_dir / "loco_generalization_summary.csv", rows)
    return rows


def openbbq_summary(openbbq_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    data = load_json(openbbq_dir / "overall_metrics.json")
    if not data:
        write_csv(out_dir / "openbbq_transfer_summary.csv", [])
        return []
    overall = data.get("overall", {})
    row = {
        "dataset": "Open-BBQ",
        "n_total": overall.get("n_total", data.get("n_total")),
        "n_ambig": overall.get("n_ambig"),
        "n_disambig": overall.get("n_disambig"),
        "accuracy_amb": overall.get("accuracy_amb"),
        "accuracy_dis": overall.get("accuracy_dis"),
        "bias_score_amb": overall.get("bias_score_amb"),
        "bias_score_dis": overall.get("bias_score_dis"),
        "false_abstention_rate": overall.get("false_abstention_rate"),
        "parse_fail_rate": overall.get("parse_fail_rate"),
        "threshold_amb": (data.get("thresholds_per_condition") or {}).get("ambig"),
        "threshold_dis": (data.get("thresholds_per_condition") or {}).get("disambig"),
    }
    rows = [row]
    write_csv(out_dir / "openbbq_transfer_summary.csv", rows)
    return rows


def cross_llm_summary(qwen_dir: Path, mistral_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in (("qwen", qwen_dir / "summary.json"), ("mistral", mistral_dir / "summary.json")):
        data = load_json(path)
        if not data:
            continue
        agg = data.get("aggregate", {})
        row = {"model": name, "seeds": ",".join(str(s) for s in data.get("seeds", []))}
        for metric in ("accuracy_amb", "accuracy_dis", "false_abstention_rate", "bias_score_amb", "parse_fail_rate"):
            stats = agg.get(metric, {})
            row[f"{metric}_mean"] = stats.get("mean")
            row[f"{metric}_std"] = stats.get("std")
            row[f"{metric}_n"] = stats.get("n")
        rows.append(row)
    write_csv(out_dir / "cross_llm_summary.csv", rows)
    return rows


def write_reproducibility(out_dir: Path) -> None:
    text = """# Reproducibility Commands

Environment:

```bash
python -m pip install -r requirements.txt
```

Main clean suite:

```bash
python scripts/run_clean_experiments.py \\
  --seeds 42 123 456 789 999 \\
  --out-dir results/v2/clean_experiments \\
  --run-signal-ablation
```

Clean LOCO:

```bash
python scripts/run_loco_clean.py \\
  --seeds 42 123 456 789 999 \\
  --out-dir results/v2/acceptance_package/loco
```

Open-BBQ transfer:

```bash
python -m src.transfer.run_open_bbq \\
  --max-samples 300 \\
  --out-dir results/v2/acceptance_package/open_bbq \\
  --force --model main
```

Cross-LLM 5-seed summaries from existing signals:

```bash
python -m src.analysis.multi_seed --version v2 --model qwen \\
  --seeds 42,123,456,789,999 \\
  --out-dir results/v2/cross_llm/qwen/multi_seed_5seed

python -m src.analysis.multi_seed --version v2 --model mistral \\
  --seeds 42,123,456,789,999 \\
  --out-dir results/v2/cross_llm/mistral/multi_seed_5seed
```

Build paper/appendix tables:

```bash
python scripts/build_acceptance_report.py
```
"""
    (out_dir / "reproducibility.md").write_text(text, encoding="utf-8")


def write_claim_language(out_dir: Path) -> None:
    text = """# Claim Language

Strong claim:

> The proposed method preserves high ambiguous-context abstention accuracy while substantially improving disambiguated-context utility and reducing false abstention, without relying on oracle condition labels at test time.

Generalization claim:

> Leave-one-category-out and Open-BBQ transfer experiments indicate that the behavior is not explained solely by category memorization or by tuning to the original BBQ split.

SAE/s7 wording:

> The SAE-derived signal is included in the full signal set and explicitly audited through signal masking. Its isolated ablation effect is small, suggesting that the final behavior is driven by the combined decision mechanism rather than by a single SAE feature.

Avoid:

- Do not claim lowest ambiguous residual bias.
- Do not claim s7 is the main driver.
- Do not treat Fairsteer as a primary full-coverage baseline when overlap is small.
- Do not claim significant improvement over self-debiasing on ambiguous accuracy alone.
"""
    (out_dir / "claim_language.md").write_text(text, encoding="utf-8")


def write_markdown_report(
    out_dir: Path,
    baseline_rows: list[dict[str, Any]],
    loco_rows: list[dict[str, Any]],
    openbbq_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Acceptance Package Report",
        "",
        "This package collects the reviewer-defense experiments and paper-ready appendix tables.",
        "",
        "## Main Metrics",
        "",
        "| System | subset | n | acc_amb | acc_dis | FAR | abs_bias_amb |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    main_rows = [row for row in baseline_rows if row.get("system") != "fairsteer"]
    aux_rows = [row for row in baseline_rows if row.get("system") == "fairsteer"]
    for row in main_rows:
        lines.append(
            f"| {row.get('system')} | {row.get('subset')} | {row.get('n_total_mean')} | "
            f"{row.get('accuracy_amb')} | {row.get('accuracy_dis')} | {row.get('FAR')} | {row.get('bias_abs_amb')} |"
        )
    if aux_rows:
        lines += [
            "",
            "Auxiliary limited-overlap comparison:",
            "",
            "| System | subset | n | acc_amb | acc_dis | FAR | note |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for row in aux_rows:
            lines.append(
                f"| {row.get('system')} | {row.get('subset')} | {row.get('n_total_mean')} | "
                f"{row.get('accuracy_amb')} | {row.get('accuracy_dis')} | {row.get('FAR')} | "
                "limited matched-ID overlap |"
            )

    lines += [
        "",
        "## Generalization",
        "",
        "| Variant | folds | acc_amb | acc_dis | FAR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in loco_rows:
        lines.append(
            f"| {row.get('variant')} | {row.get('n_folds')} | "
            f"{fmt(row.get('accuracy_amb_mean'))}+/-{fmt(row.get('accuracy_amb_std'))} | "
            f"{fmt(row.get('accuracy_dis_mean'))}+/-{fmt(row.get('accuracy_dis_std'))} | "
            f"{fmt(row.get('false_abstention_rate_mean'))}+/-{fmt(row.get('false_abstention_rate_std'))} |"
        )

    lines += ["", "## Open-BBQ Transfer", ""]
    if openbbq_rows:
        row = openbbq_rows[0]
        lines.append(
            f"Open-BBQ n={row.get('n_total')}: acc_amb={fmt(row.get('accuracy_amb'))}, "
            f"acc_dis={fmt(row.get('accuracy_dis'))}, FAR={fmt(row.get('false_abstention_rate'))}."
        )
    else:
        lines.append("Open-BBQ result not found yet.")

    lines += [
        "",
        "## Cross-LLM",
        "",
        "| Model | seeds | acc_amb | acc_dis | FAR |",
        "|---|---|---:|---:|---:|",
    ]
    for row in cross_rows:
        lines.append(
            f"| {row.get('model')} | {row.get('seeds')} | "
            f"{fmt(row.get('accuracy_amb_mean'))}+/-{fmt(row.get('accuracy_amb_std'))} | "
            f"{fmt(row.get('accuracy_dis_mean'))}+/-{fmt(row.get('accuracy_dis_std'))} | "
            f"{fmt(row.get('false_abstention_rate_mean'))}+/-{fmt(row.get('false_abstention_rate_std'))} |"
        )

    lines += [
        "",
        "## Generated Tables",
        "",
        "- `main_and_baseline_metrics.csv`",
        "- `loco_generalization_summary.csv`",
        "- `openbbq_transfer_summary.csv`",
        "- `cross_llm_summary.csv`",
        "- `residual_bias_counts.csv`",
        "- `paired_tests_summary.csv`",
        "- `threshold_audit_summary.csv`",
        "- `signal_ablation_summary.csv`",
        "- `reproducibility.md`",
        "- `claim_language.md`",
    ]
    (out_dir / "acceptance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = Path(args.clean_dir)
    baseline_rows = baseline_coverage(clean_dir, out_dir)
    residual_bias_counts(clean_dir, Path(args.sampled_dir), out_dir)
    paired_tests_summary(clean_dir, out_dir)
    threshold_audit_summary(clean_dir, out_dir)
    signal_ablation_summary(clean_dir, out_dir)
    loco_rows = loco_summary(Path(args.loco_dir), out_dir)
    openbbq_rows = openbbq_summary(Path(args.openbbq_dir), out_dir)
    cross_rows = cross_llm_summary(Path(args.qwen_dir), Path(args.mistral_dir), out_dir)
    write_reproducibility(out_dir)
    write_claim_language(out_dir)
    write_markdown_report(out_dir, baseline_rows, loco_rows, openbbq_rows, cross_rows)
    print(f"Saved acceptance report: {out_dir / 'acceptance_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
