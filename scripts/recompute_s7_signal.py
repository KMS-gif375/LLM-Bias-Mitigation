#!/usr/bin/env python3
"""
Recompute only s7_sae_feature in existing signal files.

This is useful when SAE bias feature indices change but s1-s6 should remain
identical. The script reads an existing full signal file (or a full .bak file if
the current file is partial), recomputes s7 with the configured SAE, and atomically
replaces the current signal file after the category is complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_pipeline import (
    _load_bias_sae_features,
    _load_items,
    _maybe_load_sae,
    _stage_output_dir,
    load_config,
    select_model_block,
    setup_logging,
)
from src.signals.prompts import PROMPT_BUILDERS
from src.signals.sae_feature import compute_sae_signal
from src.utils.data_loader import DEFAULT_CATEGORIES_V2
from src.utils.llm_utils import LLMWrapper

logger = logging.getLogger("recompute_s7")


def _apply_version(config: dict, version: str, model: str) -> None:
    if version == "v1":
        return
    if version == "v2":
        config["data"]["sampled_dir"] = "data/sampled_v2"
        config["data"]["samples_per_category"] = 1000
        config["output"]["results_dir"] = "results/v2"
    elif version == "smoke":
        config["data"]["sampled_dir"] = "data/sampled_smoke"
        config["data"]["samples_per_category"] = 5
        config["output"]["results_dir"] = "results/smoke_e2e"
    elif version == "mini":
        config["data"]["sampled_dir"] = "data/sampled_mini"
        config["data"]["samples_per_category"] = 100
        config["output"]["results_dir"] = "results/v2_mini"
    else:
        raise ValueError(f"unknown version: {version}")

    config["data"]["categories"] = list(DEFAULT_CATEGORIES_V2)
    if model != "main":
        base = config["output"]["results_dir"]
        config["output"]["results_dir"] = f"{base}/cross_llm/{model}"


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _select_source_signal(path: Path, expected_rows: int) -> Path:
    current_count = _line_count(path)
    if current_count == expected_rows:
        return path

    backups = sorted(
        path.parent.glob(f"{path.name}.bak.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for backup in backups:
        if _line_count(backup) == expected_rows:
            logger.warning(
                "  current signal is partial (%s rows); using backup source: %s",
                current_count,
                backup,
            )
            return backup

    raise FileNotFoundError(
        f"no full signal source for {path} (current rows={current_count}, "
        f"expected={expected_rows})"
    )


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def recompute_category(
    *,
    config: dict,
    model: str,
    category: str,
    llm: LLMWrapper,
    sae,
    bias_features: list[int],
    strict: bool,
    backup: bool,
) -> dict:
    n_per_cat = int(config["data"].get("samples_per_category", 300))
    signals_dir = _stage_output_dir(config, model, "signals")
    out_path = signals_dir / f"{category}_signals.jsonl"

    items = _load_items(config, category, n_per_cat)
    for item in items:
        item.setdefault("category", category)
    item_by_id = {str(item["example_id"]): item for item in items}

    source_path = _select_source_signal(out_path, expected_rows=len(items))
    records = _load_jsonl(source_path)
    if len(records) != len(items):
        raise ValueError(
            f"{category}: source rows={len(records)} != items={len(items)}"
        )

    logger.info("[%s] source=%s rows=%d", category, source_path, len(records))
    new_records: list[dict] = []
    non_null = 0
    errors = 0

    for rec in tqdm(records, desc=f"s7 {category}"):
        item = item_by_id.get(str(rec.get("example_id")))
        if item is None:
            msg = f"{category}: item not found for example_id={rec.get('example_id')}"
            if strict:
                raise KeyError(msg)
            logger.warning(msg)
            errors += 1
            new_records.append(rec)
            continue

        try:
            s7 = compute_sae_signal(
                item=item,
                llm=llm,
                sae=sae,
                prompt_builder=PROMPT_BUILDERS["vanilla"],
                bias_feature_indices=bias_features,
            )
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            logger.warning("s7 failed for %s/%s: %s", category, rec.get("example_id"), exc)
            s7 = None
            errors += 1

        rec = dict(rec)
        signals = dict(rec.get("signals", {}))
        signals["s7_sae_feature"] = s7
        rec["signals"] = signals
        if s7 is not None:
            non_null += 1
        new_records.append(rec)

    tmp_path = out_path.with_suffix(f"{out_path.suffix}.s7tmp.{int(time.time())}")
    _save_jsonl(new_records, tmp_path)

    if backup and out_path.exists():
        backup_path = out_path.with_suffix(f"{out_path.suffix}.s7bak.{int(time.time())}")
        shutil.copy2(out_path, backup_path)
        logger.info("  backup current: %s", backup_path)

    os.replace(tmp_path, out_path)
    logger.info(
        "[%s] wrote %s rows=%d s7_non_null=%d errors=%d",
        category,
        out_path,
        len(new_records),
        non_null,
        errors,
    )
    return {
        "category": category,
        "rows": len(new_records),
        "s7_non_null": non_null,
        "errors": errors,
        "source": str(source_path),
        "out": str(out_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute only s7_sae_feature")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--version", default="v2", choices=("v1", "v2", "smoke", "mini"))
    parser.add_argument("--model", default="main", choices=("main", "gemma", "qwen", "mistral"))
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--sae-bias-features", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.log_dir)
    logger.info("args: %s", vars(args))

    config = load_config(args.config)
    _apply_version(config, args.version, args.model)

    if args.model == "qwen":
        raise ValueError("qwen has no SAE; s7 should remain None")

    model_cfg = select_model_block(config, args.model)
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )
    sae = _maybe_load_sae(config, args.model, llm)
    if sae is None:
        raise RuntimeError("SAE failed to load; cannot recompute s7")
    bias_features = _load_bias_sae_features(
        config,
        args.model,
        explicit_path=args.sae_bias_features,
    )
    if not bias_features:
        raise RuntimeError("bias SAE feature list is empty")

    categories = args.categories or config["data"]["categories"]
    summary = []
    for category in categories:
        summary.append(
            recompute_category(
                config=config,
                model=args.model,
                category=category,
                llm=llm,
                sae=sae,
                bias_features=bias_features,
                strict=args.strict,
                backup=not args.no_backup,
            )
        )

    summary_path = (
        _stage_output_dir(config, args.model, "signals")
        / f"s7_recompute_summary_{int(time.time())}.json"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
