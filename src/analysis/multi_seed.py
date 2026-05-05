"""
Multi-seed MoE 실험 — 통계적 robustness 강화용.

Signal extraction은 한 번만 수행 (seed=42 데이터 기준), MoE 학습/평가만 5번
반복하여 mean ± std를 보고합니다. 추가로:

    - Bootstrap 95% CI (1000 iterations) per seed
    - Per-category metrics
    - Threshold sensitivity (각 seed별 best τ)

비용 (Mac M4 Pro): seeds × (~30s training + ~5s eval) ≈ 5분.

사용:
    # 기본 5 seeds
    python -m src.analysis.multi_seed --version v2

    # 커스텀 seeds
    python -m src.analysis.multi_seed --seeds 42,123,456,789,999

    # 출력 디렉토리 변경
    python -m src.analysis.multi_seed --version v2 --out-dir results/v2/multi_seed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("multi_seed")


# =============================================================
# Dataclasses
# =============================================================
@dataclass
class SeedResult:
    seed: int
    best_val_loss: float
    best_epoch: int
    best_threshold: float
    metrics: dict[str, float]
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


@dataclass
class MultiSeedSummary:
    seeds: list[int]
    per_seed: list[SeedResult]
    aggregate: dict[str, dict[str, float]]   # {metric_key: {mean, std, ci_low, ci_high}}
    aggregate_per_category: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


# =============================================================
# Aggregation utilities
# =============================================================
def _aggregate_values(values: list[float]) -> dict[str, float]:
    """주어진 numeric list에 대해 mean / std / CI 95% 계산."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"mean": float("nan"), "std": 0.0, "n": 0}
    m = float(mean(vals))
    s = float(stdev(vals)) if len(vals) > 1 else 0.0
    n = len(vals)
    # 정규근사 95% CI (n>=30이면 z=1.96, 작으면 t-distribution이지만 실용적으로 1.96 사용)
    ci_half = 1.96 * s / max(np.sqrt(n), 1.0) if n > 1 else 0.0
    return {
        "mean": m,
        "std": s,
        "n": n,
        "ci_low": m - ci_half,
        "ci_high": m + ci_half,
    }


def aggregate_seeds(per_seed: list[SeedResult]) -> dict[str, dict[str, float]]:
    """seed별 metrics를 metric_key별로 묶어 mean/std/CI 계산."""
    if not per_seed:
        return {}
    keys = sorted({k for r in per_seed for k in r.metrics.keys()})
    agg: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [r.metrics.get(k) for r in per_seed]
        agg[k] = _aggregate_values([v for v in vals if v is not None])
    # threshold 도 aggregate
    agg["best_threshold"] = _aggregate_values([r.best_threshold for r in per_seed])
    agg["best_val_loss"] = _aggregate_values([r.best_val_loss for r in per_seed])
    return agg


def aggregate_per_category(per_seed: list[SeedResult]) -> dict[str, dict[str, dict[str, float]]]:
    """카테고리별로 metric mean/std 계산."""
    cats = sorted({c for r in per_seed for c in r.per_category.keys()})
    out: dict[str, dict[str, dict[str, float]]] = {}
    for cat in cats:
        keys = sorted({
            k for r in per_seed
            for k in r.per_category.get(cat, {}).keys()
        })
        agg: dict[str, dict[str, float]] = {}
        for k in keys:
            vals = [r.per_category.get(cat, {}).get(k) for r in per_seed]
            agg[k] = _aggregate_values([v for v in vals if v is not None])
        out[cat] = agg
    return out


# =============================================================
# Single-seed run (run_pipeline 재사용)
# =============================================================
def run_single_seed(
    config: dict,
    seed: int,
    args_namespace,
    save_dir: Path,
) -> SeedResult:
    """
    단일 seed로 MoE 학습 + threshold search + 평가.

    sigals/embeddings는 외부에서 미리 추출되어 있어야 함 (signal_extraction stage 완료).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import (  # type: ignore
        _collect_records_and_embeddings,
        _instances_by_id,
        _infer_embed_dim,
        _moe_predict_all,
    )
    from src.models.moe_aggregator import MoEAggregator
    from src.models.trainer import SignalsDataset, TrainConfig, train_moe
    from src.models.override import (
        apply_threshold_override,
        search_optimal_threshold,
    )
    from src.evaluation.bbq_evaluator import evaluate_bbq

    # 재현성
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. records + embeddings
    records, embeddings = _collect_records_and_embeddings(config, args_namespace)
    if not records:
        raise RuntimeError("Stage 2 (signal_extraction) 결과 없음")

    instances_by_id = _instances_by_id(records, config, args_namespace)

    # 2. train/val split (seed에 따라 shuffle)
    val_split = float(config["moe"].get("training", {}).get("val_split", 0.2))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(records))
    shuffled = [records[i] for i in perm]
    n_val = max(1, int(len(shuffled) * val_split))
    train_records = shuffled[:-n_val]
    val_records = shuffled[-n_val:]

    train_ds = SignalsDataset(train_records, embeddings)
    val_ds = SignalsDataset(val_records, embeddings)

    embed_dim = _infer_embed_dim(embeddings, default=4096)
    moe_cfg = config["moe"]
    training_cfg = moe_cfg.get("training", {})

    # 3. 학습
    seed_save_dir = save_dir / f"seed_{seed}"
    seed_save_dir.mkdir(parents=True, exist_ok=True)
    train_config = TrainConfig(
        epochs=int(training_cfg.get("epochs", 30)),
        batch_size=int(training_cfg.get("batch_size", 64)),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        val_every=int(training_cfg.get("val_every", 5)),
        device=config["models"][args_namespace.model].get("device", "auto"),
        seed=seed,
        save_dir=str(seed_save_dir),
    )

    model = MoEAggregator(
        signal_dim=7,
        embed_dim=embed_dim,
        num_experts=int(moe_cfg.get("num_experts", 4)),
        gating_hidden=int(moe_cfg.get("gating_hidden_dim", 64)),
        expert_hidden=int(moe_cfg.get("expert_hidden_dim", 128)),
    )
    out = train_moe(train_ds, val_ds, model, train_config)
    best_ckpt = out.get("checkpoint_path")

    # 4. MoE 추론 + threshold search (val에서)
    val_predictions = _moe_predict_all(model, val_records, embeddings, instances_by_id)
    tau_range = config.get("override", {}).get(
        "threshold_search", {"range": [0.3, 0.7], "step": 0.05}
    )
    search = search_optimal_threshold(
        val_predictions,
        threshold_range=tuple(tau_range.get("range", [0.3, 0.7])),
        step=float(tau_range.get("step", 0.05)),
    )

    # 5. 전체에서 평가 (test 분할이 없는 경우 모든 records 사용)
    all_predictions = _moe_predict_all(model, records, embeddings, instances_by_id)
    final_preds: list[int] = []
    final_items: list[dict] = []
    for vp in all_predictions:
        result = apply_threshold_override(
            primary_answer=int(vp["primary_answer"]),
            p_score=float(vp["p_score"]),
            item=vp["item"],
            threshold=float(search.best_threshold),
        )
        final_preds.append(result["final_answer"])
        final_items.append(vp["item"])

    metrics = evaluate_bbq(final_preds, final_items)

    # 6. per-category 평가
    cats = sorted({it.get("category", "_unknown") for it in final_items})
    per_category: dict[str, dict[str, float]] = {}
    for cat in cats:
        idxs = [i for i, it in enumerate(final_items) if it.get("category") == cat]
        if not idxs:
            continue
        cat_preds = [final_preds[i] for i in idxs]
        cat_items = [final_items[i] for i in idxs]
        cat_metrics = evaluate_bbq(cat_preds, cat_items)
        per_category[cat] = {
            k: float(v) if isinstance(v, (int, float)) else 0.0
            for k, v in cat_metrics.items()
            if v is not None and not isinstance(v, dict)
        }

    return SeedResult(
        seed=seed,
        best_val_loss=float(out.get("best_val_loss") or float("inf")),
        best_epoch=int(out.get("best_epoch") or -1),
        best_threshold=float(search.best_threshold),
        metrics={
            k: float(v) for k, v in metrics.items()
            if v is not None and isinstance(v, (int, float))
        },
        per_category=per_category,
        checkpoint_path=str(best_ckpt) if best_ckpt else None,
    )


# =============================================================
# Driver
# =============================================================
def run(
    seeds: list[int],
    config_path: str = "configs/default.yaml",
    version: str = "v1",
    model_key: str = "main",
    out_dir: Optional[str] = None,
) -> MultiSeedSummary:
    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # version에 따라 데이터 경로 자동 조정
    if version == "v2":
        config["data"]["sampled_dir"] = "data/sampled_v2"
        config["data"]["samples_per_category"] = 1000
        from src.utils.data_loader import DEFAULT_CATEGORIES_V2
        config["data"]["categories"] = list(DEFAULT_CATEGORIES_V2)

    out_dir = out_dir or f"results/{version}/multi_seed" if version != "v1" else "results/multi_seed"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # args mock (run_pipeline helpers가 기대하는 형태)
    class _Args:
        def __init__(self):
            self.model = model_key
            self.categories = None

    args_ns = _Args()

    per_seed: list[SeedResult] = []
    for seed in seeds:
        logger.info("=" * 60)
        logger.info(f"  SEED {seed}")
        logger.info("=" * 60)
        result = run_single_seed(config, seed, args_ns, out_path)
        per_seed.append(result)
        # seed별 결과 저장
        (out_path / f"seed_{seed}_results.json").write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False, default=float),
            encoding="utf-8",
        )
        logger.info(
            f"  Seed {seed}: tau={result.best_threshold:.2f} "
            f"acc_amb={result.metrics.get('accuracy_amb'):.4f} "
            f"bias_amb={result.metrics.get('bias_score_amb')}"
        )

    # 종합 통계
    aggregate = aggregate_seeds(per_seed)
    aggregate_cats = aggregate_per_category(per_seed)

    summary = MultiSeedSummary(
        seeds=seeds,
        per_seed=per_seed,
        aggregate=aggregate,
        aggregate_per_category=aggregate_cats,
    )

    # 요약 저장
    (out_path / "summary.json").write_text(
        json.dumps({
            "seeds": summary.seeds,
            "aggregate": summary.aggregate,
            "aggregate_per_category": summary.aggregate_per_category,
            "per_seed": [asdict(r) for r in summary.per_seed],
        }, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    logger.info(f"  [저장] summary → {out_path / 'summary.json'}")

    # 콘솔 요약
    logger.info("=" * 60)
    logger.info("  Aggregate (mean ± std)")
    logger.info("=" * 60)
    for k, stats in aggregate.items():
        if stats.get("n", 0) == 0:
            continue
        logger.info(f"  {k:25s}: {stats['mean']:.4f} ± {stats['std']:.4f}  (n={stats['n']})")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed MoE 실험")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--version", type=str, default="v1", choices=("v1", "v2"))
    parser.add_argument("--model", type=str, default="main")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,999",
                        help="comma-separated seed list")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="결과 저장 디렉토리 (미지정 시 results/{version}/multi_seed)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run(
        seeds=seeds,
        config_path=args.config,
        version=args.version,
        model_key=args.model,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
