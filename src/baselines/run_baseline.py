"""
Generic baseline runner — DeCAP, FairSteer, Composite Prompting.

`src/evaluation/baselines.py`의 BASELINE_REGISTRY를 그대로 재사용하여
1-pass baselines를 동일한 평가 형식으로 실행합니다.

사용:
    python -m src.baselines.run_baseline --method decap --eval
    python -m src.baselines.run_baseline --method composite_prompting --eval
    python -m src.baselines.run_baseline --method fairsteer --eval
        # FairSteer는 사전 학습 steering vector 부재 시 vanilla로 fallback

    # smoke test
    python -m src.baselines.run_baseline --method decap --max-samples 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger("baseline_runner")


def _load_all_items(
    config: dict,
    categories: list[str],
    max_samples: Optional[int] = None,
) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import _load_items  # type: ignore

    n_per_cat = max_samples or config["data"].get("samples_per_category", 300)
    items: list[dict] = []
    for cat in categories:
        for it in _load_items(config, cat, n_per_cat=n_per_cat):
            it.setdefault("category", cat)
            items.append(it)
    return items


METHOD_REFERENCE = {
    "self_debiasing_reprompting": "Gallegos et al., NAACL 2025",
    "decap": "Bae et al., 2025",
    "fairsteer": "Li et al., 2025",
    "composite_prompting": "공정성 + CoT + 역할 통합 (in-house)",
}


def run(
    method: str,
    config_path: str = "configs/default.yaml",
    categories: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
    out_dir: Optional[str] = None,
    skip_existing: bool = True,
    fairsteer_vector_path: Optional[str] = None,
) -> dict:
    """
    Baseline을 실행하고 metrics를 저장합니다.

    Args:
        method: BASELINE_REGISTRY 키 ("decap", "composite_prompting", "fairsteer", ...).
        fairsteer_vector_path: FairSteer용 steering vector(.pt). None이면 vanilla fallback.

    Returns:
        평가 metrics dict.
    """
    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cats = categories or config["data"]["categories"]
    out_dir = out_dir or f"results/baselines/{method}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    final_json = out_path / "final.json"
    if skip_existing and final_json.exists():
        logger.info(f"  [skip] {final_json} 이미 존재")
        return json.loads(final_json.read_text(encoding="utf-8"))

    items = _load_all_items(config, cats, max_samples=max_samples)
    logger.info(f"  Loaded {len(items)} instances from {len(cats)} categories")
    if not items:
        raise RuntimeError("BBQ items 없음 — `python -m src.utils.data_loader --all` 먼저")

    from src.utils.llm_utils import LLMWrapper
    from src.evaluation.baselines import BASELINE_REGISTRY

    if method not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. 가능: {list(BASELINE_REGISTRY)}")
    runner = BASELINE_REGISTRY[method]

    model_cfg = config["models"]["main"]
    logger.info(f"  Loading LLM: {model_cfg['name']}")
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )

    # 메서드별 추가 인자
    extra_kwargs: dict = {}
    if method == "fairsteer":
        if fairsteer_vector_path and Path(fairsteer_vector_path).exists():
            import torch
            sv = torch.load(fairsteer_vector_path, map_location="cpu", weights_only=True)
            extra_kwargs["steering_vector"] = sv
            extra_kwargs["layer_idx"] = config.get("fairsteer", {}).get("layer", 16)
            extra_kwargs["alpha"] = config.get("fairsteer", {}).get("alpha", 1.0)
        else:
            logger.warning(
                f"  FairSteer steering vector 없음 (path={fairsteer_vector_path}) "
                "— vanilla로 fallback"
            )

    t0 = time.time()
    raw_predictions = runner(items, llm, **extra_kwargs)
    elapsed = time.time() - t0
    logger.info(f"  {method} inference 완료: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # raw 저장
    preds_path = out_path / "predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for item, pred in zip(items, raw_predictions):
            f.write(json.dumps({
                "example_id": item["example_id"],
                "category": item.get("category"),
                "context_condition": item.get("context_condition"),
                "label": item.get("label"),
                "prediction_text": pred,
            }, ensure_ascii=False) + "\n")
    logger.info(f"  [저장] raw predictions → {preds_path}")

    from src.evaluation.bbq_evaluator import evaluate_bbq

    metrics = evaluate_bbq(raw_predictions, items)
    by_cat: dict[str, dict] = {}
    for cat in cats:
        cat_items = [it for it in items if it.get("category") == cat]
        cat_preds = [p for it, p in zip(items, raw_predictions) if it.get("category") == cat]
        if cat_items:
            by_cat[cat] = evaluate_bbq(cat_preds, cat_items)

    payload = {
        "method": method,
        "reference": METHOD_REFERENCE.get(method, "unknown"),
        "model": model_cfg["name"],
        "n_instances": len(items),
        "elapsed_seconds": elapsed,
        "fairsteer_used_vector": method == "fairsteer" and "steering_vector" in extra_kwargs,
        "overall": metrics,
        "per_category": by_cat,
    }
    final_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    logger.info(f"  [저장] metrics → {final_json}")
    logger.info(
        f"  Overall: acc_amb={metrics.get('accuracy_amb'):.4f} "
        f"acc_dis={metrics.get('accuracy_dis'):.4f} "
        f"bias_amb={metrics.get('bias_score_amb')}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline runner (DeCAP / FairSteer / Composite)")
    parser.add_argument("--method", type=str, required=True,
                        choices=list(METHOD_REFERENCE.keys()))
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--eval", action="store_true",
                        help="전체 평가 (full data). 미지정 시 --max-samples 필요.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="기존 final.json 무시하고 재실행")
    parser.add_argument("--fairsteer-vector", type=str, default=None,
                        help="FairSteer steering vector (.pt) 경로")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.eval and args.max_samples is None:
        logger.error("--eval 또는 --max-samples N 중 하나를 지정하세요.")
        return 2

    run(
        method=args.method,
        config_path=args.config,
        categories=args.categories,
        max_samples=args.max_samples,
        out_dir=args.out_dir,
        skip_existing=not args.force,
        fairsteer_vector_path=args.fairsteer_vector,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
