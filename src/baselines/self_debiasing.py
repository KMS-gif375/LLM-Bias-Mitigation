"""
Baseline: Self-Debiasing-Reprompting (Gallegos et al., NAACL 2025).

알고리즘:
    1. 1차 답변: vanilla prompt (system: helpful assistant + 객관식)
    2. 2차 prompt: 1차 답을 보여주고 "stereotypes에 의존하지 않았는지 검토,
       모호한 context면 unknown 선택"하라고 재프롬프팅
    3. 2차 답변을 최종 답으로 사용

본 파일은 src/evaluation/baselines.py의 `run_self_debiasing_reprompting()`을
재사용하면서, full BBQ 데이터에서 평가하고 결과를 우리 method와 동일한 형식
(`results/baselines/self_debiasing/final.json`)으로 저장합니다.

사용:
    # 빠른 테스트 (카테고리당 5개)
    python -m src.baselines.self_debiasing --max-samples 5

    # 전체 평가 (Stage 1 + 평가, ~2-3시간)
    python -m src.baselines.self_debiasing --eval

    # 특정 카테고리만
    python -m src.baselines.self_debiasing --eval --categories Age Gender_identity
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

logger = logging.getLogger("self_debiasing")


# =============================================================
# Helper: parquet → category items (run_pipeline의 _load_items 재사용)
# =============================================================
def _load_all_items(
    config: dict,
    categories: list[str],
    max_samples: Optional[int] = None,
) -> list[dict]:
    """카테고리별 BBQ instance를 모두 모아 단일 리스트로 반환."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import _load_items  # type: ignore

    n_per_cat = max_samples or config["data"].get("samples_per_category", 300)
    items: list[dict] = []
    for cat in categories:
        for it in _load_items(config, cat, n_per_cat=n_per_cat):
            it.setdefault("category", cat)
            items.append(it)
    return items


# =============================================================
# Driver
# =============================================================
def run(
    config_path: str = "configs/default.yaml",
    categories: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
    out_dir: str = "results/baselines/self_debiasing",
    skip_existing: bool = True,
    version: str = "v1",
) -> dict:
    """
    Self-Debiasing-Reprompting을 BBQ에서 실행하고 metrics 저장.

    Args:
        config_path: YAML config 경로.
        categories: 평가할 카테고리 리스트. None이면 config 전체.
        max_samples: 카테고리당 최대 샘플 수 (None = 전체).
        out_dir: 결과 저장 디렉토리.
        skip_existing: 이미 final.json이 있으면 건너뛰기.

    Returns:
        평가 metrics dict (BBQ 표준).
    """
    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if version in ("v2", "smoke", "mini"):
        from src.utils.data_loader import DEFAULT_CATEGORIES_V2
        config["data"]["sampled_dir"] = {
            "v2": "data/sampled_v2",
            "smoke": "data/sampled_smoke",
            "mini": "data/sampled_mini",
        }[version]
        config["data"]["samples_per_category"] = {
            "v2": 1000, "smoke": 5, "mini": 100,
        }[version]
        config["data"]["categories"] = list(DEFAULT_CATEGORIES_V2)

    cats = categories or config["data"]["categories"]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    final_json = out_path / "final.json"
    if skip_existing and final_json.exists():
        logger.info(f"  [skip] {final_json} 이미 존재")
        return json.loads(final_json.read_text(encoding="utf-8"))

    # 1. 데이터 로드
    items = _load_all_items(config, cats, max_samples=max_samples)
    logger.info(f"  Loaded {len(items)} instances from {len(cats)} categories")
    if not items:
        raise RuntimeError("BBQ items 없음 — `python -m src.utils.data_loader --all` 먼저")

    # 2. LLM 로드 (Llama-3.1-8B)
    from src.utils.llm_utils import LLMWrapper

    model_cfg = config["models"]["main"]
    logger.info(f"  Loading LLM: {model_cfg['name']}")
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )

    # 3. Self-Debiasing-Reprompting 실행
    from src.evaluation.baselines import run_self_debiasing_reprompting

    t0 = time.time()
    raw_predictions = run_self_debiasing_reprompting(items, llm)
    elapsed = time.time() - t0
    logger.info(f"  Self-Debiasing inference 완료: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 4. raw 예측 저장
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

    # 5. BBQ metrics 계산
    from src.evaluation.bbq_evaluator import evaluate_bbq

    metrics = evaluate_bbq(raw_predictions, items)

    # 카테고리별 metrics도 추가
    by_cat: dict[str, dict] = {}
    for cat in cats:
        cat_items = [it for it in items if it.get("category") == cat]
        cat_preds = [p for it, p in zip(items, raw_predictions) if it.get("category") == cat]
        if cat_items:
            by_cat[cat] = evaluate_bbq(cat_preds, cat_items)

    payload = {
        "method": "self_debiasing_reprompting",
        "reference": "Gallegos et al., NAACL 2025",
        "model": model_cfg["name"],
        "n_instances": len(items),
        "elapsed_seconds": elapsed,
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
    parser = argparse.ArgumentParser(
        description="Self-Debiasing-Reprompting baseline (Gallegos et al., 2025)"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--version", type=str, default="v1",
        choices=("v1", "v2", "smoke", "mini"),
        help="data version (v1=7×300, v2=9×1000, smoke=9×5, mini=9×100)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="전체 평가 (full data, ~2-3시간). 미지정 시 max-samples 필요.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="카테고리당 최대 샘플 수 (smoke test용)",
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", default=None,
        help="평가할 BBQ 카테고리 (기본: config 전체)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/baselines/self_debiasing",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="기존 final.json 무시하고 재실행",
    )
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
        config_path=args.config,
        categories=args.categories,
        max_samples=args.max_samples,
        out_dir=args.out_dir,
        skip_existing=not args.force,
        version=args.version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
