"""
Cross-LLM CLI runner — Gemma-2-9B / Qwen-2.5-7B에서 primary answer 추출.

본 모듈은 cross-LLM transfer를 위한 최소 CLI:
    1. 지정 모델 (gemma / qwen) 로드 (HF 다운로드 약 15GB / 15GB)
    2. BBQ instances에 vanilla prompt로 primary answer 추출
    3. (옵션) 우리 학습된 MoE checkpoint로 transfer 평가

본격적인 Cross-LLM signal extraction (s5 bias-head 재식별, s7 SAE 재식별 등)은
src/cross_llm/{gemma,qwen}_pipeline.py 함수를 직접 호출. 본 wrapper는 vanilla
inference + 평가 정도만 제공.

CLI:
    # Gemma-2-9B-It에서 BBQ vanilla 답변
    python -m src.cross_llm.run_cross_llm --model gemma --max-samples 10

    # Qwen-2.5-7B에서 BBQ vanilla 답변
    python -m src.cross_llm.run_cross_llm --model qwen --max-samples 10

    # 특정 카테고리만
    python -m src.cross_llm.run_cross_llm --model gemma --max-samples 10 \\
        --categories Age Gender_identity

NOTE: smoke test에서는 모델 다운로드 시간이 길어 권장하지 않음.
Stage 5 평가용으로 별도 실행 권장.
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
from tqdm import tqdm

logger = logging.getLogger("cross_llm")


def run(
    model_name: str,
    config_path: str = "configs/default.yaml",
    categories: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
    out_dir: Optional[str] = None,
    version: str = "v1",
) -> dict:
    """
    Cross-LLM (gemma 또는 qwen)에서 BBQ vanilla primary answer 추출.

    Args:
        model_name: "gemma" 또는 "qwen".
        version: "v1"/"v2"/"smoke" — 데이터 로드 경로 결정.
    """
    if model_name not in ("gemma", "qwen"):
        raise ValueError(f"Unknown model: {model_name}. 가능: 'gemma', 'qwen'")

    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # version에 따라 데이터 경로
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

    out_dir = out_dir or f"results/cross_llm/{model_name}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import _load_items  # type: ignore

    cats = categories or config["data"]["categories"]
    n_per_cat = max_samples or config["data"].get("samples_per_category", 300)
    items: list[dict] = []
    for cat in cats:
        for it in _load_items(config, cat, n_per_cat=n_per_cat):
            it.setdefault("category", cat)
            items.append(it)
    logger.info(f"  Loaded {len(items)} instances from {len(cats)} categories")

    if not items:
        logger.error("BBQ items 없음")
        return {"error": "no_items"}

    # 모델 로드
    from src.utils.llm_utils import LLMWrapper

    model_cfg = config["models"][model_name]
    logger.info(f"  Loading {model_name}: {model_cfg['name']}")
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )

    # vanilla inference
    from src.signals.prompts import PROMPT_BUILDERS
    from src.evaluation.bbq_evaluator import evaluate_bbq, parse_prediction

    builder = PROMPT_BUILDERS["vanilla"]
    t0 = time.time()
    predictions: list[str] = []
    for item in tqdm(items, desc=f"{model_name} vanilla"):
        try:
            sys_msg, usr_msg = builder(item)
            out = llm.generate(
                user_message=usr_msg, system_message=sys_msg,
                max_new_tokens=model_cfg.get("max_new_tokens", 64),
                temperature=model_cfg.get("temperature", 0.0),
            )
            predictions.append(out.text)
        except Exception as e:
            logger.warning(f"  실패 (id={item.get('example_id')}): {e}")
            predictions.append("")
    elapsed = time.time() - t0
    logger.info(f"  Inference 완료: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 평가
    metrics = evaluate_bbq(predictions, items)
    per_cat: dict[str, dict] = {}
    for cat in cats:
        idxs = [i for i, it in enumerate(items) if it.get("category") == cat]
        if not idxs:
            continue
        cat_preds = [predictions[i] for i in idxs]
        cat_items = [items[i] for i in idxs]
        per_cat[cat] = evaluate_bbq(cat_preds, cat_items)

    # 저장 — predictions.csv + final.json
    preds_csv = out_path / "predictions.csv"
    with open(preds_csv, "w", encoding="utf-8") as f:
        f.write("example_id,category,context_condition,label,parsed_pred,raw_text\n")
        for it, p in zip(items, predictions):
            parsed = parse_prediction(p)
            raw = (p or "").replace("\n", " ").replace(",", ";")[:200]
            f.write(
                f"{it.get('example_id','')},{it.get('category','')},"
                f"{it.get('context_condition','')},{it.get('label',-1)},"
                f"{parsed},\"{raw}\"\n"
            )

    payload = {
        "method": f"cross_llm_{model_name}_vanilla",
        "model": model_cfg["name"],
        "n_total": len(items),
        "elapsed_seconds": elapsed,
        "overall": metrics,
        "per_category": per_cat,
    }
    final_json = out_path / "final.json"
    final_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    logger.info(f"  [저장] {final_json}")
    logger.info(
        f"  Overall: acc_amb={metrics.get('accuracy_amb'):.4f} "
        f"bias_amb={metrics.get('bias_score_amb')}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-LLM vanilla evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=("gemma", "qwen"))
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--version", type=str, default="v1",
                        choices=("v1", "v2", "smoke", "mini"))
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run(
        model_name=args.model,
        config_path=args.config,
        categories=args.categories,
        max_samples=args.max_samples,
        out_dir=args.out_dir,
        version=args.version,
    )
    return 2 if isinstance(result, dict) and "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
