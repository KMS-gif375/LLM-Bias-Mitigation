"""
End-to-End 파이프라인 실행 스크립트.

전체 흐름:
    Stage 0: 데이터 샘플링 (별도: python -m src.utils.sampling)
    Stage 1: 4-Prompt Inference -> results/signals/{model}/{category}_stage1.jsonl
    Stage 2: 7-Signal Extraction -> results/signals/{model}/{category}_signals.jsonl
    Stage 3: MoE 학습 -> results/moe/{model}/best.pt
    Stage 4: Threshold Override + 평가 -> results/evaluation/{model}/final.json

사용법:
    # 전체 파이프라인 (메인 모델만)
    python run_pipeline.py --model main

    # 특정 stage만
    python run_pipeline.py --model main --stages 1,2

    # 특정 카테고리만
    python run_pipeline.py --model main --categories Age Gender_identity
"""

import argparse
import json
from pathlib import Path

import yaml

from src.signals.extract_all import extract_signals_batch
from src.signals.inference import run_4prompt_inference
from src.signals.sae_feature import SAEWrapper
from src.utils.data_loader import load_sampled
from src.utils.llm_utils import LLMWrapper


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage1_inference(config: dict, args: argparse.Namespace) -> None:
    """Stage 1: 4-Prompt Inference 실행."""
    print("\n" + "=" * 60)
    print("[STAGE 1] 4-Prompt Inference")
    print("=" * 60)

    model_cfg = config["models"][args.model]
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg["dtype"],
        device=model_cfg["device"],
    )

    output_dir = Path(config["output"]["signals_dir"]) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = args.categories or config["data"]["categories"]
    for category in categories:
        print(f"\n[카테고리] {category}")
        items = load_sampled(config["data"]["sampled_dir"], category)
        for item in items:
            item.setdefault("category", category)

        out_path = output_dir / f"{category}_stage1.jsonl"
        run_4prompt_inference(
            items=items,
            llm=llm,
            output_path=out_path,
            max_new_tokens=model_cfg.get("max_new_tokens", 64),
            temperature=model_cfg.get("temperature", 0.0),
        )


def stage2_signals(config: dict, args: argparse.Namespace) -> None:
    """Stage 2: 7-Signal Extraction 실행."""
    print("\n" + "=" * 60)
    print("[STAGE 2] 7-Signal Extraction")
    print("=" * 60)

    model_cfg = config["models"][args.model]
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg["dtype"],
        device=model_cfg["device"],
    )

    # SAE 로드 (옵션)
    sae = None
    if args.model in ("main",) and config.get("sae", {}).get("llama"):
        sae_cfg = config["sae"]["llama"]
        try:
            sae = SAEWrapper(
                sae_repo=sae_cfg["repo"],
                layer=sae_cfg["layer"],
                device=str(llm.device),
            )
        except Exception as e:
            print(f"  [WARN] SAE 로드 실패, s7 신호 비활성화: {e}")

    signals_dir = Path(config["output"]["signals_dir"]) / args.model
    categories = args.categories or config["data"]["categories"]

    for category in categories:
        print(f"\n[카테고리] {category}")
        items = load_sampled(config["data"]["sampled_dir"], category)
        for item in items:
            item.setdefault("category", category)

        stage1_path = signals_dir / f"{category}_stage1.jsonl"
        if not stage1_path.exists():
            print(f"  [skip] Stage 1 결과 없음: {stage1_path}")
            continue

        with open(stage1_path, "r", encoding="utf-8") as f:
            stage1_results = [json.loads(line) for line in f if line.strip()]

        out_path = signals_dir / f"{category}_signals.jsonl"
        extract_signals_batch(
            items=items,
            stage1_results=stage1_results,
            llm=llm,
            sae=sae,
            output_path=out_path,
            n_consistency_samples=config["signals"]["s4_consistency"]["n_samples"],
        )


def stage3_train_moe(config: dict, args: argparse.Namespace) -> None:
    """Stage 3: MoE 학습."""
    print("\n" + "=" * 60)
    print("[STAGE 3] MoE Aggregator Training")
    print("=" * 60)
    print("  [TODO] 학습 노트북(notebooks/03_moe_training.ipynb)에서 수행 권장")
    print("  필요 입력: signals JSONL + question embedding 캐시")


def stage4_evaluate(config: dict, args: argparse.Namespace) -> None:
    """Stage 4: Threshold Override + 평가."""
    print("\n" + "=" * 60)
    print("[STAGE 4] Threshold Override + Evaluation")
    print("=" * 60)
    print("  [TODO] 평가 노트북(notebooks/04_evaluation.ipynb)에서 수행 권장")
    print("  필요 입력: 학습된 MoE 모델 + signals JSONL + 원본 BBQ items")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End 파이프라인")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="설정 파일",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="main",
        choices=["main", "gemma", "qwen"],
        help="LLM 선택",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="실행할 카테고리 (생략 시 전체)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="1,2,3,4",
        help="실행할 stage (쉼표 구분, 예: 1,2)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stages = set(args.stages.split(","))

    if "1" in stages:
        stage1_inference(config, args)
    if "2" in stages:
        stage2_signals(config, args)
    if "3" in stages:
        stage3_train_moe(config, args)
    if "4" in stages:
        stage4_evaluate(config, args)

    print("\n" + "=" * 60)
    print("[OK] 파이프라인 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
