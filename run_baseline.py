"""
베이스라인 실험 실행 스크립트

바닐라 프롬프트(디바이어싱 없음)로 4개 모델의 기본 편향 수준과 정확도를 측정합니다.
이 결과가 이후 모든 실험의 비교 기준선이 됩니다.

실행 방법:
    python run_baseline.py                          # 전체 모델, 순환 순열 적용
    python run_baseline.py --no_permutation          # 순환 순열 없이 원본 그대로
    python run_baseline.py --model gpt4o_mini        # 특정 모델만 실행
    python run_baseline.py --category Age            # 특정 카테고리만 실행
    python run_baseline.py --max_samples 100         # 카테고리당 최대 100개만 (테스트용)
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from src.models.openrouter_client import OpenRouterClient
from src.prompts.vanilla import build_vanilla_prompt
from src.evaluation.metrics import parse_model_answer, evaluate_all
from src.utils.data_loader import load_bbq_category, filter_by_context, apply_cyclic_permutations


def load_config(config_path="configs/experiment_config.yaml"):
    """실험 설정 파일을 로드합니다."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_model_on_items(client, model_id, items, temperature=0, max_tokens=256):
    """
    모델에게 BBQ 질문들을 보내고 답변을 수집합니다.

    Args:
        client: OpenRouterClient 인스턴스
        model_id: 모델 ID (예: "openai/gpt-4o")
        items: BBQ 항목 리스트
        temperature: 생성 온도 (0 = 결정적)
        max_tokens: 최대 토큰 수

    Returns:
        predictions: 파싱된 답변 리스트 (0, 1, 2, 또는 -1)
        raw_answers: 모델의 원본 답변 리스트
    """
    predictions = []
    raw_answers = []

    for item in tqdm(items, desc=f"  질문 처리 중", leave=False):
        system_msg, user_msg = build_vanilla_prompt(item)

        try:
            raw_answer = client.ask(
                model_id=model_id,
                user_message=user_msg,
                system_message=system_msg,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            # API 에러 시 빈 답변 처리 (실험 중단 방지)
            print(f"\n    [에러] {e}")
            raw_answer = ""

        parsed = parse_model_answer(raw_answer)
        predictions.append(parsed)
        raw_answers.append(raw_answer)

    return predictions, raw_answers


def run_baseline_for_model(client, model_name, model_id, config, args):
    """
    하나의 모델에 대해 베이스라인 실험을 실행합니다.

    Returns:
        모델의 전체 결과 딕셔너리
    """
    categories = args.category if args.category else config["categories"]
    if isinstance(categories, str):
        categories = [categories]

    model_results = {
        "model_name": model_name,
        "model_id": model_id,
        "categories": {},
        "overall": {},
    }

    # 전체 결과를 모으기 위한 리스트
    all_ambig_items = []
    all_ambig_preds = []
    all_disambig_items = []
    all_disambig_preds = []

    for category in categories:
        print(f"\n  📂 카테고리: {category}")

        # 데이터 로드
        try:
            data = load_bbq_category(config["data"]["raw_dir"], category)
        except FileNotFoundError as e:
            print(f"    [건너뜀] {e}")
            continue

        # 모호 / 비모호로 분리
        ambig_items = filter_by_context(data, "ambig")
        disambig_items = filter_by_context(data, "disambig")

        # 샘플 수 제한 (테스트용, 순환 순열 적용 전에 제한)
        if args.max_samples:
            ambig_items = ambig_items[:args.max_samples]
            disambig_items = disambig_items[:args.max_samples]

        # 순환 순열 적용 (위치 편향 제거)
        if not args.no_permutation:
            ambig_items = apply_cyclic_permutations(ambig_items)
            disambig_items = apply_cyclic_permutations(disambig_items)
            print(f"    모호: {len(ambig_items)}개, 비모호: {len(disambig_items)}개 (순환순열 3x 적용)")
        else:
            print(f"    모호: {len(ambig_items)}개, 비모호: {len(disambig_items)}개 (순환순열 미적용)")

        # 모호 맥락 실행
        print(f"    [모호 맥락 실행 중...]")
        ambig_preds, ambig_raws = run_model_on_items(
            client, model_id, ambig_items,
            temperature=config["experiment"]["temperature"],
            max_tokens=config["experiment"]["max_tokens"],
        )

        # 비모호 맥락 실행
        print(f"    [비모호 맥락 실행 중...]")
        disambig_preds, disambig_raws = run_model_on_items(
            client, model_id, disambig_items,
            temperature=config["experiment"]["temperature"],
            max_tokens=config["experiment"]["max_tokens"],
        )

        # 카테고리별 평가
        cat_metrics = evaluate_all(
            ambig_items, ambig_preds,
            disambig_items, disambig_preds,
        )

        # 파싱 실패 비율도 기록 (모델이 이상한 답변을 얼마나 하는지)
        total_preds = ambig_preds + disambig_preds
        parse_fail_rate = total_preds.count(-1) / len(total_preds) if total_preds else 0

        cat_metrics["parse_fail_rate"] = round(parse_fail_rate, 4)
        cat_metrics["num_ambig"] = len(ambig_items)
        cat_metrics["num_disambig"] = len(disambig_items)

        model_results["categories"][category] = cat_metrics

        # 결과 출력
        print(f"    ✅ 결과:")
        print(f"       편향 점수:     {cat_metrics['bias_score']}")
        print(f"       정확도(모호):   {cat_metrics['accuracy_ambig']}")
        print(f"       정확도(비모호): {cat_metrics['accuracy_disambig']}")
        print(f"       Diff-Bias:     {cat_metrics['diff_bias']}")
        print(f"       파싱실패율:     {cat_metrics['parse_fail_rate']}")

        # 전체 집계용 데이터 누적
        all_ambig_items.extend(ambig_items)
        all_ambig_preds.extend(ambig_preds)
        all_disambig_items.extend(disambig_items)
        all_disambig_preds.extend(disambig_preds)

    # 전체 카테고리 종합 결과
    if all_ambig_items:
        model_results["overall"] = evaluate_all(
            all_ambig_items, all_ambig_preds,
            all_disambig_items, all_disambig_preds,
        )

    return model_results


def save_results(results, output_dir):
    """결과를 JSON 파일로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perm_suffix = results.get("experiment", "baseline").split("baseline_vanilla_")[-1]
    filename = f"baseline_{perm_suffix}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="BBQ 베이스라인 실험")
    parser.add_argument("--model", type=str, default=None,
                        help="특정 모델만 실행 (예: gpt4o, gpt35, llama3_70b, mistral_7b)")
    parser.add_argument("--category", type=str, nargs="+", default=None,
                        help="특정 카테고리만 실행 (예: Age Gender_identity)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="카테고리당 최대 샘플 수 (테스트용)")
    parser.add_argument("--no_permutation", action="store_true",
                        help="순환 순열을 적용하지 않고 원본 그대로 실행 (비교 실험용)")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                        help="설정 파일 경로")
    args = parser.parse_args()

    # 설정 로드
    config = load_config(args.config)

    # API 클라이언트 생성
    client = OpenRouterClient()

    # 실행할 모델 결정
    if args.model:
        # 특정 모델만 실행
        models_to_run = {args.model: config["models"][args.model]}
    else:
        # 전체 모델 실행
        models_to_run = config["models"]

    # 순환 순열 적용 여부 표시
    perm_label = "no_permutation" if args.no_permutation else "with_permutation"
    print(f"\n🔧 순환 순열: {'미적용' if args.no_permutation else '적용'}")

    # 전체 결과를 담을 딕셔너리
    all_results = {
        "experiment": f"baseline_vanilla_{perm_label}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "temperature": config["experiment"]["temperature"],
            "max_tokens": config["experiment"]["max_tokens"],
            "max_samples": args.max_samples,
            "cyclic_permutation": not args.no_permutation,
        },
        "models": {},
    }

    # 각 모델별로 실험 실행
    for model_name, model_id in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"🤖 모델: {model_name} ({model_id})")
        print(f"{'='*60}")

        model_results = run_baseline_for_model(
            client, model_name, model_id, config, args
        )
        all_results["models"][model_name] = model_results

        # 종합 결과 출력
        if model_results["overall"]:
            print(f"\n  📊 {model_name} 종합 결과:")
            for k, v in model_results["overall"].items():
                print(f"     {k}: {v}")

    # 결과 저장
    save_results(all_results, config["data"]["results_dir"])

    print(f"\n{'='*60}")
    print("✅ 베이스라인 실험 완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
