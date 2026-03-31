"""
베이스라인 실험 실행 스크립트

바닐라 프롬프트(디바이어싱 없음)로 4개 모델의 기본 편향 수준과 정확도를 측정합니다.
이 결과가 이후 모든 실험의 비교 기준선이 됩니다.

실행 방법:
    python run_baseline.py                                          # 전체 모델, 순환 순열 적용
    python run_baseline.py --concurrency 20                        # 동시 20개 호출 (더 빠름)
    python run_baseline.py --no_permutation                         # 순환 순열 없이 원본 그대로
    python run_baseline.py --model gpt4o_mini                       # 특정 모델만 실행
    python run_baseline.py --category Age                           # 특정 카테고리만 실행
    python run_baseline.py --max_samples 5                          # 카테고리당 최대 5개만 (테스트용)
    python run_baseline.py --output_dir data/results/baseline_vanilla  # 저장 경로 지정
    python run_baseline.py --resume                                 # 이전 실행 이어하기
"""

import argparse
import asyncio
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


def get_checkpoint_path(output_dir, perm_label, model_name=None):
    """체크포인트 파일 경로를 반환합니다. 모델별로 분리됩니다."""
    os.makedirs(output_dir, exist_ok=True)
    if model_name:
        return Path(output_dir) / f"checkpoint_baseline_{perm_label}_{model_name}.json"
    return Path(output_dir) / f"checkpoint_baseline_{perm_label}.json"


def load_checkpoint(checkpoint_path):
    """체크포인트 파일이 있으면 로드합니다."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path, results):
    """현재까지의 결과를 체크포인트 파일로 저장합니다."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def run_model_on_items(client, model_id, items, temperature=0, max_tokens=256, concurrency=10):
    """
    모델에게 BBQ 질문들을 비동기 배치로 보내고 답변을 수집합니다.

    Args:
        client: OpenRouterClient 인스턴스
        model_id: 모델 ID (예: "openai/gpt-4o")
        items: BBQ 항목 리스트
        temperature: 생성 온도 (0 = 결정적)
        max_tokens: 최대 토큰 수
        concurrency: 동시 호출 수 (기본 10)

    Returns:
        predictions: 파싱된 답변 리스트 (0, 1, 2, 또는 -1)
        raw_answers: 모델의 원본 답변 리스트
    """
    # 프롬프트 생성
    request_list = []
    for item in items:
        system_msg, user_msg = build_vanilla_prompt(item)
        request_list.append((system_msg, user_msg))

    # 비동기 배치 호출
    print(f"      (동시 {concurrency}개씩 비동기 호출)")
    raw_answers = await client.ask_batch(
        model_id=model_id,
        request_list=request_list,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )

    # 파싱
    predictions = [parse_model_answer(ans) for ans in raw_answers]

    return predictions, raw_answers


async def run_baseline_for_model(client, model_name, model_id, config, args,
                                  checkpoint_path, all_results):
    """
    하나의 모델에 대해 베이스라인 실험을 실행합니다.
    이미 완료된 카테고리는 건너뜁니다.

    Returns:
        모델의 전체 결과 딕셔너리
    """
    categories = args.category if args.category else config["categories"]
    if isinstance(categories, str):
        categories = [categories]

    # 이전 체크포인트에서 이 모델의 결과 복원
    existing = all_results.get("models", {}).get(model_name, None)
    if existing:
        model_results = existing
        completed = set(model_results.get("categories", {}).keys())
        if completed:
            print(f"    [skip] 이전 진행 복원: {len(completed)}개 카테고리 완료 ({', '.join(completed)})")
    else:
        model_results = {
            "model_name": model_name,
            "model_id": model_id,
            "categories": {},
            "overall": {},
        }
        completed = set()

    # 전체 결과를 모으기 위한 리스트
    all_ambig_items = []
    all_ambig_preds = []
    all_disambig_items = []
    all_disambig_preds = []

    for category in categories:
        # 이미 완료된 카테고리 건너뛰기
        if category in completed:
            print(f"\n  [카테고리] 카테고리: {category} — [skip] 이미 완료, 건너뜀")
            continue

        print(f"\n  [카테고리] 카테고리: {category}")

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
        ambig_preds, ambig_raws = await run_model_on_items(
            client, model_id, ambig_items,
            temperature=config["experiment"]["temperature"],
            max_tokens=config["experiment"]["max_tokens"],
            concurrency=args.concurrency,
        )

        # 비모호 맥락 실행
        print(f"    [비모호 맥락 실행 중...]")
        disambig_preds, disambig_raws = await run_model_on_items(
            client, model_id, disambig_items,
            temperature=config["experiment"]["temperature"],
            max_tokens=config["experiment"]["max_tokens"],
            concurrency=args.concurrency,
        )

        # 카테고리별 평가
        cat_metrics = evaluate_all(
            ambig_items, ambig_preds,
            disambig_items, disambig_preds,
        )

        # 파싱 실패 비율도 기록
        total_preds = ambig_preds + disambig_preds
        parse_fail_rate = total_preds.count(-1) / len(total_preds) if total_preds else 0

        cat_metrics["parse_fail_rate"] = round(parse_fail_rate, 4)
        cat_metrics["num_ambig"] = len(ambig_items)
        cat_metrics["num_disambig"] = len(disambig_items)

        model_results["categories"][category] = cat_metrics

        # 결과 출력
        print(f"    [OK] 결과:")
        print(f"       편향 점수:     {cat_metrics['bias_score']}")
        print(f"       정확도(모호):   {cat_metrics['accuracy_ambig']}")
        print(f"       정확도(비모호): {cat_metrics['accuracy_disambig']}")
        print(f"       Diff-Bias:     {cat_metrics['diff_bias']}")
        print(f"       파싱실패율:     {cat_metrics['parse_fail_rate']}")

        # 체크포인트 저장 (카테고리 완료마다)
        all_results["models"][model_name] = model_results
        save_checkpoint(checkpoint_path, all_results)
        print(f"    [SAVE] 체크포인트 저장 완료")

    # 전체 카테고리에 대해 종합 결과 재계산
    # (체크포인트에서 복원한 카테고리 포함)
    for category in categories:
        if category not in model_results["categories"]:
            continue

        try:
            data = load_bbq_category(config["data"]["raw_dir"], category)
        except FileNotFoundError:
            continue

        ambig_items = filter_by_context(data, "ambig")
        disambig_items = filter_by_context(data, "disambig")

        if args.max_samples:
            ambig_items = ambig_items[:args.max_samples]
            disambig_items = disambig_items[:args.max_samples]

        if not args.no_permutation:
            ambig_items = apply_cyclic_permutations(ambig_items)
            disambig_items = apply_cyclic_permutations(disambig_items)

        all_ambig_items.extend(ambig_items)
        all_disambig_items.extend(disambig_items)

    # overall은 카테고리별 metrics에서 계산 (전체 재실행 없이)
    if model_results["categories"]:
        # 카테고리별 가중 평균 계산
        total_ambig = sum(m["num_ambig"] for m in model_results["categories"].values())
        total_disambig = sum(m["num_disambig"] for m in model_results["categories"].values())

        if total_ambig > 0 and total_disambig > 0:
            weighted_bias = 0
            weighted_acc_ambig = 0
            weighted_acc_disambig = 0
            weighted_diff_bias = 0
            diff_bias_total = 0

            for cat, m in model_results["categories"].items():
                w_a = m["num_ambig"] / total_ambig
                w_d = m["num_disambig"] / total_disambig
                if m["bias_score"] is not None:
                    weighted_bias += m["bias_score"] * w_a
                weighted_acc_ambig += m["accuracy_ambig"] * w_a
                weighted_acc_disambig += m["accuracy_disambig"] * w_d
                if m["diff_bias"] is not None:
                    weighted_diff_bias += m["diff_bias"] * w_d
                    diff_bias_total += w_d

            model_results["overall"] = {
                "bias_score": round(weighted_bias, 4),
                "accuracy_ambig": round(weighted_acc_ambig, 4),
                "accuracy_disambig": round(weighted_acc_disambig, 4),
                "diff_bias": round(weighted_diff_bias / diff_bias_total, 4) if diff_bias_total > 0 else None,
            }

    return model_results


def save_results(results, output_dir):
    """최종 결과를 JSON 파일로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perm_suffix = results.get("experiment", "baseline").split("baseline_vanilla_")[-1]
    filename = f"baseline_{perm_suffix}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] 최종 결과 저장: {filepath}")
    return filepath


async def async_main():
    parser = argparse.ArgumentParser(description="BBQ 베이스라인 실험")
    parser.add_argument("--model", type=str, default=None,
                        help="특정 모델만 실행 (예: gpt4o_mini, gpt35, llama3_70b, mistral_24b)")
    parser.add_argument("--category", type=str, nargs="+", default=None,
                        help="특정 카테고리만 실행 (예: Age Gender_identity)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="카테고리당 최대 샘플 수 (테스트용)")
    parser.add_argument("--no_permutation", action="store_true",
                        help="순환 순열을 적용하지 않고 원본 그대로 실행 (비교 실험용)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="동시 API 호출 수 (기본 10, 높을수록 빠르지만 rate limit 주의)")
    parser.add_argument("--resume", action="store_true",
                        help="이전 실행의 체크포인트에서 이어하기")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="결과 저장 경로 (기본: config의 results_dir)")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                        help="설정 파일 경로")
    args = parser.parse_args()

    # 설정 로드
    config = load_config(args.config)

    # API 클라이언트 생성
    client = OpenRouterClient()

    # 실행할 모델 결정
    if args.model:
        models_to_run = {args.model: config["models"][args.model]}
    else:
        models_to_run = config["models"]

    # 출력 경로 결정
    output_dir = args.output_dir if args.output_dir else config["data"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 순환 순열 적용 여부 표시
    perm_label = "no_permutation" if args.no_permutation else "with_permutation"
    print(f"\n[설정] 순환 순열: {'미적용' if args.no_permutation else '적용'}")
    print(f"[SPEED] 동시 호출 수: {args.concurrency}")
    print(f"[설정] 저장 경로: {output_dir}")

    # 각 모델별로 실험 실행
    for model_name, model_id in models_to_run.items():
        # 모델별 체크포인트 경로 (충돌 방지)
        checkpoint_path = get_checkpoint_path(output_dir, perm_label, model_name)

        # 체크포인트 복원 또는 새로 시작
        if args.resume and checkpoint_path.exists():
            all_results = load_checkpoint(checkpoint_path)
            print(f"[LOAD] 체크포인트 복원: {checkpoint_path}")
        else:
            all_results = {
                "experiment": f"baseline_vanilla_{perm_label}",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "temperature": config["experiment"]["temperature"],
                    "max_tokens": config["experiment"]["max_tokens"],
                    "max_samples": args.max_samples,
                    "cyclic_permutation": not args.no_permutation,
                    "concurrency": args.concurrency,
                },
                "models": {},
            }
            if not args.resume and checkpoint_path.exists():
                print(f"[WARN] 기존 체크포인트 발견. 이어하려면 --resume 옵션을 추가하세요.")

        # 이미 모든 카테고리가 완료된 모델 건너뛰기
        categories = args.category if args.category else config["categories"]
        if isinstance(categories, str):
            categories = [categories]

        existing_model = all_results.get("models", {}).get(model_name, {})
        completed_cats = set(existing_model.get("categories", {}).keys())
        remaining = [c for c in categories if c not in completed_cats]

        if args.resume and not remaining:
            print(f"\n{'='*60}")
            print(f"[MODEL] {model_name} -- [skip] 전체 완료, 건너뜀")
            print(f"{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f"[MODEL] {model_name} ({model_id})")
        if args.resume and completed_cats:
            print(f"   남은 카테고리: {len(remaining)}개 / {len(categories)}개")
        print(f"{'='*60}")

        model_results = await run_baseline_for_model(
            client, model_name, model_id, config, args,
            checkpoint_path, all_results,
        )
        all_results["models"][model_name] = model_results

        # 종합 결과 출력
        if model_results.get("overall"):
            print(f"\n  [RESULT] {model_name} 종합 결과:")
            for k, v in model_results["overall"].items():
                print(f"     {k}: {v}")

        # 모델별 최종 결과 저장
        save_results(all_results, output_dir)

        # 체크포인트 삭제 (해당 모델 정상 완료 시)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[DEL] 체크포인트 삭제 ({model_name} 완료)")

    print(f"\n{'='*60}")
    print("[OK] 베이스라인 실험 완료!")
    print(f"{'='*60}")


def main():
    # Windows에서 asyncio 호환성 설정
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
