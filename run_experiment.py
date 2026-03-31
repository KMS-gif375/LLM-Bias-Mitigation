"""
프롬프트 기법 실험 실행 스크립트

5가지 프롬프트 기법으로 모델의 편향 수준과 정확도를 측정합니다.
바닐라(A)는 베이스라인, B~E는 디바이어싱 기법입니다.

실행 방법:
    python run_experiment.py --prompt vanilla                    # (A) 바닐라
    python run_experiment.py --prompt fairness_instruction       # (B) 공정성 지시문
    python run_experiment.py --prompt cot_debiasing              # (C) CoT 디바이어싱
    python run_experiment.py --prompt role_based                 # (D) 역할 기반
    python run_experiment.py --prompt composite                  # (E) 복합 프롬프팅
    python run_experiment.py --prompt all                        # 전체 프롬프트 순차 실행
    python run_experiment.py --prompt fairness_instruction --model gpt4o_mini  # 특정 모델만
    python run_experiment.py --prompt cot_debiasing --resume     # 이어하기
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from src.models.openrouter_client import OpenRouterClient
from src.prompts.vanilla import build_vanilla_prompt
from src.prompts.fairness_instruction import build_fairness_instruction_prompt
from src.prompts.cot_debiasing import build_cot_debiasing_prompt
from src.prompts.role_based import build_role_based_prompt
from src.prompts.composite import build_composite_prompt
from src.evaluation.metrics import parse_model_answer, evaluate_all
from src.utils.data_loader import load_bbq_category, filter_by_context, apply_cyclic_permutations


# 프롬프트 기법 매핑
PROMPT_BUILDERS = {
    "vanilla": build_vanilla_prompt,
    "fairness_instruction": build_fairness_instruction_prompt,
    "cot_debiasing": build_cot_debiasing_prompt,
    "role_based": build_role_based_prompt,
    "composite": build_composite_prompt,
}

PROMPT_LABELS = {
    "vanilla": "(A) Vanilla Baseline",
    "fairness_instruction": "(B) Fairness Instruction",
    "cot_debiasing": "(C) CoT Debiasing",
    "role_based": "(D) Role-Based",
    "composite": "(E) Composite",
}


def load_config(config_path="configs/experiment_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_checkpoint_path(output_dir, prompt_name, model_name):
    os.makedirs(output_dir, exist_ok=True)
    return Path(output_dir) / f"checkpoint_{prompt_name}_{model_name}.json"


def load_checkpoint(checkpoint_path):
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path, results):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def run_model_on_items(client, model_id, items, prompt_builder, concurrency=10,
                              temperature=0, max_tokens=256):
    """모델에게 BBQ 질문들을 비동기 배치로 보내고 답변을 수집합니다."""
    request_list = []
    for item in items:
        system_msg, user_msg = prompt_builder(item)
        request_list.append((system_msg, user_msg))

    print(f"      (동시 {concurrency}개씩 비동기 호출)")
    raw_answers = await client.ask_batch(
        model_id=model_id,
        request_list=request_list,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )

    predictions = [parse_model_answer(ans) for ans in raw_answers]
    return predictions, raw_answers


async def run_experiment_for_model(client, model_name, model_id, config, args,
                                    prompt_name, prompt_builder, checkpoint_path, all_results):
    """하나의 모델에 대해 실험을 실행합니다."""
    categories = args.category if args.category else config["categories"]
    if isinstance(categories, str):
        categories = [categories]

    existing = all_results.get("models", {}).get(model_name, None)
    if existing:
        model_results = existing
        completed = set(model_results.get("categories", {}).keys())
        if completed:
            print(f"    [skip] 이전 진행 복원: {len(completed)}개 카테고리 완료")
    else:
        model_results = {
            "model_name": model_name,
            "model_id": model_id,
            "categories": {},
            "overall": {},
        }
        completed = set()

    for category in categories:
        if category in completed:
            print(f"\n  [카테고리] {category} -- [skip] 이미 완료")
            continue

        print(f"\n  [카테고리] {category}")

        try:
            data = load_bbq_category(config["data"]["raw_dir"], category)
        except FileNotFoundError as e:
            print(f"    [건너뜀] {e}")
            continue

        ambig_items = filter_by_context(data, "ambig")
        disambig_items = filter_by_context(data, "disambig")

        if args.max_samples:
            ambig_items = ambig_items[:args.max_samples]
            disambig_items = disambig_items[:args.max_samples]

        if not args.no_permutation:
            ambig_items = apply_cyclic_permutations(ambig_items)
            disambig_items = apply_cyclic_permutations(disambig_items)
            print(f"    모호: {len(ambig_items)}개, 비모호: {len(disambig_items)}개 (순환순열 3x)")
        else:
            print(f"    모호: {len(ambig_items)}개, 비모호: {len(disambig_items)}개")

        print(f"    [모호 맥락 실행 중...]")
        ambig_preds, _ = await run_model_on_items(
            client, model_id, ambig_items, prompt_builder,
            concurrency=args.concurrency,
            temperature=config["experiment"]["temperature"],
            max_tokens=config["experiment"]["max_tokens"],
        )

        print(f"    [비모호 맥락 실행 중...]")
        disambig_preds, _ = await run_model_on_items(
            client, model_id, disambig_items, prompt_builder,
            concurrency=args.concurrency,
            temperature=config["experiment"]["temperature"],
            max_tokens=config["experiment"]["max_tokens"],
        )

        cat_metrics = evaluate_all(ambig_items, ambig_preds, disambig_items, disambig_preds)

        total_preds = ambig_preds + disambig_preds
        parse_fail_rate = total_preds.count(-1) / len(total_preds) if total_preds else 0
        cat_metrics["parse_fail_rate"] = round(parse_fail_rate, 4)
        cat_metrics["num_ambig"] = len(ambig_items)
        cat_metrics["num_disambig"] = len(disambig_items)

        model_results["categories"][category] = cat_metrics

        print(f"    [OK] 결과:")
        print(f"       편향 점수:     {cat_metrics['bias_score']}")
        print(f"       정확도(모호):   {cat_metrics['accuracy_ambig']}")
        print(f"       정확도(비모호): {cat_metrics['accuracy_disambig']}")
        print(f"       Diff-Bias:     {cat_metrics['diff_bias']}")
        print(f"       파싱실패율:     {cat_metrics['parse_fail_rate']}")

        all_results["models"][model_name] = model_results
        save_checkpoint(checkpoint_path, all_results)
        print(f"    [SAVE] 체크포인트 저장 완료")

    # 종합 결과 계산 (가중 평균)
    if model_results["categories"]:
        total_ambig = sum(m["num_ambig"] for m in model_results["categories"].values())
        total_disambig = sum(m["num_disambig"] for m in model_results["categories"].values())

        if total_ambig > 0 and total_disambig > 0:
            weighted_bias = 0
            weighted_acc_ambig = 0
            weighted_acc_disambig = 0
            weighted_diff_bias = 0
            diff_bias_total = 0

            for m in model_results["categories"].values():
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
    os.makedirs(output_dir, exist_ok=True)
    prompt_name = results.get("prompt_method", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prompt_name}_{timestamp}.json"
    filepath = Path(output_dir) / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] 최종 결과 저장: {filepath}")
    return filepath


async def run_single_prompt(args, config, client, prompt_name):
    """단일 프롬프트 기법으로 실험을 실행합니다."""
    prompt_builder = PROMPT_BUILDERS[prompt_name]
    prompt_label = PROMPT_LABELS[prompt_name]
    output_dir = args.output_dir if args.output_dir else config["data"]["results_dir"]

    print(f"\n{'='*60}")
    print(f"[PROMPT] {prompt_label}")
    print(f"{'='*60}")

    if args.model:
        models_to_run = {args.model: config["models"][args.model]}
    else:
        models_to_run = config["models"]

    for model_name, model_id in models_to_run.items():
        checkpoint_path = get_checkpoint_path(output_dir, prompt_name, model_name)

        if args.resume and checkpoint_path.exists():
            all_results = load_checkpoint(checkpoint_path)
            print(f"[LOAD] 체크포인트 복원: {checkpoint_path}")
        else:
            all_results = {
                "experiment": f"prompt_{prompt_name}",
                "prompt_method": prompt_name,
                "prompt_label": prompt_label,
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

        # 이미 완료된 모델 건너뛰기
        categories = args.category if args.category else config["categories"]
        if isinstance(categories, str):
            categories = [categories]
        existing_model = all_results.get("models", {}).get(model_name, {})
        completed_cats = set(existing_model.get("categories", {}).keys())
        remaining = [c for c in categories if c not in completed_cats]

        if args.resume and not remaining:
            print(f"\n[MODEL] {model_name} -- [skip] 전체 완료")
            continue

        print(f"\n[MODEL] {model_name} ({model_id})")
        if args.resume and completed_cats:
            print(f"   남은 카테고리: {len(remaining)}개 / {len(categories)}개")

        model_results = await run_experiment_for_model(
            client, model_name, model_id, config, args,
            prompt_name, prompt_builder, checkpoint_path, all_results,
        )
        all_results["models"][model_name] = model_results

        if model_results.get("overall"):
            print(f"\n  [RESULT] {model_name} 종합 결과:")
            for k, v in model_results["overall"].items():
                print(f"     {k}: {v}")

        save_results(all_results, output_dir)

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[DEL] 체크포인트 삭제 ({model_name} 완료)")


async def async_main():
    parser = argparse.ArgumentParser(description="BBQ 프롬프트 기법 실험")
    parser.add_argument("--prompt", type=str, required=True,
                        choices=list(PROMPT_BUILDERS.keys()) + ["all"],
                        help="프롬프트 기법 (vanilla, fairness_instruction, cot_debiasing, role_based, composite, all)")
    parser.add_argument("--model", type=str, default=None,
                        help="특정 모델만 실행")
    parser.add_argument("--category", type=str, nargs="+", default=None,
                        help="특정 카테고리만 실행")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="카테고리당 최대 샘플 수 (테스트용)")
    parser.add_argument("--no_permutation", action="store_true",
                        help="순환 순열 미적용")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="동시 API 호출 수 (기본 10)")
    parser.add_argument("--resume", action="store_true",
                        help="이전 실행의 체크포인트에서 이어하기")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="결과 저장 경로")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                        help="설정 파일 경로")
    args = parser.parse_args()

    config = load_config(args.config)
    client = OpenRouterClient()

    print(f"[설정] 순환 순열: {'미적용' if args.no_permutation else '적용'}")
    print(f"[SPEED] 동시 호출 수: {args.concurrency}")

    if args.prompt == "all":
        for prompt_name in PROMPT_BUILDERS:
            await run_single_prompt(args, config, client, prompt_name)
    else:
        await run_single_prompt(args, config, client, args.prompt)

    print(f"\n{'='*60}")
    print("[OK] 실험 완료!")
    print(f"{'='*60}")


def main():
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
