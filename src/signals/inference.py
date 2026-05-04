"""
Stage 1: 4-Prompt Inference.

같은 BBQ 질문에 대해 4가지 prompt variant로 LLM 응답을 수집합니다.
응답 텍스트와 함께 첫 토큰 logit을 저장하여 후속 신호 추출에 활용합니다.

체크포인트 단위는 (category, prompt_variant)이며, 중간에 끊겨도 이어서 실행 가능합니다.
"""

import json
import re
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from src.signals.prompts import PROMPT_BUILDERS
from src.utils.llm_utils import LLMWrapper


# 답변 파싱 정규식: (A), A., A 등 다양한 형식 허용
_ANSWER_PATTERN = re.compile(r"\b([ABC])\b")


def parse_answer(text: str) -> int:
    """
    LLM 응답 텍스트에서 답 인덱스를 파싱합니다.

    Args:
        text: 모델 응답 텍스트.

    Returns:
        0 (A), 1 (B), 2 (C), -1 (파싱 실패).
    """
    if not text:
        return -1
    match = _ANSWER_PATTERN.search(text.upper())
    if not match:
        return -1
    return {"A": 0, "B": 1, "C": 2}[match.group(1)]


def run_4prompt_inference(
    items: list[dict],
    llm: LLMWrapper,
    output_path: Path,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    save_every: int = 50,
) -> list[dict]:
    """
    한 카테고리의 모든 instance에 4개 prompt로 추론을 수행합니다.

    Args:
        items: 샘플링된 BBQ instance 리스트.
        llm: LLMWrapper 인스턴스.
        output_path: 결과 JSONL 저장 경로.
        max_new_tokens: 생성 최대 토큰 수.
        temperature: 0이면 greedy, >0이면 sampling.
        save_every: N개마다 중간 저장 (체크포인트).

    Returns:
        instance별 4-prompt 결과 리스트:
        [
            {
                "example_id": ...,
                "responses": {
                    "vanilla": {"text": ..., "answer": 0, "logprobs": {...}},
                    "debiasing_instruction": {...},
                    "cot": {...},
                    "counterfactual_swap": {...},
                }
            },
            ...
        ]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 체크포인트: 이미 처리된 example_id 로드
    completed_ids: set = set()
    results: list[dict] = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    results.append(rec)
                    completed_ids.add(rec["example_id"])
        print(f"  [resume] {len(completed_ids)}개 이미 완료, 이어서 실행")

    pending = [item for item in items if item["example_id"] not in completed_ids]
    if not pending:
        print(f"  [skip] 전체 완료")
        return results

    pbar = tqdm(pending, desc=f"4-Prompt Inference")
    for i, item in enumerate(pbar):
        record = {
            "example_id": item["example_id"],
            "category": item.get("category"),
            "context_condition": item.get("context_condition"),
            "question_polarity": item.get("question_polarity"),
            "label": item.get("label"),
            "responses": {},
        }

        for variant, builder in PROMPT_BUILDERS.items():
            system_msg, user_msg = builder(item)

            out = llm.generate(
                user_message=user_msg,
                system_message=system_msg,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                return_logits=False,
            )

            # 선택지 logprob 별도 계산 (s3 신뢰도용)
            logprobs = llm.get_answer_logprobs(
                user_message=user_msg,
                choice_tokens=["A", "B", "C"],
                system_message=system_msg,
            )

            record["responses"][variant] = {
                "text": out.text,
                "answer": parse_answer(out.text),
                "logprobs": logprobs,
            }

        results.append(record)

        # 중간 저장
        if (i + 1) % save_every == 0:
            _save_jsonl(results, output_path)

    _save_jsonl(results, output_path)
    return results


def _save_jsonl(records: list[dict], path: Path) -> None:
    """결과를 JSONL로 저장합니다."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
