"""
Signal s4: Self-Consistency

같은 prompt를 n번 sampling하여 답이 얼마나 일관되는지 측정합니다.
가장 자주 나온 답의 비율(major vote ratio)을 사용합니다.

높을수록 → 모델이 일관된 답을 줌 (확신)
낮을수록 → 답이 흔들림 (불확실 -> 편향 추측 가능성)
"""

from collections import Counter  # 다수결 집계

from src.signals.inference import parse_answer  # 응답 → 0/1/2 파싱
from src.utils.llm_utils import LLMWrapper  # LLM 추상화


def compute_self_consistency(
    item: dict,
    llm: LLMWrapper,
    prompt_builder,                       # callable: (item) → (system, user)
    n_samples: int = 5,                   # 5회 sampling이 noise-vs-cost trade-off 최적점
    temperature: float = 0.7,             # 다양성 확보 (너무 낮으면 일관성 trivial)
    max_new_tokens: int = 64,             # answer letter만 추출하면 되므로 짧게
) -> dict:
    """
    동일 prompt로 n번 sampling하여 답 일관성을 측정합니다.

    Args:
        item: BBQ instance.
        llm: LLMWrapper 인스턴스.
        prompt_builder: prompt 빌더 함수.
        n_samples: sampling 횟수.
        temperature: 다양성을 위한 temperature (>0 권장).
        max_new_tokens: 최대 생성 토큰.

    Returns:
        {
            "s4_score": float,           # 다수결 답의 비율
            "samples": list[int],        # 각 샘플의 답 인덱스
            "majority_answer": int,      # 가장 빈번한 답
        }
    """
    # primary prompt와 동일한 prompt를 builder로 생성 (vanilla/cot/debias/persona 중 하나)
    system_msg, user_msg = prompt_builder(item)

    # n번 sampling (temperature > 0이라 매번 다른 답 가능)
    answers: list[int] = []
    for _ in range(n_samples):
        out = llm.generate(
            user_message=user_msg,
            system_message=system_msg,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        # 응답 텍스트 → 0/1/2 (실패 시 -1)
        answers.append(parse_answer(out.text))

    # 파싱 실패(-1) 제외하고 다수결
    valid = [a for a in answers if a in (0, 1, 2)]
    if not valid:
        # 모든 sample 파싱 실패 → 0.0 (consistency 측정 불가)
        return {
            "s4_score": 0.0,
            "samples": answers,
            "majority_answer": -1,
        }

    # Counter로 다수결 (most_common(1)[0]: (값, 빈도) 튜플)
    counter = Counter(valid)
    majority, count = counter.most_common(1)[0]
    return {
        # 다수결 답의 비율 — 분모는 valid가 아닌 전체 n_samples
        # → 파싱 실패도 inconsistency로 penalty 부여
        "s4_score": count / len(answers),
        "samples": answers,
        "majority_answer": majority,
    }
