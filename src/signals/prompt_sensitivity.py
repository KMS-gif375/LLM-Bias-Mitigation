"""
Signal s6: Prompt Sensitivity

4가지 prompt variant의 답이 얼마나 일관되는지 측정합니다.
4개 답이 모두 같으면 모델이 prompt에 robust하고, 다르면 prompt에 의존적입니다.

높을수록 → 4 prompt 모두 같은 답 (robust, 편향 낮음 가능성)
낮을수록 → prompt마다 답이 다름 (불안정, 편향 추측 가능성)

NOTE: s4(self-consistency)와 다름.
    s4 = 같은 prompt, n번 sampling
    s6 = 다른 prompt, 1번씩
"""

from collections import Counter


def compute_prompt_sensitivity(prompt_responses: dict[str, dict]) -> dict:
    """
    4-prompt 응답에서 답 일관성을 계산합니다.

    Args:
        prompt_responses: Stage 1 결과의 "responses" 필드.
            {
                "vanilla": {"answer": 0, ...},
                "debiasing_instruction": {"answer": 0, ...},
                "cot": {"answer": 1, ...},
                "counterfactual_swap": {"answer": 0, ...},
            }

    Returns:
        {
            "s6_score": float,        # 다수결 답의 비율
            "majority_answer": int,
            "answers": list[int],
            "n_unique": int,          # 고유한 답의 개수 (1=완전 일관, 3=완전 분산)
        }
    """
    answers = [r["answer"] for r in prompt_responses.values()]
    valid = [a for a in answers if a in (0, 1, 2)]

    if not valid:
        return {
            "s6_score": 0.0,
            "majority_answer": -1,
            "answers": answers,
            "n_unique": 0,
        }

    counter = Counter(valid)
    majority, count = counter.most_common(1)[0]

    return {
        "s6_score": count / len(answers),
        "majority_answer": majority,
        "answers": answers,
        "n_unique": len(set(valid)),
    }
