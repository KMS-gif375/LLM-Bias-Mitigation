"""
Signal s3: Self-Confidence

선택지(A/B/C) token에 대한 모델의 logit 기반 신뢰도를 계산합니다.
선택된 답 토큰의 softmax 확률을 사용합니다.

높을수록 → 모델이 답에 자신 있음
낮을수록 → 모델이 불확실 -> 편향 추측 가능성
"""

import math
from typing import Optional


def compute_confidence_from_logprobs(
    logprobs: dict[str, float],
    chosen_answer: int,
) -> float:
    """
    A/B/C logprob에서 선택된 답의 softmax 확률을 계산합니다.

    Args:
        logprobs: {"A": -1.2, "B": -3.4, "C": -2.1} 형식.
        chosen_answer: 모델이 선택한 답 (0, 1, 2).

    Returns:
        0.0 ~ 1.0 확률값.
    """
    if chosen_answer not in (0, 1, 2):
        return 0.0

    letter = ["A", "B", "C"][chosen_answer]
    if letter not in logprobs:
        return 0.0

    # softmax: exp(x_i) / sum(exp(x_j))
    values = [logprobs.get(k, float("-inf")) for k in ("A", "B", "C")]
    max_v = max(v for v in values if v != float("-inf"))
    exps = [math.exp(v - max_v) if v != float("-inf") else 0.0 for v in values]
    total = sum(exps)
    if total == 0.0:
        return 0.0

    idx = {"A": 0, "B": 1, "C": 2}[letter]
    return exps[idx] / total


def compute_entropy(logprobs: dict[str, float]) -> float:
    """
    A/B/C 분포의 entropy를 계산합니다 (불확실성 척도).

    Args:
        logprobs: A/B/C logprob 딕셔너리.

    Returns:
        nats 단위 entropy (0이면 완전 확신, 최대 ln(3) ≈ 1.099).
    """
    values = [logprobs.get(k, float("-inf")) for k in ("A", "B", "C")]
    max_v = max(v for v in values if v != float("-inf"))
    exps = [math.exp(v - max_v) if v != float("-inf") else 0.0 for v in values]
    total = sum(exps)
    if total == 0.0:
        return 0.0

    probs = [e / total for e in exps]
    return -sum(p * math.log(p) for p in probs if p > 0)
