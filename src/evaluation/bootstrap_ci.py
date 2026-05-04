"""
Bootstrap 기반 신뢰구간(CI) 및 paired bootstrap p-value 계산.

논문에서 보고할 1000-bootstrap 95% CI를 계산합니다.
또한 두 시스템(예: baseline vs ours)의 차이가 통계적으로 유의한지
paired bootstrap test로 검증합니다.
"""

from typing import Callable

import numpy as np


def bootstrap_ci(
    items: list[dict],
    predictions: list[int],
    metric_fn: Callable[[list[dict], list[int]], float],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap resampling으로 metric의 신뢰구간을 추정합니다.

    Args:
        items: BBQ instance 리스트.
        predictions: 예측 답 리스트.
        metric_fn: (items, predictions) -> float 함수.
        n_iterations: bootstrap 반복 횟수.
        confidence_level: 신뢰 수준 (예: 0.95).
        seed: 랜덤 시드.

    Returns:
        {
            "mean": float,
            "lower": float,
            "upper": float,
            "samples": list[float],
        }
    """
    rng = np.random.RandomState(seed)
    n = len(items)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "samples": []}

    samples: list[float] = []
    for _ in range(n_iterations):
        indices = rng.randint(0, n, size=n)
        boot_items = [items[i] for i in indices]
        boot_preds = [predictions[i] for i in indices]
        score = metric_fn(boot_items, boot_preds)
        if score is not None:
            samples.append(score)

    if not samples:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "samples": []}

    samples_arr = np.array(samples)
    alpha = (1 - confidence_level) / 2
    lower = float(np.quantile(samples_arr, alpha))
    upper = float(np.quantile(samples_arr, 1 - alpha))

    return {
        "mean": float(samples_arr.mean()),
        "lower": lower,
        "upper": upper,
        "samples": samples,
    }


def paired_bootstrap_pvalue(
    items: list[dict],
    predictions_a: list[int],
    predictions_b: list[int],
    metric_fn: Callable[[list[dict], list[int]], float],
    n_iterations: int = 1000,
    seed: int = 42,
    direction: str = "greater",
) -> dict:
    """
    Paired bootstrap test로 두 예측의 metric 차이 유의성을 검증합니다.

    H0: metric(A) == metric(B)
    H1: metric(A) > metric(B) (direction="greater") 또는 양측

    Args:
        items: BBQ instance 리스트.
        predictions_a: 시스템 A의 예측.
        predictions_b: 시스템 B의 예측.
        metric_fn: 비교할 metric 함수.
        n_iterations: bootstrap 횟수.
        seed: 랜덤 시드.
        direction: "greater", "less", "two_sided".

    Returns:
        {
            "diff_observed": float,    # metric(A) - metric(B) 관측치
            "p_value": float,
            "diff_samples": list[float],
        }
    """
    rng = np.random.RandomState(seed)
    n = len(items)

    score_a = metric_fn(items, predictions_a) or 0.0
    score_b = metric_fn(items, predictions_b) or 0.0
    diff_observed = score_a - score_b

    diff_samples: list[float] = []
    for _ in range(n_iterations):
        indices = rng.randint(0, n, size=n)
        boot_items = [items[i] for i in indices]
        boot_a = [predictions_a[i] for i in indices]
        boot_b = [predictions_b[i] for i in indices]

        sa = metric_fn(boot_items, boot_a) or 0.0
        sb = metric_fn(boot_items, boot_b) or 0.0
        diff_samples.append(sa - sb)

    diff_arr = np.array(diff_samples)

    if direction == "greater":
        p = float((diff_arr <= 0).mean())
    elif direction == "less":
        p = float((diff_arr >= 0).mean())
    else:  # two_sided
        p = float((np.abs(diff_arr) >= abs(diff_observed)).mean())

    return {
        "diff_observed": diff_observed,
        "p_value": p,
        "diff_samples": diff_samples,
    }
