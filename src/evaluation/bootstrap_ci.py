"""
Bootstrap 기반 신뢰구간 (CI) 및 paired bootstrap p-value.

논문에서 보고할 1000-bootstrap 95% CI를 계산합니다.
또한 두 시스템(예: baseline vs ours)의 차이가 통계적으로 유의한지
paired bootstrap test로 검증합니다.

API:
    bootstrap_ci(predictions, instances, metric_fn, ...) -> dict
    paired_bootstrap_pvalue(predictions_a, predictions_b, instances, metric_fn, ...) -> dict
    metric_for(metric_name) -> Callable             # bbq_evaluator의 metric을 fn으로 변환
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.evaluation.bbq_evaluator import (
    compute_accuracy,
    compute_bias_score,
    compute_false_abstention_rate,
    parse_prediction,
)


# Metric 시그니처: (instances, pred_indices) -> float | None
MetricFn = Callable[[list[dict], list[int]], float | None]


# =============================================================
# Metric registry
# =============================================================
def _accuracy_amb(instances: list[dict], preds: list[int]) -> float:
    return _split_then_compute(instances, preds, "ambig", compute_accuracy)


def _accuracy_dis(instances: list[dict], preds: list[int]) -> float:
    return _split_then_compute(instances, preds, "disambig", compute_accuracy)


def _bias_score_amb(instances: list[dict], preds: list[int]) -> float | None:
    return _split_then_compute(instances, preds, "ambig", compute_bias_score)


def _bias_score_dis(instances: list[dict], preds: list[int]) -> float | None:
    return _split_then_compute(instances, preds, "disambig", compute_bias_score)


def _false_abstention(instances: list[dict], preds: list[int]) -> float:
    return _split_then_compute(instances, preds, "disambig", compute_false_abstention_rate)


_METRIC_REGISTRY: dict[str, MetricFn] = {
    "accuracy_amb": _accuracy_amb,
    "accuracy_dis": _accuracy_dis,
    "bias_score_amb": _bias_score_amb,
    "bias_score_dis": _bias_score_dis,
    "false_abstention_rate": _false_abstention,
}


def _split_then_compute(
    instances: list[dict],
    preds: list[int],
    condition: str,
    fn: Callable,
) -> float | None:
    """맥락 조건으로 필터링 후 metric 계산."""
    pairs = [
        (item, p) for item, p in zip(instances, preds)
        if item.get("context_condition") == condition
    ]
    if not pairs:
        return 0.0
    items, ps = zip(*pairs)
    return fn(list(items), list(ps))


def metric_for(name: str) -> MetricFn:
    """metric 이름을 함수로 변환 (bootstrap_ci에 전달)."""
    if name not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric: {name}. 사용 가능: {list(_METRIC_REGISTRY)}"
        )
    return _METRIC_REGISTRY[name]


# =============================================================
# 1. Bootstrap CI
# =============================================================
def bootstrap_ci(
    predictions: list[str | int],
    instances: list[dict],
    metric_fn: MetricFn,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap resampling으로 metric의 95% CI를 추정합니다.

    Args:
        predictions: 예측 답 (list[str]/list[int]/혼합).
        instances: BBQ instance 리스트.
        metric_fn: (instances, pred_indices) -> float | None.
        n_iterations: bootstrap 횟수.
        confidence_level: 신뢰 수준.
        seed: 랜덤 시드.

    Returns:
        {
            "mean": float,
            "lower": float,
            "upper": float,
            "samples": list[float],
            "n_valid_samples": int,
        }

    Raises:
        ValueError: predictions와 instances 길이 불일치.
    """
    if len(predictions) != len(instances):
        raise ValueError(
            f"길이 불일치: predictions={len(predictions)}, instances={len(instances)}"
        )

    pred_indices = [parse_prediction(p) for p in predictions]
    n = len(instances)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "samples": [], "n_valid_samples": 0}

    rng = np.random.RandomState(seed)
    samples: list[float] = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        boot_items = [instances[i] for i in idx]
        boot_preds = [pred_indices[i] for i in idx]
        score = metric_fn(boot_items, boot_preds)
        if score is not None:
            samples.append(float(score))

    if not samples:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "samples": [], "n_valid_samples": 0}

    arr = np.array(samples)
    alpha = (1 - confidence_level) / 2
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(arr, alpha)),
        "upper": float(np.quantile(arr, 1 - alpha)),
        "samples": samples,
        "n_valid_samples": len(samples),
    }


# =============================================================
# 2. Paired Bootstrap
# =============================================================
def paired_bootstrap_pvalue(
    predictions_a: list[str | int],
    predictions_b: list[str | int],
    instances: list[dict],
    metric_fn: MetricFn,
    n_iterations: int = 1000,
    seed: int = 42,
    direction: str = "greater",
) -> dict:
    """
    Paired bootstrap test로 두 시스템의 metric 차이 유의성을 검증합니다.

    H0: metric(A) == metric(B)
    H1: metric(A) > metric(B) (direction="greater")
        metric(A) < metric(B) (direction="less")
        |diff| > observed (direction="two_sided")

    같은 bootstrap index를 두 시스템에 적용하여 paired sampling합니다.

    Args:
        predictions_a: 시스템 A의 예측.
        predictions_b: 시스템 B의 예측.
        instances: BBQ instance.
        metric_fn: metric 함수.
        n_iterations: bootstrap 횟수.
        seed: 랜덤 시드.
        direction: "greater" | "less" | "two_sided".

    Returns:
        {
            "diff_observed": float,    # metric(A) - metric(B)
            "p_value": float,
            "diff_samples": list[float],
        }
    """
    n = len(instances)
    if not (len(predictions_a) == len(predictions_b) == n):
        raise ValueError(
            f"길이 불일치: A={len(predictions_a)}, B={len(predictions_b)}, "
            f"instances={n}"
        )

    pred_a = [parse_prediction(p) for p in predictions_a]
    pred_b = [parse_prediction(p) for p in predictions_b]

    score_a = metric_fn(instances, pred_a) or 0.0
    score_b = metric_fn(instances, pred_b) or 0.0
    diff_observed = float(score_a - score_b)

    rng = np.random.RandomState(seed)
    diff_samples: list[float] = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        boot_items = [instances[i] for i in idx]
        boot_a = [pred_a[i] for i in idx]
        boot_b = [pred_b[i] for i in idx]
        sa = metric_fn(boot_items, boot_a) or 0.0
        sb = metric_fn(boot_items, boot_b) or 0.0
        diff_samples.append(float(sa - sb))

    arr = np.array(diff_samples)
    if direction == "greater":
        p = float((arr <= 0).mean())
    elif direction == "less":
        p = float((arr >= 0).mean())
    elif direction == "two_sided":
        p = float((np.abs(arr) >= abs(diff_observed)).mean())
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return {
        "diff_observed": diff_observed,
        "p_value": p,
        "diff_samples": diff_samples,
    }
