"""
Effect size computation utilities.

논문에서 우리 method vs baseline의 차이를 보고할 때 statistical significance
(p-value)와 practical significance (effect size)를 함께 보고하기 위함.

API:
    cohens_d(group1, group2)  → standardized mean difference
    interpret_cohens_d(d)     → "small" / "medium" / "large" 등 라벨
"""

from __future__ import annotations

import math
from typing import Iterable


def cohens_d(group1: Iterable[float], group2: Iterable[float]) -> float:
    """
    Cohen's d (pooled SD 기반).

    d = (mean1 - mean2) / pooled_sd
    pooled_sd = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    Args:
        group1: 1번 group의 metric 값들 (예: ours seed별 acc_amb).
        group2: 2번 group의 metric 값들 (예: baseline 결과들 또는 단일 값 리스트).

    Returns:
        Cohen's d (positive = group1이 더 큼).
        n1+n2 < 4면 0.0 (분산 추정 불가).
    """
    g1 = list(group1)
    g2 = list(group2)
    n1, n2 = len(g1), len(g2)
    if n1 + n2 < 4 or n1 == 0 or n2 == 0:
        return 0.0

    mean1 = sum(g1) / n1
    mean2 = sum(g2) / n2

    if n1 > 1:
        var1 = sum((x - mean1) ** 2 for x in g1) / (n1 - 1)
    else:
        var1 = 0.0
    if n2 > 1:
        var2 = sum((x - mean2) ** 2 for x in g2) / (n2 - 1)
    else:
        var2 = 0.0

    df = n1 + n2 - 2
    if df <= 0:
        return 0.0

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
    if pooled_var <= 0:
        return 0.0

    pooled_sd = math.sqrt(pooled_var)
    return (mean1 - mean2) / pooled_sd


def interpret_cohens_d(d: float) -> str:
    """
    Cohen (1988) 가이드라인:
        |d| < 0.2  → negligible
        0.2 ≤ |d| < 0.5  → small
        0.5 ≤ |d| < 0.8  → medium
        |d| ≥ 0.8  → large
    """
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"
