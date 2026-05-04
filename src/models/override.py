"""
Stage 4: Threshold Override

MoE가 출력한 신뢰도 p를 임계값 tau와 비교하여 최종 답을 결정합니다.

규칙:
    p >= tau:  primary_answer 유지
    p <  tau:  unknown 인덱스로 override

기능:
    - apply_threshold_override(): 단일 instance 적용
    - search_optimal_threshold(): grid search (전역 단일 tau)
    - search_optimal_threshold_per_category(): 카테고리별 다른 tau
    - risk_coverage_curve(): risk-coverage tradeoff 분석
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================
# 1. Apply override
# =============================================================
def find_unknown_index(item: dict) -> int:
    """
    BBQ instance에서 Unknown 선택지 인덱스를 찾습니다.

    Args:
        item: BBQ instance.

    Returns:
        Unknown 선택지 인덱스 (0/1/2). 못 찾으면 2.
    """
    answer_info = item.get("answer_info", {})
    if isinstance(answer_info, str):
        # parquet 저장 후 JSON 문자열로 들어올 수 있음
        import json
        try:
            answer_info = json.loads(answer_info)
        except json.JSONDecodeError:
            return 2

    for i in range(3):
        info = answer_info.get(f"ans{i}", []) if isinstance(answer_info, dict) else []
        if len(info) >= 2 and info[1] == "unknown":
            return i
    return 2


def apply_threshold_override(
    primary_answer: int,
    p_score: float,
    item: dict,
    threshold: float = 0.5,
) -> dict:
    """
    threshold override 적용.

    Args:
        primary_answer: 모델 답 (0/1/2).
        p_score: MoE confidence ∈ [0, 1].
        item: BBQ instance.
        threshold: tau.

    Returns:
        {"final_answer": int, "overridden": bool, "p_score": float}.
    """
    if p_score >= threshold or primary_answer == -1:
        return {
            "final_answer": primary_answer,
            "overridden": False,
            "p_score": p_score,
        }
    unknown_idx = find_unknown_index(item)
    return {
        "final_answer": unknown_idx,
        "overridden": True,
        "p_score": p_score,
    }


# =============================================================
# 2. Threshold search
# =============================================================
@dataclass
class ThresholdSearchResult:
    best_threshold: float
    best_score: float
    metric_used: str
    all_scores: dict[float, float]


def search_optimal_threshold(
    val_predictions: list[dict],
    metric: str = "balanced_accuracy",
    threshold_range: tuple[float, float] = (0.05, 0.95),
    step: float = 0.025,
) -> ThresholdSearchResult:
    """
    Validation set에서 최적 threshold를 grid search합니다.

    Args:
        val_predictions: [{
            "primary_answer": int,
            "p_score": float,
            "item": dict (BBQ instance with label, context_condition),
        }, ...]
        metric: "accuracy_amb" | "accuracy_dis" | "balanced_accuracy" | "macro_acc".
        threshold_range: (min, max) 탐색 범위.
        step: 탐색 간격.

    Returns:
        ThresholdSearchResult.
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    scores: dict[float, float] = {}

    for tau_raw in thresholds:
        tau = round(float(tau_raw), 4)
        scores[tau] = _eval_threshold(val_predictions, tau, metric)

    if not scores:
        return ThresholdSearchResult(0.5, 0.0, metric, {})

    best_tau = max(scores, key=scores.get)
    return ThresholdSearchResult(
        best_threshold=best_tau,
        best_score=scores[best_tau],
        metric_used=metric,
        all_scores=scores,
    )


def search_optimal_threshold_per_category(
    val_predictions: list[dict],
    metric: str = "balanced_accuracy",
    threshold_range: tuple[float, float] = (0.05, 0.95),
    step: float = 0.025,
) -> dict[str, ThresholdSearchResult]:
    """
    카테고리별로 별도의 threshold를 grid search합니다.

    Returns:
        {category: ThresholdSearchResult}.
    """
    by_cat: dict[str, list[dict]] = {}
    for pred in val_predictions:
        cat = pred["item"].get("category", "_unknown")
        by_cat.setdefault(cat, []).append(pred)

    results: dict[str, ThresholdSearchResult] = {}
    for cat, preds in by_cat.items():
        results[cat] = search_optimal_threshold(
            preds, metric=metric, threshold_range=threshold_range, step=step,
        )
        logger.info(
            f"  [{cat:25s}] best_tau={results[cat].best_threshold:.3f} "
            f"score={results[cat].best_score:.4f} (n={len(preds)})"
        )
    return results


def _eval_threshold(
    val_predictions: list[dict],
    tau: float,
    metric: str,
) -> float:
    """
    한 threshold에 대해 metric을 계산합니다.
    """
    correct_amb = 0
    correct_dis = 0
    total_amb = 0
    total_dis = 0

    for pred in val_predictions:
        item = pred["item"]
        result = apply_threshold_override(
            primary_answer=pred["primary_answer"],
            p_score=pred["p_score"],
            item=item,
            threshold=tau,
        )
        label = item.get("label", -1)
        cond = item.get("context_condition", "")
        is_correct = result["final_answer"] == label

        if cond == "ambig":
            total_amb += 1
            correct_amb += int(is_correct)
        elif cond == "disambig":
            total_dis += 1
            correct_dis += int(is_correct)

    acc_amb = correct_amb / total_amb if total_amb > 0 else 0.0
    acc_dis = correct_dis / total_dis if total_dis > 0 else 0.0

    if metric == "accuracy_amb":
        return acc_amb
    if metric == "accuracy_dis":
        return acc_dis
    if metric in ("balanced_accuracy", "macro_acc"):
        return (acc_amb + acc_dis) / 2.0
    raise ValueError(f"Unknown metric: {metric}")


# =============================================================
# 3. Risk-Coverage Curve
# =============================================================
@dataclass
class RiskCoveragePoint:
    threshold: float
    coverage: float        # p >= tau인 sample 비율
    risk: float            # 그 sample들 중 오답 비율
    error_rate: float      # 전체 sample 기준 오답 비율 (override 적용 후)


def risk_coverage_curve(
    val_predictions: list[dict],
    threshold_range: tuple[float, float] = (0.0, 1.0),
    step: float = 0.025,
) -> list[RiskCoveragePoint]:
    """
    Risk-Coverage curve를 계산합니다.

    Coverage: p >= tau인 sample의 비율 (모델 답을 keep하는 비율).
    Risk: keep된 sample들 중에서 모델 답이 틀린 비율.
    Error rate: override 적용 후 전체 정확도의 보수.

    Args:
        val_predictions: search_optimal_threshold와 동일.
        threshold_range: 탐색 범위.
        step: 간격.

    Returns:
        RiskCoveragePoint 리스트 (threshold 오름차순).
    """
    if not val_predictions:
        return []

    p_scores = np.array([p["p_score"] for p in val_predictions])
    primary = np.array([p["primary_answer"] for p in val_predictions])
    labels = np.array([p["item"].get("label", -1) for p in val_predictions])
    unknown_idx = np.array([find_unknown_index(p["item"]) for p in val_predictions])

    n = len(val_predictions)
    points: list[RiskCoveragePoint] = []
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)

    for tau_raw in thresholds:
        tau = round(float(tau_raw), 4)
        keep_mask = p_scores >= tau

        # Coverage
        coverage = float(keep_mask.mean())

        # Risk: kept sample 중 오답 비율
        if keep_mask.sum() > 0:
            kept_correct = (primary[keep_mask] == labels[keep_mask]).sum()
            risk = 1.0 - (kept_correct / keep_mask.sum())
        else:
            risk = 0.0

        # Error rate: override 적용 후 전체 오답 비율
        final = np.where(keep_mask, primary, unknown_idx)
        error_rate = float((final != labels).mean())

        points.append(RiskCoveragePoint(
            threshold=tau,
            coverage=coverage,
            risk=float(risk),
            error_rate=error_rate,
        ))

    return points


def best_threshold_from_rc_curve(
    points: Iterable[RiskCoveragePoint],
    objective: str = "min_error",
) -> RiskCoveragePoint:
    """
    Risk-Coverage curve에서 최적 threshold를 선택합니다.

    Args:
        points: risk_coverage_curve 결과.
        objective: "min_error" (전체 오류 최소화),
                   "min_risk_at_full_coverage" (coverage=1에서의 risk).

    Returns:
        선택된 RiskCoveragePoint.
    """
    points = list(points)
    if not points:
        raise ValueError("빈 risk-coverage curve")

    if objective == "min_error":
        return min(points, key=lambda p: p.error_rate)
    if objective == "min_risk_at_full_coverage":
        # coverage=1.0인 경우만 보고 risk 최소화
        full = [p for p in points if p.coverage >= 0.99]
        return min(full, key=lambda p: p.risk) if full else min(points, key=lambda p: p.threshold)
    raise ValueError(f"Unknown objective: {objective}")
