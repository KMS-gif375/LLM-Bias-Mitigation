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
    threshold override 적용 (단일 τ 버전 — backward-compat).

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


def apply_per_condition_override(
    primary_answer: int,
    p_score: float,
    item: dict,
    thresholds: dict[str, float],
) -> dict:
    """
    Per-condition threshold override.

    BBQ의 ambig/disambig은 정답 분포가 정반대 (ambig 정답 ≈ unknown,
    disambig 정답 ≈ 구체 인물). 단일 τ로는 한쪽이 좋아지면 다른쪽이 나빠지는
    trade-off가 생기므로 condition별로 다른 τ를 사용한다.

    Args:
        primary_answer: 모델 답 (0/1/2).
        p_score: MoE confidence ∈ [0, 1].
        item: BBQ instance ('context_condition' 필요).
        thresholds: {"ambig": float, "disambig": float} (그 외 condition은
            unknown으로 간주하고 default 0.5 사용).

    Returns:
        {"final_answer": int, "overridden": bool, "p_score": float,
         "condition": str, "threshold_used": float}.
    """
    cond = item.get("context_condition", "")
    tau = thresholds.get(cond, thresholds.get("default", 0.5))

    if p_score >= tau or primary_answer == -1:
        return {
            "final_answer": primary_answer,
            "overridden": False,
            "p_score": p_score,
            "condition": cond,
            "threshold_used": tau,
        }
    unknown_idx = find_unknown_index(item)
    return {
        "final_answer": unknown_idx,
        "overridden": True,
        "p_score": p_score,
        "condition": cond,
        "threshold_used": tau,
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


@dataclass
class PerConditionSearchResult:
    """ambig/disambig 각각에 대한 best τ + 통합 metric."""

    thresholds: dict[str, float]            # {"ambig": τ_amb, "disambig": τ_dis}
    per_condition_scores: dict[str, dict]   # {"ambig": {τ: score}, "disambig": {τ: score}}
    combined_score: float                   # 두 condition 합산 metric
    metric_used: str


def search_optimal_threshold_per_condition(
    val_predictions: list[dict],
    metric_amb: str = "accuracy_amb",
    metric_dis: str = "accuracy_dis",
    threshold_range: tuple[float, float] = (0.05, 0.95),
    step: float = 0.025,
) -> PerConditionSearchResult:
    """
    ambig/disambig 각각 독립적으로 best τ를 grid search한다.

    원리:
        - ambig 정답은 보통 "unknown" → 높은 τ로 적극 abstain (stereotype 차단)
        - disambig 정답은 구체 인물 → 낮은 τ로 model 답을 보존
        - 두 condition을 분리 최적화하면 단일 τ보다 항상 우월하거나 동등.

    Args:
        val_predictions: search_optimal_threshold와 동일 schema. item에
            'context_condition' 필요.
        metric_amb: ambig에서 최적화할 metric (보통 "accuracy_amb" 또는
            "neg_bias_abs_amb"). bias 절대값 최소화하려면 후자.
        metric_dis: disambig metric (보통 "accuracy_dis").
        threshold_range, step: grid search 범위.

    Returns:
        PerConditionSearchResult.
    """
    # ambig/disambig으로 분리
    amb_preds = [p for p in val_predictions if p["item"].get("context_condition") == "ambig"]
    dis_preds = [p for p in val_predictions if p["item"].get("context_condition") == "disambig"]

    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    amb_scores: dict[float, float] = {}
    dis_scores: dict[float, float] = {}

    # ambig grid search
    for tau_raw in thresholds:
        tau = round(float(tau_raw), 4)
        amb_scores[tau] = _eval_threshold_for_condition(amb_preds, tau, metric_amb, "ambig")
        dis_scores[tau] = _eval_threshold_for_condition(dis_preds, tau, metric_dis, "disambig")

    if not amb_scores or not dis_scores:
        return PerConditionSearchResult(
            thresholds={"ambig": 0.5, "disambig": 0.5},
            per_condition_scores={"ambig": amb_scores, "disambig": dis_scores},
            combined_score=0.0,
            metric_used=f"{metric_amb}+{metric_dis}",
        )

    best_tau_amb = max(amb_scores, key=amb_scores.get)
    best_tau_dis = max(dis_scores, key=dis_scores.get)

    # 통합 score = 두 best score의 macro average
    combined = (amb_scores[best_tau_amb] + dis_scores[best_tau_dis]) / 2.0

    logger.info(
        f"  [per-condition τ] ambig: τ={best_tau_amb:.3f} ({metric_amb}={amb_scores[best_tau_amb]:.4f})  "
        f"disambig: τ={best_tau_dis:.3f} ({metric_dis}={dis_scores[best_tau_dis]:.4f})"
    )

    return PerConditionSearchResult(
        thresholds={"ambig": best_tau_amb, "disambig": best_tau_dis},
        per_condition_scores={"ambig": amb_scores, "disambig": dis_scores},
        combined_score=combined,
        metric_used=f"{metric_amb}+{metric_dis}",
    )


def _eval_threshold_for_condition(
    preds: list[dict],
    tau: float,
    metric: str,
    condition: str,
) -> float:
    """
    한 condition의 predictions에 대해 단일 τ 적용 후 metric 계산.

    metric 옵션:
        accuracy_amb / accuracy_dis: 정답률
        neg_bias_abs_amb: -|bias_score_amb| (bias 작을수록 좋음 → max로 변환)
        neg_false_abst: -false_abstention_rate (적을수록 좋음, dis용)
    """
    if not preds:
        return 0.0

    correct = 0
    total = 0
    n_stereo = 0
    n_anti = 0
    n_abstained = 0  # disambig에서 unknown으로 override

    # is_stereotyped_answer는 evaluation 모듈에 있으므로 lazy import
    from src.evaluation.bbq_evaluator import is_stereotyped_answer

    for pred in preds:
        item = pred["item"]
        result = apply_threshold_override(
            primary_answer=pred["primary_answer"],
            p_score=pred["p_score"],
            item=item,
            threshold=tau,
        )
        label = item.get("label", -1)
        is_correct = result["final_answer"] == label

        total += 1
        if is_correct:
            correct += 1

        # bias 계산용
        kind = is_stereotyped_answer(item, result["final_answer"])
        if kind == "stereotyped":
            n_stereo += 1
        elif kind == "anti_stereotyped":
            n_anti += 1
        if condition == "disambig" and result["overridden"]:
            n_abstained += 1

    acc = correct / total if total > 0 else 0.0

    if metric in ("accuracy_amb", "accuracy_dis"):
        return acc
    if metric == "neg_bias_abs_amb":
        if n_stereo + n_anti == 0:
            return 0.0
        bias = 2 * (n_stereo / (n_stereo + n_anti)) - 1
        return -abs(bias)
    if metric == "neg_false_abst":
        if total == 0:
            return 0.0
        return -(n_abstained / total)
    raise ValueError(f"Unknown metric for per-condition: {metric}")


# 사용자 main repo의 자체 구현 alias — 호환성 유지
# (이미 새 search_optimal_threshold_per_condition + apply_per_condition_override가
# 위에 정의되어 있고 worktree 형식이 모든 callsite와 통합됨)
def apply_threshold_override_per_condition(
    primary_answer: int,
    p_score: float,
    item: dict,
    thresholds_by_cond: dict[str, float],
    default_threshold: float = 0.5,
) -> dict:
    """condition별 별도 tau 적용 — apply_per_condition_override의 backward-compat alias."""
    if not thresholds_by_cond and default_threshold:
        thresholds_by_cond = {"ambig": default_threshold, "disambig": default_threshold}
    return apply_per_condition_override(
        primary_answer=primary_answer,
        p_score=p_score,
        item=item,
        thresholds=thresholds_by_cond,
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
