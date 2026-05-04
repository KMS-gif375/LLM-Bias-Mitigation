"""
Stage 4: Threshold Override

MoE가 출력한 확신도 p를 임계값 tau와 비교하여 최종 답을 결정합니다.

Logic:
    if p >= tau:
        final_answer = primary_answer  (모델 답 유지)
    else:
        final_answer = unknown_index    (Unknown으로 override)

이 단계는 학습 가능한 파라미터가 없는 deterministic 후처리입니다.
tau는 validation set에서 grid search로 결정합니다.
"""

import numpy as np


def find_unknown_index(item: dict) -> int:
    """
    BBQ instance에서 Unknown 선택지의 인덱스를 찾습니다.

    Args:
        item: BBQ instance.

    Returns:
        Unknown 선택지 인덱스 (0, 1, 2). 못 찾으면 2 (BBQ 기본값).
    """
    answer_info = item.get("answer_info", {})
    for i in range(3):
        info = answer_info.get(f"ans{i}", [])
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
    임계값 override를 적용합니다.

    Args:
        primary_answer: 모델의 원래 답 (0, 1, 2).
        p_score: MoE 출력 확신도 ∈ [0, 1].
        item: BBQ instance.
        threshold: tau (이 값 미만이면 Unknown override).

    Returns:
        {
            "final_answer": int,
            "overridden": bool,
            "p_score": float,
        }
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


def search_optimal_threshold(
    val_predictions: list[dict],
    metric: str = "accuracy_amb",
    threshold_range: tuple[float, float] = (0.3, 0.7),
    step: float = 0.05,
) -> dict:
    """
    Validation set에서 최적 threshold를 grid search합니다.

    Args:
        val_predictions: [{
            "primary_answer": int,
            "p_score": float,
            "item": dict (BBQ instance 포함),
        }, ...]
        metric: 최적화할 지표 ("accuracy_amb", "accuracy_dis", "balanced").
        threshold_range: (min, max) 탐색 범위.
        step: 탐색 간격.

    Returns:
        {
            "best_threshold": float,
            "best_score": float,
            "all_scores": dict[float, float],
        }
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    scores: dict[float, float] = {}

    for tau in thresholds:
        tau = round(float(tau), 4)
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
                if is_correct:
                    correct_amb += 1
            elif cond == "disambig":
                total_dis += 1
                if is_correct:
                    correct_dis += 1

        acc_amb = correct_amb / total_amb if total_amb > 0 else 0.0
        acc_dis = correct_dis / total_dis if total_dis > 0 else 0.0

        if metric == "accuracy_amb":
            scores[tau] = acc_amb
        elif metric == "accuracy_dis":
            scores[tau] = acc_dis
        else:  # balanced
            scores[tau] = (acc_amb + acc_dis) / 2

    best_tau = max(scores, key=scores.get)
    return {
        "best_threshold": best_tau,
        "best_score": scores[best_tau],
        "all_scores": scores,
    }
