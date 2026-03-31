"""
편향 점수 필터링 (Bias Score Filter) 후처리 (B)

모델의 답변이 고정관념 방향이고 신뢰도가 낮은 경우,
자동으로 Unknown으로 교체합니다.

원리:
1. 모델의 답변이 고정관념 대상 그룹을 선택했는지 확인
2. 해당 답변의 신뢰도(logprob 기반)가 임계값 미만인지 확인
3. 두 조건을 모두 만족하면 → Unknown(가장 높은 인덱스)으로 교체

출처: Schick, T., et al. (2021). "Self-Diagnosis and Self-Debiasing." TACL.
"""

from src.evaluation.metrics import is_stereotyped_answer


def find_unknown_index(item):
    """
    Unknown 선택지의 인덱스를 찾습니다.
    answer_info에서 그룹 태그가 'unknown'인 선택지를 반환합니다.
    """
    answer_info = item.get("answer_info", {})
    for i in range(3):
        info = answer_info.get(f"ans{i}", [])
        if len(info) >= 2 and info[1] == "unknown":
            return i
    # 찾지 못하면 기본값 2 (BBQ에서 ans2가 보통 Unknown)
    return 2


def apply_bias_score_filter(items, predictions, confidence_scores=None, threshold=0.7):
    """
    편향 점수 필터링을 적용합니다.

    고정관념 방향의 답변 중 신뢰도가 낮은 것을 Unknown으로 교체합니다.
    confidence_scores가 없으면 고정관념 방향 답변을 모두 Unknown으로 교체합니다.

    Args:
        items: BBQ 항목 리스트
        predictions: 모델 답변 리스트 (0, 1, 2, -1)
        confidence_scores: 답변별 신뢰도 리스트 (0.0~1.0), None이면 신뢰도 무시
        threshold: 신뢰도 임계값 (이 값 미만이면 필터링)

    Returns:
        filtered_predictions: 필터링된 답변 리스트
        filter_count: 필터링된 답변 수
    """
    filtered_predictions = []
    filter_count = 0

    for i, (item, pred) in enumerate(zip(items, predictions)):
        # 파싱 실패는 그대로 유지
        if pred == -1:
            filtered_predictions.append(pred)
            continue

        # 고정관념 방향인지 확인
        stereotype_type = is_stereotyped_answer(item, pred)

        if stereotype_type == "stereotyped":
            # 신뢰도 기반 필터링
            if confidence_scores is not None:
                if confidence_scores[i] < threshold:
                    unknown_idx = find_unknown_index(item)
                    filtered_predictions.append(unknown_idx)
                    filter_count += 1
                else:
                    filtered_predictions.append(pred)
            else:
                # 신뢰도 정보 없으면 모호 맥락에서만 필터링
                context_condition = item.get("context_condition", "")
                if context_condition == "ambig":
                    unknown_idx = find_unknown_index(item)
                    filtered_predictions.append(unknown_idx)
                    filter_count += 1
                else:
                    filtered_predictions.append(pred)
        else:
            filtered_predictions.append(pred)

    return filtered_predictions, filter_count
