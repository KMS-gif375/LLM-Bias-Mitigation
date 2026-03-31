"""
교차 모델 검증 (Cross-Model Verification) 후처리 (C)

주요 모델의 답변을 보조 모델이 검증합니다.
모호한 맥락에서 두 모델이 서로 다른 인구통계학적 그룹을 선택하면
Unknown으로 변경합니다.

원리:
1. 주요 모델(primary)이 답변을 생성
2. 보조 모델(secondary)이 같은 질문에 답변
3. 모호 맥락에서 두 모델이 서로 다른 그룹 답변을 선택하면 → Unknown으로 교체
4. 비모호 맥락에서는 주요 모델의 답변을 유지 (정확도 보존)
"""

from src.evaluation.metrics import is_stereotyped_answer, parse_model_answer


def find_unknown_index(item):
    """Unknown 선택지의 인덱스를 찾습니다."""
    answer_info = item.get("answer_info", {})
    for i in range(3):
        info = answer_info.get(f"ans{i}", [])
        if len(info) >= 2 and info[1] == "unknown":
            return i
    return 2


def apply_cross_model_verification(items, primary_preds, secondary_preds):
    """
    교차 모델 검증을 적용합니다.

    모호 맥락에서 두 모델이 서로 다른 인구통계학적 그룹 답변을 선택하면
    Unknown으로 교체합니다. 비모호 맥락에서는 주요 모델의 답변을 유지합니다.

    Args:
        items: BBQ 항목 리스트
        primary_preds: 주요 모델의 답변 리스트
        secondary_preds: 보조 모델의 답변 리스트

    Returns:
        verified_predictions: 검증된 답변 리스트
        override_count: Unknown으로 교체된 답변 수
    """
    verified_predictions = []
    override_count = 0

    for item, p_pred, s_pred in zip(items, primary_preds, secondary_preds):
        context_condition = item.get("context_condition", "")

        # 비모호 맥락: 주요 모델 답변 유지 (정확도 보존)
        if context_condition == "disambig":
            verified_predictions.append(p_pred)
            continue

        # 파싱 실패: 그대로 유지
        if p_pred == -1 or s_pred == -1:
            verified_predictions.append(p_pred)
            continue

        # 두 모델이 같은 답변: 그대로 유지
        if p_pred == s_pred:
            verified_predictions.append(p_pred)
            continue

        # 두 모델이 다른 답변: 각각의 고정관념 방향 확인
        p_type = is_stereotyped_answer(item, p_pred)
        s_type = is_stereotyped_answer(item, s_pred)

        # 둘 다 Unknown이 아닌 그룹 답변인데 서로 다르면 → Unknown으로 교체
        if p_type != "unknown" and s_type != "unknown" and p_pred != s_pred:
            unknown_idx = find_unknown_index(item)
            verified_predictions.append(unknown_idx)
            override_count += 1
        else:
            verified_predictions.append(p_pred)

    return verified_predictions, override_count


async def get_secondary_predictions(client, secondary_model_id, items, prompt_builder,
                                     concurrency=10, temperature=0, max_tokens=256):
    """
    보조 모델로 답변을 생성합니다.

    Args:
        client: OpenRouterClient 인스턴스
        secondary_model_id: 보조 모델 ID
        items: BBQ 항목 리스트
        prompt_builder: 프롬프트 빌더 함수
        concurrency: 동시 호출 수
        temperature: 생성 온도
        max_tokens: 최대 토큰 수

    Returns:
        predictions: 보조 모델의 답변 리스트
        raw_answers: 보조 모델의 원본 답변 리스트
    """
    request_list = []
    for item in items:
        system_msg, user_msg = prompt_builder(item)
        request_list.append((system_msg, user_msg))

    raw_answers = await client.ask_batch(
        model_id=secondary_model_id,
        request_list=request_list,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )

    predictions = [parse_model_answer(ans) for ans in raw_answers]
    return predictions, raw_answers
