"""
Signal s5: Bias-head Activation

모델 내부의 특정 attention head가 demographic 정보에 과도하게 주의를
기울이는지 측정합니다. 사전에 식별된 "bias head" 인덱스의 attention
weight를 demographic token에 대해 합산합니다.

Bias head 식별 방법 (오프라인):
    1. 다수의 BBQ 샘플에서 layer별 head별 attention 패턴 추출
    2. demographic token (그룹명)에 대한 평균 attention이 높은 head 선정
    3. config의 head_indices에 기록

높을수록 → bias head가 demographic token에 강하게 반응 (편향 의존)
낮을수록 → demographic token 무시 (편향 낮음)

NOTE: 본 모듈은 골격이며, 실제 head 인덱스 식별은 별도 노트북에서 수행합니다.
"""

from typing import Optional

import torch

from src.utils.llm_utils import LLMWrapper


def identify_demographic_token_indices(
    item: dict,
    llm: LLMWrapper,
    prompt: str,
) -> list[int]:
    """
    프롬프트에서 demographic group을 가리키는 토큰 인덱스를 찾습니다.

    Args:
        item: BBQ instance.
        llm: LLMWrapper.
        prompt: 토크나이즈된 프롬프트 문자열.

    Returns:
        demographic token 위치 리스트.
    """
    answer_info = item.get("answer_info", {})
    demographic_terms: set[str] = set()

    for i in range(3):
        info = answer_info.get(f"ans{i}", [])
        if len(info) >= 2 and info[1] != "unknown":
            demographic_terms.add(info[0].lower())

    if not demographic_terms:
        return []

    # 토큰 단위로 매칭
    tokens = llm.tokenizer.tokenize(prompt)
    indices: list[int] = []
    for i, tok in enumerate(tokens):
        clean = tok.lstrip("Ġ▁ ").lower()
        if any(term in clean for term in demographic_terms):
            indices.append(i)
    return indices


def compute_bias_head_activation(
    item: dict,
    llm: LLMWrapper,
    prompt_builder,
    head_indices: list[tuple[int, int]],
) -> float:
    """
    지정된 (layer, head) 페어들의 demographic-token attention을 측정합니다.

    Args:
        item: BBQ instance.
        llm: LLMWrapper.
        prompt_builder: prompt 빌더.
        head_indices: [(layer_idx, head_idx), ...] 리스트.

    Returns:
        0.0 ~ 1.0 평균 attention 가중치.

    NOTE: HuggingFace 모델에서 attention 추출은 output_attentions=True 필요.
          Mac MPS에서는 일부 모델에서 미지원이므로 CUDA 권장.
    """
    if not head_indices:
        return 0.0

    system_msg, user_msg = prompt_builder(item)
    prompt = llm.build_chat_prompt(user_msg, system_msg)
    inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.device)

    demographic_token_idx = identify_demographic_token_indices(item, llm, prompt)
    if not demographic_token_idx:
        return 0.0

    with torch.inference_mode():
        outputs = llm.model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions  # tuple of (batch, n_heads, seq, seq)
    if attentions is None:
        return 0.0

    last_token_idx = inputs["input_ids"].shape[1] - 1
    scores: list[float] = []

    for layer_idx, head_idx in head_indices:
        if layer_idx >= len(attentions):
            continue
        attn = attentions[layer_idx][0, head_idx]  # (seq, seq)
        # 마지막 토큰이 demographic token에 보이는 attention 합
        attn_to_demographic = sum(
            attn[last_token_idx, j].item()
            for j in demographic_token_idx
            if j < attn.shape[1]
        )
        scores.append(attn_to_demographic)

    return sum(scores) / len(scores) if scores else 0.0
