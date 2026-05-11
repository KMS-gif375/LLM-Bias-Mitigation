"""
Signal s5: Bias-head Activation

모델 내부의 특정 attention head가 demographic 정보에 과도하게 주의를
기울이는지 측정합니다. 사전에 식별된 "bias head" 인덱스의 attention
weight를 demographic token에 대해 합산합니다.

Bias head 식별 (Contrastive):
    1. Stage 1 결과에서 stereotype 답한 케이스 vs anti-stereotype 답한 케이스 분리.
    2. 각 (layer, head)에 대해 demographic token으로 향한 attention 평균 계산.
    3. Contrastive score = mean(stereo_attn) - mean(anti_attn).
    4. Top-N head 선정 → results/bias_heads.json 캐싱.

높을수록 → bias head가 demographic token에 강하게 반응 (편향 의존)
낮을수록 → demographic token 무시 (편향 낮음)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from src.evaluation.bbq_evaluator import is_stereotyped_answer
from src.utils.llm_utils import LLMWrapper

logger = logging.getLogger(__name__)


# 디폴트 캐시 경로 (Llama / "main" 모델용)
DEFAULT_BIAS_HEADS_PATH = Path("results/bias_heads.json")


def bias_heads_path_for(model_key: str = "main") -> Path:
    """
    모델별 bias_heads.json 경로.

    - "main" (Llama): results/bias_heads.json (legacy 유지)
    - "gemma": results/cross_llm/gemma/bias_heads.json
    - "qwen": results/cross_llm/qwen/bias_heads.json
    - 기타: results/cross_llm/{model_key}/bias_heads.json
    """
    if model_key == "main":
        return DEFAULT_BIAS_HEADS_PATH
    return Path(f"results/cross_llm/{model_key}/bias_heads.json")


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


# =============================================================
# Bias-head identification (offline, contrastive)
# =============================================================
def _attention_to_demographic_per_head(
    item: dict,
    llm: LLMWrapper,
    prompt_builder,
) -> Optional[torch.Tensor]:
    """
    한 instance에 대해 (n_layers, n_heads) attention-to-demographic 행렬을 반환.

    Returns:
        (n_layers, n_heads) 텐서. demographic token이 없거나 attention이 없으면 None.
    """
    system_msg, user_msg = prompt_builder(item)
    prompt = llm.build_chat_prompt(user_msg, system_msg)
    inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.device)

    demo_idx = identify_demographic_token_indices(item, llm, prompt)
    if not demo_idx:
        return None

    with torch.inference_mode():
        outputs = llm.model(**inputs, output_attentions=True, return_dict=True)

    attentions = outputs.attentions  # tuple of (batch, n_heads, seq, seq)
    if not attentions:
        return None

    last_idx = inputs["input_ids"].shape[1] - 1
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    seq_len = attentions[0].shape[-1]

    # 유효한 demographic token 인덱스만 사용
    valid_demo = [j for j in demo_idx if j < seq_len]
    if not valid_demo:
        return None

    # 결과 행렬: 마지막 토큰 → demographic token 평균 attention
    result = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    for layer in range(n_layers):
        # (heads, seq, seq) → 마지막 토큰 행에서 demographic 컬럼들의 평균
        attn_layer = attentions[layer][0]  # (heads, seq, seq)
        sliced = attn_layer[:, last_idx, :][:, valid_demo]  # (heads, n_demo)
        result[layer] = sliced.mean(dim=-1).float().cpu()

    return result


def identify_bias_heads(
    bbq_train_data: list[dict],
    stage1_results: list[dict],
    llm: LLMWrapper,
    prompt_builder,
    primary_prompt: str = "vanilla",
    n_top: int = 20,
    save_path: str | Path = DEFAULT_BIAS_HEADS_PATH,
    max_samples: Optional[int] = None,
) -> list[tuple[int, int]]:
    """
    Contrastive 방법으로 bias-relevant attention heads를 식별합니다.

    1. Stage 1 결과의 모델 답변(primary_prompt)을 stereotype/anti-stereotype으로 분류.
    2. 각 (layer, head)에서 demographic token으로 향한 attention 평균을 측정.
    3. Contrastive score = mean(stereo) - mean(anti).
    4. score 내림차순 상위 n_top개의 (layer, head) 반환.

    Args:
        bbq_train_data: BBQ instance 리스트 (item에 example_id, answer_info,
            additional_metadata, question_polarity, context_condition 필요).
        stage1_results: Stage 1 결과 리스트 (각 dict에 example_id, responses).
        llm: LLMWrapper.
        prompt_builder: prompt 빌더 (PROMPT_BUILDERS[primary_prompt]).
        primary_prompt: stereotype 분류에 사용할 prompt 변형.
        n_top: 선택할 head 수.
        save_path: 결과 캐시 저장 경로 (None이면 저장 안 함).
        max_samples: 분석에 사용할 최대 샘플 수 (None이면 전체).

    Returns:
        [(layer, head), ...] n_top개. 분류 가능 샘플이 없으면 빈 리스트.
    """
    stage1_by_id = {r["example_id"]: r["responses"] for r in stage1_results}

    stereo_acc: list[torch.Tensor] = []
    anti_acc: list[torch.Tensor] = []

    pool = bbq_train_data if max_samples is None else bbq_train_data[:max_samples]
    for item in tqdm(pool, desc="Bias-head identification"):
        ex_id = item.get("example_id")
        if ex_id not in stage1_by_id:
            continue

        primary = stage1_by_id[ex_id].get(primary_prompt, {})
        ans = primary.get("answer")
        try:
            answer_idx = int(ans)
        except (TypeError, ValueError):
            continue

        kind = is_stereotyped_answer(item, answer_idx)
        if kind not in ("stereotyped", "anti_stereotyped"):
            continue  # unknown 답변 또는 분류 불가는 contrastive 분리에 부적합

        attn_matrix = _attention_to_demographic_per_head(item, llm, prompt_builder)
        if attn_matrix is None:
            continue

        if kind == "stereotyped":
            stereo_acc.append(attn_matrix)
        else:
            anti_acc.append(attn_matrix)

    if not stereo_acc or not anti_acc:
        logger.warning(
            f"  [bias-head] contrastive 샘플 부족 "
            f"(stereo={len(stereo_acc)}, anti={len(anti_acc)}) — bias-head 식별 불가"
        )
        return []

    stereo_mean = torch.stack(stereo_acc).mean(dim=0)
    anti_mean = torch.stack(anti_acc).mean(dim=0)
    diff = stereo_mean - anti_mean  # (n_layers, n_heads)

    # 상위 n_top
    flat = diff.flatten()
    top_vals, top_idx = flat.topk(min(n_top, flat.numel()))
    n_heads = diff.shape[1]
    head_indices: list[tuple[int, int]] = [
        (int(i // n_heads), int(i % n_heads)) for i in top_idx.tolist()
    ]

    logger.info(
        f"  [bias-head] {len(head_indices)}개 식별 "
        f"(stereo={len(stereo_acc)}, anti={len(anti_acc)} 샘플 사용)"
    )

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "head_indices": head_indices,
            "scores": top_vals.tolist(),
            "n_stereotyped": len(stereo_acc),
            "n_anti_stereotyped": len(anti_acc),
            "method": "contrastive_attention_to_demographic",
            "primary_prompt": primary_prompt,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"  [bias-head] 저장: {path}")

    return head_indices


def load_bias_heads(
    path: str | Path = DEFAULT_BIAS_HEADS_PATH,
) -> list[tuple[int, int]]:
    """
    캐싱된 bias-head 인덱스를 로드합니다. 파일이 없으면 빈 리스트 반환.
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [tuple(h) for h in data.get("head_indices", [])]
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"  [bias-head] 캐시 로드 실패 ({p}): {e}")
        return []
