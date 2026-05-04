"""
Cross-LLM Transfer: Gemma-2-9B-It pipeline.

Llama에서 학습된 시스템을 Gemma에 그대로 전이(transfer)합니다.

차이점:
    1. 모델: Llama-3.1-8B → Gemma-2-9B-It (RunPod CUDA 권장)
    2. SAE: Llama-Scope → Gemma Scope (DeepMind, google/gemma-scope-9b-it-res)
    3. Bias-head 재식별: Llama 그대로 적용 불가 → Gemma의 head/layer 분포에서 재식별
    4. Bias SAE feature 재식별: Gemma Scope에서 stereotype-correlated feature 다시 선정

설계:
    - LLMWrapper는 model-agnostic이므로 그대로 사용
    - signal extractor도 그대로 (s1~s6은 입력만 다름; s7만 SAE 차이)
    - MoE aggregator는 그대로 사용 (model-agnostic embedding 권장)

전이 시 embedding 정합성 노트:
    - Gemma hidden_dim=3584, Llama=4096이므로 LLM hidden을 직접 사용하면 dim mismatch.
    - 권장: sentence-transformers (model-agnostic 384-dim) 사용 → 본 모듈도 default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch
    from src.utils.llm_utils import LLMWrapper
    from src.signals.sae_feature import SAEWrapper

logger = logging.getLogger(__name__)


# =============================================================
# Config
# =============================================================
@dataclass
class GemmaConfig:
    """Gemma-2-9B 전용 설정."""

    model_name: str = "google/gemma-2-9b-it"
    dtype: str = "bfloat16"
    device: str = "auto"

    # Gemma Scope SAE
    sae_repo: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 20
    sae_top_k: int = 50

    # Bias head (Gemma용 재식별 결과 — 빈 리스트면 자동 0)
    bias_head_indices: list[tuple[int, int]] = field(default_factory=list)

    # Bias SAE feature 인덱스 (Gemma Scope에서 재식별)
    bias_sae_features: list[int] = field(default_factory=list)


# =============================================================
# Bias-head 재식별 (오프라인 절차 헬퍼)
# =============================================================
def identify_bias_heads_contrastive(
    llm: "LLMWrapper",
    bbq_instances: list[dict],
    n_layers_to_scan: Optional[int] = None,
    top_k: int = 8,
) -> list[tuple[int, int]]:
    """
    Bias-head를 contrastive 방식으로 재식별합니다.

    Algorithm:
        1. 같은 instance에 대해 (a) 원본 context, (b) demographic-swap context를 각각 forward.
        2. 각 (layer, head)에서 demographic token에 대한 attention 차이를 측정.
        3. 차이가 가장 큰 head k개를 bias-head로 선정.

    Args:
        llm: LLMWrapper.
        bbq_instances: 식별용 sample 리스트 (50~200개 권장).
        n_layers_to_scan: 검사할 layer 수 (None이면 전체).
        top_k: 선정할 head 개수.

    Returns:
        [(layer_idx, head_idx), ...] 리스트.

    Note:
        실제 contrastive 식별 코드는 attention extraction이 필요합니다.
        본 함수는 인터페이스이며, 노트북에서 단계별로 검증한 후 채워 넣으세요.
        구현 가이드는 notebooks/cross_llm_bias_head.ipynb 참조.
    """
    logger.warning(
        "[identify_bias_heads_contrastive] 골격 함수입니다. "
        "Gemma용 attention extraction은 노트북에서 단계별 검증 후 구현하세요."
    )
    return []


def identify_bias_sae_features(
    llm: "LLMWrapper",
    sae: "SAEWrapper",
    bbq_instances: list[dict],
    top_k: int = 50,
) -> list[int]:
    """
    Stereotype-correlated SAE feature를 재식별합니다.

    Algorithm:
        1. 각 instance에 대해 stereotype 답을 강제하는 prompt forward.
        2. SAE feature activation 추출.
        3. activation이 높고 anti-stereotype 답에서는 낮은 feature 선정.

    Args:
        llm: LLMWrapper.
        sae: SAEWrapper (Gemma Scope).
        bbq_instances: 식별용 sample.
        top_k: 선정할 feature 수.

    Returns:
        bias-related feature 인덱스 리스트.
    """
    logger.warning(
        "[identify_bias_sae_features] 골격 함수. "
        "노트북에서 단계별 검증 후 구현하세요."
    )
    return []


# =============================================================
# Gemma용 LLM + SAE 로드
# =============================================================
def load_gemma_pipeline(config: GemmaConfig) -> tuple["LLMWrapper", Optional["SAEWrapper"]]:
    """
    Gemma-2-9B + Gemma Scope SAE를 로드합니다.

    Args:
        config: GemmaConfig.

    Returns:
        (llm_wrapper, sae_wrapper). SAE 로드 실패 시 sae는 None.
    """
    from src.utils.llm_utils import LLMWrapper
    from src.signals.sae_feature import SAEWrapper

    llm = LLMWrapper(
        model_name=config.model_name,
        dtype=config.dtype,
        device=config.device,
    )

    sae: Optional[SAEWrapper] = None
    try:
        sae = SAEWrapper(
            sae_repo=config.sae_repo,
            layer=config.sae_layer,
            device=str(llm.device),
        )
    except Exception as e:
        logger.warning(f"[Gemma Scope] SAE 로드 실패, s7 비활성화: {e}")

    return llm, sae


# =============================================================
# Gemma용 신호 추출
# =============================================================
def extract_signals_gemma(
    instances: list[dict],
    stage1_results: list[dict],
    config: GemmaConfig,
    llm: "LLMWrapper",
    sae: Optional["SAEWrapper"],
    output_path,
    primary_prompt: str = "vanilla",
    n_consistency_samples: int = 5,
):
    """
    Gemma용 7-signal 추출 (extract_signals_batch wrapper).

    Args:
        instances: BBQ instance 리스트.
        stage1_results: Gemma로 돌린 Stage 1 결과.
        config: GemmaConfig (bias_head_indices, bias_sae_features 사용).
        llm: Gemma LLMWrapper.
        sae: Gemma Scope SAE (없으면 s7=None).
        output_path: 결과 저장 경로.

    Returns:
        signal records 리스트.
    """
    from src.signals.extract_all import extract_signals_batch

    return extract_signals_batch(
        items=instances,
        stage1_results=stage1_results,
        llm=llm,
        sae=sae,
        output_path=output_path,
        n_consistency_samples=n_consistency_samples,
        bias_head_indices=config.bias_head_indices,
        bias_sae_features=config.bias_sae_features,
        primary_prompt=primary_prompt,
    )
