"""
Cross-LLM Transfer: Qwen-2.5-7B-Instruct pipeline.

Qwen은 SAE가 공개되지 않았으므로 s7 (SAE Feature Activation)을 사용할 수 없습니다.
두 가지 운영 방식 중 선택:

    Option A: 6-signal 전용 MoE
        - signal_dim=6 으로 별도 MoE 학습
        - 가장 직접적이지만 Llama 모델로의 transfer가 깨짐

    Option B: 0-padding (권장)
        - 7-signal MoE를 그대로 사용하되 s7 자리에 0 (또는 학습된 default) 채움
        - signal_temperature가 학습된 0의 영향을 자연스럽게 줄임
        - Llama 학습 시스템 그대로 transfer 가능 → 본 연구의 핵심 가설 검증

기본 정책: Option B (0-padding).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.utils.llm_utils import LLMWrapper

logger = logging.getLogger(__name__)


# =============================================================
# Config
# =============================================================
@dataclass
class QwenConfig:
    """Qwen-2.5-7B 전용 설정."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "bfloat16"
    device: str = "auto"

    # Bias-head (Qwen용 재식별 결과)
    bias_head_indices: list[tuple[int, int]] = ()

    # SAE 미지원 → s7 운영 방식
    s7_strategy: str = "zero_padding"     # "zero_padding" | "drop"
    s7_default_value: float = 0.0


# =============================================================
# Load Qwen
# =============================================================
def load_qwen_pipeline(config: QwenConfig) -> "LLMWrapper":
    """
    Qwen-2.5-7B를 로드합니다 (SAE 없음).

    Args:
        config: QwenConfig.

    Returns:
        LLMWrapper.
    """
    from src.utils.llm_utils import LLMWrapper

    return LLMWrapper(
        model_name=config.model_name,
        dtype=config.dtype,
        device=config.device,
    )


# =============================================================
# Qwen용 신호 추출 (s7 = None)
# =============================================================
def extract_signals_qwen(
    instances: list[dict],
    stage1_results: list[dict],
    config: QwenConfig,
    llm: "LLMWrapper",
    output_path,
    primary_prompt: str = "vanilla",
    n_consistency_samples: int = 5,
):
    """
    Qwen용 6-signal 추출 (sae=None으로 s7 자동 None).

    Args:
        instances: BBQ instance.
        stage1_results: Stage 1 결과.
        config: QwenConfig.
        llm: Qwen LLMWrapper.
        output_path: 저장 경로.

    Returns:
        signal records 리스트 (s7_sae_feature는 None).
    """
    from src.signals.extract_all import extract_signals_batch

    return extract_signals_batch(
        items=instances,
        stage1_results=stage1_results,
        llm=llm,
        sae=None,                                  # SAE 없음 → s7 자동 None
        output_path=output_path,
        n_consistency_samples=n_consistency_samples,
        bias_head_indices=list(config.bias_head_indices),
        bias_sae_features=[],
        primary_prompt=primary_prompt,
    )


# =============================================================
# 6-signal MoE 운영 (Option A)
# =============================================================
def make_six_signal_moe(
    embed_dim: int = 384,
    num_experts: int = 4,
    gating_hidden: int = 64,    # default.yaml과 통일
    expert_hidden: int = 128,
    dropout: float = 0.1,
):
    """
    6-signal 전용 MoE (s7 제외).

    Note:
        signal 순서: s1, s2, s3, s4, s5, s6 (s7 제외).
        signals_dict_to_tensor 호출 시 keys를 명시해야 합니다.

    Args:
        embed_dim: question embedding 차원.
        num_experts: cluster 수.
        gating_hidden / expert_hidden / dropout: MoE 하이퍼파라미터.

    Returns:
        MoEAggregator (signal_dim=6).
    """
    from src.models.moe_aggregator import MoEAggregator

    return MoEAggregator(
        signal_dim=6,
        embed_dim=embed_dim,
        num_experts=num_experts,
        gating_hidden=gating_hidden,
        expert_hidden=expert_hidden,
        dropout=dropout,
    )


SIX_SIGNAL_KEYS: tuple[str, ...] = (
    "s1_evidence",
    "s2_counterfactual",
    "s3_confidence",
    "s4_consistency",
    "s5_bias_head",
    "s6_prompt_sensitivity",
)


def signals_dict_to_six_signal_tensor(signals: dict, fill_none: float = 0.0):
    """
    signals dict → (6,) tensor (s7 제외).
    """
    from src.models.moe_aggregator import signals_dict_to_tensor

    return signals_dict_to_tensor(signals, keys=SIX_SIGNAL_KEYS, fill_none=fill_none)
