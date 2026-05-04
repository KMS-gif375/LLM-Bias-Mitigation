"""
Stage 3: MoE Aggregator

7개 신호와 question embedding을 입력받아 "정답 확신도" p ∈ [0,1]을 출력합니다.

구조:
    [Question Embedding (384)] -> Gating Network -> 4 Cluster Weights (softmax)
    [7 Signals] ----+
                    |-> 4 Expert MLPs (각각 7->1)
                    +
                    -> Soft Routing (가중평균) -> p

Cluster 정의:
    1. Lexically-Substitutable: Gender, Religion (단어 substitution이 swap에 충분)
    2. Numerically-Verifiable: Age, SES (숫자/명시적 정보로 검증 가능)
    3. Cultural-Contextual: Race (문화적 맥락 의존)
    4. Identity-Sensitive: Disability, Sexual_orientation (정체성 민감)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """
    Question embedding을 받아 4개 cluster의 가중치를 출력합니다.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 64,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (batch, embed_dim) question embedding.

        Returns:
            (batch, num_experts) softmax 가중치.
        """
        return F.softmax(self.net(embedding), dim=-1)


class ExpertMLP(nn.Module):
    """
    7개 신호를 받아 하나의 확신도 점수를 출력합니다.
    """

    def __init__(
        self,
        signal_dim: int = 7,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signals: (batch, signal_dim) signal vector.

        Returns:
            (batch, 1) raw logit (sigmoid 적용 전).
        """
        return self.net(signals)


class MoEAggregator(nn.Module):
    """
    MoE 통합 모델: gating + 4 experts -> soft routing -> sigmoid.
    """

    def __init__(
        self,
        signal_dim: int = 7,
        embed_dim: int = 384,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        gating_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts

        self.gating = GatingNetwork(
            embed_dim=embed_dim,
            hidden_dim=gating_hidden_dim,
            num_experts=num_experts,
        )
        self.experts = nn.ModuleList([
            ExpertMLP(signal_dim=signal_dim, hidden_dim=expert_hidden_dim)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        signals: torch.Tensor,
        embedding: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            signals: (batch, signal_dim) 7-signal vector.
            embedding: (batch, embed_dim) question embedding.
            return_gate: True면 gating weight도 반환.

        Returns:
            (batch,) p ∈ [0, 1] 확신도.
            return_gate=True면 (p, gate_weights).
        """
        gate_weights = self.gating(embedding)  # (batch, num_experts)

        expert_outs = torch.cat([
            expert(signals) for expert in self.experts
        ], dim=-1)  # (batch, num_experts)

        # Soft routing: sum(weight * expert_out)
        weighted = (gate_weights * expert_outs).sum(dim=-1)  # (batch,)
        p = torch.sigmoid(weighted)

        if return_gate:
            return p, gate_weights
        return p


def signals_dict_to_tensor(
    signals: dict,
    keys: tuple[str, ...] = (
        "s1_evidence",
        "s2_counterfactual",
        "s3_confidence",
        "s4_consistency",
        "s5_bias_head",
        "s6_prompt_sensitivity",
        "s7_sae_feature",
    ),
    fill_none: float = 0.0,
) -> torch.Tensor:
    """
    신호 딕셔너리를 고정 순서의 텐서로 변환합니다.

    Args:
        signals: extract_signals_for_item의 "signals" 필드.
        keys: 사용할 신호 키 (순서 고정).
        fill_none: None 값 (SAE 미지원)을 채울 값.

    Returns:
        (signal_dim,) 텐서.
    """
    values = [signals.get(k) for k in keys]
    values = [fill_none if v is None else float(v) for v in values]
    return torch.tensor(values, dtype=torch.float32)
