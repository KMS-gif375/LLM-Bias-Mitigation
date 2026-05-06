"""
Stage 3: MoE Aggregator

7개 신호와 question embedding(LLM hidden state)을 입력받아
"정답 신뢰도" p ∈ [0, 1]을 출력합니다.

구조:
    [q_embed (4096)] -> Gating Network -> 4 cluster weights (softmax)
    [signals (7) | q_embed] -> 4 Expert MLPs -> 각 (batch, 1)
    p = sigmoid( sum_k gate_w[k] * expert_out[k] )

Cluster:
    1. Lexically-Substitutable (Gender, Religion)
    2. Numerically-Verifiable (Age, SES)
    3. Cultural-Contextual (Race)
    4. Identity-Sensitive (Disability, Sexual_orientation)

Loss = BCE(p, label) + lambda_bias * BiasPenalty + lambda_lb * LoadBalance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================
# 1. Components
# =============================================================
class GatingNetwork(nn.Module):
    """
    Question embedding을 받아 4개 cluster의 가중치를 출력합니다.

    구조: Linear(embed_dim, hidden) -> ReLU -> Linear(hidden, num_experts) -> softmax.
    """

    def __init__(
        self,
        embed_dim: int = 4096,
        hidden_dim: int = 128,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, q_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_embed: (batch, embed_dim).

        Returns:
            (batch, num_experts) softmax 가중치 (행 합 = 1).
        """
        return F.softmax(self.net(q_embed), dim=-1)


class ExpertMLP(nn.Module):
    """
    각 expert는 [signals | q_embed]를 받아 scalar logit을 출력합니다.

    구조: Linear(signal_dim + embed_dim, hidden) -> ReLU -> Dropout -> Linear(hidden, 1).
    """

    def __init__(
        self,
        signal_dim: int = 7,
        embed_dim: int = 4096,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = signal_dim + embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, signals: torch.Tensor, q_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signals: (batch, signal_dim).
            q_embed: (batch, embed_dim).

        Returns:
            (batch, 1) raw logit (sigmoid 적용 전).
        """
        x = torch.cat([signals, q_embed], dim=-1)
        return self.net(x)


# =============================================================
# 2. MoEAggregator
# =============================================================
@dataclass
class MoEOutput:
    """forward 결과 컨테이너 (Loss 계산 시 추가 정보 포함)."""

    p: torch.Tensor                # (batch,) confidence
    gate_w: torch.Tensor           # (batch, num_experts)
    expert_outs: torch.Tensor      # (batch, num_experts) raw logits
    normalized_signals: torch.Tensor  # (batch, signal_dim)


class MoEAggregator(nn.Module):
    """
    Mixture of Experts aggregator.

    Args:
        signal_dim: 신호 개수 (기본 7).
        embed_dim: question embedding 차원 (Llama-3.1-8B의 4096).
        num_experts: cluster 수 (기본 4).
        gating_hidden: gating network hidden 차원.
        expert_hidden: expert MLP hidden 차원.
        dropout: expert dropout 확률.
        signal_temperature_init: per-signal scaling 초기값.
    """

    def __init__(
        self,
        signal_dim: int = 7,
        embed_dim: int = 4096,
        num_experts: int = 4,
        gating_hidden: int = 64,    # gating은 작아도 충분 (q_embed → K logits)
        expert_hidden: int = 128,   # expert는 더 큰 capacity (signals + q_embed → 1 logit)
        dropout: float = 0.1,
        signal_temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.signal_dim = signal_dim
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.gating_hidden = gating_hidden
        self.expert_hidden = expert_hidden
        self.dropout = dropout

        # Per-signal learnable temperature (signal scaling)
        self.signal_temperature = nn.Parameter(
            torch.full((signal_dim,), signal_temperature_init, dtype=torch.float32)
        )

        self.gating = GatingNetwork(
            embed_dim=embed_dim,
            hidden_dim=gating_hidden,
            num_experts=num_experts,
        )
        self.experts = nn.ModuleList([
            ExpertMLP(
                signal_dim=signal_dim,
                embed_dim=embed_dim,
                hidden_dim=expert_hidden,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        signals: torch.Tensor,
        q_embed: torch.Tensor,
        forced_gate: Optional[torch.Tensor] = None,
    ) -> MoEOutput:
        """
        Args:
            signals: (batch, signal_dim) — 7개 신호 벡터.
            q_embed: (batch, embed_dim) — question embedding.
            forced_gate: (batch, num_experts) — taxonomy 기반 hard/soft routing
                강제용. 주어지면 학습된 gating 대신 이 값을 사용. cluster_ablation
                에서 taxonomy 효과를 의미 있게 측정하기 위한 옵션 (없으면 일반
                soft-MoE).

        Returns:
            MoEOutput.

        Raises:
            ValueError: 입력 차원 불일치.
        """
        if signals.dim() != 2 or signals.shape[1] != self.signal_dim:
            raise ValueError(
                f"signals shape mismatch: expected (B, {self.signal_dim}), "
                f"got {tuple(signals.shape)}"
            )
        if q_embed.dim() != 2 or q_embed.shape[1] != self.embed_dim:
            raise ValueError(
                f"q_embed shape mismatch: expected (B, {self.embed_dim}), "
                f"got {tuple(q_embed.shape)}"
            )

        # 1. 신호 정규화 (per-signal learnable scaling)
        norm_signals = signals * self.signal_temperature

        # 2. Gating — forced_gate가 주어지면 그 값을 사용, 아니면 학습된 gating
        if forced_gate is not None:
            if forced_gate.shape != (signals.shape[0], self.num_experts):
                raise ValueError(
                    f"forced_gate shape mismatch: expected "
                    f"({signals.shape[0]}, {self.num_experts}), got {tuple(forced_gate.shape)}"
                )
            gate_w = forced_gate.to(dtype=q_embed.dtype, device=q_embed.device)
        else:
            gate_w = self.gating(q_embed)           # (B, K)

        # 3. Experts
        expert_logits = torch.cat([
            expert(norm_signals, q_embed)
            for expert in self.experts
        ], dim=-1)                                   # (B, K)

        # 4. Soft routing
        weighted_logit = (gate_w * expert_logits).sum(dim=-1)  # (B,)
        p = torch.sigmoid(weighted_logit)

        return MoEOutput(
            p=p,
            gate_w=gate_w,
            expert_outs=expert_logits,
            normalized_signals=norm_signals,
        )


# =============================================================
# 3. Loss Functions
# =============================================================
def bce_loss(p: torch.Tensor, label: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Binary cross-entropy on confidence p vs label (1=correct, 0=incorrect).
    """
    p_clamped = p.clamp(min=eps, max=1 - eps)
    return -(label * torch.log(p_clamped) + (1 - label) * torch.log(1 - p_clamped)).mean()


def bias_penalty(
    p: torch.Tensor,
    is_ambig: torch.Tensor,
    is_stereotype: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    모호 맥락에서 stereotype 답을 선택한 instance에 대해 p가 작아지도록 페널티.

    이러한 instance는 threshold override로 unknown 처리되어야 하므로,
    p_target = 0이 이상적입니다.

    Args:
        p: (batch,) MoE 출력 confidence.
        is_ambig: (batch,) 모호 맥락이면 1.
        is_stereotype: (batch,) 모델 답이 stereotype 방향이면 1.
        eps: 안정성용.

    Returns:
        scalar loss.
    """
    mask = is_ambig * is_stereotype                  # (B,) 1 if ambig & stereotyped
    if mask.sum() == 0:
        return torch.zeros((), device=p.device, dtype=p.dtype)

    # -log(1 - p) for masked instances
    p_clamped = p.clamp(min=eps, max=1 - eps)
    loss_per_sample = -torch.log(1 - p_clamped)
    return (loss_per_sample * mask).sum() / mask.sum().clamp(min=1)


def load_balance_loss(gate_w: torch.Tensor) -> torch.Tensor:
    """
    Expert collapse 방지를 위한 load balancing loss.

    - importance[k] = gate_w[:, k].mean() (배치 평균 사용량)
    - target = 1/K (균등 사용)
    - loss = K * sum_k (importance[k] - 1/K)^2
        = CV^2 * K   (계수 분산 제곱과 비례)

    Returns:
        scalar loss (낮을수록 expert가 균등하게 사용됨).
    """
    if gate_w.dim() != 2:
        raise ValueError(f"gate_w must be (B, K), got {gate_w.shape}")
    num_experts = gate_w.shape[1]
    importance = gate_w.mean(dim=0)                  # (K,)
    target = 1.0 / num_experts
    return num_experts * ((importance - target) ** 2).sum()


def total_loss(
    output: MoEOutput,
    label: torch.Tensor,
    is_ambig: torch.Tensor,
    is_stereotype: torch.Tensor,
    lambda_bias: float = 0.5,
    lambda_lb: float = 0.1,
) -> dict[str, torch.Tensor]:
    """
    3가지 loss의 가중합을 계산합니다.

    Args:
        output: MoEAggregator의 forward 출력.
        label: (batch,) 1=정답, 0=오답.
        is_ambig: (batch,) 1=모호 맥락.
        is_stereotype: (batch,) 1=stereotype 방향 답.
        lambda_bias: bias penalty 가중치.
        lambda_lb: load balance 가중치 (cluster collapse 방지).

    Returns:
        {"total": ..., "bce": ..., "bias": ..., "lb": ...}.
    """
    bce = bce_loss(output.p, label)
    bias = bias_penalty(output.p, is_ambig, is_stereotype)
    lb = load_balance_loss(output.gate_w)
    total = bce + lambda_bias * bias + lambda_lb * lb
    return {"total": total, "bce": bce, "bias": bias, "lb": lb}


# =============================================================
# 4. Helpers
# =============================================================
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

    Qwen 등 SAE 미지원 모델의 경우 s7이 None → fill_none으로 채워집니다.
    """
    values = [signals.get(k) for k in keys]
    values = [fill_none if v is None else float(v) for v in values]
    return torch.tensor(values, dtype=torch.float32)


def predict_p(
    model: MoEAggregator,
    signals: torch.Tensor,
    q_embed: torch.Tensor,
    device: Optional[torch.device] = None,
) -> float:
    """
    단일 instance에 대한 추론 (eval 모드).

    Args:
        model: 학습된 MoEAggregator.
        signals: (signal_dim,) 신호 벡터.
        q_embed: (embed_dim,) embedding.
        device: torch device.

    Returns:
        p ∈ [0, 1].
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    with torch.inference_mode():
        s = signals.unsqueeze(0).to(device)
        e = q_embed.unsqueeze(0).to(device)
        out = model(s, e)
    return out.p.item()
