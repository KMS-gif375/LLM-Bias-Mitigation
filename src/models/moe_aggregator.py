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

from __future__ import annotations  # forward reference 허용

from dataclasses import dataclass    # MoEOutput 컨테이너
from typing import Optional

import torch                          # PyTorch tensor
import torch.nn as nn                 # 신경망 모듈
import torch.nn.functional as F       # softmax 등 함수형 ops


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
        embed_dim: int = 4096,    # question embedding 차원 (Llama hidden = 4096, all-MiniLM = 384)
        hidden_dim: int = 128,    # gating 중간 layer 차원 — 작게 (간단한 routing)
        num_experts: int = 4,     # K=4 (BBQ taxonomy 4 clusters)
    ) -> None:
        super().__init__()
        # Sequential: q_embed → hidden → num_experts logits (softmax 는 forward 에서)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),      # 4096 → 128
            nn.ReLU(),                              # 비선형성
            nn.Linear(hidden_dim, num_experts),    # 128 → 4 (각 expert 의 raw logit)
        )

    def forward(self, q_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_embed: (batch, embed_dim).

        Returns:
            (batch, num_experts) softmax 가중치 (행 합 = 1).
        """
        # softmax(dim=-1): 마지막 차원 (num_experts) 따라 정규화 → 각 row 합 = 1
        # 결과: 각 sample 의 4 expert 가중치 확률 분포
        return F.softmax(self.net(q_embed), dim=-1)


class ExpertMLP(nn.Module):
    """
    각 expert는 [signals | q_embed]를 받아 scalar logit을 출력합니다.

    구조: Linear(signal_dim + embed_dim, hidden) -> ReLU -> Dropout -> Linear(hidden, 1).
    """

    def __init__(
        self,
        signal_dim: int = 7,      # 7 신호 (s1~s7)
        embed_dim: int = 4096,    # question embedding 차원
        hidden_dim: int = 64,     # expert 중간 layer 차원
        dropout: float = 0.1,     # dropout — overfitting 방지
    ) -> None:
        super().__init__()
        # Expert 입력 = signals concat embedding (signal 만 쓰는 게 아니라 context 도 봄)
        in_dim = signal_dim + embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),   # (7+4096) → 64
            nn.ReLU(),                        # 비선형성
            nn.Dropout(dropout),              # 학습 시만 dropout 적용
            nn.Linear(hidden_dim, 1),         # 64 → 1 (scalar logit, sigmoid 적용 전)
        )

    def forward(self, signals: torch.Tensor, q_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signals: (batch, signal_dim).
            q_embed: (batch, embed_dim).

        Returns:
            (batch, 1) raw logit (sigmoid 적용 전).
        """
        # signals 와 embedding 을 가로로 concat → 한 expert 의 입력
        x = torch.cat([signals, q_embed], dim=-1)
        return self.net(x)


# =============================================================
# 2. MoEAggregator
# =============================================================
@dataclass
class MoEOutput:
    """forward 결과 컨테이너 (Loss 계산 시 추가 정보 포함)."""

    p: torch.Tensor                # (batch,) confidence — 최종 sigmoid 출력
    gate_w: torch.Tensor           # (batch, num_experts) — gating 가중치 (Load balance loss 용)
    expert_outs: torch.Tensor      # (batch, num_experts) raw logits — 각 expert 가 낸 점수
    normalized_signals: torch.Tensor  # (batch, signal_dim) — temperature 적용 후 신호


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
        # 하이퍼파라미터 저장 — 추론 시 입력 차원 검증용
        self.signal_dim = signal_dim
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.gating_hidden = gating_hidden
        self.expert_hidden = expert_hidden
        self.dropout = dropout

        # Per-signal learnable temperature (signal scaling) — 각 신호의 importance 자동 학습
        # nn.Parameter 라서 backward 시 gradient 받아 업데이트됨
        # 초기값 1.0 → 학습 끝나면 신호별로 0.5~2 등 다른 값으로 수렴
        # Per-signal learnable temperature (signal scaling)
        self.signal_temperature = nn.Parameter(
            torch.full((signal_dim,), signal_temperature_init, dtype=torch.float32)
        )

        # Gating network (q_embed → 4 weights)
        self.gating = GatingNetwork(
            embed_dim=embed_dim,
            hidden_dim=gating_hidden,
            num_experts=num_experts,
        )
        # 4 개의 expert MLP — nn.ModuleList 로 묶어야 PyTorch 가 자식 module 로 인식
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
        forced_gate: Optional[torch.Tensor] = None,  # cluster_ablation 에서 hard routing 강제용
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
        # 입력 차원 검증 — 자주 디버깅하던 부분이라 명시적으로 체크
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
        # signal_temperature 가 학습 가능 → 어떤 신호가 중요한지 자동 학습
        # 예: s6 (prompt-sens) 가 가장 중요하면 temperature[5] ≈ 2.0 으로 커짐
        norm_signals = signals * self.signal_temperature

        # 2. Gating — forced_gate가 주어지면 그 값을 사용, 아니면 학습된 gating
        # forced_gate: cluster_ablation 에서 taxonomy 기반 hard routing 강제 (예: Age → expert 1)
        if forced_gate is not None:
            if forced_gate.shape != (signals.shape[0], self.num_experts):
                raise ValueError(
                    f"forced_gate shape mismatch: expected "
                    f"({signals.shape[0]}, {self.num_experts}), got {tuple(forced_gate.shape)}"
                )
            # device + dtype 맞춤 (gating 결과와 호환되게)
            gate_w = forced_gate.to(dtype=q_embed.dtype, device=q_embed.device)
        else:
            # 일반 경우 — 학습된 gating network 사용
            gate_w = self.gating(q_embed)           # (B, K)

        # 3. Experts — 4 expert 각각 입력에 대해 scalar logit 출력
        # list comprehension 후 concat → (batch, K)
        # 각 expert.forward 가 (B, 1) 반환 → cat dim=-1 으로 K 개 합침
        expert_logits = torch.cat([
            expert(norm_signals, q_embed)
            for expert in self.experts
        ], dim=-1)                                   # (B, K)

        # 4. Soft routing — gating 가중치로 expert logits 평균
        # weighted_logit[b] = Σ_k gate_w[b, k] * expert_logits[b, k]
        # 결과는 (batch,) shape
        weighted_logit = (gate_w * expert_logits).sum(dim=-1)  # (B,)
        # sigmoid 로 [0, 1] 범위 확률 변환 — final confidence
        p = torch.sigmoid(weighted_logit)

        # 모든 중간 결과 함께 return — Loss 계산 시 load balance 등에 사용
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
    # log(0) 방지용 clamp — p 가 정확히 0/1 이면 -inf 발생
    p_clamped = p.clamp(min=eps, max=1 - eps)
    # 표준 BCE: -[y log p + (1-y) log(1-p)]
    return -(label * torch.log(p_clamped) + (1 - label) * torch.log(1 - p_clamped)).mean()


def bias_penalty(
    p: torch.Tensor,
    is_ambig: torch.Tensor,           # (batch,) 모호 맥락 여부 1/0
    is_stereotype: torch.Tensor,      # (batch,) stereotype 방향 답 여부 1/0
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
    # AND mask: ambig 이면서 stereotype 답한 instance 만 페널티
    # 다른 instance (정답이거나 anti-stereo) 는 영향 없음
    mask = is_ambig * is_stereotype                  # (B,) 1 if ambig & stereotyped
    # mask 가 모두 0 이면 (해당 batch 에 bias-slip 없음) → loss = 0
    if mask.sum() == 0:
        return torch.zeros((), device=p.device, dtype=p.dtype)

    # -log(1 - p) for masked instances
    # p → 1 일수록 loss → ∞, p → 0 일수록 loss → 0
    # 즉 stereotype 답한 instance 의 p 를 낮추도록 학습 → override 가 작동하게
    p_clamped = p.clamp(min=eps, max=1 - eps)
    loss_per_sample = -torch.log(1 - p_clamped)
    # masked 만 평균 — clamp(min=1) 은 mask.sum()=0 division 방어 (위에서 이미 처리됐지만 안전)
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
    # 각 expert 의 평균 사용량 (batch 차원 평균)
    importance = gate_w.mean(dim=0)                  # (K,)
    # 이상적 분포: 각 expert 가 1/K 비율로 동등 사용
    target = 1.0 / num_experts
    # MSE 형태 — importance 가 target 에서 멀어질수록 loss 증가
    # K 곱하는 이유: K 가 커도 절댓값 비교 가능하게 정규화
    return num_experts * ((importance - target) ** 2).sum()


def total_loss(
    output: MoEOutput,
    label: torch.Tensor,
    is_ambig: torch.Tensor,
    is_stereotype: torch.Tensor,
    lambda_bias: float = 0.5,    # bias penalty 가중치 (config 에서 조정 가능)
    lambda_lb: float = 0.1,      # load balance 가중치
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
    # 1. 정답 맞힘 loss (주된 학습 signal)
    bce = bce_loss(output.p, label)
    # 2. ambig + stereotype 답 페널티 (override 작동하도록 p 낮추기)
    bias = bias_penalty(output.p, is_ambig, is_stereotype)
    # 3. expert 균등 사용 강제 (cluster collapse 방지)
    lb = load_balance_loss(output.gate_w)
    # 합산 — 디버깅용으로 component 별 반환
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
    # dict → list (keys 순서 고정 — 학습/추론 일관성 필수)
    values = [signals.get(k) for k in keys]
    # None 처리 (Qwen 의 s7 등 — SAE 가 없는 모델)
    values = [fill_none if v is None else float(v) for v in values]
    # float32 tensor 변환
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
    # eval 모드 — dropout / batch norm 등 무력화
    model.eval()
    # device 자동 추론 — model 첫 parameter 의 device 사용
    if device is None:
        device = next(model.parameters()).device

    # inference_mode: no_grad 보다 더 빠르고 메모리 효율적
    with torch.inference_mode():
        # 단일 instance → (1, ...) batch shape 으로 변환
        s = signals.unsqueeze(0).to(device)
        e = q_embed.unsqueeze(0).to(device)
        out = model(s, e)
    # tensor → Python float
    return out.p.item()
