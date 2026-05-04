"""
Stacking Ablation: Baseline + 7-Signal Pipeline.

baseline의 출력을 우리 4-Stage 파이프라인의 Stage 1 결과로 대체하고,
나머지 Stage 2-4 (signal extraction → MoE → threshold override)를 적용하여
"baseline 단독" vs "baseline + 우리 method"의 추가 이득을 측정합니다.

흐름:
    [Baseline] instances -> primary_answers (str)
    [Stage 2]  primary_answers + instances -> 7 signals
    [Stage 3]  signals + q_embeddings -> MoE -> p
    [Stage 4]  p + threshold -> final_answer (override 적용)

비교:
    - baseline 단독: primary_answers
    - baseline + ours: final_answers (override 후)
    - bootstrap_ci로 통계적 유의성 검증

본 모듈은 stage-by-stage 결합 함수와 평가 함수를 제공합니다.
실제 신호 추출, MoE 추론은 외부에서 주입받습니다 (의존 역전).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch
from tqdm import tqdm

from src.evaluation.bbq_evaluator import evaluate_bbq, parse_prediction
from src.evaluation.bootstrap_ci import metric_for, paired_bootstrap_pvalue
from src.models.moe_aggregator import predict_p, signals_dict_to_tensor
from src.models.override import apply_threshold_override

if TYPE_CHECKING:
    from src.models.moe_aggregator import MoEAggregator

logger = logging.getLogger(__name__)


# 신호 추출기 인터페이스
# (instance, primary_answer_idx) -> {"s1_evidence": float, ..., "s7_sae_feature": float | None}
SignalExtractor = Callable[[dict, int], dict]

# Embedding 추출기 인터페이스
# instance -> torch.Tensor (embed_dim,)
EmbeddingExtractor = Callable[[dict], torch.Tensor]


# =============================================================
# 1. Baseline 답을 우리 pipeline으로 보강
# =============================================================
@dataclass
class StackingResult:
    """단일 instance의 stacking 결과."""

    primary_answer: int           # baseline 출력 (parse 후)
    p_score: float                # MoE 신뢰도
    final_answer: int             # threshold override 후
    overridden: bool              # override 발생 여부
    signals: dict                 # 추출된 7개 신호


def stack_one(
    instance: dict,
    baseline_answer: str,
    moe_model: "MoEAggregator",
    signal_extractor: SignalExtractor,
    embedding_extractor: EmbeddingExtractor,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> StackingResult:
    """
    하나의 instance에 대해 baseline 답에 우리 pipeline을 적용합니다.

    Args:
        instance: BBQ instance.
        baseline_answer: baseline의 raw 답변 텍스트.
        moe_model: 학습된 MoEAggregator.
        signal_extractor: (instance, primary_answer_idx) -> signals dict.
        embedding_extractor: instance -> embedding 텐서.
        threshold: override 임계값.
        device: torch device.

    Returns:
        StackingResult.
    """
    primary_idx = parse_prediction(baseline_answer)

    # 1. Stage 2: signal 추출 (baseline 답을 primary_answer로 사용)
    signals = signal_extractor(instance, primary_idx)

    # 2. Stage 3: MoE 추론
    sig_tensor = signals_dict_to_tensor(signals)
    embed = embedding_extractor(instance)
    p = predict_p(moe_model, sig_tensor, embed, device=device)

    # 3. Stage 4: Threshold override
    override = apply_threshold_override(
        primary_answer=primary_idx,
        p_score=p,
        item=instance,
        threshold=threshold,
    )

    return StackingResult(
        primary_answer=primary_idx,
        p_score=p,
        final_answer=override["final_answer"],
        overridden=override["overridden"],
        signals=signals,
    )


def stack_baseline_with_pipeline(
    instances: list[dict],
    baseline_answers: list[str],
    moe_model: "MoEAggregator",
    signal_extractor: SignalExtractor,
    embedding_extractor: EmbeddingExtractor,
    threshold: float = 0.5,
    show_progress: bool = True,
) -> list[StackingResult]:
    """
    배치로 stacking을 수행합니다.

    Args:
        instances: BBQ instance 리스트.
        baseline_answers: 같은 길이의 baseline 답변 리스트.
        moe_model: 학습된 MoE.
        signal_extractor / embedding_extractor: 신호/임베딩 추출 함수.
        threshold: override 임계값.
        show_progress: tqdm.

    Returns:
        StackingResult 리스트.

    Raises:
        ValueError: 길이 불일치.
    """
    if len(instances) != len(baseline_answers):
        raise ValueError(
            f"길이 불일치: instances={len(instances)}, "
            f"baseline_answers={len(baseline_answers)}"
        )

    moe_model.eval()
    device = next(moe_model.parameters()).device

    results: list[StackingResult] = []
    iterator = zip(instances, baseline_answers)
    if show_progress:
        iterator = tqdm(iterator, total=len(instances), desc="Stacking")

    for inst, ans in iterator:
        results.append(stack_one(
            instance=inst,
            baseline_answer=ans,
            moe_model=moe_model,
            signal_extractor=signal_extractor,
            embedding_extractor=embedding_extractor,
            threshold=threshold,
            device=device,
        ))

    return results


# =============================================================
# 2. Baseline alone vs Baseline + Ours 비교
# =============================================================
@dataclass
class StackingComparison:
    """baseline alone vs baseline+ours 비교 결과."""

    baseline_metrics: dict[str, float]
    stacked_metrics: dict[str, float]
    delta: dict[str, float]                 # stacked - baseline
    p_values: dict[str, float]              # paired bootstrap p-value
    n_overridden: int
    override_rate: float


def compare_baseline_vs_stacked(
    instances: list[dict],
    baseline_answers: list[str],
    stacking_results: list[StackingResult],
    metrics: tuple[str, ...] = (
        "accuracy_amb",
        "accuracy_dis",
        "bias_score_amb",
        "false_abstention_rate",
    ),
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> StackingComparison:
    """
    Baseline 단독 결과와 stacking 결과의 metric 차이를 비교합니다.

    Args:
        instances: BBQ instance.
        baseline_answers: baseline raw 답변.
        stacking_results: stack_baseline_with_pipeline 결과.
        metrics: 비교할 metric 이름.
        n_bootstrap: paired bootstrap 횟수.
        seed: 랜덤 시드.

    Returns:
        StackingComparison.
    """
    if len(instances) != len(baseline_answers) or len(instances) != len(stacking_results):
        raise ValueError("길이 불일치")

    baseline_preds = [str(a) for a in baseline_answers]
    stacked_preds = [r.final_answer for r in stacking_results]

    # 1. metric 계산
    baseline_metrics = evaluate_bbq(baseline_preds, instances)
    stacked_metrics = evaluate_bbq([str(p) for p in stacked_preds], instances)

    # 2. delta
    delta = {}
    for m in metrics:
        b = baseline_metrics.get(m)
        s = stacked_metrics.get(m)
        if b is not None and s is not None:
            delta[m] = s - b

    # 3. paired bootstrap p-value (각 metric에 대해)
    p_values: dict[str, float] = {}
    for m in metrics:
        try:
            metric_fn = metric_for(m)
        except ValueError:
            continue
        # bias_score는 절대값이 작아져야 좋음 → direction "less"로 검정
        # accuracy/false_abstention 등은 상황에 따라 다름; 모두 두 측 검정으로
        result = paired_bootstrap_pvalue(
            predictions_a=stacked_preds,
            predictions_b=baseline_preds,
            instances=instances,
            metric_fn=metric_fn,
            n_iterations=n_bootstrap,
            seed=seed,
            direction="two_sided",
        )
        p_values[m] = result["p_value"]

    n_overridden = sum(1 for r in stacking_results if r.overridden)
    override_rate = n_overridden / len(stacking_results) if stacking_results else 0.0

    return StackingComparison(
        baseline_metrics=baseline_metrics,
        stacked_metrics=stacked_metrics,
        delta=delta,
        p_values=p_values,
        n_overridden=n_overridden,
        override_rate=override_rate,
    )


# =============================================================
# 3. 결과 포맷팅 헬퍼
# =============================================================
def format_comparison_table(comp: StackingComparison) -> str:
    """비교 결과를 사람이 읽기 좋은 표로 포맷."""
    lines = [
        "=" * 76,
        f"{'Metric':<25} {'Baseline':>12} {'Stacked':>12} {'Delta':>10} {'p-value':>10}",
        "-" * 76,
    ]
    for m in comp.delta:
        b = comp.baseline_metrics.get(m, 0.0) or 0.0
        s = comp.stacked_metrics.get(m, 0.0) or 0.0
        d = comp.delta[m]
        p = comp.p_values.get(m, float("nan"))
        lines.append(
            f"{m:<25} {b:>12.4f} {s:>12.4f} {d:>+10.4f} {p:>10.4f}"
        )
    lines += [
        "-" * 76,
        f"Override 발생: {comp.n_overridden} / {comp.n_overridden + (len(comp.delta) and 0)} "
        f"({comp.override_rate * 100:.1f}%)",
        "=" * 76,
    ]
    return "\n".join(lines)
