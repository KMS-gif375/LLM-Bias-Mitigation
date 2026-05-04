"""
11.6 Qualitative Analysis

논문 부록과 본문 case study를 위한 정성 분석:

    1. top_sae_max_activating_examples()
       — 각 Top-K bias SAE feature에 대해, 가장 강하게 활성된 BBQ 샘플 추출.
       (feature가 어떤 demographic context에 반응하는지 인간이 해석 가능)

    2. top_bias_head_attention_examples()
       — Top bias-head의 attention map을 demographic token에 대해 시각화 데이터로 제공.

    3. failure_cases()
       — 모델이 override 후에도 여전히 틀린 샘플을 카테고리별로 분류.
       — over-correction (정답을 unknown으로 가린 경우)
       — under-correction (편향된 답을 그대로 둔 경우)

각 함수는 "데이터" 위주로 반환합니다. 시각화는 visualization.py에서 별도 처리.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.evaluation.bbq_evaluator import (
    is_stereotyped_answer,
    parse_prediction,
)
from src.models.override import find_unknown_index

logger = logging.getLogger(__name__)


# =============================================================
# 1. SAE feature max-activating examples
# =============================================================
@dataclass
class FeatureExample:
    """단일 max-activating sample."""

    example_id: str
    activation: float
    category: str
    context_short: str
    question: str
    primary_answer: int
    label: int


def top_sae_max_activating_examples(
    activations: np.ndarray,
    feature_indices: Sequence[int],
    instances_by_id: dict[str, dict],
    example_ids: Sequence[str],
    top_n: int = 10,
    context_max_chars: int = 200,
) -> dict[int, list[FeatureExample]]:
    """
    각 feature에 대해 활성도가 가장 높은 top_n개의 BBQ 샘플 추출.

    Args:
        activations: (n_samples, n_features) feature activation matrix.
        feature_indices: 분석할 feature 인덱스 리스트.
        instances_by_id: {example_id: BBQ instance dict}.
        example_ids: activations의 row 순서에 대응되는 example_id (길이 n_samples).
        top_n: feature당 상위 몇 개 sample 선택.
        context_max_chars: context 미리보기 글자 수.

    Returns:
        {feature_index: [FeatureExample, ...]}.
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D, got {activations.shape}")
    if len(example_ids) != activations.shape[0]:
        raise ValueError("example_ids 길이와 activations 행 수 불일치")

    out: dict[int, list[FeatureExample]] = {}
    for fidx in feature_indices:
        if fidx >= activations.shape[1]:
            logger.warning(f"  [SAE] feature index {fidx} 범위 초과 — skip")
            continue
        col = activations[:, fidx]
        top_rows = np.argsort(col)[::-1][:top_n]

        examples: list[FeatureExample] = []
        for row in top_rows:
            ex_id = example_ids[row]
            inst = instances_by_id.get(ex_id, {})
            ctx = (inst.get("context") or "")[:context_max_chars]
            examples.append(FeatureExample(
                example_id=ex_id,
                activation=float(col[row]),
                category=inst.get("category", "unknown"),
                context_short=ctx,
                question=inst.get("question", ""),
                primary_answer=int(inst.get("_primary_answer", -1)),
                label=int(inst.get("label", -1)),
            ))
        out[int(fidx)] = examples
    return out


# =============================================================
# 2. Bias-head attention examples
# =============================================================
@dataclass
class BiasHeadExample:
    """Bias-head attention 시각화 데이터."""

    example_id: str
    layer: int
    head: int
    tokens: list[str]
    attention_to_demographic: list[float]
    bias_score: float                           # avg attention to demographic tokens


def top_bias_head_attention_examples(
    head_attention_records: list[dict],
    layer: int,
    head: int,
    top_n: int = 10,
) -> list[BiasHeadExample]:
    """
    특정 (layer, head)에서 demographic-token attention이 가장 큰 sample 추출.

    Args:
        head_attention_records: 각 sample에 대해 다음 키를 가진 dict 리스트.
            - example_id: str
            - tokens: list[str]
            - attention_per_layer_head: dict[(layer, head)] -> dict
                - "demographic_indices": list[int]
                - "attention_to_demographic": list[float] (각 token의 attention)
        layer / head: 분석할 layer/head.
        top_n: 상위 몇 개 sample.

    Returns:
        list[BiasHeadExample] (bias_score 내림차순).
    """
    extracted: list[BiasHeadExample] = []
    key = (layer, head)
    for rec in head_attention_records:
        attn_dict = rec.get("attention_per_layer_head", {})
        att = attn_dict.get(key) or attn_dict.get(f"{layer}_{head}")
        if not att:
            continue

        att_to_demo = att.get("attention_to_demographic", [])
        if not att_to_demo:
            continue

        bias_score = float(np.mean(att_to_demo))
        extracted.append(BiasHeadExample(
            example_id=rec["example_id"],
            layer=layer,
            head=head,
            tokens=list(rec.get("tokens", [])),
            attention_to_demographic=list(att_to_demo),
            bias_score=bias_score,
        ))

    extracted.sort(key=lambda e: e.bias_score, reverse=True)
    return extracted[:top_n]


# =============================================================
# 3. Failure case analysis
# =============================================================
@dataclass
class FailureCase:
    example_id: str
    category: str
    context_condition: str        # "ambig" | "disambig"
    question_polarity: str        # "neg" | "nonneg"
    primary_answer: int
    final_answer: int
    label: int
    p_score: float
    failure_type: str             # "over_correction" | "under_correction" | "wrong_kept"
    stereotype_kind: Optional[str] = None
    context_short: str = ""


@dataclass
class FailureAnalysis:
    over_correction: list[FailureCase] = field(default_factory=list)
    under_correction: list[FailureCase] = field(default_factory=list)
    wrong_kept: list[FailureCase] = field(default_factory=list)

    def by_category(self) -> dict[str, dict[str, int]]:
        """카테고리별 failure 종류별 카운트."""
        out: dict[str, dict[str, int]] = {}
        for kind, cases in (
            ("over_correction", self.over_correction),
            ("under_correction", self.under_correction),
            ("wrong_kept", self.wrong_kept),
        ):
            for c in cases:
                out.setdefault(c.category, {"over_correction": 0,
                                             "under_correction": 0,
                                             "wrong_kept": 0})
                out[c.category][kind] += 1
        return out


def failure_cases(
    val_predictions: list[dict],
    threshold: float = 0.5,
    top_n_per_type: int = 20,
    context_max_chars: int = 200,
) -> FailureAnalysis:
    """
    Override 적용 후에도 틀린 샘플을 분류합니다.

    분류 정의:
        - over_correction: primary가 정답이었는데 override가 unknown으로 바꿔서 틀림.
        - under_correction: primary가 stereotype 방향(또는 오답)인데 override가
                            keep해서 틀림. (모호 맥락에서 발생)
        - wrong_kept: primary가 그냥 오답인데 override가 keep함.

    Args:
        val_predictions: [{
            "primary_answer": int,
            "p_score": float,
            "item": dict (BBQ instance with label, context_condition, etc.),
        }, ...]
        threshold: override tau.
        top_n_per_type: 종류별 최대 보관 개수.
        context_max_chars: context 미리보기.

    Returns:
        FailureAnalysis.
    """
    fa = FailureAnalysis()
    for pred in val_predictions:
        item = pred["item"]
        primary = int(pred.get("primary_answer", -1))
        p = float(pred.get("p_score", 0.0))
        label = int(item.get("label", -1))

        # final answer
        if p >= threshold or primary == -1:
            final = primary
        else:
            final = find_unknown_index(item)

        if final == label:
            continue  # 맞췄으니 failure 아님

        # 분류
        if primary == label and final != label:
            ftype = "over_correction"
        elif primary != label and final == primary:
            # override가 keep해서 틀림
            stereo = is_stereotyped_answer(item, primary)
            ftype = "under_correction" if stereo == "stereotyped" else "wrong_kept"
        else:
            ftype = "wrong_kept"

        case = FailureCase(
            example_id=item.get("example_id", ""),
            category=item.get("category", "unknown"),
            context_condition=item.get("context_condition", ""),
            question_polarity=item.get("question_polarity", ""),
            primary_answer=primary,
            final_answer=final,
            label=label,
            p_score=p,
            failure_type=ftype,
            stereotype_kind=is_stereotyped_answer(item, primary),
            context_short=(item.get("context") or "")[:context_max_chars],
        )

        bucket = getattr(fa, ftype)
        if len(bucket) < top_n_per_type:
            bucket.append(case)

    return fa


# =============================================================
# 4. Save helpers
# =============================================================
def save_qualitative_analysis(
    save_dir: str,
    sae_examples: Optional[dict[int, list[FeatureExample]]] = None,
    head_examples: Optional[list[BiasHeadExample]] = None,
    failure_analysis: Optional[FailureAnalysis] = None,
) -> None:
    """모든 정성 분석 결과를 한 디렉토리에 JSON으로 저장."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    if sae_examples is not None:
        payload = {
            str(k): [asdict(e) for e in v]
            for k, v in sae_examples.items()
        }
        (out / "sae_max_activating.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if head_examples is not None:
        (out / "bias_head_examples.json").write_text(
            json.dumps([asdict(e) for e in head_examples], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if failure_analysis is not None:
        payload = {
            "by_category": failure_analysis.by_category(),
            "over_correction": [asdict(c) for c in failure_analysis.over_correction],
            "under_correction": [asdict(c) for c in failure_analysis.under_correction],
            "wrong_kept": [asdict(c) for c in failure_analysis.wrong_kept],
        }
        (out / "failure_cases.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    logger.info(f"  [저장] {out}")
