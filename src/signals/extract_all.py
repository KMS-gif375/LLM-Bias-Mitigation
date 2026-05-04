"""
Stage 2: 7개 신호 통합 추출 스크립트.

Stage 1 결과(4-prompt 응답)와 BBQ instance를 입력받아 7개 신호를 모두 추출하고
JSONL로 저장합니다.

각 신호는 [0, 1] 범위로 정규화되어 Stage 3 MoE의 입력으로 사용됩니다.
"""

import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.signals.bias_head import compute_bias_head_activation, load_bias_heads
from src.signals.confidence import compute_confidence_from_logprobs
from src.signals.consistency import compute_self_consistency
from src.signals.counterfactual import compute_s2_for_item
from src.signals.evidence import compute_evidence
from src.signals.prompts import PROMPT_BUILDERS
from src.signals.prompt_sensitivity import compute_prompt_sensitivity
from src.signals.sae_feature import SAEWrapper, compute_sae_signal
from src.utils.llm_utils import LLMWrapper


def extract_signals_for_item(
    item: dict,
    stage1_responses: dict,
    llm: LLMWrapper,
    sae: Optional[SAEWrapper] = None,
    n_consistency_samples: int = 5,
    bias_head_indices: Optional[list[tuple[int, int]]] = None,
    bias_sae_features: Optional[list[int]] = None,
    primary_prompt: str = "vanilla",
) -> dict:
    """
    하나의 instance에 대해 7개 신호를 모두 계산합니다.

    Args:
        item: BBQ instance.
        stage1_responses: Stage 1 결과의 "responses" 필드.
        llm: 메인 LLM.
        sae: SAE wrapper (None이면 s7=None).
        n_consistency_samples: s4 sampling 횟수.
        bias_head_indices: s5용 (layer, head) 페어 리스트.
        bias_sae_features: s7용 SAE feature 인덱스.
        primary_prompt: s1, s3 등 단일 prompt가 필요한 경우 사용할 variant.

    Returns:
        {
            "example_id": ...,
            "primary_answer": int,
            "signals": {
                "s1_evidence": float,
                "s2_counterfactual": float,
                "s3_confidence": float,
                "s4_consistency": float,
                "s5_bias_head": float,
                "s6_prompt_sensitivity": float,
                "s7_sae_feature": float | None,
            }
        }
    """
    primary = stage1_responses[primary_prompt]
    primary_answer = primary["answer"]

    # Stage 1은 answer를 "0"/"1"/"2" 문자열로 저장하므로 두 형식으로 변환
    _IDX_TO_LETTER = {"0": "A", "1": "B", "2": "C", 0: "A", 1: "B", 2: "C"}
    answer_letter = _IDX_TO_LETTER.get(primary_answer, str(primary_answer))
    try:
        answer_idx = int(primary_answer)
    except (TypeError, ValueError):
        answer_idx = -1

    # s1: Evidence (letter 필요)
    s1 = compute_evidence(item, answer_letter, llm)

    # s2: Counterfactual (int 필요)
    s2_result = compute_s2_for_item(
        item=item,
        original_answer=answer_idx,
        llm=llm,
        prompt_builder=PROMPT_BUILDERS[primary_prompt],
    )
    s2 = s2_result["s2_score"]

    # s3: Confidence (logprobs가 문자열로 저장된 경우 dict로 복원)
    raw_logprobs = primary.get("logprobs", {})
    if isinstance(raw_logprobs, str):
        try:
            import ast as _ast
            raw_logprobs = _ast.literal_eval(raw_logprobs)
        except (ValueError, SyntaxError):
            raw_logprobs = {}
    s3 = compute_confidence_from_logprobs(raw_logprobs, answer_idx)

    # s4: Self-consistency
    s4_result = compute_self_consistency(
        item=item,
        llm=llm,
        prompt_builder=PROMPT_BUILDERS[primary_prompt],
        n_samples=n_consistency_samples,
    )
    s4 = s4_result["s4_score"]

    # s5: Bias-head activation. 명시적 인자 없으면 캐싱된 bias_heads.json 사용.
    head_idx_to_use = bias_head_indices
    if not head_idx_to_use:
        head_idx_to_use = load_bias_heads()
    s5 = compute_bias_head_activation(
        item=item,
        llm=llm,
        prompt_builder=PROMPT_BUILDERS[primary_prompt],
        head_indices=head_idx_to_use or [],
    )

    # s6: Prompt sensitivity (Stage 1 결과 활용, LLM 호출 없음)
    s6_result = compute_prompt_sensitivity(stage1_responses)
    s6 = s6_result["s6_score"]

    # s7: SAE feature (옵션). 로드/추론 실패 시 None으로 처리하고 다른 신호는 살림.
    if sae is None:
        s7 = None
    else:
        try:
            s7 = compute_sae_signal(
                item=item,
                llm=llm,
                sae=sae,
                prompt_builder=PROMPT_BUILDERS[primary_prompt],
                bias_feature_indices=bias_sae_features or [],
            )
        except Exception as _sae_exc:  # noqa: BLE001
            # 한 번 실패하면 동일 SAE로 재시도해도 똑같이 실패하므로 SAE 자체 무효화
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"  s7 SAE 신호 계산 실패 — 이번 카테고리 s7=None으로 처리: {_sae_exc}"
            )
            s7 = None

    return {
        "example_id": item["example_id"],
        "category": item.get("category"),
        "context_condition": item.get("context_condition"),
        "label": item.get("label"),
        "primary_answer": primary_answer,
        "signals": {
            "s1_evidence": s1,
            "s2_counterfactual": s2,
            "s3_confidence": s3,
            "s4_consistency": s4,
            "s5_bias_head": s5,
            "s6_prompt_sensitivity": s6,
            "s7_sae_feature": s7,
        },
    }


def extract_signals_batch(
    items: list[dict],
    stage1_results: list[dict],
    llm: LLMWrapper,
    sae: Optional[SAEWrapper],
    output_path: Path,
    save_every: int = 25,
    **kwargs,
) -> list[dict]:
    """
    카테고리 단위로 7개 신호를 추출하고 저장합니다.

    Args:
        items: 샘플링된 instance 리스트.
        stage1_results: Stage 1 결과 (instance와 같은 순서).
        llm: 메인 LLM.
        sae: SAE wrapper (선택).
        output_path: JSONL 저장 경로.
        save_every: N개마다 체크포인트 저장.
        **kwargs: extract_signals_for_item에 전달되는 추가 인자.

    Returns:
        신호 추출 결과 리스트.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 체크포인트 로드
    completed: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed[rec["example_id"]] = rec
        print(f"  [resume] {len(completed)}개 신호 추출 완료, 이어서 실행")

    # example_id -> stage1 responses 매핑
    stage1_by_id = {r["example_id"]: r["responses"] for r in stage1_results}

    results: list[dict] = list(completed.values())
    pending = [item for item in items if item["example_id"] not in completed]

    pbar = tqdm(pending, desc="Signal Extraction")
    for i, item in enumerate(pbar):
        if item["example_id"] not in stage1_by_id:
            continue

        rec = extract_signals_for_item(
            item=item,
            stage1_responses=stage1_by_id[item["example_id"]],
            llm=llm,
            sae=sae,
            **kwargs,
        )
        results.append(rec)

        if (i + 1) % save_every == 0:
            _save(results, output_path)

    _save(results, output_path)
    return results


def _save(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
