"""
Stage 2: 7개 신호 통합 추출 스크립트.

Stage 1 결과(4-prompt 응답)와 BBQ instance를 입력받아 7개 신호를 모두 추출하고
JSONL로 저장합니다.

각 신호는 [0, 1] 범위로 정규화되어 Stage 3 MoE의 입력으로 사용됩니다.
"""

import json  # JSONL 입출력
import logging  # 경고/에러 로깅
from pathlib import Path  # 경로 객체
from typing import Optional  # 선택형 타입

from tqdm import tqdm  # 진행률 표시

# 평가 유틸: 답변이 stereotype 방향인지 판정 (bias_penalty 학습용)
from src.evaluation.bbq_evaluator import is_stereotyped_answer
# 7개 신호 함수들
from src.signals.bias_head import compute_bias_head_activation, load_bias_heads  # s5
from src.signals.confidence import compute_confidence_from_logprobs  # s3
from src.signals.consistency import compute_self_consistency  # s4
from src.signals.counterfactual import compute_s2_for_item  # s2
from src.signals.evidence import compute_evidence  # s1
from src.signals.prompts import PROMPT_BUILDERS  # vanilla/cot/debias/persona builder dict
from src.signals.prompt_sensitivity import compute_prompt_sensitivity  # s6
from src.signals.sae_feature import SAEWrapper, compute_sae_signal  # s7
from src.utils.llm_utils import LLMWrapper  # LLM 추상화

logger = logging.getLogger(__name__)  # 모듈 전용 logger

# 모듈 레벨에서 한 번만 경고하기 위한 플래그 (반복 카테고리 처리 시 spam 방지)
_BIAS_HEAD_WARNING_EMITTED = False  # s5 bias_heads.json 미존재 경고
_SAE_FEATURE_WARNING_EMITTED = False  # s7 bias_sae_features 미식별 경고


def extract_signals_for_item(
    item: dict,  # BBQ instance 한 건
    stage1_responses: dict,  # 해당 instance의 4-prompt 응답 dict
    llm: LLMWrapper,  # 메인 LLM (Llama/Qwen/Mistral)
    sae: Optional[SAEWrapper] = None,  # SAE wrapper (s7 계산용, 없으면 s7=None)
    n_consistency_samples: int = 5,  # s4 self-consistency sampling 횟수
    bias_head_indices: Optional[list[tuple[int, int]]] = None,  # s5용 (layer, head) 페어
    bias_sae_features: Optional[list[int]] = None,  # s7용 SAE feature 인덱스 리스트
    primary_prompt: str = "vanilla",  # 기본 prompt variant (s1, s3, s4, s5에서 사용)
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
    primary = stage1_responses[primary_prompt]  # primary prompt 응답 dict
    primary_answer = primary["answer"]  # 모델이 고른 답 (문자열 "0"/"1"/"2" 또는 int)

    # Stage 1은 answer를 "0"/"1"/"2" 문자열로 저장하므로 두 형식으로 변환
    # → s1 evidence는 letter("A"/"B"/"C"), s2 counterfactual은 int 형식을 요구
    _IDX_TO_LETTER = {"0": "A", "1": "B", "2": "C", 0: "A", 1: "B", 2: "C"}
    answer_letter = _IDX_TO_LETTER.get(primary_answer, str(primary_answer))  # letter 변환
    try:
        answer_idx = int(primary_answer)  # int 변환
    except (TypeError, ValueError):
        # 응답이 파싱 불가능한 경우 -1 (downstream에서 unknown으로 처리)
        answer_idx = -1

    # ─── s1: Evidence ────────────────────────────────────────────────
    # 답의 근거가 context에 명시되어 있는지 entailment 기반 점수 [0,1]
    s1 = compute_evidence(item, answer_letter, llm)

    # ─── s2: Counterfactual ──────────────────────────────────────────
    # 사회집단 단어를 swap 했을 때 답이 바뀌면 bias 가능성 ↑
    s2_result = compute_s2_for_item(
        item=item,
        original_answer=answer_idx,
        llm=llm,
        prompt_builder=PROMPT_BUILDERS[primary_prompt],
    )
    s2 = s2_result["s2_score"]

    # ─── s3: Confidence ──────────────────────────────────────────────
    # logprobs가 문자열로 저장된 경우(JSONL round-trip 손상) dict로 복원
    raw_logprobs = primary.get("logprobs", {})
    if isinstance(raw_logprobs, str):
        try:
            import ast as _ast  # eval보다 안전한 literal_eval
            raw_logprobs = _ast.literal_eval(raw_logprobs)
        except (ValueError, SyntaxError):
            # 복원 실패 시 빈 dict (compute_confidence에서 fallback 처리)
            raw_logprobs = {}
    s3 = compute_confidence_from_logprobs(raw_logprobs, answer_idx)

    # ─── s4: Self-consistency ────────────────────────────────────────
    # 같은 prompt를 temperature 샘플링으로 N번 실행해 답 일치율 측정
    s4_result = compute_self_consistency(
        item=item,
        llm=llm,
        prompt_builder=PROMPT_BUILDERS[primary_prompt],
        n_samples=n_consistency_samples,
    )
    s4 = s4_result["s4_score"]

    # ─── s5: Bias-head activation ────────────────────────────────────
    # 명시적 인자 없으면 캐싱된 bias_heads.json 사용.
    # 캐시도 비어있으면 s5는 0으로 떨어지므로 한 번만 loud warning을 띄운다.
    global _BIAS_HEAD_WARNING_EMITTED
    head_idx_to_use = bias_head_indices
    if not head_idx_to_use:
        head_idx_to_use = load_bias_heads()  # results/bias_heads.json 로드
    if not head_idx_to_use and not _BIAS_HEAD_WARNING_EMITTED:
        # 한 번만 경고 (카테고리당 1회 X, 전체 runtime에서 1회만)
        logger.error(
            "  [s5] bias_heads 인덱스가 비어 있음 — s5_bias_head 신호가 항상 0으로 "
            "떨어집니다. `python -m src.signals.bias_head` 또는 identify_bias_heads()를 "
            "먼저 실행하여 results/bias_heads.json을 생성하세요. "
            "(이 신호 없이 학습/평가를 진행하면 7-signal 주장이 6-signal로 축소됨)"
        )
        _BIAS_HEAD_WARNING_EMITTED = True
    s5 = compute_bias_head_activation(
        item=item,
        llm=llm,
        prompt_builder=PROMPT_BUILDERS[primary_prompt],
        head_indices=head_idx_to_use or [],  # 빈 리스트면 0 반환
    )

    # ─── s6: Prompt sensitivity ──────────────────────────────────────
    # Stage 1 결과 4-prompt 답 분산만 사용 → 추가 LLM 호출 없음 (효율적)
    s6_result = compute_prompt_sensitivity(stage1_responses)
    s6 = s6_result["s6_score"]

    # ─── s7: SAE feature (옵션) ──────────────────────────────────────
    # 로드/추론 실패 시 None으로 처리하고 다른 신호는 살림 (full pipeline 보존)
    global _SAE_FEATURE_WARNING_EMITTED
    if sae is None:
        # SAE 미로드 (메모리 절약 등)
        s7 = None
    else:
        # bias-related SAE feature가 식별되지 않은 경우 fallback이 단순 top-k 평균을
        # 쓰므로 "bias 관련"이라는 신호 의미가 옅어진다. 한 번만 경고.
        if not bias_sae_features and not _SAE_FEATURE_WARNING_EMITTED:
            logger.error(
                "  [s7] bias_sae_features가 비어 있음 — top-k activation magnitude "
                "fallback으로 동작합니다. 이는 편향 무관 신호일 수 있으므로 식별된 "
                "feature 인덱스를 전달하는 것을 권장."
            )
            _SAE_FEATURE_WARNING_EMITTED = True
        try:
            s7 = compute_sae_signal(
                item=item,
                llm=llm,
                sae=sae,
                prompt_builder=PROMPT_BUILDERS[primary_prompt],
                bias_feature_indices=bias_sae_features or [],
            )
        except Exception as _sae_exc:  # noqa: BLE001
            # 한 번 실패하면 동일 SAE로 재시도해도 똑같이 실패하므로 SAE 자체 무효화 X
            # 대신 이 sample만 s7=None으로 표시 (다음 sample 재시도)
            logger.warning(
                f"  s7 SAE 신호 계산 실패 — 이번 카테고리 s7=None으로 처리: {_sae_exc}"
            )
            s7 = None

    # bias_penalty 학습용: stereotype 방향 답이면 1, 아니면 0.
    # is_stereotyped_answer는 -1/unknown/판단불가도 처리하므로 안전.
    stereo_kind = is_stereotyped_answer(item, answer_idx)
    is_stereotype = 1.0 if stereo_kind == "stereotyped" else 0.0

    # 결과 dict 반환 (Stage 3 MoE 입력 + 분석용 메타데이터 포함)
    return {
        "example_id": item["example_id"],  # BBQ instance 식별자
        "category": item.get("category"),  # 9개 카테고리 중 하나
        "context_condition": item.get("context_condition"),  # "ambig" or "disambig"
        "label": item.get("label"),  # 정답 인덱스
        "primary_answer": primary_answer,  # 모델 응답
        "is_stereotype": is_stereotype,  # bias_penalty target
        "signals": {
            "s1_evidence": s1,
            "s2_counterfactual": s2,
            "s3_confidence": s3,
            "s4_consistency": s4,
            "s5_bias_head": s5,
            "s6_prompt_sensitivity": s6,
            "s7_sae_feature": s7,  # None일 수 있음 (MoE input에서 0 또는 mask로 처리)
        },
    }


def extract_signals_batch(
    items: list[dict],  # 카테고리 내 instance 리스트
    stage1_results: list[dict],  # Stage 1 결과 (instance와 같은 순서)
    llm: LLMWrapper,  # 메인 LLM
    sae: Optional[SAEWrapper],  # SAE wrapper (선택)
    output_path: Path,  # JSONL 저장 경로 (Path 객체)
    save_every: int = 25,  # 체크포인트 저장 주기
    **kwargs,  # extract_signals_for_item에 전달되는 추가 인자
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
    # 출력 디렉토리 미리 생성 (이미 있어도 OK)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── 체크포인트 로드 (resumable 실행) ────────────────────────────
    # 이전 실행이 중간에 끊겼다면 example_id 기준으로 완료분을 스킵
    completed: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # 빈 줄 스킵
                    rec = json.loads(line)
                    completed[rec["example_id"]] = rec
        print(f"  [resume] {len(completed)}개 신호 추출 완료, 이어서 실행")

    # example_id -> stage1 responses 매핑 (O(1) lookup)
    stage1_by_id = {r["example_id"]: r["responses"] for r in stage1_results}

    # 결과 리스트: 이미 완료된 것 + 새로 처리할 것
    results: list[dict] = list(completed.values())
    # pending: 완료되지 않은 instance만 추출 (resume 시 중복 작업 방지)
    pending = [item for item in items if item["example_id"] not in completed]

    pbar = tqdm(pending, desc="Signal Extraction")  # 진행률 바
    for i, item in enumerate(pbar):
        # Stage 1 응답 없는 instance는 건너뜀 (드물지만 누락 가능)
        if item["example_id"] not in stage1_by_id:
            continue

        # 7개 신호 모두 계산
        rec = extract_signals_for_item(
            item=item,
            stage1_responses=stage1_by_id[item["example_id"]],
            llm=llm,
            sae=sae,
            **kwargs,
        )
        results.append(rec)

        # save_every마다 체크포인트 저장 (재실행 안정성)
        if (i + 1) % save_every == 0:
            _save(results, output_path)

    # 최종 저장 (pending이 save_every 배수가 아닐 때 마지막 chunk 저장 보장)
    _save(results, output_path)
    return results


def _save(records: list[dict], path: Path) -> None:
    """JSONL 저장 (전체 덮어쓰기 — append가 아니므로 resume 시 중복 없음)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            # ensure_ascii=False: 한글 카테고리(KoBBQ)도 그대로 저장
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
