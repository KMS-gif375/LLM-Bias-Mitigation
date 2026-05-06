"""
Transfer 스크립트 공통 helper.

기능:
    - resolve_thresholds: per-condition threshold 해결 (source eval에서 자동 로드)
    - apply_composite_keys: 카테고리 간 example_id 충돌 자동 감지 + 처리

Zero-shot transfer는 source(in-distribution) val에서 학습한 thresholds를
사용하는 게 자연스러우므로, 가능하면 results/evaluation/main/final.json에서
thresholds를 자동 로드한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_thresholds(
    threshold: float = 0.5,
    threshold_amb: Optional[float] = None,
    threshold_dis: Optional[float] = None,
    source_eval_path: Optional[str] = "results/evaluation/main/final.json",
) -> dict[str, float]:
    """
    Transfer 평가용 per-condition thresholds 결정.

    우선순위:
        1. threshold_amb / threshold_dis 둘 다 명시 → 사용
        2. source_eval_path의 final.json에 thresholds 필드 있음 → 사용
        3. fallback: ambig=threshold (legacy), disambig=threshold

    Args:
        threshold: 단일 τ (legacy/fallback).
        threshold_amb: 명시적 ambig τ.
        threshold_dis: 명시적 disambig τ.
        source_eval_path: source 평가 결과 경로.

    Returns:
        {"ambig": float, "disambig": float}.
    """
    # 1. 명시적 per-condition
    if threshold_amb is not None and threshold_dis is not None:
        logger.info(
            f"  [threshold] explicit per-condition: "
            f"amb={threshold_amb} dis={threshold_dis}"
        )
        return {"ambig": float(threshold_amb), "disambig": float(threshold_dis)}

    # 2. source eval의 thresholds 자동 로드
    if source_eval_path:
        path = Path(source_eval_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                ths = data.get("thresholds")
                if isinstance(ths, dict) and "ambig" in ths and "disambig" in ths:
                    logger.info(
                        f"  [threshold] auto-loaded from {path}: "
                        f"amb={ths['ambig']} dis={ths['disambig']}"
                    )
                    return {
                        "ambig": float(ths["ambig"]),
                        "disambig": float(ths["disambig"]),
                    }
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"  [threshold] source eval load 실패: {e}")

    # 3. legacy single τ fallback
    logger.info(f"  [threshold] legacy single τ: {threshold} (per-condition 미설정)")
    return {"ambig": float(threshold), "disambig": float(threshold)}


def apply_composite_keys(
    items: list[dict],
    raw_embeddings: dict,
) -> tuple[dict, dict]:
    """
    items + raw embeddings를 composite key (category::ex_id) 기반 dict로 변환.

    카테고리 간 example_id 충돌이 있으면 자동 감지하고 composite key로 격리한다.
    충돌이 없으면 무해하게 작동.

    Args:
        items: BBQ instance 리스트. 각 item에 'example_id'와 'category' 필요.
        raw_embeddings: cache_embeddings 결과 — {ex_id: tensor} (raw key).
            이 dict는 만약 collision이 있었다면 이미 일부 손실됐을 수 있음.

    Returns:
        (composite_embeddings, items_by_ukey):
            composite_embeddings: {f"{cat}::{ex_id}": tensor}
            items_by_ukey: {f"{cat}::{ex_id}": item}
    """
    composite_emb: dict = {}
    items_by_ukey: dict = {}
    n_missing_emb = 0
    n_collision = 0
    raw_id_to_cats: dict = {}

    for it in items:
        ex_id = it.get("example_id")
        cat = it.get("category", "_unknown")
        if ex_id is None:
            continue
        ukey = f"{cat}::{ex_id}"

        # raw_id_to_cats로 cross-cat collision 추적
        raw_id_to_cats.setdefault(ex_id, set()).add(cat)

        if ukey in items_by_ukey:
            n_collision += 1   # 동일 카테고리 내부 ex_id 중복 (드물지만 가능)
        items_by_ukey[ukey] = it

        if ex_id in raw_embeddings:
            composite_emb[ukey] = raw_embeddings[ex_id]
        else:
            n_missing_emb += 1

    cross_cat_dups = sum(1 for cats in raw_id_to_cats.values() if len(cats) > 1)
    if cross_cat_dups > 0:
        logger.warning(
            f"  [composite-key] {cross_cat_dups}개 raw ex_id가 여러 카테고리에 등장 → "
            f"composite key로 격리 (raw lookup이었다면 일부 instance가 잘못된 embedding을 받았을 것)"
        )
    if n_missing_emb > 0:
        logger.warning(f"  [composite-key] {n_missing_emb}개 item embedding 누락")
    if n_collision > 0:
        logger.warning(f"  [composite-key] {n_collision}개 동일 카테고리 내 ex_id 중복")

    return composite_emb, items_by_ukey


def make_unique_id(item: dict) -> str:
    """item에서 composite key 생성 — transfer 스크립트의 lookup 통일용."""
    cat = item.get("category", "_unknown")
    ex_id = item.get("example_id")
    return f"{cat}::{ex_id}"


def stratified_sample_per_category(
    items: list[dict],
    max_samples: int,
    stratify_key: str = "context_condition",
    seed: int = 42,
) -> list[dict]:
    """
    카테고리당 max_samples 개로 제한 — context_condition (ambig/disambig)을
    균등하게 stratify하고 셔플.

    이전 버전 버그:
        items = []
        for cat, lst in by_cat.items():
            items.extend(lst[:max_samples])
    → 데이터가 ambig 다음 disambig 순서면 첫 N개가 모두 ambig → acc_dis=0.

    Open-BBQ에서 발견됨 (ambig 29192 + disambig 29192이지만 파일 내 순서가
    ambig 먼저 → max_samples=50이면 모두 ambig만 뽑힘).

    Args:
        items: BBQ instance 리스트. 'category'와 stratify_key 필드 필요.
        max_samples: 카테고리당 최대 샘플 수 (None/0이면 그대로 반환).
        stratify_key: 균등 샘플링 기준 (default "context_condition").
        seed: 랜덤 시드 (재현성).

    Returns:
        stratified + shuffled item 리스트.
    """
    import random
    from collections import defaultdict

    if max_samples is None or max_samples <= 0:
        return items

    rng = random.Random(seed)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_cat[it.get("category", "_unknown")].append(it)

    sampled: list[dict] = []
    for cat in sorted(by_cat):
        lst = by_cat[cat]
        # context_condition 별 그룹화
        by_strat: dict[str, list[dict]] = defaultdict(list)
        for it in lst:
            by_strat[it.get(stratify_key, "_unknown")].append(it)

        # 그룹별 균등 분배 (예: amb 25 + dis 25 = 50)
        keys_sorted = sorted(by_strat)
        if not keys_sorted:
            continue
        per_group = max_samples // len(keys_sorted)
        remainder = max_samples - per_group * len(keys_sorted)

        cat_sampled: list[dict] = []
        for i, key in enumerate(keys_sorted):
            take = per_group + (1 if i < remainder else 0)
            group_items = by_strat[key][:]
            rng.shuffle(group_items)
            cat_sampled.extend(group_items[:take])

        rng.shuffle(cat_sampled)
        sampled.extend(cat_sampled)

    logger.info(
        f"  [stratified-sample] {len(sampled)} items from {len(by_cat)} categories "
        f"(max {max_samples}/cat, stratified by {stratify_key})"
    )
    return sampled
