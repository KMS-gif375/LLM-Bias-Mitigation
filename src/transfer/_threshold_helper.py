"""
Transfer 스크립트 공통 helper.

기능:
    - resolve_thresholds: per-condition threshold 해결 (source eval에서 자동 로드)
    - apply_composite_keys: 카테고리 간 example_id 충돌 자동 감지 + 처리

Zero-shot transfer는 source(in-distribution) validation에서 선택한 thresholds를
사용하므로, 해당 backbone의 current evaluation artifact를 먼저 찾고 main/legacy
artifact를 차례로 fallback한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_SCHEMAS = (
    "thresholds",
    "thresholds_per_condition",
    "thresholds_resolved",
)
ROUTING_MODE = "oracle_target_condition"
CONDITION_SOURCE = "target_dataset.context_condition"
CACHE_PROVENANCE_SCHEMA_VERSION = 1


def _coerce_threshold_pair(value: object) -> tuple[Optional[dict[str, float]], str]:
    """Return a finite [0, 1] threshold pair or a diagnostic string."""
    if not isinstance(value, dict):
        return None, "value is not an object"
    missing = [key for key in ("ambig", "disambig") if key not in value]
    if missing:
        return None, f"missing keys: {', '.join(missing)}"

    converted: dict[str, float] = {}
    for key in ("ambig", "disambig"):
        raw = value[key]
        if isinstance(raw, bool):
            return None, f"{key} is boolean, not numeric"
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return None, f"{key} is not numeric: {raw!r}"
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            return None, f"{key} must be finite and in [0, 1]: {raw!r}"
        converted[key] = number
    return converted, ""


def resolve_thresholds_with_provenance(
    threshold: float = 0.5,
    threshold_amb: Optional[float] = None,
    threshold_dis: Optional[float] = None,
    source_eval_path: Optional[str] = None,
    model_key: str = "main",
) -> dict[str, Any]:
    """Resolve transfer thresholds and report their artifact provenance.

    The ``thresholds`` member is the same value returned by
    :func:`resolve_thresholds`. ``source_path`` and ``schema`` are populated
    only when the pair came from a serialized evaluation artifact.
    """
    if (threshold_amb is None) != (threshold_dis is None):
        raise ValueError(
            "threshold_amb and threshold_dis must be provided together"
        )
    if threshold_amb is not None and threshold_dis is not None:
        thresholds, error = _coerce_threshold_pair({
            "ambig": threshold_amb,
            "disambig": threshold_dis,
        })
        if thresholds is None:
            raise ValueError(f"invalid explicit per-condition thresholds: {error}")
        logger.info(
            "  [threshold] explicit per-condition: amb=%s dis=%s",
            thresholds["ambig"],
            thresholds["disambig"],
        )
        return {
            "thresholds": thresholds,
            "source": "explicit_per_condition",
            "source_path": None,
            "schema": None,
            "requested_model_key": model_key,
        }

    if source_eval_path is not None:
        candidates = [Path(source_eval_path)]
    else:
        candidates: list[Path] = []
        if model_key != "main":
            candidates.append(
                REPO_ROOT
                / f"results/v2/cross_llm/{model_key}/evaluation/{model_key}/final.json"
            )
        candidates.extend([
            REPO_ROOT / "results/v2/evaluation/main/final.json",
            REPO_ROOT / "results/evaluation/main/final.json",
        ])

    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "  [threshold] source eval load failed (%s): %s; trying next candidate",
                path,
                exc,
            )
            continue
        if not isinstance(data, dict):
            logger.warning(
                "  [threshold] %s is not a JSON object; trying next candidate",
                path,
            )
            continue

        saw_schema = False
        for schema in THRESHOLD_SCHEMAS:
            if schema not in data:
                continue
            saw_schema = True
            thresholds, error = _coerce_threshold_pair(data[schema])
            if thresholds is None:
                logger.warning(
                    "  [threshold] invalid %s in %s (%s); trying another schema/candidate",
                    schema,
                    path,
                    error,
                )
                continue
            resolved_path = str(path.expanduser().resolve())
            logger.info(
                "  [threshold] auto-loaded from %s[%s]: amb=%s dis=%s",
                path,
                schema,
                thresholds["ambig"],
                thresholds["disambig"],
            )
            return {
                "thresholds": thresholds,
                "source": "source_eval_artifact",
                "source_path": resolved_path,
                "schema": schema,
                "requested_model_key": model_key,
            }
        if not saw_schema:
            logger.warning(
                "  [threshold] %s has no supported threshold schema; "
                "trying next candidate",
                path,
            )

    thresholds, error = _coerce_threshold_pair({
        "ambig": threshold,
        "disambig": threshold,
    })
    if thresholds is None:
        raise ValueError(f"invalid legacy scalar threshold: {error}")
    logger.info(
        "  [threshold] legacy single threshold: %s (per-condition unset)",
        threshold,
    )
    return {
        "thresholds": thresholds,
        "source": "legacy_scalar_fallback",
        "source_path": None,
        "schema": None,
        "requested_model_key": model_key,
    }


def resolve_thresholds(
    threshold: float = 0.5,
    threshold_amb: Optional[float] = None,
    threshold_dis: Optional[float] = None,
    source_eval_path: Optional[str] = None,
    model_key: str = "main",
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
    resolution = resolve_thresholds_with_provenance(
        threshold=threshold,
        threshold_amb=threshold_amb,
        threshold_dis=threshold_dis,
        source_eval_path=source_eval_path,
        model_key=model_key,
    )
    return resolution["thresholds"]


def _file_sha256(path: str | Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _resolved_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def build_transfer_cache_provenance(
    *,
    runner: str,
    config_path: str,
    model_key: str,
    threshold: float,
    threshold_amb: Optional[float],
    threshold_dis: Optional[float],
    threshold_resolution: Mapping[str, Any],
    data_dir: Optional[str] = None,
    categories: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
    moe_ckpt: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the request identity used to validate a raw-transfer cache."""
    config_resolved = _resolved_path(config_path)
    return {
        "schema_version": CACHE_PROVENANCE_SCHEMA_VERSION,
        "status": "verified",
        "runner": runner,
        "routing_mode": ROUTING_MODE,
        "condition_source": CONDITION_SOURCE,
        "config_path": config_resolved,
        "config_sha256": _file_sha256(config_resolved) if config_resolved else None,
        "model_key": model_key,
        "threshold_request": {
            "threshold": float(threshold),
            "threshold_amb": None if threshold_amb is None else float(threshold_amb),
            "threshold_dis": None if threshold_dis is None else float(threshold_dis),
        },
        "threshold_resolution": dict(threshold_resolution),
        "data_dir": _resolved_path(data_dir),
        "categories": sorted(str(category) for category in categories)
        if categories
        else None,
        "max_samples": max_samples,
        "moe_ckpt_request": _resolved_path(moe_ckpt),
        "extra": dict(extra or {}),
    }


def atomic_write_json(path: str | Path, payload: object) -> None:
    """Durably replace a JSON artifact without exposing a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=float)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(tmp_name, path.stat().st_mode)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_transfer_cache(
    path: str | Path,
    expected_provenance: Mapping[str, Any],
    *,
    cache_logger: Optional[logging.Logger] = None,
) -> Optional[dict[str, Any]]:
    """Load a compatible transfer result, migrating legacy metadata atomically.

    A cache with verified provenance must exactly match the current request.
    Older caches cannot be validated retroactively, so they are labeled
    ``legacy_unverified`` on disk and returned only with a prominent warning.
    """
    log = cache_logger or logger
    path = Path(path)
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("  [cache] cannot read %s (%s); recomputing", path, exc)
        return None
    if not isinstance(cached, dict):
        log.warning("  [cache] %s is not a JSON object; recomputing", path)
        return None

    expected_routing = expected_provenance.get("routing_mode", ROUTING_MODE)
    expected_condition = expected_provenance.get(
        "condition_source", CONDITION_SOURCE
    )
    for key, expected in (
        ("routing_mode", expected_routing),
        ("condition_source", expected_condition),
    ):
        observed = cached.get(key)
        if observed is not None and observed != expected:
            log.warning(
                "  [cache] %s has incompatible %s=%r (requested %r); recomputing",
                path,
                key,
                observed,
                expected,
            )
            return None

    changed = False
    if "routing_mode" not in cached:
        cached["routing_mode"] = expected_routing
        changed = True
    if "condition_source" not in cached:
        cached["condition_source"] = expected_condition
        changed = True

    observed_provenance = cached.get("cache_provenance")
    if observed_provenance is None:
        cached["cache_provenance"] = {
            "schema_version": CACHE_PROVENANCE_SCHEMA_VERSION,
            "status": "legacy_unverified",
            "routing_mode": expected_routing,
            "condition_source": expected_condition,
            "warning": (
                "The original model, threshold, and input request was not recorded; "
                "cache compatibility cannot be verified. Re-run with --force for "
                "a verified artifact."
            ),
        }
        atomic_write_json(path, cached)
        log.warning(
            "  [cache] %s is legacy and stale/unverified; metadata was migrated "
            "atomically. Returning it because skip_existing=True; use --force "
            "to recompute.",
            path,
        )
        return cached

    if not isinstance(observed_provenance, dict):
        log.warning("  [cache] %s has malformed provenance; recomputing", path)
        return None
    if observed_provenance.get("status") == "legacy_unverified":
        if changed:
            atomic_write_json(path, cached)
        log.warning(
            "  [cache] %s remains stale/unverified; returning it because "
            "skip_existing=True. Use --force to recompute.",
            path,
        )
        return cached

    expected = dict(expected_provenance)
    if observed_provenance != expected:
        differing = sorted(
            key
            for key in set(observed_provenance) | set(expected)
            if observed_provenance.get(key) != expected.get(key)
        )
        log.warning(
            "  [cache] %s provenance mismatch in %s; recomputing and "
            "invalidating stage caches",
            path,
            ", ".join(differing) or "unknown fields",
        )
        return None

    if changed:
        atomic_write_json(path, cached)
    log.info("  [cache] verified compatible result: %s", path)
    return cached


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
