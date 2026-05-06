#!/usr/bin/env python3
"""
End-to-End Smoke Test 결과 검증.

run_smoke_e2e.sh 실행 후 모든 stage의 출력 파일 존재 + 비어있지 않음 확인.
일부 핵심 JSON은 schema 간이 검증 (필수 키 존재).

CLI:
    python scripts/verify_smoke_e2e.py
    python scripts/verify_smoke_e2e.py --out-dir results/smoke_e2e
    python scripts/verify_smoke_e2e.py --strict   # 한 개라도 실패 시 exit 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]


# out_dir → 매칭되는 data_dir
_OUT_TO_DATA = {
    "smoke_e2e": "sampled_smoke",
    "v2_mini":   "sampled_mini",
    "v2":        "sampled_v2",
}


def _data_dir_for(out_dir: Path) -> str:
    """out_dir 이름에서 매칭되는 data_dir 자동 추정."""
    name = out_dir.name
    return _OUT_TO_DATA.get(name, "sampled_smoke")


# stage 이름 -> (상대 경로 또는 placeholder, 필수 JSON 키 또는 None)
# {out_dir}/x → out_dir/x, {data_dir} → ROOT/data/sampled_*, results/x → ROOT/results/x
EXPECTED_TEMPLATE: list[tuple[str, str, Optional[list[str]]]] = [
    # Phase 1: Data + Signals
    ("1/22 Data sampling",       "{data_dir}",                                       None),
    # Stage 1 (4-prompt inference)은 signals/main/ 안에 *_stage1.jsonl
    ("2/22 Stage1 Inference",    "{out_dir}/signals/main",                           None),
    ("3/22 Stage2 Signals",      "{out_dir}/signals",                                None),
    ("4/22 Bias-head identify",  "results/bias_heads.json",                          ["head_indices", "scores"]),
    # Phase 2: MoE + Threshold + Ablation
    ("6/22 Multi-seed",          "{out_dir}/multi_seed",                             None),
    ("7/22 MoE+Eval+Ablation",   "{out_dir}/evaluation",                             None),
    ("7/22 Ablation main",       "{out_dir}/ablation/main",                          None),
    ("8/22 Threshold sweep",     "{out_dir}/thresholds",                             None),
    # Phase 3: Baselines
    ("11/22 Composite",          "{out_dir}/baselines/composite/final.json",         ["overall"]),
    ("12/22 Self-Debiasing",     "{out_dir}/baselines/self_debiasing/final.json",    ["overall"]),
    ("13/22 DeCAP",              "{out_dir}/baselines/decap/final.json",             ["overall"]),
    ("14/22 FairSteer",          "{out_dir}/baselines/fairsteer/final.json",         ["overall"]),
    # Phase 4: SAE Layer
    ("15/22 SAE layers",         "{out_dir}/sae_layers",                             None),
    # Phase 6: Transfer
    ("18/22 ImplicitBBQ",        "{out_dir}/transfer/implicit_bbq",                  None),
    ("19/22 Open-BBQ",           "{out_dir}/transfer/open_bbq",                      None),
    ("20/22 KoBBQ",              "{out_dir}/transfer/kobbq",                         None),
    # Phase 7: Statistics + Figures
    ("21/22 Qualitative",        "{out_dir}/qualitative",                            None),
    ("22/22 Paper figures",      "{out_dir}/figures",                                None),
]

EXPECTED_CROSS_LLM_TEMPLATE: list[tuple[str, str, Optional[list[str]]]] = [
    ("16/22 Cross-LLM Gemma",    "{out_dir}/cross_llm/gemma/final.json", ["overall"]),
    ("17/22 Cross-LLM Qwen",     "{out_dir}/cross_llm/qwen/final.json",  ["overall"]),
]


def _resolve(rel: str, out_dir: Path) -> Path:
    """경로 해석:
       - {data_dir} → ROOT/data/sampled_<X>
       - {out_dir}/x → out_dir/x (out_dir 자체는 ROOT/results/<X>)
       - 'results/' 시작 → ROOT 기준
       - 그 외 → out_dir 기준 (legacy)
    """
    if "{data_dir}" in rel:
        data_name = _data_dir_for(out_dir)
        return ROOT / "data" / data_name / rel.replace("{data_dir}", "").lstrip("/")
    if "{out_dir}" in rel:
        sub = rel.replace("{out_dir}", "").lstrip("/")
        return out_dir / sub if sub else out_dir
    if rel.startswith(("data/", "results/")):
        return ROOT / rel
    return out_dir / rel


def _check_path(target: Path) -> tuple[bool, str]:
    """target (file 또는 dir) 존재 + 비어있지 않음 검증."""
    if not target.exists():
        return False, f"NOT FOUND: {target}"
    if target.is_file():
        if target.stat().st_size == 0:
            return False, f"EMPTY: {target}"
        return True, f"OK ({target.stat().st_size} bytes)"
    # directory
    contents = list(target.iterdir())
    if not contents:
        return False, f"EMPTY DIR: {target}"
    # 최소 1개 file 또는 sub-dir 존재
    n_files = sum(1 for c in contents if c.is_file())
    n_dirs = sum(1 for c in contents if c.is_dir())
    return True, f"OK ({n_files} files, {n_dirs} dirs)"


def _check_json_keys(target: Path, required_keys: list[str]) -> tuple[bool, str]:
    """JSON 파일이면 필수 키 존재 검증."""
    if not target.is_file() or target.suffix != ".json":
        return True, "skip (not json)"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"JSON parse error: {e}"
    missing = [k for k in required_keys if k not in data]
    if missing:
        return False, f"missing keys: {missing}"
    return True, f"keys OK: {required_keys}"


def verify_smoke_test_e2e(
    out_dir: str = "results/smoke_e2e",
    include_cross_llm: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """
    모든 stage 결과 검증.

    Returns:
        {"passed": int, "failed": int, "total": int, "details": [...]}
    """
    out_path = ROOT / out_dir if not Path(out_dir).is_absolute() else Path(out_dir)
    expected = list(EXPECTED_TEMPLATE)
    if include_cross_llm:
        expected.extend(EXPECTED_CROSS_LLM_TEMPLATE)

    print("=" * 72)
    print(f" Smoke E2E 결과 검증")
    print(f" out_dir: {out_path}")
    print(f" cross_llm: {'YES' if include_cross_llm else 'SKIP'}")
    print("=" * 72)

    details: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0

    for stage, rel, req_keys in expected:
        target = _resolve(rel, out_path)
        ok, msg = _check_path(target)
        if ok and req_keys:
            ok2, msg2 = _check_json_keys(target, req_keys)
            ok = ok and ok2
            msg = f"{msg} | {msg2}"

        marker = "[OK]  " if ok else "[FAIL]"
        print(f" {marker} {stage:<28} {msg}")
        details.append({"stage": stage, "path": str(target), "ok": ok, "msg": msg})
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    print("-" * 72)
    total = n_pass + n_fail
    print(f" 총: {n_pass}/{total} stage 통과 ({n_fail} 실패)")
    if n_fail == 0:
        print(" 모든 stage 정상 작동! → 9 × 1000 풀 런 진행 가능")
    else:
        print(f" {n_fail}개 stage 실패. 풀 런 전 점검 필요.")
    print("=" * 72)

    summary = {
        "passed": n_pass,
        "failed": n_fail,
        "total": total,
        "details": details,
    }

    # 결과 JSON 저장
    report_path = out_path / "verify_report.json"
    out_path.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f" 검증 리포트: {report_path}")

    if strict and n_fail > 0:
        sys.exit(1)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify End-to-End smoke test outputs")
    parser.add_argument("--out-dir", type=str, default="results/smoke_e2e",
                        help="smoke test output directory")
    parser.add_argument("--include-cross-llm", action="store_true",
                        help="RUN_CROSS_LLM=1 로 실행한 경우 cross-llm stage도 검증")
    parser.add_argument("--strict", action="store_true",
                        help="실패 시 exit 1")
    args = parser.parse_args()

    summary = verify_smoke_test_e2e(
        out_dir=args.out_dir,
        include_cross_llm=args.include_cross_llm,
        strict=args.strict,
    )
    return 0 if summary["failed"] == 0 else (1 if args.strict else 0)


if __name__ == "__main__":
    sys.exit(main())
