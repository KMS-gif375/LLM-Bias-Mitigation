#!/usr/bin/env python3
"""
Data leakage 자동 감사 — 코드 패턴 검사로 누설 가능 지점 발견.

검사 카테고리:
  L1. Train/Eval data overlap (학습 → 동일 데이터로 평가)
  L2. Threshold tuning on eval set (τ search on test)
  L3. Feature selection on full corpus (bias-head, SAE feature)
  L4. Multi-seed split consistency (seed별 split이 안정적인지)
  L5. Baseline data leakage (baseline 내부 학습?)
  L6. Embedding cache leakage (캐시가 다른 split에서 fit?)

각 검사는 grep/AST로 패턴 매칭 + 의심 라인 보고.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Finding:
    """누설 의심 발견."""
    def __init__(self, severity, category, file, line, snippet, note):
        self.severity = severity   # HIGH / MED / LOW / INFO
        self.category = category
        self.file = file
        self.line = line
        self.snippet = snippet
        self.note = note

    def __str__(self):
        rel = self.file.relative_to(ROOT) if hasattr(self.file, "relative_to") else self.file
        return f"  [{self.severity}] [{self.category}] {rel}:{self.line}\n      {self.snippet}\n      → {self.note}"


def find_files(patterns):
    files = []
    for pat in patterns:
        files.extend(ROOT.glob(pat))
    return [f for f in files if "venv" not in str(f) and ".claude" not in str(f)]


def check_collect_records_use(file: Path) -> list[Finding]:
    """L1: _collect_records_and_embeddings 결과가 학습+평가 둘 다에 쓰이는지."""
    findings = []
    text = file.read_text()
    lines = text.split("\n")
    in_eval_fn = False
    has_collect = False
    has_train_call = False
    has_eval_metric = False

    for i, line in enumerate(lines, 1):
        if re.search(r"def run_(evaluation|moe_training|ablation)\b", line):
            in_eval_fn = line.split("def ")[1].split("(")[0]
        if "_collect_records_and_embeddings" in line and "def " not in line:
            findings.append(Finding(
                "INFO", "L1", file, i, line.strip(),
                f"records 로드 — 다음 stage가 학습 + 평가 둘 다 같은 records 쓰는지 확인",
            ))
    return findings


def check_threshold_search_input(file: Path) -> list[Finding]:
    """L2: search_optimal_threshold가 무엇을 받는지 (val? all?)"""
    findings = []
    text = file.read_text()
    lines = text.split("\n")
    for i, line in enumerate(lines, 1):
        m = re.search(r"search_optimal_threshold(_per_condition)?\(", line)
        if m:
            # 다음 한 줄 더 보고 인자 확인
            context = " ".join(lines[i-1:i+3])
            if "val_predictions" in context or "val_preds" in context:
                # 'val_' prefix를 쓰지만 실제 9000 전부일 수도 있음
                # caller 추적 필요
                findings.append(Finding(
                    "MED", "L2", file, i,
                    line.strip(),
                    "τ search 입력. val_predictions 변수명이 실제로 val split인지 caller에서 확인 필요",
                ))
    return findings


def check_feature_selection_pool(files: list[Path]) -> list[Finding]:
    """L3: bias-head, SAE feature 식별이 full corpus 쓰는지."""
    findings = []
    targets = {
        "bias_head.py": ["identify_bias_heads"],
        "sae_ablation.py": ["identify_bias_features"],
        "sae_feature.py": ["identify"],
    }
    for f in files:
        if f.name not in targets:
            continue
        text = f.read_text()
        for fn_name in targets[f.name]:
            if fn_name in text:
                # 함수 정의 찾고 인자 확인
                pat = re.compile(rf"def {fn_name}\w*\([^)]*\)", re.MULTILINE)
                for m in pat.finditer(text):
                    sig = m.group()
                    if "train" in sig.lower() and "data" in sig.lower():
                        findings.append(Finding(
                            "LOW", "L3", f,
                            text[:m.start()].count("\n") + 1,
                            sig[:120] + "...",
                            "feature 선택 함수가 'train_data' 받음 — caller가 진짜 train 부분만 넘기는지 확인",
                        ))
                    else:
                        findings.append(Finding(
                            "MED", "L3", f,
                            text[:m.start()].count("\n") + 1,
                            sig[:120] + "...",
                            "feature 선택 함수가 data 부분 명시 없음 — full corpus 사용 의심",
                        ))
    return findings


def check_baselines_no_training(files: list[Path]) -> list[Finding]:
    """L5: baseline 내부에 학습 단계 있는지 (있으면 별도 split 필요)."""
    findings = []
    train_patterns = [
        (r"learn_steering_vector|steering_vector\s*=", "FairSteer steering vector 학습"),
        (r"train_alpha|tune_alpha", "FairSteer alpha tuning"),
        (r"def train\b|fit\(|\.fit\(", "일반적 fit/train 호출"),
    ]
    for f in files:
        if "baselines" not in str(f) or f.name == "run_baseline.py":
            continue
        text = f.read_text()
        for pat, desc in train_patterns:
            for m in re.finditer(pat, text):
                line_no = text[:m.start()].count("\n") + 1
                line = text.split("\n")[line_no - 1].strip()
                findings.append(Finding(
                    "INFO", "L5", f, line_no, line[:120],
                    f"{desc} — train/eval 분리 확인 필요",
                ))
    return findings


def check_cache_invalidation(files: list[Path]) -> list[Finding]:
    """L6: embeddings/checkpoint cache가 split 무관하게 재사용되는지."""
    findings = []
    cache_patterns = [
        r"_embeddings\.pt|cache_embeddings|emb_cache",
        r"moe_best\.pt|_find_latest_checkpoint",
    ]
    for f in files:
        if not f.suffix == ".py":
            continue
        text = f.read_text()
        for pat in cache_patterns:
            for m in re.finditer(pat, text):
                line_no = text[:m.start()].count("\n") + 1
                if line_no > len(text.split("\n")):
                    continue
                line = text.split("\n")[line_no - 1].strip()
                if line.startswith("#") or '"""' in line:
                    continue
                findings.append(Finding(
                    "INFO", "L6", f, line_no, line[:120],
                    "cache/checkpoint 참조 — split-aware 재생성 필요한지 확인",
                ))
                break  # one per file
    return findings


def check_multi_seed_consistency(file: Path) -> list[Finding]:
    """L4: multi_seed가 seed별 일관된 split을 쓰는지."""
    findings = []
    text = file.read_text()
    if "random_state" not in text and "seed" not in text:
        findings.append(Finding(
            "MED", "L4", file, 1, "(no random_state/seed)",
            "multi-seed module이 seed 사용 안 함 — split 재현성 의심",
        ))
    return findings


def main():
    src_files = find_files(["run_pipeline.py", "src/**/*.py", "scripts/*.py"])
    src_files = [f for f in src_files if f.suffix == ".py"]

    print("=" * 78)
    print(" 데이터 누설 자동 감사")
    print("=" * 78)
    print(f"  검사 파일 수: {len(src_files)}")

    all_findings = []

    # L1: collect_records 사용
    print("\n[L1] _collect_records_and_embeddings 사용 패턴")
    for f in src_files:
        all_findings.extend(check_collect_records_use(f))

    # L2: threshold search input
    print("[L2] threshold τ search 입력 검사")
    for f in src_files:
        all_findings.extend(check_threshold_search_input(f))

    # L3: feature selection pool
    print("[L3] feature selection (bias-head, SAE) data pool 검사")
    all_findings.extend(check_feature_selection_pool(src_files))

    # L4: multi-seed consistency
    print("[L4] multi-seed split 일관성")
    ms_file = ROOT / "src" / "analysis" / "multi_seed.py"
    if ms_file.exists():
        all_findings.extend(check_multi_seed_consistency(ms_file))

    # L5: baseline training
    print("[L5] baseline 내부 학습 단계")
    all_findings.extend(check_baselines_no_training(src_files))

    # L6: cache invalidation
    print("[L6] embedding/checkpoint cache 재사용")
    all_findings.extend(check_cache_invalidation(src_files))

    # 정렬 + 요약
    sev_order = {"HIGH": 0, "MED": 1, "LOW": 2, "INFO": 3}
    all_findings.sort(key=lambda f: (sev_order.get(f.severity, 99), f.category, str(f.file)))

    print("\n" + "=" * 78)
    print(" Findings")
    print("=" * 78)

    by_sev = {"HIGH": 0, "MED": 0, "LOW": 0, "INFO": 0}
    for f in all_findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1

    for sev in ["HIGH", "MED", "LOW", "INFO"]:
        sev_findings = [f for f in all_findings if f.severity == sev]
        if not sev_findings:
            continue
        print(f"\n--- {sev} ({len(sev_findings)}건) ---")
        for f in sev_findings[:20]:  # 카테고리당 20개 cap
            print(f)
        if len(sev_findings) > 20:
            print(f"  ... +{len(sev_findings)-20}건")

    print("\n" + "=" * 78)
    print(f" Summary: HIGH={by_sev['HIGH']}, MED={by_sev['MED']}, LOW={by_sev['LOW']}, INFO={by_sev['INFO']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
