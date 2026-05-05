"""
Open-BBQ (zhaoliu0914/LLM-Bias-Benchmark, Open-DeBias 2025) preparation.

GitHub: https://github.com/zhaoliu0914/LLM-Bias-Benchmark

저장소 형식:
    data/{category}_{condition}_multiple_choice_{model}.jsonl
        — OpenAI batch API 요청 형식 (system/user content)
    metadata/{category}_{condition}_metadata.jsonl
        — custom_id, question_polarity, answer_info, label, target_bias

본 모듈은 원본 데이터를 BBQ schema로 변환하여 우리 평가 파이프라인
(src/transfer/run_openbias.py 등)에서 그대로 사용할 수 있게 합니다.

변환 매핑:
    custom_id → example_id (string)
    custom_id 접두사 → category (예: "age" → "Age", "race_ethnicity" → "Race_ethnicity")
    custom_id 중간 → context_condition ("ambiguous" → "ambig", "disambiguated" → "disambig")
    metadata → answer_info, label, question_polarity
    user content (data 파일) → context + question + options 파싱
    target_bias → additional_metadata.stereotyped_groups 도출

CLI:
    # 자동: 저장소 clone + 변환
    python -m src.data.prepare_open_bbq --auto

    # 이미 clone한 저장소에서 변환
    python -m src.data.prepare_open_bbq --repo-dir /tmp/LLM-Bias-Benchmark

    # 출력 디렉토리 변경
    python -m src.data.prepare_open_bbq --auto --out-dir data/open_bbq/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("prepare_open_bbq")


REPO_URL = "https://github.com/zhaoliu0914/LLM-Bias-Benchmark.git"

# custom_id prefix → BBQ category 표준 명칭
CATEGORY_MAPPING: dict[str, str] = {
    "age": "Age",
    "disability_status": "Disability_status",
    "gender_identity": "Gender_identity",
    "nationality": "Nationality",
    "physical_appearance": "Physical_appearance",
    "race_ethnicity": "Race_ethnicity",
    "race_x_gender": "Race_x_gender",
    "race_x_ses": "Race_x_SES",
    "religion": "Religion",
    "ses": "SES",
    "sexual_orientation": "Sexual_orientation",
}


def _normalize_category(prefix: str) -> str:
    return CATEGORY_MAPPING.get(prefix, prefix)


def _parse_custom_id(custom_id: str) -> Optional[tuple[str, str, int]]:
    """
    'age-ambiguous-1' → ('Age', 'ambig', 1).
    Returns None on failure.
    """
    # 알려진 카테고리 prefix 우선 매칭 (longest match)
    # custom_id 형식: "{category}-{condition}-{index}"
    # category는 underscore 포함 가능 (예: "disability_status-ambiguous-1").
    sorted_prefixes = sorted(CATEGORY_MAPPING.keys(), key=len, reverse=True)
    for raw_prefix in sorted_prefixes:
        prefix_with_dash = raw_prefix + "-"
        if custom_id.startswith(prefix_with_dash):
            rest = custom_id[len(prefix_with_dash):]
            cat = CATEGORY_MAPPING[raw_prefix]
            m = re.match(r"^(ambiguous|disambiguated)-(\d+)$", rest)
            if not m:
                return None
            cond = "ambig" if m.group(1) == "ambiguous" else "disambig"
            idx = int(m.group(2))
            return cat, cond, idx
    return None


# user content 형식: "...context... \n Who ...? \n (0) opt0 (1) opt1 (2) opt2 "
_OPTION_PATTERN = re.compile(r"\((\d+)\)\s*([^()]+?)(?=\s*\(\d+\)|$)", re.DOTALL)


def _parse_user_content(content: str) -> Optional[dict]:
    """
    user message에서 context, question, ans0/1/2 추출.

    형식 (예시):
        "I saw a grandson and their grandfather last week ... \n
         Who was not comfortable using the phone? \n
         (0) grandfather (1) grandson (2) Can't be determined "
    """
    if not content:
        return None
    text = content.strip()
    # Question은 보통 " ? " 로 끝나거나 "?" 다음 newline
    # \n으로 split 후 마지막 줄에 "(0) ... (1) ... (2) ..."
    parts = text.split("\n")
    # 마지막 줄에 옵션이 있을 가능성 — back-search
    options_line = None
    options_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if "(0)" in parts[i] and "(1)" in parts[i] and "(2)" in parts[i]:
            options_line = parts[i].strip()
            options_idx = i
            break
    if options_idx is None:
        return None

    # question은 options 직전 줄
    question = ""
    for i in range(options_idx - 1, -1, -1):
        line = parts[i].strip()
        if line:
            question = line
            break
    # context는 question 이전 모든 줄
    context_lines = []
    if options_idx >= 2:
        for i in range(options_idx - 1):
            stripped = parts[i].strip()
            if stripped and stripped != question:
                context_lines.append(stripped)
    context = " ".join(context_lines).strip()

    # 옵션 파싱 (정규식)
    matches = _OPTION_PATTERN.findall(options_line)
    options = {int(num): opt.strip() for num, opt in matches}
    if not all(i in options for i in (0, 1, 2)):
        return None

    return {
        "context": context,
        "question": question,
        "ans0": options[0],
        "ans1": options[1],
        "ans2": options[2],
    }


# =============================================================
# Repo clone
# =============================================================
def ensure_repo(repo_dir: str | Path) -> Path:
    repo_path = Path(repo_dir)
    if repo_path.exists() and any(repo_path.iterdir()):
        logger.info(f"  [repo] 기존 디렉토리 사용: {repo_path}")
        return repo_path

    logger.info(f"  [repo] git clone {REPO_URL} → {repo_path}")
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", REPO_URL, str(repo_path)],
        check=True,
    )
    return repo_path


# =============================================================
# Conversion
# =============================================================
def convert(
    repo_dir: str | Path,
    out_dir: str | Path = "data/open_bbq",
    model_filter: str = "gpt4o",
) -> dict:
    """
    Open-BBQ 데이터를 BBQ schema JSONL로 변환합니다.

    Args:
        repo_dir: clone된 저장소 경로.
        out_dir: 출력 디렉토리 (data/open_bbq/).
        model_filter: 어느 모델용 prompt 파일을 사용할지 (paraphrase는 동일).
                      default "gpt4o".

    Returns:
        {category: n_records} 통계.
    """
    repo_path = Path(repo_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata_dir = repo_path / "metadata"
    data_dir = repo_path / "data"
    if not metadata_dir.exists() or not data_dir.exists():
        raise FileNotFoundError(
            f"repo 구조 비정상: {metadata_dir} 또는 {data_dir} 없음"
        )

    # 1. metadata 통합 로드 (custom_id → metadata)
    metadata_by_id: dict[str, dict] = {}
    for meta_file in sorted(metadata_dir.glob("*_metadata.jsonl")):
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    metadata_by_id[rec["custom_id"]] = rec
    logger.info(f"  metadata loaded: {len(metadata_by_id)} entries")

    # 2. data 통합 로드 (custom_id → user content)
    data_files = sorted(data_dir.glob(f"*_multiple_choice_{model_filter}.jsonl"))
    if not data_files:
        # fallback
        data_files = sorted(data_dir.glob("*_multiple_choice_*.jsonl"))
        logger.warning(f"  model_filter={model_filter} 매칭 없음 — 모든 model 사용")

    user_content_by_id: dict[str, str] = {}
    for df in data_files:
        with open(df, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec.get("custom_id")
                if not cid or cid in user_content_by_id:
                    continue  # 중복 skip
                msgs = rec.get("body", {}).get("messages", [])
                user_msg = next(
                    (m["content"] for m in msgs if m.get("role") == "user"),
                    "",
                )
                if user_msg:
                    user_content_by_id[cid] = user_msg
    logger.info(f"  data files: {len(data_files)} files, {len(user_content_by_id)} unique entries")

    # 3. 변환 + 카테고리별 저장
    by_cat: dict[str, list[dict]] = defaultdict(list)
    n_skipped_no_metadata = 0
    n_skipped_parse = 0

    for cid, user_content in user_content_by_id.items():
        meta = metadata_by_id.get(cid)
        if not meta:
            n_skipped_no_metadata += 1
            continue

        parsed_id = _parse_custom_id(cid)
        if not parsed_id:
            continue
        category, cond, _ = parsed_id

        parsed_uc = _parse_user_content(user_content)
        if not parsed_uc:
            n_skipped_parse += 1
            continue

        # stereotyped_groups derivation
        target_bias = meta.get("target_bias")
        stereotyped_groups: list[str] = []
        ans_info = meta.get("answer_info", {})
        if isinstance(target_bias, int) and target_bias in (0, 1):
            tb_info = ans_info.get(f"ans{target_bias}", [])
            if len(tb_info) >= 2:
                stereotyped_groups = [tb_info[1]]

        bbq_record = {
            "example_id": cid,
            "category": category,
            "context_condition": cond,
            "question_polarity": meta.get("question_polarity", "neg"),
            "context": parsed_uc["context"],
            "question": parsed_uc["question"],
            "ans0": parsed_uc["ans0"],
            "ans1": parsed_uc["ans1"],
            "ans2": parsed_uc["ans2"],
            "label": int(meta.get("label", -1)),
            "answer_info": ans_info,
            "additional_metadata": {
                "stereotyped_groups": stereotyped_groups,
                "source": "open_bbq",
                "target_bias": target_bias,
            },
        }
        by_cat[category].append(bbq_record)

    # 4. 카테고리별 JSONL 저장
    stats: dict[str, int] = {}
    for cat, recs in by_cat.items():
        cat_path = out_path / f"{cat}.jsonl"
        with open(cat_path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[cat] = len(recs)
        logger.info(f"  [{cat}] {len(recs)} → {cat_path}")

    summary = {
        "out_dir": str(out_path),
        "model_filter": model_filter,
        "categories": stats,
        "total": sum(stats.values()),
        "skipped_no_metadata": n_skipped_no_metadata,
        "skipped_parse": n_skipped_parse,
    }
    (out_path / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"\n  완료: total={summary['total']}, "
        f"skipped(metadata)={n_skipped_no_metadata}, "
        f"skipped(parse)={n_skipped_parse}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Open-BBQ → BBQ schema converter")
    parser.add_argument("--auto", action="store_true",
                        help="자동으로 git clone (default tmp 위치)")
    parser.add_argument("--repo-dir", type=str, default="/tmp/LLM-Bias-Benchmark",
                        help="이미 clone한 저장소 경로 (--auto 시 clone 위치)")
    parser.add_argument("--out-dir", type=str, default="data/open_bbq")
    parser.add_argument("--model-filter", type=str, default="gpt4o",
                        help="어느 모델용 prompt 파일 사용 (default gpt4o)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.auto:
        ensure_repo(args.repo_dir)

    convert(
        repo_dir=args.repo_dir,
        out_dir=args.out_dir,
        model_filter=args.model_filter,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
