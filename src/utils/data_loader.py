"""
BBQ 데이터 로더 및 샘플링 유틸리티.

기능:
    - BBQ 원본 데이터셋 git clone으로 다운로드
    - 7개 카테고리 선택 + 카테고리당 random sampling (seed 고정)
    - context_condition (ambig/disambig)에 대한 stratified sampling
    - Train/Val/Test split (70/15/15)
    - parquet 형식으로 저장 (재현성 + 빠른 로드)

함수 시그니처:
    download_bbq(data_dir) -> None
    sample_bbq(data_dir, categories, n_per_category, seed) -> pd.DataFrame
    split_data(df, train_ratio, val_ratio, test_ratio) -> (train, val, test)
    load_split(sampled_dir, split) -> pd.DataFrame
    format_question(item) -> str

CLI 사용법:
    python -m src.utils.data_loader --download
    python -m src.utils.data_loader --sample
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BBQ_REPO_URL = "https://github.com/nyu-mll/BBQ.git"

# v1: 7 카테고리 (법적 보호 특성 기반) — 기존 실험 호환
DEFAULT_CATEGORIES: list[str] = [
    "Gender_identity",
    "Race_ethnicity",
    "Age",
    "Religion",
    "Disability_status",
    "SES",
    "Sexual_orientation",
]

# v2: BBQ 9 카테고리 전체 (Nationality, Physical_appearance 추가)
DEFAULT_CATEGORIES_V2: list[str] = DEFAULT_CATEGORIES + [
    "Nationality",
    "Physical_appearance",
]


def get_categories(version: str = "v1") -> list[str]:
    """version에 따라 default 카테고리 리스트 반환.

    smoke: 9 카테고리 (v2와 동일) — smoke test용으로 카테고리당 5개 샘플링
    mini: 9 카테고리 (v2와 동일) — 검증용 9 × 100
    v2: 9 카테고리, 카테고리당 1000
    v1: 7 카테고리 (법적 보호 특성), 카테고리당 300
    """
    if version in ("v2", "smoke", "mini"):
        return list(DEFAULT_CATEGORIES_V2)
    return list(DEFAULT_CATEGORIES)


# =============================================================
# 1. 다운로드
# =============================================================
def download_bbq(data_dir: str) -> None:
    """
    BBQ 원본 데이터셋을 git clone으로 다운로드합니다.

    BBQ repo의 data/ 폴더 안에 있는 JSONL 파일들을 data_dir로 복사합니다.
    이미 데이터가 존재하면 다시 받지 않습니다.

    Args:
        data_dir: BBQ JSONL 파일들이 저장될 디렉토리.

    Raises:
        RuntimeError: git clone 실패 시.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # 이미 충분히 받았는지 확인
    existing = list(data_path.glob("*.jsonl"))
    if len(existing) >= 9:  # BBQ는 9개 단일 + 2개 교차 카테고리
        logger.info(f"[다운로드] {len(existing)}개 JSONL 이미 존재, 건너뜀")
        return

    # 임시 clone 위치
    tmp_clone = data_path.parent / "_bbq_tmp_clone"
    if tmp_clone.exists():
        shutil.rmtree(tmp_clone)

    logger.info(f"[다운로드] BBQ repo clone -> {tmp_clone}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", BBQ_REPO_URL, str(tmp_clone)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"BBQ git clone 실패: {e.stderr.decode('utf-8', errors='ignore')}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "git이 설치되어 있지 않습니다. Mac: `brew install git` 또는 Xcode CLT 설치."
        ) from e

    # JSONL만 복사
    src_data_dir = tmp_clone / "data"
    if not src_data_dir.exists():
        raise RuntimeError(f"BBQ data/ 폴더 없음: {src_data_dir}")

    copied = 0
    for jsonl in src_data_dir.glob("*.jsonl"):
        dest = data_path / jsonl.name
        shutil.copy2(jsonl, dest)
        copied += 1
        logger.info(f"  [복사] {jsonl.name}")

    # 임시 clone 정리
    shutil.rmtree(tmp_clone, ignore_errors=True)

    logger.info(f"[다운로드 완료] {copied}개 JSONL -> {data_path}")


# =============================================================
# 2. 로드 및 샘플링
# =============================================================
def load_bbq_category(data_dir: str | Path, category: str) -> list[dict]:
    """
    한 카테고리의 BBQ JSONL 파일을 로드합니다.

    Args:
        data_dir: BBQ JSONL 디렉토리.
        category: 카테고리 이름 (예: "Age").

    Returns:
        BBQ instance 리스트.

    Raises:
        FileNotFoundError: 파일 없음.
    """
    file_path = Path(data_dir) / f"{category}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"BBQ 파일 없음: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        items: list[dict] = []
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec.setdefault("category", category)
                items.append(rec)
        return items


def sample_bbq(
    data_dir: str,
    categories: list[str] | None = None,
    n_per_category: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    카테고리별 random sampling을 수행하여 통합 DataFrame을 만듭니다.

    context_condition (ambig/disambig)에 대해 stratified sampling을 수행하여
    두 조건이 균등하게 포함되도록 합니다.

    Args:
        data_dir: BBQ JSONL 디렉토리.
        categories: 사용할 카테고리 리스트. None이면 DEFAULT_CATEGORIES 사용.
        n_per_category: 카테고리당 샘플 수.
        seed: 랜덤 시드 (재현성).

    Returns:
        샘플링된 instance를 담은 DataFrame.
        컬럼: example_id, category, context_condition, question_polarity,
              context, question, ans0, ans1, ans2, label, answer_info,
              additional_metadata.
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES

    rng = np.random.RandomState(seed)
    all_records: list[dict] = []

    for category in categories:
        try:
            items = load_bbq_category(data_dir, category)
        except FileNotFoundError as e:
            logger.warning(f"  [건너뜀] {e}")
            continue

        sampled = _stratified_sample(
            items, n=n_per_category, rng=rng, stratify_key="context_condition"
        )
        all_records.extend(sampled)
        logger.info(f"  {category:20s}: {len(items):>6,} -> {len(sampled)}")

    df = pd.DataFrame(all_records)
    logger.info(f"[샘플링 완료] 총 {len(df):,}개 instance")
    return df


def _stratified_sample(
    items: list[dict],
    n: int,
    rng: np.random.RandomState,
    stratify_key: str = "context_condition",
) -> list[dict]:
    """
    필드 값별로 균등하게 샘플링합니다.

    Args:
        items: 전체 instance 리스트.
        n: 샘플링할 총 개수.
        rng: numpy RandomState (재현성).
        stratify_key: 균등 샘플링 기준 필드.

    Returns:
        n개 샘플 리스트 (그룹 균등).
    """
    groups: dict[str, list[dict]] = {}
    for item in items:
        key = item.get(stratify_key, "_unknown")
        groups.setdefault(key, []).append(item)

    keys_sorted = sorted(groups)
    per_group = n // len(keys_sorted)
    remainder = n - per_group * len(keys_sorted)

    sampled: list[dict] = []
    for i, key in enumerate(keys_sorted):
        take = per_group + (1 if i < remainder else 0)
        group_items = groups[key]
        if take >= len(group_items):
            sampled.extend(group_items)
        else:
            indices = rng.choice(len(group_items), size=take, replace=False)
            sampled.extend(group_items[j] for j in indices)

    # 카테고리 순서 섞기
    indices = rng.permutation(len(sampled))
    return [sampled[i] for i in indices][:n]


# =============================================================
# 3. Split (Train/Val/Test)
# =============================================================
def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_cols: tuple[str, ...] = ("category", "context_condition"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train/Val/Test split을 수행합니다.

    카테고리와 맥락 조건에 대해 stratified split을 수행하여
    각 split이 동일한 분포를 가지도록 합니다.

    Args:
        df: 샘플링된 instance DataFrame.
        train_ratio: train 비율 (기본 0.7).
        val_ratio: val 비율 (기본 0.15).
        test_ratio: test 비율 (기본 0.15).
        seed: 랜덤 시드.
        stratify_cols: stratify 기준 컬럼.

    Returns:
        (train_df, val_df, test_df) 튜플.

    Raises:
        ValueError: 비율 합이 1이 아닌 경우.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"비율 합이 1이 아닙니다: {total}")

    rng = np.random.RandomState(seed)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    # stratify 그룹별로 split
    for _, group in df.groupby(list(stratify_cols), observed=True):
        n = len(group)
        if n == 0:
            continue

        indices = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # test는 나머지 전부

        train_parts.append(group.iloc[indices[:n_train]])
        val_parts.append(group.iloc[indices[n_train:n_train + n_val]])
        test_parts.append(group.iloc[indices[n_train + n_val:]])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(
        f"[Split] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )
    return train_df, val_df, test_df


# =============================================================
# 4. 저장 및 로드
# =============================================================
def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sampled_dir: str | Path,
) -> dict[str, Path]:
    """
    Train/Val/Test split을 parquet으로 저장합니다.

    parquet은 JSONL보다 빠르게 로드되고 컬럼 단위 압축이 가능합니다.

    Args:
        train_df, val_df, test_df: 각 split DataFrame.
        sampled_dir: 저장 디렉토리.

    Returns:
        {"train": path, "val": path, "test": path} 딕셔너리.
    """
    out_dir = Path(sampled_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out_dir / "train.parquet",
        "val": out_dir / "val.parquet",
        "test": out_dir / "test.parquet",
    }

    for name, df in zip(("train", "val", "test"), (train_df, val_df, test_df)):
        # dict 컬럼은 parquet 저장을 위해 JSON 문자열로 직렬화
        df_to_save = df.copy()
        for col in df_to_save.columns:
            if df_to_save[col].dtype == object:
                sample = df_to_save[col].dropna().iloc[0] if not df_to_save[col].dropna().empty else None
                if isinstance(sample, (dict, list)):
                    df_to_save[col] = df_to_save[col].apply(
                        lambda x: json.dumps(x, ensure_ascii=False) if x is not None else None
                    )
        df_to_save.to_parquet(paths[name], index=False, compression="snappy")
        logger.info(f"  [저장] {name}: {paths[name]} ({len(df):,}개)")

    return paths


def load_split(sampled_dir: str | Path, split: str) -> pd.DataFrame:
    """
    저장된 split을 로드합니다.

    Args:
        sampled_dir: parquet 저장 디렉토리.
        split: "train", "val", "test" 중 하나.

    Returns:
        해당 split의 DataFrame. dict 컬럼은 JSON 문자열에서 복원.
    """
    path = Path(sampled_dir) / f"{split}.parquet"
    df = pd.read_parquet(path)

    # JSON 문자열 컬럼을 dict/list로 복원
    for col in ("answer_info", "additional_metadata"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
    return df


# =============================================================
# 5. 프롬프트 포맷팅
# =============================================================
def format_question(item: dict | pd.Series) -> str:
    """
    BBQ instance를 LLM 입력용 프롬프트 문자열로 변환합니다.

    Args:
        item: BBQ instance (dict 또는 pandas Row).

    Returns:
        Context + Question + (A)/(B)/(C) 선택지로 구성된 프롬프트.
    """
    return (
        f"Context: {item['context']}\n"
        f"Question: {item['question']}\n"
        f"(A) {item['ans0']}\n"
        f"(B) {item['ans1']}\n"
        f"(C) {item['ans2']}\n"
        f"Answer:"
    )


# =============================================================
# CLI
# =============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="BBQ 데이터 다운로드 + 샘플링 + Split")
    parser.add_argument("--data-dir", type=str, default="data/bbq",
                        help="BBQ 원본 JSONL 디렉토리")
    parser.add_argument("--sampled-dir", type=str, default=None,
                        help="샘플링 결과 parquet 디렉토리. 미지정 시 version에 따라 자동")
    parser.add_argument("--n-per-category", type=int, default=None,
                        help="카테고리당 샘플 수. 미지정 시 version에 따라 자동 (v1=300, v2=1000)")
    parser.add_argument("--version", type=str, default="v1",
                        choices=("v1", "v2", "smoke", "mini"),
                        help="실험 버전. v1=7×300, v2=9×1000, smoke=9×5, mini=9×100 (검증)")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    parser.add_argument("--download", action="store_true",
                        help="BBQ 원본 다운로드")
    parser.add_argument("--sample", action="store_true",
                        help="샘플링 + split 수행")
    parser.add_argument("--all", action="store_true",
                        help="다운로드 + 샘플링 + split 모두 수행")
    args = parser.parse_args()

    # version 기반 default 적용
    if args.sampled_dir is None:
        args.sampled_dir = {
            "v1": "data/sampled",
            "v2": "data/sampled_v2",
            "smoke": "data/sampled_smoke",
            "mini": "data/sampled_mini",
        }[args.version]
    if args.n_per_category is None:
        args.n_per_category = {"v1": 300, "v2": 1000, "smoke": 5, "mini": 100}[args.version]
    cats = get_categories(args.version)

    if args.all or args.download:
        logger.info("=" * 60)
        logger.info("STEP 1: BBQ 다운로드")
        logger.info("=" * 60)
        download_bbq(args.data_dir)

    if args.all or args.sample:
        logger.info("=" * 60)
        logger.info(f"STEP 2: 카테고리별 샘플링 (version={args.version}, "
                    f"{len(cats)} cats × {args.n_per_category})")
        logger.info("=" * 60)
        df = sample_bbq(
            data_dir=args.data_dir,
            categories=cats,
            n_per_category=args.n_per_category,
            seed=args.seed,
        )

        logger.info("=" * 60)
        logger.info("STEP 3: Train/Val/Test Split")
        logger.info("=" * 60)
        train_df, val_df, test_df = split_data(
            df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed,
        )

        logger.info("=" * 60)
        logger.info("STEP 4: parquet 저장")
        logger.info("=" * 60)
        save_splits(train_df, val_df, test_df, args.sampled_dir)

    logger.info("[완료]")


if __name__ == "__main__":
    main()
