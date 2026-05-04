"""
BBQ 데이터 로더 및 샘플링 유틸리티.

BBQ JSONL 파일을 로드하고 카테고리별로 random sampling을 수행합니다.
재현성을 위해 seed를 고정합니다.
"""

import json
import random
from pathlib import Path
from typing import Optional


def load_bbq_category(bbq_dir: str, category: str) -> list[dict]:
    """
    BBQ JSONL 파일에서 한 카테고리의 모든 instance를 로드합니다.

    Args:
        bbq_dir: BBQ JSONL 파일들이 있는 디렉토리 경로.
        category: 카테고리 이름 (예: "Age", "Gender_identity").

    Returns:
        BBQ instance 리스트 (각 항목은 dict).

    Raises:
        FileNotFoundError: 해당 카테고리 JSONL 파일이 없는 경우.
    """
    file_path = Path(bbq_dir) / f"{category}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"BBQ 파일 없음: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_category(
    items: list[dict],
    n: int,
    seed: int = 42,
    stratify_by: Optional[str] = "context_condition",
) -> list[dict]:
    """
    카테고리에서 n개 instance를 random sampling합니다.

    stratify_by가 주어지면 해당 필드 값별로 균등 샘플링합니다.
    예: context_condition이면 ambig/disambig 각각 n/2개.

    Args:
        items: 전체 instance 리스트.
        n: 샘플링할 개수.
        seed: 랜덤 시드.
        stratify_by: 균등 샘플링 기준 필드 (None이면 단순 랜덤).

    Returns:
        샘플링된 instance 리스트 (정확히 n개, 재현 가능).
    """
    rng = random.Random(seed)

    if stratify_by is None:
        return rng.sample(items, min(n, len(items)))

    # 필드 값별로 그룹핑
    groups: dict[str, list[dict]] = {}
    for item in items:
        key = item.get(stratify_by, "_unknown")
        groups.setdefault(key, []).append(item)

    # 각 그룹에서 n / len(groups) 개씩 샘플링
    per_group = n // len(groups)
    remainder = n - per_group * len(groups)

    sampled: list[dict] = []
    for i, (key, group_items) in enumerate(sorted(groups.items())):
        take = per_group + (1 if i < remainder else 0)
        sampled.extend(rng.sample(group_items, min(take, len(group_items))))

    rng.shuffle(sampled)
    return sampled[:n]


def load_sampled(sampled_dir: str, category: str) -> list[dict]:
    """
    이전에 샘플링되어 저장된 instance를 로드합니다.

    Args:
        sampled_dir: 샘플링 결과 저장 디렉토리.
        category: 카테고리 이름.

    Returns:
        샘플링된 instance 리스트.
    """
    file_path = Path(sampled_dir) / f"{category}_sampled.jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_sampled(items: list[dict], sampled_dir: str, category: str) -> Path:
    """
    샘플링 결과를 JSONL로 저장합니다.

    Args:
        items: 저장할 instance 리스트.
        sampled_dir: 저장 디렉토리.
        category: 카테고리 이름.

    Returns:
        저장된 파일 경로.
    """
    Path(sampled_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(sampled_dir) / f"{category}_sampled.jsonl"

    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return file_path


def format_question(item: dict) -> str:
    """
    BBQ instance를 LLM 입력용 질문 문자열로 변환합니다.

    Args:
        item: BBQ instance.

    Returns:
        Context + Question + 선택지로 구성된 프롬프트 문자열.
    """
    return (
        f"Context: {item['context']}\n"
        f"Question: {item['question']}\n"
        f"(A) {item['ans0']}\n"
        f"(B) {item['ans1']}\n"
        f"(C) {item['ans2']}\n"
        f"Answer:"
    )
