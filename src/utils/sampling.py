"""
BBQ 데이터 샘플링 스크립트.

7개 카테고리 각각에서 300개씩, 총 2,100개 instance를 샘플링합니다.
context_condition (ambig/disambig)에 대해 균등 샘플링하여 평가의 균형을 보장합니다.

사용법:
    python -m src.utils.sampling
"""

import argparse
from pathlib import Path

import yaml

from src.utils.data_loader import load_bbq_category, sample_category, save_sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="BBQ 데이터 샘플링")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="설정 파일 경로",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bbq_dir = config["data"]["bbq_dir"]
    sampled_dir = config["data"]["sampled_dir"]
    n_per_cat = config["data"]["samples_per_category"]
    seed = config["seed"]
    categories = config["data"]["categories"]

    print(f"[샘플링 시작] {len(categories)}개 카테고리, 카테고리당 {n_per_cat}개")
    print(f"  - BBQ 원본 경로: {bbq_dir}")
    print(f"  - 저장 경로: {sampled_dir}")
    print(f"  - Seed: {seed}")

    total_sampled = 0
    for category in categories:
        try:
            items = load_bbq_category(bbq_dir, category)
        except FileNotFoundError as e:
            print(f"  [건너뜀] {category}: {e}")
            continue

        sampled = sample_category(
            items,
            n=n_per_cat,
            seed=seed,
            stratify_by="context_condition",
        )
        path = save_sampled(sampled, sampled_dir, category)
        total_sampled += len(sampled)
        print(f"  {category:20s}: {len(items):>6,} -> {len(sampled)} ({path.name})")

    print(f"\n[완료] 총 {total_sampled}개 instance 샘플링됨.")


if __name__ == "__main__":
    main()
