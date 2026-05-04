"""
프로젝트 디렉토리 구조 초기화 스크립트.

데이터, 결과, 로그 등의 디렉토리를 생성하고
.gitkeep을 두어 빈 디렉토리도 git에 포함되도록 합니다.

사용법:
    python setup_project.py
    python setup_project.py --root /path/to/project
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# 생성할 디렉토리 목록
DIRECTORIES: list[str] = [
    "data/bbq",
    "data/sampled",
    "data/cache",
    "src/signals",
    "src/models",
    "src/evaluation",
    "src/utils",
    "notebooks",
    "configs",
    "results/signals",
    "results/moe",
    "results/evaluation",
    "results/figures",
    "tests",
    "logs",
]

# .gitkeep을 둘 디렉토리 (빈 상태로 git에 포함하기 위함)
GITKEEP_DIRS: list[str] = [
    "data/bbq",
    "data/sampled",
    "data/cache",
    "results/signals",
    "results/moe",
    "results/evaluation",
    "results/figures",
    "logs",
]


def create_directories(root: Path) -> None:
    """
    프로젝트 디렉토리 구조를 생성합니다.

    Args:
        root: 프로젝트 루트 경로.
    """
    for d in DIRECTORIES:
        path = root / d
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  [디렉토리] {path}")


def create_gitkeeps(root: Path) -> None:
    """
    빈 디렉토리에 .gitkeep 파일을 생성합니다.

    Args:
        root: 프로젝트 루트 경로.
    """
    for d in GITKEEP_DIRS:
        gitkeep = root / d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
            logger.info(f"  [.gitkeep] {gitkeep}")


def main() -> None:
    parser = argparse.ArgumentParser(description="프로젝트 구조 초기화")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="프로젝트 루트 디렉토리 (기본: 현재 디렉토리)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    logger.info(f"[프로젝트 초기화] 루트: {root}")

    create_directories(root)
    create_gitkeeps(root)

    logger.info("[완료] 디렉토리 구조 생성 완료")


if __name__ == "__main__":
    main()
