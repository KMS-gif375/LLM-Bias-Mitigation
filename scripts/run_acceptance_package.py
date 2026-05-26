#!/usr/bin/env python3
"""
Orchestrate the reviewer-defense experiment package.

Default steps:
  1. Clean LOCO 5 seeds
  2. Open-BBQ fresh transfer
  3. Qwen/Mistral 5-seed summaries from existing signals
  4. Paper/appendix report generation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full acceptance-defense package.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["loco", "openbbq", "crossllm", "report"],
        choices=["loco", "openbbq", "crossllm", "report"],
    )
    parser.add_argument("--seeds", default="42,123,456,789,999")
    parser.add_argument("--version", default="v2")
    parser.add_argument("--model", default="main")
    parser.add_argument("--results-dir", default="results/v2")
    parser.add_argument("--package-dir", default="results/v2/acceptance_package")
    parser.add_argument(
        "--openbbq-max-samples",
        type=int,
        default=300,
        help="Per-category cap for Open-BBQ; 300 gives about 3,300 examples over 11 categories.",
    )
    parser.add_argument("--force-openbbq", action="store_true", default=True)
    parser.add_argument("--no-force-openbbq", dest="force_openbbq", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print()
    print("=" * 80)
    print(" ".join(cmd))
    print("=" * 80)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    package_dir = Path(args.package_dir)
    seeds_space = args.seeds.replace(",", " ").split()
    seeds_csv = ",".join(seeds_space)

    if "loco" in args.steps:
        run_cmd(
            [
                sys.executable,
                "scripts/run_loco_clean.py",
                "--model",
                args.model,
                "--results-dir",
                args.results_dir,
                "--out-dir",
                str(package_dir / "loco"),
                "--seeds",
                *seeds_space,
            ],
            dry_run=args.dry_run,
        )

    if "openbbq" in args.steps:
        cmd = [
            sys.executable,
            "-m",
            "src.transfer.run_open_bbq",
            "--max-samples",
            str(args.openbbq_max_samples),
            "--out-dir",
            str(package_dir / "open_bbq"),
            "--model",
            args.model,
        ]
        if args.force_openbbq:
            cmd.append("--force")
        run_cmd(cmd, dry_run=args.dry_run)

    if "crossllm" in args.steps:
        for model in ("qwen", "mistral"):
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "src.analysis.multi_seed",
                    "--version",
                    args.version,
                    "--model",
                    model,
                    "--seeds",
                    seeds_csv,
                    "--out-dir",
                    f"results/v2/cross_llm/{model}/multi_seed_5seed",
                ],
                dry_run=args.dry_run,
            )

    if "report" in args.steps:
        run_cmd(
            [
                sys.executable,
                "scripts/build_acceptance_report.py",
                "--out-dir",
                str(package_dir / "report"),
                "--loco-dir",
                str(package_dir / "loco"),
                "--openbbq-dir",
                str(package_dir / "open_bbq"),
            ],
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
