#!/usr/bin/env python3
"""Generate compact revision figures for the IEEE Access manuscript."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", format="png")


def pareto_tradeoff(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    set_style()
    rows = [
        ("Primary only", 0.0717, 0.5596, "#7f7f7f", "o", (8, 9), "left"),
        ("Composite", 0.2449, 0.6843, "#b0b0b0", "s", (9, -9), "left"),
        ("DeCAP", 0.2419, 0.8057, "#888888", "s", (9, 2), "left"),
        ("MoE single τ", 0.1325, 0.9494, "#2b6cb0", "^", (8, -15), "left"),
        ("MoE predicted", 0.0843, 0.9946, "#2b6cb0", "D", (10, -18), "left"),
        ("Condition-only", 0.0726, 1.0000, "#1b9e77", "o", (-7, 22), "center"),
        ("Hybrid", 0.0723, 0.9979, "#d95f02", "P", (5, -34), "left"),
    ]

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for label, far, acc_amb, color, marker, offset, ha in rows:
        ax.scatter(far, acc_amb, s=28, color=color, marker=marker, edgecolor="black", linewidth=0.35, zorder=3)
        ax.annotate(
            label,
            xy=(far, acc_amb),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.8,
            ha=ha,
            arrowprops=dict(arrowstyle="-", lw=0.35, color="#666666"),
        )

    ax.set_xlabel("False Abstention Rate")
    ax.set_ylabel("Ambiguous Accuracy")
    ax.set_xlim(0.04, 0.285)
    ax.set_ylim(0.52, 1.015)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    save(fig, out_path)
    plt.close(fig)


def routing_heatmap(out_dir: Path) -> None:
    """Regenerate the routing heatmap with neutral expert labels."""
    from src.analysis.qualitative import run_cluster_routing_heatmap

    run_cluster_routing_heatmap(out_dir)
    src_pdf = out_dir / "cluster_routing_heatmap.pdf"
    src_png = out_dir / "cluster_routing_heatmap.png"
    dst_pdf = out_dir / "fig5_cluster_routing.pdf"
    dst_png = out_dir / "fig5_cluster_routing.png"
    if src_pdf.exists():
        src_pdf.replace(dst_pdf)
    if src_png.exists():
        src_png.replace(dst_png)


def main() -> int:
    paper_dir = ROOT / "paper" / "ieee_access" / "figures"
    docs_dir = ROOT / "docs" / "figures"
    pareto_tradeoff(paper_dir / "fig_tradeoff_pareto.pdf")
    routing_heatmap(paper_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ["fig_tradeoff_pareto.pdf", "fig_tradeoff_pareto.png", "fig5_cluster_routing.pdf", "fig5_cluster_routing.png"]:
        src = paper_dir / name
        if src.exists():
            shutil.copy2(src, docs_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
