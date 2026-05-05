"""
Paper figure generator — 논문용 PDF 7종.

각 figure는 results/ 의 데이터 (있는 만큼) 으로 생성. 데이터 부재 시 graceful skip.

Figure 1: Pipeline overview (ASCII/box diagram)
Figure 2: SAE bias-feature identification (3-method overlap)
Figure 3: MoE architecture diagram
Figure 4: Main BBQ results — bar chart (Ours vs baselines)
Figure 5: Cluster routing heatmap (category × cluster)
Figure 6: Open-set transfer (ImplicitBBQ + Open-BBQ + KoBBQ)
Figure 7: Qualitative analysis (top SAE features + max-activating examples)

CLI:
    python -m src.paper.figures --all
    python -m src.paper.figures --figs 4 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("paper.figures")

# 색상 팔레트 (consistent across figures)
COLORS = {
    "vanilla": "#808080",
    "composite": "#1f77b4",
    "self_debiasing": "#d62728",
    "decap": "#ff7f0e",
    "fairsteer": "#9467bd",
    "ours": "#2ca02c",
}


def _set_paper_style():
    try:
        import matplotlib as mpl
        mpl.rcParams.update({
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
    except ImportError:
        pass


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=200)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    logger.info(f"  [저장] {path}")


# =============================================================
# Figure 1: Pipeline overview
# =============================================================
def fig1_pipeline(save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("  matplotlib 미설치 — skip"); return

    _set_paper_style()
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.set_axis_off()

    boxes = [
        ("BBQ\nInstance", 0.5, 1.5, "#e8e8e8"),
        ("Stage 1:\n4-Prompt\nInference", 2.5, 1.5, "#ffe4b5"),
        ("Stage 2:\n7-Signal\nExtraction", 4.5, 1.5, "#ffd1a4"),
        ("Stage 3:\nMoE\nAggregator\n(4 Experts)", 6.5, 1.5, "#ffbb88"),
        ("Stage 4:\nThreshold\nOverride", 8.5, 1.5, "#ffa066"),
        ("Final\nAnswer", 10.5, 1.5, "#88dd88"),
    ]
    for text, x, y, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x - 0.7, y - 0.7), 1.4, 1.4,
            boxstyle="round,pad=0.04", linewidth=1.2,
            facecolor=color, edgecolor="black",
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=10)

    # arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + 0.7
        x2 = boxes[i + 1][1] - 0.7
        ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))

    # labels under boxes
    annotations = [
        (2.5, 0.4, "vanilla / debias /\ncot / cf-swap"),
        (4.5, 0.4, "s1-s7 signals"),
        (6.5, 0.4, "p ∈ [0,1]\nrouted by category"),
        (8.5, 0.4, "p < τ → unknown"),
    ]
    for x, y, text in annotations:
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                color="#555555", style="italic")

    ax.set_title("System Pipeline: 4-Stage Mechanism-Aware Debiasing", fontsize=13, pad=12)
    _save(fig, save_path)


# =============================================================
# Figure 2: SAE bias-feature identification (3 methods)
# =============================================================
def fig2_sae_identification(save_path: Path) -> None:
    """3-method 합집합/교집합 시각화. results/sae_layers/features_layer*.json 사용."""
    feat_files = sorted(Path("results").rglob("features_layer*.json"))
    if not feat_files:
        logger.warning("  features_layer*.json 없음 — skip"); return

    # layer 15 우선
    target = next(
        (f for f in feat_files if "layer15" in f.name), feat_files[0]
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    methods = data.get("method_features", {})
    if not methods:
        logger.warning("  method_features 없음 — skip"); return

    sets = {k: set(v) for k, v in methods.items()}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Bar: per-method count + union
    method_names = list(sets.keys()) + ["union"]
    counts = [len(sets[k]) for k in sets] + [len(set.union(*sets.values()))]
    colors_b = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(method_names)]
    ax1.bar(method_names, counts, color=colors_b, edgecolor="black")
    for i, c in enumerate(counts):
        ax1.text(i, c + 1, str(c), ha="center", fontsize=9)
    ax1.set_ylabel("# Bias features")
    ax1.set_title("Per-method count + union")

    # Pairwise overlap matrix
    keys = list(sets.keys())
    n = len(keys)
    matrix = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            inter = len(sets[ki] & sets[kj])
            matrix[i, j] = inter

    im = ax2.imshow(matrix, cmap="Blues")
    ax2.set_xticks(range(n)); ax2.set_xticklabels(keys, rotation=15)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(keys)
    ax2.set_title("Pairwise overlap (count)")
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{int(matrix[i,j])}", ha="center", va="center",
                     fontsize=10,
                     color="white" if matrix[i,j] > matrix.max()/2 else "black")
    fig.colorbar(im, ax=ax2)

    fig.suptitle(f"SAE Bias-Feature Identification ({target.stem})", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)


# =============================================================
# Figure 3: MoE architecture
# =============================================================
def fig3_moe_architecture(save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    _set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.set_axis_off()

    # Inputs
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 4.3), 1.6, 0.7, boxstyle="round,pad=0.04",
        facecolor="#e8e8e8", edgecolor="black"))
    ax.text(1.1, 4.65, "Signals\ns₁..s₇", ha="center", va="center", fontsize=10)

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 3.3), 1.6, 0.7, boxstyle="round,pad=0.04",
        facecolor="#e8e8e8", edgecolor="black"))
    ax.text(1.1, 3.65, "Question\nEmbedding", ha="center", va="center", fontsize=10)

    # Gating
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.0, 3.3), 2.0, 1.7, boxstyle="round,pad=0.04",
        facecolor="#ffd1a4", edgecolor="black"))
    ax.text(4.0, 4.15, "Gating Network\nsoftmax → 4 weights", ha="center",
            va="center", fontsize=10)

    # 4 Experts
    experts = ["Lex-Sub", "Numeric", "Cultural", "Identity"]
    for i, name in enumerate(experts):
        y = 5 - i * 1.2
        ax.add_patch(mpatches.FancyBboxPatch(
            (6.0, y - 0.4), 1.7, 0.8, boxstyle="round,pad=0.04",
            facecolor="#ffbb88", edgecolor="black"))
        ax.text(6.85, y, f"Expert: {name}", ha="center", va="center", fontsize=10)

    # Sum
    ax.add_patch(mpatches.Circle((9.0, 3.0), 0.4, facecolor="#88dd88", edgecolor="black"))
    ax.text(9.0, 3.0, "Σ", ha="center", va="center", fontsize=14, weight="bold")

    # Output
    ax.text(9.0, 1.7, "p ∈ [0, 1]\nconfidence", ha="center", va="center",
            fontsize=10, style="italic")

    # Arrows
    for i, _ in enumerate(experts):
        y = 5 - i * 1.2
        ax.annotate("", xy=(8.6, 3.0), xytext=(7.7, y),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#888888"))

    ax.annotate("", xy=(3.0, 4.15), xytext=(1.9, 3.65),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    for i in range(4):
        y = 5 - i * 1.2
        ax.annotate("", xy=(6.0, y), xytext=(1.9, 4.65),
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="#aaaaaa"))

    ax.annotate("", xy=(9.0, 2.5), xytext=(9.0, 2.6),
                arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_title("Mechanism-Aware MoE Aggregator", fontsize=13)
    _save(fig, save_path)


# =============================================================
# Figure 4: Main BBQ results bar chart
# =============================================================
def fig4_main_results(save_path: Path) -> None:
    """Ours vs baselines |bias_score_amb| bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _set_paper_style()

    # 사용 가능한 결과 수집
    methods_data: list[tuple[str, float, float, str]] = []  # (label, |bias|, FAR, color)

    # Ours
    ours_path = Path("results/evaluation/main/final.json")
    if ours_path.exists():
        d = json.loads(ours_path.read_text(encoding="utf-8"))
        m = d.get("metrics", {})
        bias = m.get("bias_score_amb")
        far = m.get("false_abstention_rate", 0.0)
        if bias is not None:
            methods_data.append(("Ours\n(τ=0.65)", abs(float(bias)), float(far), COLORS["ours"]))

    # Baselines
    for name, color, dirname in [
        ("Self-Debiasing", COLORS["self_debiasing"], "self_debiasing"),
        ("DeCAP", COLORS["decap"], "decap"),
        ("FairSteer", COLORS["fairsteer"], "fairsteer"),
        ("Composite", COLORS["composite"], "composite_prompting"),
    ]:
        p = Path(f"results/baselines/{dirname}/final.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d.get("overall", {})
        bias = m.get("bias_score_amb")
        far = m.get("false_abstention_rate", 0.0)
        if bias is not None:
            methods_data.append((name, abs(float(bias)), float(far), color))

    if len(methods_data) < 2:
        logger.warning("  비교 가능한 baseline 부족 — skip"); return

    # Sort by |bias| ascending (lower is better)
    methods_data.sort(key=lambda r: r[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    labels = [m[0] for m in methods_data]
    biases = [m[1] for m in methods_data]
    fars = [m[2] for m in methods_data]
    colors = [m[3] for m in methods_data]

    x = np.arange(len(labels))
    bars1 = ax1.bar(x, biases, color=colors, edgecolor="black")
    for bar, v in zip(bars1, biases):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("|Bias Score (ambig)|")
    ax1.set_title("Bias Reduction (lower = better)")
    ax1.grid(axis="y", linestyle=":", alpha=0.3)

    bars2 = ax2.bar(x, fars, color=colors, edgecolor="black")
    for bar, v in zip(bars2, fars):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("False Abstention Rate")
    ax2.set_title("Over-correction (lower = better)")
    ax2.grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle(f"BBQ Main Results (Llama-3.1-8B, n=2,097)", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)


# =============================================================
# Figure 5: Cluster routing heatmap
# =============================================================
def fig5_cluster_routing(save_path: Path) -> None:
    """src/analysis/qualitative.py의 routing heatmap을 그대로 호출."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.analysis.qualitative import run_cluster_routing_heatmap
    out_dir = save_path.parent
    run_cluster_routing_heatmap(out_dir)
    # rename if needed
    src = out_dir / "cluster_routing_heatmap.pdf"
    if src.exists() and src != save_path:
        src.rename(save_path)
        logger.info(f"  rename {src} → {save_path}")


# =============================================================
# Figure 6: Open-set transfer
# =============================================================
def fig6_open_set(save_path: Path) -> None:
    """ImplicitBBQ-style + Open-BBQ + KoBBQ overall metrics 비교."""
    transfer_results: dict[str, dict] = {}
    for name, p in [
        ("ImplicitBBQ-style", "results/transfer/implicit_bbq/overall_metrics.json"),
        ("Open-BBQ", "results/transfer/open_bbq/overall_metrics.json"),
        ("KoBBQ", "results/transfer/kobbq/overall_metrics.json"),
    ]:
        if Path(p).exists():
            transfer_results[name] = json.loads(Path(p).read_text(encoding="utf-8"))

    if not transfer_results:
        logger.warning("  transfer 결과 없음 — skip"); return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = list(transfer_results.keys())
    accs_amb = [transfer_results[m]["overall"].get("accuracy_amb", 0) for m in methods]
    accs_dis = [transfer_results[m]["overall"].get("accuracy_dis", 0) for m in methods]
    biases = [
        abs(transfer_results[m]["overall"].get("bias_score_amb") or 0) for m in methods
    ]

    x = np.arange(len(methods))
    width = 0.27
    ax.bar(x - width, accs_amb, width, label="acc_amb", color="#2ca02c", edgecolor="black")
    ax.bar(x, accs_dis, width, label="acc_dis", color="#1f77b4", edgecolor="black")
    ax.bar(x + width, biases, width, label="|bias_amb|", color="#d62728", edgecolor="black")

    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("Open-set Transfer (zero-shot)")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


# =============================================================
# Figure 7: Qualitative SAE features
# =============================================================
def fig7_qualitative(save_path: Path) -> None:
    """Top SAE features bar chart by Δs7 contribution (sae_layer_comparison output)."""
    cmp_csv = Path("results/sae_layers/comparison.csv")
    if not cmp_csv.exists():
        # fallback to results/v1/sae_layers/...
        for p in Path("results").rglob("sae_layers/comparison.csv"):
            cmp_csv = p
            break
    if not cmp_csv.exists():
        logger.warning("  sae_layers/comparison.csv 없음 — skip"); return

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _set_paper_style()
    df = pd.read_csv(cmp_csv)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    layers = df["layer"].astype(int).tolist()
    deltas = df["s7_delta_loss"].astype(float).tolist()
    bars = ax.bar(layers, deltas, color="#9467bd", edgecolor="black")
    for bar, L in zip(bars, layers):
        if L == 15:
            bar.set_color("#ff7f0e")  # 현재 default 강조

    best_idx = max(range(len(deltas)), key=lambda i: deltas[i])
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(2.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ val_loss (s7 contribution)")
    ax.set_title("SAE Layer Comparison — s7 contribution per layer")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    for x, d in zip(layers, deltas):
        ax.text(x, d + (max(deltas) - min(deltas)) * 0.02, f"{d:+.4f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)


# =============================================================
# Driver
# =============================================================
FIG_FNS = {
    1: fig1_pipeline,
    2: fig2_sae_identification,
    3: fig3_moe_architecture,
    4: fig4_main_results,
    5: fig5_cluster_routing,
    6: fig6_open_set,
    7: fig7_qualitative,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper figure generator")
    parser.add_argument("--all", action="store_true", help="모든 figure 생성")
    parser.add_argument("--figs", type=int, nargs="+", default=None,
                        help="생성할 figure 번호 (1-7)")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(FIG_FNS.keys()) if args.all else (args.figs or [])
    if not targets:
        logger.error("--all 또는 --figs 1 4 5 ...")
        return 2

    for n in targets:
        if n not in FIG_FNS:
            logger.warning(f"  unknown figure: {n}")
            continue
        logger.info(f"\n=== Figure {n} ===")
        save_path = out_dir / f"fig{n}_{FIG_FNS[n].__name__.replace('fig' + str(n) + '_', '')}.pdf"
        try:
            FIG_FNS[n](save_path)
        except Exception as e:
            logger.error(f"  Figure {n} 실패: {e}")

    logger.info(f"\n완료: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
