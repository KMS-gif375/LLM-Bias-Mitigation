"""
11.5 Visualization

논문용 figure 5종을 PDF(고화질)로 저장합니다.

Figures:
    1. plot_cluster_routing_heatmap()    — 카테고리 × cluster gating weight 평균.
    2. plot_bias_head_heatmap()          — layer × head bias attention score.
    3. plot_sae_feature_activation()     — Top-K bias feature의 카테고리별 평균 활성도.
    4. plot_baseline_comparison_bar()    — accuracy_amb / bias_amb 등 baseline 비교 막대.
    5. plot_risk_coverage_curve()        — threshold 변화에 따른 risk-coverage tradeoff.

각 함수는 fig 객체를 반환하여 호출자가 추가 조정 가능. PDF 저장은 보조 함수
save_pdf()로 일괄 처리.

사용 예시:
    from src.ablation.visualization import (
        plot_cluster_routing_heatmap, save_pdf,
    )
    fig = plot_cluster_routing_heatmap(routing_matrix, categories, clusters)
    save_pdf(fig, "results/figures/fig1_cluster_routing.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# matplotlib은 선택적 import (설치 안 되어 있어도 모듈 import는 가능)
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    _MPL_OK = True
except ImportError:
    plt = None  # type: ignore
    mpl = None  # type: ignore
    _MPL_OK = False


# =============================================================
# 0. Style helpers
# =============================================================
def set_paper_style() -> None:
    """논문용 default 스타일 (Helvetica 비슷한 sans-serif, vector PDF)."""
    if not _MPL_OK:
        return
    mpl.rcParams.update({
        "pdf.fonttype": 42,             # vector font in PDF
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_pdf(fig, path: str | Path) -> None:
    """Fig을 PDF로 저장 (디렉토리 자동 생성)."""
    if not _MPL_OK or fig is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="pdf", bbox_inches="tight")
    logger.info(f"  [저장] {p}")


def _require_mpl():
    if not _MPL_OK:
        raise RuntimeError("matplotlib 미설치. `pip install matplotlib`")


# =============================================================
# 1. Cluster routing heatmap
# =============================================================
def plot_cluster_routing_heatmap(
    routing_matrix: np.ndarray,
    categories: Sequence[str],
    cluster_labels: Sequence[str],
    title: str = "Cluster routing (gating weight) per category",
):
    """
    카테고리 × cluster의 평균 gating weight 히트맵.

    Args:
        routing_matrix: (n_categories, n_clusters) — 각 카테고리의 평균 gate_w.
        categories: 카테고리 이름 (행 라벨).
        cluster_labels: cluster 이름 (열 라벨).
        title: 그래프 제목.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    set_paper_style()

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    im = ax.imshow(routing_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(cluster_labels)))
    ax.set_xticklabels(cluster_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_title(title)

    # Cell 안에 숫자
    for i in range(routing_matrix.shape[0]):
        for j in range(routing_matrix.shape[1]):
            v = routing_matrix[i, j]
            color = "white" if v < 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label="Gate weight")
    fig.tight_layout()
    return fig


# =============================================================
# 2. Bias-head layer × head heatmap
# =============================================================
def plot_bias_head_heatmap(
    head_scores: np.ndarray,
    title: str = "Bias-attention score per (layer, head)",
):
    """
    각 (layer, head)의 demographic-token attention 평균 히트맵.

    Args:
        head_scores: (n_layers, n_heads) matrix.
        title: 제목.

    Returns:
        Figure.
    """
    _require_mpl()
    set_paper_style()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(head_scores, aspect="auto", cmap="magma")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Avg attention to demographic token")
    fig.tight_layout()
    return fig


# =============================================================
# 3. SAE feature activation
# =============================================================
def plot_sae_feature_activation(
    feature_means: np.ndarray,
    feature_indices: Sequence[int],
    categories: Sequence[str],
    title: str = "Top-K SAE feature activation per category",
):
    """
    Top-K bias feature의 카테고리별 평균 활성도 히트맵.

    Args:
        feature_means: (n_categories, K) matrix.
        feature_indices: 표시할 SAE feature 인덱스 (열 라벨).
        categories: 카테고리 이름 (행 라벨).
        title: 제목.

    Returns:
        Figure.
    """
    _require_mpl()
    set_paper_style()

    fig, ax = plt.subplots(figsize=(max(5.0, 0.3 * len(feature_indices)), 4.0))
    im = ax.imshow(feature_means, aspect="auto", cmap="cividis")

    ax.set_xticks(range(len(feature_indices)))
    ax.set_xticklabels([str(i) for i in feature_indices], rotation=90, fontsize=7)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("SAE feature index")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="Avg activation")
    fig.tight_layout()
    return fig


# =============================================================
# 4. Baseline comparison bar chart
# =============================================================
def plot_baseline_comparison_bar(
    methods: Sequence[str],
    metrics: dict[str, Sequence[float]],
    metric_labels: Optional[dict[str, str]] = None,
    title: str = "Baseline comparison",
):
    """
    여러 metric을 grouped bar chart로 표시.

    Args:
        methods: x축 라벨 (메서드 이름).
        metrics: {metric_name: [val_for_method1, val_for_method2, ...]}.
        metric_labels: 표시용 metric 라벨 매핑.
        title: 제목.

    Returns:
        Figure.
    """
    _require_mpl()
    set_paper_style()

    metric_labels = metric_labels or {}
    metric_names = list(metrics.keys())
    n_methods = len(methods)
    n_metrics = len(metric_names)

    if n_methods == 0 or n_metrics == 0:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * n_methods), 3.5))
    width = 0.8 / max(n_metrics, 1)
    x = np.arange(n_methods)

    for i, m in enumerate(metric_names):
        vals = list(metrics[m])
        if len(vals) != n_methods:
            logger.warning(f"  [bar] metric '{m}' 값 개수 불일치 — skip")
            continue
        offset = (i - (n_metrics - 1) / 2) * width
        ax.bar(x + offset, vals, width=width,
               label=metric_labels.get(m, m))

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


# =============================================================
# 5. Risk-coverage curve
# =============================================================
def plot_risk_coverage_curve(
    points: Sequence,            # list of RiskCoveragePoint or dict
    title: str = "Risk-Coverage curve",
):
    """
    src.models.override.risk_coverage_curve()의 출력을 그립니다.

    Args:
        points: RiskCoveragePoint 리스트 또는 같은 필드의 dict 리스트.
                각 항목은 .threshold / .coverage / .risk / .error_rate 또는
                대응 dict 키를 가져야 함.
        title: 제목.

    Returns:
        Figure.
    """
    _require_mpl()
    set_paper_style()

    def _get(p, key):
        return getattr(p, key) if hasattr(p, key) else p[key]

    if not points:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    cov = [_get(p, "coverage") for p in points]
    risk = [_get(p, "risk") for p in points]
    err = [_get(p, "error_rate") for p in points]
    tau = [_get(p, "threshold") for p in points]

    fig, ax1 = plt.subplots(figsize=(5.0, 3.5))
    ax1.plot(cov, risk, marker="o", color="C0", label="Risk @ kept")
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Risk", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(cov, err, marker="s", color="C3", label="Total error (post-override)")
    ax2.set_ylabel("Error rate", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    ax1.set_title(title)
    ax1.grid(linestyle=":", alpha=0.5)

    # threshold annotations on a few points
    n_points = len(tau)
    if n_points > 0:
        idxs = list({0, n_points // 2, n_points - 1})
        for idx in idxs:
            ax1.annotate(
                f"τ={tau[idx]:.2f}",
                xy=(cov[idx], risk[idx]),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
            )

    fig.tight_layout()
    return fig


# =============================================================
# 6. One-shot driver
# =============================================================
def render_all(
    *,
    routing_matrix: Optional[np.ndarray] = None,
    categories: Optional[Sequence[str]] = None,
    cluster_labels: Optional[Sequence[str]] = None,
    head_scores: Optional[np.ndarray] = None,
    sae_feature_means: Optional[np.ndarray] = None,
    sae_feature_indices: Optional[Sequence[int]] = None,
    baseline_methods: Optional[Sequence[str]] = None,
    baseline_metrics: Optional[dict[str, Sequence[float]]] = None,
    rc_points: Optional[Sequence] = None,
    out_dir: str = "results/figures",
) -> dict[str, str]:
    """
    제공된 데이터로 가능한 figure를 모두 생성하고 PDF로 저장.

    Returns:
        {fig_name: saved_path}.
    """
    saved: dict[str, str] = {}
    out = Path(out_dir)

    if routing_matrix is not None and categories and cluster_labels:
        fig = plot_cluster_routing_heatmap(routing_matrix, categories, cluster_labels)
        path = out / "fig1_cluster_routing.pdf"
        save_pdf(fig, path)
        saved["cluster_routing"] = str(path)

    if head_scores is not None:
        fig = plot_bias_head_heatmap(head_scores)
        path = out / "fig2_bias_head.pdf"
        save_pdf(fig, path)
        saved["bias_head"] = str(path)

    if (sae_feature_means is not None and sae_feature_indices is not None
            and categories is not None):
        fig = plot_sae_feature_activation(
            sae_feature_means, sae_feature_indices, categories,
        )
        path = out / "fig3_sae_features.pdf"
        save_pdf(fig, path)
        saved["sae_features"] = str(path)

    if baseline_methods and baseline_metrics:
        fig = plot_baseline_comparison_bar(baseline_methods, baseline_metrics)
        path = out / "fig4_baseline_comparison.pdf"
        save_pdf(fig, path)
        saved["baseline"] = str(path)

    if rc_points:
        fig = plot_risk_coverage_curve(rc_points)
        path = out / "fig5_risk_coverage.pdf"
        save_pdf(fig, path)
        saved["risk_coverage"] = str(path)

    return saved
