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
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("paper.figures")

# 색상 팔레트 — Wong colorblind-safe palette (Nature Methods 2011)
# https://www.nature.com/articles/nmeth.1618
COLORS = {
    "vanilla": "#999999",      # gray
    "composite": "#0072B2",    # blue
    "self_debiasing": "#D55E00",  # vermilion
    "decap": "#E69F00",        # orange
    "fairsteer": "#CC79A7",    # pink
    "ours": "#009E73",         # bluish-green
}

# 학회 표준 figure 크기
SINGLE_COL = (3.5, 2.5)   # NLP/ML conference single column
DOUBLE_COL = (7.0, 5.0)   # double column / wide figures
WIDE = (10.0, 6.0)        # full-page wide figures


def _configure_korean_font(mpl) -> None:
    """Use a Korean-capable font when available so exported PDFs stay readable."""
    try:
        from matplotlib import font_manager
    except Exception:
        mpl.rcParams["axes.unicode_minus"] = False
        return

    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for font_path in candidates:
        path = Path(font_path)
        if not path.exists():
            continue
        font_manager.fontManager.addfont(str(path))
        font_name = font_manager.FontProperties(fname=str(path)).get_name()
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans", "Arial Unicode MS"]
        break
    mpl.rcParams["axes.unicode_minus"] = False


def _significance_marker(p: float) -> str:
    """p-value → asterisk 표기."""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _set_paper_style():
    """폰트 크기 + 색상 + grid 설정. IEEE Access 가독성 기준."""
    try:
        import matplotlib as mpl
        mpl.rcParams.update({
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.size": 13,            # base font
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,      # 가독성 ↑
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 15,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
        })
        mpl.rcParams["axes.unicode_minus"] = False
    except ImportError:
        pass


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".png"), format="png", bbox_inches="tight", dpi=300)
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
    fig, ax = plt.subplots(figsize=(11.5, 3.1))
    ax.set_xlim(-0.5, 11.5)
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
        ax.text(x, y, text, ha="center", va="center", fontsize=12)

    # arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + 0.7
        x2 = boxes[i + 1][1] - 0.7
        ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"))

    # labels under boxes
    annotations = [
        (2.5, 0.35, "vanilla / debias /\nCoT / cf-swap"),
        (4.5, 0.35, "s1-s7 signals"),
        (6.5, 0.35, "p ∈ [0,1]\n4-expert mixture"),
        (8.5, 0.35, "p < τ → unknown"),
    ]
    for x, y, text in annotations:
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                color="#555555", style="italic")

    ax.set_title("System Pipeline: 4-Stage Mechanism-Aware Debiasing", fontsize=15, pad=8)
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
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.set_axis_off()

    # Inputs
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 4.3), 1.6, 0.7, boxstyle="round,pad=0.04",
        facecolor="#e8e8e8", edgecolor="black"))
    ax.text(1.1, 4.65, "Signals\ns1..s7", ha="center", va="center", fontsize=10)

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
    experts = ["Expert 1\n(Lex-Sub)", "Expert 2\n(Numeric)",
               "Expert 3\n(Cultural)", "Expert 4\n(Identity)"]
    for i, name in enumerate(experts):
        y = 5 - i * 1.2
        ax.add_patch(mpatches.FancyBboxPatch(
            (6.0, y - 0.4), 1.7, 0.8, boxstyle="round,pad=0.04",
            facecolor="#ffbb88", edgecolor="black"))
        ax.text(6.85, y, name, ha="center", va="center", fontsize=10)

    # Sum (Σ box, 작고 깔끔하게)
    ax.add_patch(mpatches.Circle((9.0, 3.0), 0.4, facecolor="#88dd88", edgecolor="black", linewidth=1.2))
    ax.text(9.0, 3.0, "Σ", ha="center", va="center", fontsize=18, weight="bold")

    # Output text (Σ 아래, 출력 화살표 없이 단순)
    ax.text(9.0, 1.9, "p ∈ [0, 1]\nconfidence", ha="center", va="center",
            fontsize=11, style="italic")

    # Arrows: experts → Σ (skip if too close, 단정하게)
    for i, _ in enumerate(experts):
        y = 5 - i * 1.2
        ax.annotate("", xy=(8.55, 3.0), xytext=(7.75, y),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#666666",
                                    connectionstyle="arc3,rad=0.05"))

    # Arrows: Signals → gating
    ax.annotate("", xy=(2.95, 4.5), xytext=(1.95, 4.65),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    # Arrows: Question Embedding → gating
    ax.annotate("", xy=(2.95, 3.8), xytext=(1.95, 3.65),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    # Arrows: gating → experts (4 arrows in light gray)
    for i in range(4):
        y = 5 - i * 1.2
        ax.annotate("", xy=(5.95, y), xytext=(5.05, 4.15),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#888888"))

    ax.set_title("Mechanism-Aware MoE Aggregator", fontsize=14, pad=12)
    _save(fig, save_path)


# =============================================================
# Figure 4: Main BBQ results bar chart
# =============================================================
def _load_ours_predictions() -> Optional[tuple[list[int], list[dict]]]:
    """
    Ours predictions과 instances를 로드 (bootstrap CI 계산용).
    None이면 추정 불가.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import yaml
        with open("configs/default.yaml") as f:
            config = yaml.safe_load(f)
        from run_pipeline import (  # type: ignore
            _collect_records_and_embeddings,
            _instances_by_id,
            _infer_embed_dim,
            _moe_predict_all,
        )
        from src.models.moe_aggregator import MoEAggregator
        from src.models.override import apply_threshold_override
        import torch

        class _Args:
            model = "main"
            categories = None

        records, embeddings = _collect_records_and_embeddings(config, _Args())
        if not records:
            return None
        instances_by_id = _instances_by_id(records, config, _Args())

        ckpt_path = Path("results/moe/main/moe_best.pt")
        if not ckpt_path.exists():
            for cand in ("results/moe/main/moe_last.pt", "results/v2/moe/main/moe_best.pt"):
                if Path(cand).exists():
                    ckpt_path = Path(cand)
                    break
        if not ckpt_path.exists():
            return None
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        saved_cfg = saved.get("model_config", {})
        embed_dim = _infer_embed_dim(embeddings, default=4096)
        model = MoEAggregator(
            signal_dim=int(saved_cfg.get("signal_dim", 7)),
            embed_dim=int(saved_cfg.get("embed_dim", embed_dim)),
            num_experts=int(saved_cfg.get("num_experts", 4)),
            gating_hidden=int(saved_cfg.get("gating_hidden", 64)),
            expert_hidden=int(saved_cfg.get("expert_hidden", 128)),
        )
        model.load_state_dict(saved.get("model_state_dict", saved), strict=False)
        model.eval()

        # threshold from canonical final.json — v2 우선
        eval_path = next((p for p in [
            Path("results/v2/evaluation/main/final.json"),
            Path("results/v2_runpod/evaluation/main/final.json"),
            Path("results/evaluation/main/final.json"),
        ] if p.exists()), Path("results/evaluation/main/final.json"))
        threshold = 0.5
        if eval_path.exists():
            try:
                threshold = float(json.loads(eval_path.read_text())["threshold"])
            except Exception:
                pass

        val_preds = _moe_predict_all(model, records, embeddings, instances_by_id)
        final_preds: list[int] = []
        final_items: list[dict] = []
        for vp in val_preds:
            r = apply_threshold_override(
                primary_answer=int(vp["primary_answer"]),
                p_score=float(vp["p_score"]),
                item=vp["item"],
                threshold=threshold,
            )
            final_preds.append(r["final_answer"])
            final_items.append(vp["item"])
        return final_preds, final_items
    except Exception as e:
        logger.warning(f"  ours predictions 로드 실패: {e}")
        return None


def _load_baseline_predictions(jsonl_path: Path) -> Optional[tuple[list[int], list[dict]]]:
    """baseline predictions.jsonl → (preds, instances) — items_by_id 매칭 필요."""
    if not jsonl_path.exists():
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.bbq_evaluator import parse_prediction
    from run_pipeline import _load_items  # type: ignore
    import yaml
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    # 모든 카테고리에서 items 수집 — composite key (cat::ex_id) 사용
    # 이전 버그: items_by_id[ex_id]는 cross-cat collision 시 일부 instance 손실
    items_by_id: dict[str, dict] = {}
    n_per_cat = config["data"].get("samples_per_category", 300)
    for cat in config["data"]["categories"]:
        for it in _load_items(config, cat, n_per_cat=n_per_cat):
            it.setdefault("category", cat)
            items_by_id[f"{cat}::{it['example_id']}"] = it

    preds: list[int] = []
    items: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ex_id = rec.get("example_id")
            cat = rec.get("category", "_unknown")
            ukey = f"{cat}::{ex_id}"
            if ukey not in items_by_id:
                continue
            preds.append(parse_prediction(rec.get("prediction_text", "")))
            items.append(items_by_id[ukey])
    return (preds, items) if preds else None


def _parse_mean_std(value: str | float | int | None) -> tuple[float, float]:
    """Parse values formatted as 'mean+/-std' in acceptance-package CSVs."""
    if value is None or value == "":
        return float("nan"), 0.0
    if isinstance(value, (int, float)):
        return float(value), 0.0
    text = str(value).strip()
    if "+/-" in text:
        mean, std = text.split("+/-", 1)
        return float(mean), float(std)
    return float(text), 0.0


def fig4_main_results(save_path: Path) -> None:
    """
    Paper-safe main comparison figure.

    The older version emphasized |bias_amb|, which is unstable when the
    ambiguous residual denominator is tiny. The submission figure therefore
    foregrounds the robust claims: ambiguous accuracy, disambiguated accuracy,
    and false abstention rate. Residual bias counts remain an appendix table.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _set_paper_style()
    report_csv = Path("results/v2/acceptance_package/report/main_and_baseline_metrics.csv")
    if not report_csv.exists():
        logger.warning("  acceptance package main metrics 없음 — skip")
        return

    by_system: dict[str, dict] = {}
    with open(report_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_system[row["system"]] = row

    selected = [
        ("composite", "Composite"),
        ("decap", "DeCAP"),
        ("self_debiasing", "Self-Debias"),
        ("ours_single_tau", "Ours\nsingle τ"),
        ("ours_predicted_condition", "Ours\npredicted"),
    ]
    rows = [(key, label, by_system[key]) for key, label in selected if key in by_system]
    if len(rows) < 2:
        logger.warning("  acceptance package systems 부족 — skip")
        return

    labels = [label for _, label, _ in rows]
    acc_amb = []
    acc_amb_std = []
    acc_dis = []
    acc_dis_std = []
    far = []
    far_std = []
    for _, _, row in rows:
        m, s = _parse_mean_std(row["accuracy_amb"]); acc_amb.append(m); acc_amb_std.append(s)
        m, s = _parse_mean_std(row["accuracy_dis"]); acc_dis.append(m); acc_dis_std.append(s)
        m, s = _parse_mean_std(row["FAR"]); far.append(m); far_std.append(s)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.8, 5.4), gridspec_kw={"width_ratios": [1.48, 1.0]}
    )
    x = np.arange(len(labels))
    width = 0.36

    ours_idx = labels.index("Ours\npredicted") if "Ours\npredicted" in labels else len(labels) - 1
    for ax in (ax1, ax2):
        ax.axvspan(ours_idx - 0.5, ours_idx + 0.5, color=COLORS["ours"], alpha=0.07, zorder=0)

    ax1.bar(
        x - width / 2, acc_amb, width, yerr=acc_amb_std,
        label="Ambiguous", color="#56B4E9", edgecolor="black", linewidth=0.8,
        capsize=3, error_kw={"elinewidth": 0.8},
    )
    ax1.bar(
        x + width / 2, acc_dis, width, yerr=acc_dis_std,
        label="Disambiguated", color="#E69F00", edgecolor="black", linewidth=0.8,
        capsize=3, error_kw={"elinewidth": 0.8},
    )
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by context", pad=10)
    ax1.set_ylim(0, 1.20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.legend(frameon=True, framealpha=0.92, loc="upper left",
               ncol=2, handlelength=1.2, columnspacing=0.9)
    ax1.grid(axis="y", linestyle=":", alpha=0.35)

    bars = ax2.bar(
        x, far, yerr=far_std, color="#009E73", edgecolor="black",
        linewidth=0.8, capsize=3, error_kw={"elinewidth": 0.8},
    )
    for bar, v in zip(bars, far):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.025, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("False Abstention Rate")
    ax2.set_title("Over-abstention")
    ax2.set_ylim(0, min(1.0, max(far) * 1.18 + 0.05))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle("BBQ Main Comparison (5 seeds; Llama-3.1-8B)", fontsize=13, y=0.985)
    fig.text(
        0.5, 0.025,
        "FairSteer is omitted from the main plot because matched-ID overlap is limited (n≈15). "
        "Residual ambiguous bias is reported as raw counts/CI in the appendix.",
        ha="center", fontsize=8.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
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
    src_png = out_dir / "cluster_routing_heatmap.png"
    dst_png = save_path.with_suffix(".png")
    if src_png.exists() and src_png != dst_png:
        src_png.rename(dst_png)
        logger.info(f"  rename {src_png} → {dst_png}")


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
