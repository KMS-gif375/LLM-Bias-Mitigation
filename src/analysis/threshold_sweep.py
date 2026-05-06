"""
Threshold Sensitivity Analysis.

다양한 threshold τ에 대한 BBQ 결과 변화를 분석하고, 카테고리별 / cluster별
최적 τ를 찾습니다.

기능:
    1. global threshold_sweep()       — τ ∈ [0.30, 0.85] 사이 BBQ 지표 변화
    2. plot_risk_coverage()           — FAR vs (1 - |bias|) 곡선 PDF
    3. per_category_threshold()       — 7개 카테고리별 최적 τ
    4. per_cluster_threshold()        — 4개 cluster (default taxonomy)별 최적 τ
    5. find_optimal_threshold()       — 가중치 기반 종합 점수 최적 τ

CLI:
    python -m src.analysis.threshold_sweep --full
    python -m src.analysis.threshold_sweep --thresholds 0.3,0.5,0.7
    python -m src.analysis.threshold_sweep --no-plot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("threshold_sweep")

# matplotlib은 선택적
try:
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    plt = None
    _MPL_OK = False

# Default cluster taxonomy (cluster_ablation과 동일하게 유지)
DEFAULT_CLUSTER_MAP: dict[str, str] = {
    "Gender_identity": "lexical",
    "Religion": "lexical",
    "Age": "numerical",
    "SES": "numerical",
    "Race_ethnicity": "cultural",
    "Disability_status": "identity",
    "Sexual_orientation": "identity",
}


# =============================================================
# 1. Global threshold sweep
# =============================================================
def threshold_sweep(
    val_predictions: list[dict],
    thresholds: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """
    여러 threshold에서 BBQ 평가 지표를 한 번에 계산.

    Args:
        val_predictions: [{"primary_answer": int, "p_score": float, "item": BBQ dict}, ...]
        thresholds: 평가할 τ 리스트. None이면 0.30~0.85 (12 values).

    Returns:
        DataFrame columns: tau, acc_amb, acc_dis, bias_amb, bias_dis, far,
                            n_kept, n_overridden, parse_fail_rate
    """
    if thresholds is None:
        thresholds = [round(0.30 + 0.05 * i, 4) for i in range(12)]
    thresholds = list(thresholds)

    from src.evaluation.bbq_evaluator import evaluate_bbq
    from src.models.override import apply_threshold_override

    rows: list[dict] = []
    for tau in thresholds:
        final_preds: list[int] = []
        items: list[dict] = []
        n_overridden = 0
        for vp in val_predictions:
            item = vp["item"]
            res = apply_threshold_override(
                primary_answer=int(vp["primary_answer"]),
                p_score=float(vp["p_score"]),
                item=item,
                threshold=float(tau),
            )
            final_preds.append(res["final_answer"])
            items.append(item)
            n_overridden += int(res["overridden"])

        m = evaluate_bbq(final_preds, items)
        rows.append({
            "tau": float(tau),
            "acc_amb": float(m.get("accuracy_amb", 0.0)),
            "acc_dis": float(m.get("accuracy_dis", 0.0)),
            "bias_amb": m.get("bias_score_amb"),
            "bias_dis": m.get("bias_score_dis"),
            "far": float(m.get("false_abstention_rate", 0.0)),
            "n_kept": len(val_predictions) - n_overridden,
            "n_overridden": n_overridden,
            "parse_fail_rate": float(m.get("parse_fail_rate", 0.0)),
        })

    return pd.DataFrame(rows)


# =============================================================
# 2. Risk-Coverage curve plot (FAR vs 1-|bias|)
# =============================================================
def plot_risk_coverage(
    results_df: pd.DataFrame,
    save_path: str | Path,
    title: str = "FAR vs 1 - |bias_amb| trade-off",
) -> None:
    """
    threshold_sweep 결과 DataFrame을 받아 PDF로 저장.

    X축 = FAR (false abstention rate, 비모호 맥락에서 unknown 비율)
    Y축 = 1 - |bias_amb| (편향 적을수록 높음)
    """
    if not _MPL_OK:
        logger.warning("matplotlib 미설치 — plot 생략")
        return
    if results_df.empty:
        logger.warning("빈 DataFrame — plot 생략")
        return

    df = results_df.dropna(subset=["bias_amb"]).copy()
    df["one_minus_abs_bias"] = 1.0 - df["bias_amb"].abs()

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(df["far"], df["one_minus_abs_bias"], marker="o", linewidth=1.8, color="C0")

    # τ 값 annotation
    for _, row in df.iterrows():
        ax.annotate(
            f"τ={row['tau']:.2f}",
            xy=(row["far"], row["one_minus_abs_bias"]),
            xytext=(4, -8),
            textcoords="offset points",
            fontsize=7,
            alpha=0.8,
        )

    ax.set_xlabel("False Abstention Rate (FAR)")
    ax.set_ylabel("1 − |bias_score_amb|")
    ax.set_title(title)
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_xlim(left=-0.01)
    ax.set_ylim(top=1.01)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [저장] risk-coverage curve → {save_path}")


# =============================================================
# 3. Per-category threshold
# =============================================================
def per_category_threshold(
    val_predictions: list[dict],
    thresholds: Optional[Iterable[float]] = None,
    metric_weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    카테고리별로 별도 threshold sweep + 최적 τ 선택.

    Args:
        val_predictions: 표준 형식. item에 'category' 필드 필요.
        thresholds: τ 리스트 (None → default).
        metric_weights: find_optimal_threshold와 동일 형식.

    Returns:
        DataFrame columns: category, best_tau, score, acc_amb, acc_dis, bias_amb, far
    """
    by_cat: dict[str, list[dict]] = {}
    for vp in val_predictions:
        cat = vp["item"].get("category", "_unknown")
        by_cat.setdefault(cat, []).append(vp)

    rows: list[dict] = []
    for cat, preds in sorted(by_cat.items()):
        if not preds:
            continue
        sweep = threshold_sweep(preds, thresholds=thresholds)
        best_tau, score = _argmax_score(sweep, metric_weights)
        if best_tau is None:
            continue
        best_row = sweep[sweep["tau"] == best_tau].iloc[0]
        rows.append({
            "category": cat,
            "best_tau": best_tau,
            "score": score,
            "acc_amb": best_row["acc_amb"],
            "acc_dis": best_row["acc_dis"],
            "bias_amb": best_row["bias_amb"],
            "far": best_row["far"],
            "n": len(preds),
        })
    return pd.DataFrame(rows)


# =============================================================
# 4. Per-cluster threshold (default taxonomy)
# =============================================================
def per_cluster_threshold(
    val_predictions: list[dict],
    thresholds: Optional[Iterable[float]] = None,
    metric_weights: Optional[dict[str, float]] = None,
    cluster_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Cluster별 (lexical / numerical / cultural / identity) 최적 τ 탐색.

    가설:
        - identity (Disability, Sexual_orientation): 보수적 (τ 높음)
        - numerical (Age, SES): 덜 보수적 (τ 낮음)

    Args:
        cluster_map: {category: cluster_name}. None이면 DEFAULT_CLUSTER_MAP.

    Returns:
        DataFrame columns: cluster, best_tau, score, acc_amb, acc_dis, bias_amb, far, n
    """
    cmap = cluster_map or DEFAULT_CLUSTER_MAP
    by_cluster: dict[str, list[dict]] = {}
    for vp in val_predictions:
        cat = vp["item"].get("category", "_unknown")
        cluster = cmap.get(cat, "other")
        by_cluster.setdefault(cluster, []).append(vp)

    rows: list[dict] = []
    for cluster, preds in sorted(by_cluster.items()):
        if not preds:
            continue
        sweep = threshold_sweep(preds, thresholds=thresholds)
        best_tau, score = _argmax_score(sweep, metric_weights)
        if best_tau is None:
            continue
        best_row = sweep[sweep["tau"] == best_tau].iloc[0]
        rows.append({
            "cluster": cluster,
            "best_tau": best_tau,
            "score": score,
            "acc_amb": best_row["acc_amb"],
            "acc_dis": best_row["acc_dis"],
            "bias_amb": best_row["bias_amb"],
            "far": best_row["far"],
            "n": len(preds),
        })
    return pd.DataFrame(rows)


# =============================================================
# 5. 최적 threshold 자동 결정
# =============================================================
DEFAULT_METRIC_WEIGHTS: dict[str, float] = {
    "acc_amb": 1.0,
    "bias_amb": -1.0,         # |bias|를 최소화 (음수 가중)
    "far": -0.5,              # FAR 최소화
}


def find_optimal_threshold(
    val_predictions: list[dict],
    thresholds: Optional[Iterable[float]] = None,
    metric_weights: Optional[dict[str, float]] = None,
) -> dict:
    """
    가중치 기반 종합 점수로 최적 τ를 결정합니다.

    Score = Σ w_i × m_i(τ)
    bias_amb는 |값|을 사용 (절대값이 0에 가까울수록 좋음).

    Returns:
        {"best_tau": float, "score": float, "details": <best row dict>,
         "all_scores": [<list of (tau, score) tuples>]}
    """
    sweep = threshold_sweep(val_predictions, thresholds=thresholds)
    best_tau, score = _argmax_score(sweep, metric_weights)
    if best_tau is None:
        return {"best_tau": None, "score": None, "details": {}, "all_scores": []}

    details = sweep[sweep["tau"] == best_tau].iloc[0].to_dict()

    # 전체 점수 계산해서 함께 반환
    all_scores: list[tuple[float, float]] = []
    weights = metric_weights or DEFAULT_METRIC_WEIGHTS
    for _, row in sweep.iterrows():
        s = _row_score(row, weights)
        all_scores.append((float(row["tau"]), s))

    return {
        "best_tau": float(best_tau),
        "score": float(score),
        "details": details,
        "all_scores": all_scores,
    }


# =============================================================
# Internal: row scoring
# =============================================================
def _row_score(row: pd.Series, weights: dict[str, float]) -> float:
    """단일 row에 대한 가중 점수. bias_amb는 |값|."""
    score = 0.0
    for key, w in weights.items():
        v = row.get(key)
        if v is None or pd.isna(v):
            continue
        v = float(v)
        if key.startswith("bias"):
            v = abs(v)
        score += w * v
    return score


def _argmax_score(
    sweep: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> tuple[Optional[float], Optional[float]]:
    """가중치 기반 score 최대 τ 반환."""
    if sweep.empty:
        return None, None
    w = weights or DEFAULT_METRIC_WEIGHTS
    best_tau, best_score = None, -float("inf")
    for _, row in sweep.iterrows():
        s = _row_score(row, w)
        if s > best_score:
            best_score = s
            best_tau = float(row["tau"])
    return best_tau, best_score


# =============================================================
# CLI driver
# =============================================================
def _build_val_predictions(
    config_path: str = "configs/default.yaml",
    model_key: str = "main",
) -> list[dict]:
    """
    run_pipeline의 _moe_predict_all과 동등하게 records + embeddings + MoE를
    로드해서 val_predictions를 만듭니다.
    """
    load_dotenv()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from run_pipeline import (  # type: ignore
        _collect_records_and_embeddings,
        _instances_by_id,
        _stage_output_dir,
        _find_latest_checkpoint,
        _moe_predict_all,
        _infer_embed_dim,
    )
    from src.models.moe_aggregator import MoEAggregator

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # args mock
    class _Args:
        def __init__(self, model: str):
            self.model = model
            self.categories = None

    args = _Args(model_key)

    records, embeddings = _collect_records_and_embeddings(config, args)
    if not records:
        raise RuntimeError("Stage 2 (signal_extraction) 결과가 없음 — 먼저 파이프라인 실행")

    instances_by_id = _instances_by_id(records, config, args)

    # MoE checkpoint 로드
    moe_dir = _stage_output_dir(config, model_key, "moe")
    ckpt = _find_latest_checkpoint(moe_dir)
    embed_dim = _infer_embed_dim(embeddings, default=4096)
    moe_cfg = config["moe"]

    saved_state = None
    saved_model_cfg: dict = {}
    if ckpt and ckpt.exists():
        saved_state = torch.load(ckpt, map_location="cpu", weights_only=True)
        saved_model_cfg = saved_state.get("model_config", {}) or {}

    model = MoEAggregator(
        signal_dim=int(saved_model_cfg.get("signal_dim", 7)),
        embed_dim=int(saved_model_cfg.get("embed_dim", embed_dim)),
        num_experts=int(saved_model_cfg.get("num_experts",
                                            moe_cfg.get("num_experts", 4))),
        gating_hidden=int(saved_model_cfg.get("gating_hidden",
                                              moe_cfg.get("gating_hidden_dim", 64))),
        expert_hidden=int(saved_model_cfg.get("expert_hidden",
                                              moe_cfg.get("expert_hidden_dim", 128))),
        dropout=float(saved_model_cfg.get("dropout", 0.1)),
    )
    if saved_state is not None:
        model.load_state_dict(saved_state.get("model_state_dict", saved_state),
                              strict=False)
        logger.info(f"  MoE 체크포인트 로드: {ckpt}")
    else:
        logger.warning("  MoE 체크포인트 없음 — 미학습 모델로 진행")

    return _moe_predict_all(model, records, embeddings, instances_by_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Threshold sensitivity analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="main")
    parser.add_argument(
        "--thresholds", type=str, default=None,
        help="comma-separated τ values (예: 0.3,0.5,0.7). 생략 시 0.30~0.85 (12개)",
    )
    parser.add_argument("--full", action="store_true",
                        help="default τ grid (0.30~0.85)로 평가 (기본 동작)")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="결과 저장 디렉토리")
    parser.add_argument("--no-plot", action="store_true",
                        help="PDF 그래프 저장 생략")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    thresholds = None
    if args.thresholds:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Loading val predictions ===")
    val_predictions = _build_val_predictions(args.config, args.model)
    logger.info(f"  n_predictions: {len(val_predictions)}")

    # 1. Global sweep
    logger.info("=== Global threshold sweep ===")
    sweep_df = threshold_sweep(val_predictions, thresholds=thresholds)
    csv_path = out_dir / "threshold_sensitivity.csv"
    sweep_df.to_csv(csv_path, index=False)
    logger.info(f"  [저장] {csv_path}")
    print(sweep_df.to_string(index=False))

    # 2. Risk-coverage plot
    if not args.no_plot:
        plot_risk_coverage(sweep_df, out_dir / "risk_coverage_curve.pdf")

    # 3. Per-category
    logger.info("=== Per-category threshold ===")
    per_cat = per_category_threshold(val_predictions, thresholds=thresholds)
    cat_path = out_dir / "per_category_threshold.csv"
    per_cat.to_csv(cat_path, index=False)
    logger.info(f"  [저장] {cat_path}")
    print(per_cat.to_string(index=False))

    # 4. Per-cluster
    logger.info("=== Per-cluster threshold ===")
    per_cl = per_cluster_threshold(val_predictions, thresholds=thresholds)
    cl_path = out_dir / "per_cluster_threshold.csv"
    per_cl.to_csv(cl_path, index=False)
    logger.info(f"  [저장] {cl_path}")
    print(per_cl.to_string(index=False))

    # 5. 최적 τ 자동
    logger.info("=== Optimal τ (weighted score) ===")
    opt = find_optimal_threshold(val_predictions, thresholds=thresholds)
    logger.info(f"  best_tau = {opt['best_tau']:.3f}  score = {opt['score']:.4f}")
    (out_dir / "threshold_optimal.json").write_text(
        json.dumps(opt, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
