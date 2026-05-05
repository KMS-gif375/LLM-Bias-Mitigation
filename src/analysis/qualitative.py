"""
Qualitative Analysis CLI.

기존 src/ablation/qualitative_analysis.py와 src/ablation/visualization.py를
실제 results/ 데이터로 구동하는 CLI 통합 모듈. 새로 추가:
    - SAE feature visualization with Neuronpedia label fetch (선택)
    - Failure case 분석 (ours_only_correct / both_wrong / vanilla_only_correct)
    - Risk-coverage curve (multi-method)
    - Bias-head heatmap (Layer × Head)
    - Cluster routing heatmap (Category × Cluster)

CLI:
    # 사용 가능한 모든 정성 분석 자동 수행
    python -m src.analysis.qualitative --auto

    # 특정 분석만
    python -m src.analysis.qualitative --tasks failure_cases bias_heads_heatmap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("qualitative")


AVAILABLE_TASKS = (
    "bias_heads_heatmap",      # results/bias_heads.json → Layer × Head heatmap
    "cluster_routing_heatmap", # MoE gating → Category × Cluster heatmap
    "sae_features",            # SAE bias features + max-activating examples
    "failure_cases",           # ours vs vanilla 비교
    "risk_coverage",           # threshold_sensitivity.csv → 시각화
)


# =============================================================
# Task: bias_heads_heatmap
# =============================================================
def run_bias_heads_heatmap(out_dir: Path, n_layers: int = 32, n_heads: int = 32) -> None:
    bh_path = Path("results/bias_heads.json")
    if not bh_path.exists():
        logger.warning(f"  bias_heads.json 없음 → skip")
        return

    data = json.loads(bh_path.read_text(encoding="utf-8"))
    head_indices = data.get("head_indices", [])
    scores = data.get("scores", [])

    matrix = np.zeros((n_layers, n_heads), dtype=np.float32)
    for (L, H), s in zip(head_indices, scores or [1.0] * len(head_indices)):
        if 0 <= L < n_layers and 0 <= H < n_heads:
            matrix[int(L), int(H)] = float(s)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  matplotlib 미설치 — heatmap skip")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = "Reds" if matrix.min() >= 0 else "RdBu_r"
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title(
        f"Top-{len(head_indices)} Bias-Relevant Attention Heads (Llama-3.1-8B)"
    )
    fig.colorbar(im, ax=ax, label="Contrastive score")
    fig.tight_layout()
    save_path = out_dir / "bias_heads_heatmap.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"  [저장] {save_path}")


# =============================================================
# Task: cluster_routing_heatmap (MoE gating)
# =============================================================
def run_cluster_routing_heatmap(
    out_dir: Path,
    config_path: str = "configs/default.yaml",
) -> None:
    """학습된 MoE의 Category × Cluster routing 평균을 heatmap으로."""
    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import (  # type: ignore
        _collect_records_and_embeddings,
        _instances_by_id,
        _infer_embed_dim,
    )
    from src.models.moe_aggregator import MoEAggregator, signals_dict_to_tensor

    class _Args:
        model = "main"
        categories = None

    args_ns = _Args()
    records, embeddings = _collect_records_and_embeddings(config, args_ns)
    if not records:
        logger.warning("  signals 없음 → skip")
        return

    instances_by_id = _instances_by_id(records, config, args_ns)

    # MoE 로드
    ckpt_path = Path("results/moe/main/moe_best.pt")
    if not ckpt_path.exists():
        for cand in ("results/moe/main/moe_last.pt", "results/v2/moe/main/moe_best.pt"):
            if Path(cand).exists():
                ckpt_path = Path(cand)
                break
    if not ckpt_path.exists():
        logger.warning("  MoE checkpoint 없음 → skip")
        return

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

    # 카테고리별 평균 gate weight 계산
    cats = sorted({r.get("category", "_unknown") for r in records})
    n_experts = model.num_experts
    matrix = np.zeros((len(cats), n_experts), dtype=np.float32)
    counts = np.zeros(len(cats), dtype=np.int64)
    cat_idx = {c: i for i, c in enumerate(cats)}
    device = next(model.parameters()).device

    with torch.inference_mode():
        for rec in records:
            ex_id = rec.get("example_id")
            if ex_id not in embeddings:
                continue
            sig_t = signals_dict_to_tensor(rec.get("signals", {})).unsqueeze(0).to(device)
            emb_t = embeddings[ex_id].to(torch.float32).unsqueeze(0).to(device)
            out = model(sig_t, emb_t)
            w = out.gate_w[0].cpu().numpy()
            i = cat_idx.get(rec.get("category", "_unknown"), -1)
            if i >= 0:
                matrix[i] += w
                counts[i] += 1
    for i in range(len(cats)):
        if counts[i] > 0:
            matrix[i] /= counts[i]

    cluster_names = ("Lex-Sub", "Numeric", "Cultural", "Identity")
    if n_experts != 4:
        cluster_names = tuple(f"C{i}" for i in range(n_experts))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  matplotlib 미설치 — skip")
        return

    fig, ax = plt.subplots(figsize=(0.95 * n_experts + 3, 0.45 * len(cats) + 2))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_experts))
    ax.set_xticklabels(cluster_names, rotation=15, ha="right")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_title("Mechanism-Aware Cluster Routing")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Avg gate weight")
    fig.tight_layout()
    save_path = out_dir / "cluster_routing_heatmap.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"  [저장] {save_path}")


# =============================================================
# Task: sae_features (Neuronpedia + max-activating)
# =============================================================
def fetch_neuronpedia_label(feature_id: int, layer: int = 15,
                             release: str = "llama_scope_lxr_8x") -> Optional[str]:
    """
    Neuronpedia에서 feature label 가져오기 (실패해도 None).
    Llama-Scope 매핑:
        layer/release → "llamascope-llama-3.1-8b-base/{layer}-llamascope-res-32k"
    """
    try:
        import urllib.request
        # release_lxr_8x → 32K features
        if "8x" in release:
            sae_id = f"{layer}-llamascope-res-32k"
        elif "32x" in release:
            sae_id = f"{layer}-llamascope-res-128k"
        else:
            sae_id = f"{layer}-llamascope-res"
        url = (
            f"https://www.neuronpedia.org/api/feature/llama-3.1-8b-base/"
            f"{sae_id}/{feature_id}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "research"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("explanations", [{}])[0].get("description") or data.get("label")
    except Exception:
        return None


def run_sae_features(out_dir: Path, top_k: int = 20,
                     fetch_labels: bool = False) -> None:
    """식별된 SAE bias feature + (옵션) Neuronpedia label."""
    # 식별된 feature 로드
    feat_files = sorted(Path("results").rglob("features_layer*.json"))
    if not feat_files:
        logger.warning("  features_layer*.json 없음 (sae_layer_comparison 실행 필요) → skip")
        return

    out_subdir = out_dir / "sae_features"
    out_subdir.mkdir(parents=True, exist_ok=True)

    for ff in feat_files:
        data = json.loads(ff.read_text(encoding="utf-8"))
        bias_features = data.get("bias_features", [])[:top_k]
        if not bias_features:
            continue

        layer_match = ff.stem.replace("features_layer", "")
        try:
            layer = int(layer_match)
        except ValueError:
            layer = 15

        feature_records = []
        for fid in bias_features:
            rec = {"feature_id": fid, "layer": layer}
            if fetch_labels:
                desc = fetch_neuronpedia_label(fid, layer=layer)
                if desc:
                    rec["neuronpedia_description"] = desc
            feature_records.append(rec)

        out_path = out_subdir / f"layer_{layer}_features.json"
        out_path.write_text(
            json.dumps({
                "layer": layer,
                "top_k": top_k,
                "features": feature_records,
                "method_features": data.get("method_features", {}),
            }, indent=2, ensure_ascii=False, default=float),
            encoding="utf-8",
        )
        logger.info(f"  [저장] {out_path}")


# =============================================================
# Task: failure_cases (ours vs vanilla)
# =============================================================
def run_failure_cases(out_dir: Path, max_cases: int = 50) -> None:
    """
    Self-Debiasing baseline (또는 vanilla)과 ours 비교.
    Both 결과가 results/baselines/self_debiasing/predictions.jsonl 와
    results/evaluation/main/final.json 에 있다고 가정.
    """
    self_dbg_path = Path("results/baselines/self_debiasing/predictions.jsonl")
    ours_eval_path = Path("results/evaluation/main/final.json")

    if not ours_eval_path.exists():
        logger.warning("  results/evaluation/main/final.json 없음 → skip")
        return

    # ours predictions: stage1.jsonl + signals.jsonl 매칭으로 도출 가능하지만
    # 간단히 evaluate로부터 final_preds를 다시 만들기보다 stage1의 vanilla 답변을 baseline,
    # ours는 evaluate_bbq의 metric으로 대체 (예제 단위 비교가 어려운 경우).
    # 더 정확한 비교를 위해 results/transfer/* predictions.csv를 활용하거나
    # 별도로 ours predictions JSONL이 필요.

    if not self_dbg_path.exists():
        logger.warning("  Self-Debiasing predictions 없음 → vanilla baseline은 stage1 사용")
        # Stage 1 vanilla answer를 baseline으로
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        try:
            stage1_files = sorted(Path("results/signals/main").glob("*_stage1.jsonl"))
            vanilla_by_id: dict = {}
            for sf in stage1_files:
                with open(sf) as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            ans = rec.get("responses", {}).get("vanilla", {}).get("answer")
                            try:
                                vanilla_by_id[rec["example_id"]] = int(ans)
                            except (TypeError, ValueError):
                                pass
            logger.info(f"  Vanilla baseline: {len(vanilla_by_id)} answers from stage1")
        except Exception as e:
            logger.warning(f"  vanilla 로드 실패: {e}")
            return
    else:
        with open(self_dbg_path) as f:
            self_dbg = [json.loads(line) for line in f if line.strip()]
        from src.evaluation.bbq_evaluator import parse_prediction
        vanilla_by_id = {
            r["example_id"]: parse_prediction(r.get("prediction_text", ""))
            for r in self_dbg
        }

    # ours predictions: pipeline 직접 호출
    load_dotenv()
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import (  # type: ignore
        _collect_records_and_embeddings,
        _instances_by_id,
        _infer_embed_dim,
        _moe_predict_all,
    )
    from src.models.moe_aggregator import MoEAggregator
    from src.models.override import apply_threshold_override

    class _Args:
        model = "main"
        categories = None

    records, embeddings = _collect_records_and_embeddings(config, _Args())
    if not records:
        logger.warning("  records 없음 → skip")
        return
    instances_by_id = _instances_by_id(records, config, _Args())

    ckpt_path = Path("results/moe/main/moe_best.pt")
    if not ckpt_path.exists():
        logger.warning("  MoE 체크포인트 없음 → skip")
        return
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

    # threshold from ours_eval
    eval_data = json.loads(ours_eval_path.read_text(encoding="utf-8"))
    threshold = float(eval_data.get("threshold", 0.5))

    val_preds = _moe_predict_all(model, records, embeddings, instances_by_id)
    ours_by_id: dict[str, int] = {}
    for vp in val_preds:
        ex_id = vp["item"].get("example_id") if isinstance(vp.get("item"), dict) else None
        if ex_id is None:
            continue
        result = apply_threshold_override(
            primary_answer=int(vp["primary_answer"]),
            p_score=float(vp["p_score"]),
            item=vp["item"],
            threshold=threshold,
        )
        ours_by_id[ex_id] = int(result["final_answer"])

    # 비교
    buckets = {
        "ours_only_correct": [],
        "vanilla_only_correct": [],
        "both_wrong": [],
        "both_correct": 0,
    }
    for ex_id, item in instances_by_id.items():
        if ex_id not in ours_by_id or ex_id not in vanilla_by_id:
            continue
        gold = item.get("label", -1)
        ours = ours_by_id[ex_id]
        vani = vanilla_by_id[ex_id]
        ours_corr = ours == gold
        vani_corr = vani == gold

        case = {
            "example_id": ex_id,
            "category": item.get("category"),
            "context_condition": item.get("context_condition"),
            "context": (item.get("context") or "")[:200],
            "question": item.get("question"),
            "gold": gold,
            "vanilla_pred": vani,
            "ours_pred": ours,
        }
        if ours_corr and not vani_corr:
            buckets["ours_only_correct"].append(case)
        elif not ours_corr and vani_corr:
            buckets["vanilla_only_correct"].append(case)
        elif not ours_corr and not vani_corr:
            buckets["both_wrong"].append(case)
        else:
            buckets["both_correct"] += 1

    payload = {
        "summary": {
            "ours_only_correct": len(buckets["ours_only_correct"]),
            "vanilla_only_correct": len(buckets["vanilla_only_correct"]),
            "both_wrong": len(buckets["both_wrong"]),
            "both_correct": buckets["both_correct"],
        },
        "samples": {
            "ours_only_correct": buckets["ours_only_correct"][:max_cases],
            "vanilla_only_correct": buckets["vanilla_only_correct"][:max_cases],
            "both_wrong": buckets["both_wrong"][:max_cases],
        },
    }
    out_path = out_dir / "failure_cases.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    logger.info(
        f"  [저장] {out_path}  "
        f"ours_only={payload['summary']['ours_only_correct']}, "
        f"vani_only={payload['summary']['vanilla_only_correct']}, "
        f"both_wrong={payload['summary']['both_wrong']}, "
        f"both_correct={payload['summary']['both_correct']}"
    )


# =============================================================
# Task: risk_coverage (multi-method)
# =============================================================
def run_risk_coverage(out_dir: Path) -> None:
    """threshold_sensitivity.csv → FAR vs (1 − |bias_amb|) curve."""
    csv_path = Path("results/threshold_sensitivity.csv")
    if not csv_path.exists():
        logger.warning("  threshold_sensitivity.csv 없음 → skip")
        return

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  matplotlib/pandas 미설치 — skip")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["bias_amb"]).copy()
    df["one_minus_abs_bias"] = 1.0 - df["bias_amb"].abs()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(df["far"], df["one_minus_abs_bias"], marker="o", linewidth=2,
            color="#2ca02c", label="Ours (threshold sweep)")

    for _, row in df.iterrows():
        ax.annotate(
            f"τ={row['tau']:.2f}",
            xy=(row["far"], row["one_minus_abs_bias"]),
            xytext=(4, -10), textcoords="offset points", fontsize=8, alpha=0.7,
        )

    ax.set_xlabel("False Abstention Rate (FAR)")
    ax.set_ylabel("Bias Reduction (1 − |bias_amb|)")
    ax.set_title("Risk-Coverage Trade-off")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    save_path = out_dir / "risk_coverage_curve.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"  [저장] {save_path}")


# =============================================================
# Driver
# =============================================================
TASK_FNS = {
    "bias_heads_heatmap": run_bias_heads_heatmap,
    "cluster_routing_heatmap": run_cluster_routing_heatmap,
    "sae_features": run_sae_features,
    "failure_cases": run_failure_cases,
    "risk_coverage": run_risk_coverage,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualitative analysis CLI")
    parser.add_argument("--auto", action="store_true",
                        help="모든 task 자동 수행 (사용 가능한 데이터로)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=list(AVAILABLE_TASKS),
                        help="특정 task만 수행")
    parser.add_argument("--out-dir", type=str, default="results/qualitative")
    parser.add_argument("--neuronpedia", action="store_true",
                        help="SAE feature label fetch (Neuronpedia API)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(AVAILABLE_TASKS) if args.auto else (args.tasks or [])
    if not tasks:
        logger.error("--auto 또는 --tasks 지정")
        return 2

    for task in tasks:
        if task not in TASK_FNS:
            logger.warning(f"  알 수 없는 task: {task}")
            continue
        logger.info(f"\n=== {task} ===")
        try:
            if task == "sae_features":
                TASK_FNS[task](out_dir, fetch_labels=args.neuronpedia)
            else:
                TASK_FNS[task](out_dir)
        except Exception as e:
            logger.error(f"  {task} 실패: {e}")

    logger.info(f"\n완료: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
