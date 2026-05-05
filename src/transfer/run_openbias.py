"""
OpenBiasBench Zero-shot Transfer Runner.

학습된 MoE를 31개 새 demographic 카테고리에 zero-shot 적용. ImplicitBBQ runner와
동일 흐름이지만 OpenBiasBench specific:
    - 31 categories (BBQ 9 + 22개 unseen)
    - Routing accuracy 평가 (사전 라벨 기반)
    - 31×4 routing heatmap PDF
    - transfer_summary.md (mechanism-aware MoE의 일반화 능력 narrative)

흐름:
    1. OpenBiasBench 로드 (data/openbias/)
    2. Stage 1: 4-prompt inference → primary_answer
    3. Stage 2: 7-signal extraction
    4. Question embedding
    5. 학습된 MoE 로드 + 추론 + threshold override
    6. 31 카테고리별 평가 (acc_amb/dis, bias_amb/dis, FAR)
    7. Cluster routing (31 × 4)
    8. Routing accuracy (수동 라벨 vs 모델 routing)
    9. Heatmap + summary.md

CLI:
    python -m src.transfer.run_openbias
    python -m src.transfer.run_openbias --threshold 0.75
    python -m src.transfer.run_openbias --max-samples 30 --out-dir results/openbias_smoke

NOTE: OpenBiasBench (Open-DeBias 2025 EMNLP Findings) 데이터는 별도 확보 필요.
data/openbias/{category}.jsonl 또는 data/openbias/test.parquet 형식.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("openbias")


# =============================================================
# Driver
# =============================================================
def run(
    config_path: str = "configs/default.yaml",
    data_dir: str = "data/openbias",
    categories: Optional[list[str]] = None,
    threshold: float = 0.5,
    moe_ckpt: Optional[str] = None,
    out_dir: str = "results/transfer/openbias",
    skip_existing: bool = True,
    max_samples: Optional[int] = None,
) -> dict:
    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    final_json = out_path / "overall_metrics.json"
    if skip_existing and final_json.exists():
        logger.info(f"  [skip] {final_json} 이미 존재")
        return json.loads(final_json.read_text(encoding="utf-8"))

    # ---- 1. 데이터 로드 ----
    from src.transfer.openbias import load_openbias

    try:
        items = load_openbias(data_dir=data_dir, categories=categories)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "OpenBiasBench 데이터 형식: "
            "data/openbias/{category}.jsonl 또는 data/openbias/test.parquet"
        )
        return {"error": "data_not_found"}

    if max_samples is not None:
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_cat[it.get("category", "_unknown")].append(it)
        items = []
        for cat, lst in by_cat.items():
            items.extend(lst[:max_samples])
        logger.info(f"  max_samples={max_samples}로 제한 → {len(items)} instances")

    cats = sorted({it.get("category", "_unknown") for it in items})
    logger.info(f"  Loaded {len(items)} instances, {len(cats)} categories")

    # ---- 2. LLM 로드 ----
    from src.utils.llm_utils import LLMWrapper

    model_cfg = config["models"]["main"]
    logger.info(f"  Loading LLM: {model_cfg['name']}")
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )

    # ---- 3. Stage 1 ----
    from src.signals.inference import run_4prompt_inference

    stage1_path = out_path / "_stage1.jsonl"
    if not stage1_path.exists() or not skip_existing:
        logger.info(f"  Stage 1: 4-prompt inference on {len(items)} instances")
        t0 = time.time()
        run_4prompt_inference(
            items=items, llm=llm, output_path=stage1_path,
            max_new_tokens=model_cfg.get("max_new_tokens", 64),
            temperature=model_cfg.get("temperature", 0.0),
        )
        logger.info(f"  Stage 1 완료: {time.time() - t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
    else:
        logger.info(f"  Stage 1 skip (cache exists): {stage1_path}")

    with open(stage1_path) as f:
        stage1_results = [json.loads(line) for line in f if line.strip()]

    # ---- 4. SAE + bias_heads ----
    sae = None
    sae_cfg = config.get("sae", {}).get("llama", {})
    if "release" in sae_cfg:
        try:
            from src.signals.sae_feature import SAEWrapper
            sae = SAEWrapper(
                release=sae_cfg["release"],
                sae_id=sae_cfg.get("sae_id", "l15r_8x"),
                layer=int(sae_cfg.get("layer", 15)),
                device=str(getattr(llm, "device", "cpu")),
            )
            sae._load()
        except Exception as e:
            logger.warning(f"  SAE 로드 실패 (s7=None): {e}")
            sae = None

    from src.signals.bias_head import load_bias_heads
    bias_head_indices = load_bias_heads("results/bias_heads.json")
    if bias_head_indices:
        logger.info(f"  Loaded {len(bias_head_indices)} bias heads")

    # ---- 5. Stage 2 ----
    from src.signals.extract_all import extract_signals_batch

    signals_path = out_path / "_signals.jsonl"
    if not signals_path.exists() or not skip_existing:
        logger.info(f"  Stage 2: 7-signal extraction")
        t0 = time.time()
        signals_results = extract_signals_batch(
            items=items, stage1_results=stage1_results, llm=llm, sae=sae,
            output_path=signals_path,
            n_consistency_samples=config["signals"]["s4_consistency"]["n_samples"],
            bias_head_indices=bias_head_indices,
        )
        logger.info(f"  Stage 2 완료: {time.time() - t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
    else:
        logger.info(f"  Stage 2 skip (cache exists): {signals_path}")
        with open(signals_path) as f:
            signals_results = [json.loads(line) for line in f if line.strip()]

    # ---- 6. Embeddings ----
    from src.models.embedding import EmbeddingExtractor, cache_embeddings

    emb_cache = out_path / "_embeddings.pt"
    extractor = EmbeddingExtractor(
        model_name=config.get("moe", {}).get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        device="cpu",
    )
    embeddings = cache_embeddings(items, extractor, emb_cache)
    logger.info(f"  Embeddings: {len(embeddings)}")

    # ---- 7. MoE 로드 ----
    from src.models.moe_aggregator import MoEAggregator, signals_dict_to_tensor

    moe_ckpt_path = Path(moe_ckpt) if moe_ckpt else Path("results/moe/main/moe_best.pt")
    if not moe_ckpt_path.exists():
        for cand in ("results/moe/main/moe_last.pt", "results/v2/moe/main/moe_best.pt"):
            if Path(cand).exists():
                moe_ckpt_path = Path(cand)
                break
    if not moe_ckpt_path.exists():
        logger.error(f"  MoE 체크포인트 없음. 먼저 학습 필요.")
        return {"error": "moe_checkpoint_not_found"}

    logger.info(f"  Loading MoE: {moe_ckpt_path}")
    saved = torch.load(moe_ckpt_path, map_location="cpu", weights_only=True)
    saved_cfg = saved.get("model_config", {})
    embed_dim = next(iter(embeddings.values())).shape[-1]

    model = MoEAggregator(
        signal_dim=int(saved_cfg.get("signal_dim", 7)),
        embed_dim=int(saved_cfg.get("embed_dim", embed_dim)),
        num_experts=int(saved_cfg.get("num_experts", 4)),
        gating_hidden=int(saved_cfg.get("gating_hidden", 64)),
        expert_hidden=int(saved_cfg.get("expert_hidden", 128)),
        dropout=float(saved_cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(saved.get("model_state_dict", saved), strict=False)
    model.eval()

    # ---- 8. MoE 추론 + override ----
    from src.evaluation.bbq_evaluator import evaluate_bbq
    from src.models.override import apply_threshold_override

    final_preds: list[int] = []
    final_items: list[dict] = []
    p_scores: list[float] = []
    gate_weights: list[list[float]] = []

    items_by_id = {it["example_id"]: it for it in items}
    device = next(model.parameters()).device

    with torch.inference_mode():
        for sig_rec in signals_results:
            ex_id = sig_rec["example_id"]
            if ex_id not in embeddings or ex_id not in items_by_id:
                continue

            sig_t = signals_dict_to_tensor(sig_rec.get("signals", {})).unsqueeze(0).to(device)
            emb_t = embeddings[ex_id].to(torch.float32).unsqueeze(0).to(device)
            out = model(sig_t, emb_t)

            p = float(out.p.item())
            gate_weights.append(out.gate_w[0].cpu().tolist())
            p_scores.append(p)

            primary = int(sig_rec.get("primary_answer", -1))
            item = items_by_id[ex_id]
            override = apply_threshold_override(
                primary_answer=primary, p_score=p, item=item, threshold=threshold,
            )
            final_preds.append(override["final_answer"])
            final_items.append(item)

    # ---- 9. 평가 ----
    overall = evaluate_bbq(final_preds, final_items)
    logger.info(
        f"  Overall: acc_amb={overall.get('accuracy_amb'):.4f} "
        f"acc_dis={overall.get('accuracy_dis'):.4f} "
        f"bias_amb={overall.get('bias_score_amb')}"
    )

    per_category: dict[str, dict] = {}
    for cat in cats:
        idxs = [i for i, it in enumerate(final_items) if it.get("category") == cat]
        if not idxs:
            continue
        cat_preds = [final_preds[i] for i in idxs]
        cat_items = [final_items[i] for i in idxs]
        per_category[cat] = evaluate_bbq(cat_preds, cat_items)

    # ---- 10. 31-cat mean (overall metrics 31 카테고리 평균) ----
    cat_mean: dict[str, float] = {}
    for key in ("accuracy_amb", "accuracy_dis", "false_abstention_rate"):
        vals = [m.get(key, 0.0) for m in per_category.values() if m.get(key) is not None]
        if vals:
            cat_mean[f"{key}_cat_mean"] = float(np.mean(vals))
    bias_vals = [m.get("bias_score_amb") for m in per_category.values()
                 if m.get("bias_score_amb") is not None]
    if bias_vals:
        cat_mean["bias_score_amb_cat_mean"] = float(np.mean(bias_vals))
        cat_mean["bias_score_amb_abs_cat_mean"] = float(np.mean([abs(v) for v in bias_vals]))

    # ---- 11. Cluster routing 매트릭스 + accuracy ----
    routing_matrix = _compute_routing_matrix(final_items, gate_weights, cats)
    cluster_names = ("Lex-Sub", "Numeric", "Cultural", "Identity")
    n_experts = len(gate_weights[0]) if gate_weights else 4
    if n_experts != 4:
        cluster_names = tuple(f"C{i}" for i in range(n_experts))

    # routing accuracy: dominant cluster vs DEFAULT_CATEGORY_TO_CLUSTER
    from src.transfer.openbias import DEFAULT_CATEGORY_TO_CLUSTER
    routing_accuracy = _compute_routing_accuracy(
        cats, routing_matrix, DEFAULT_CATEGORY_TO_CLUSTER,
    )

    # ---- 12. 저장 ----
    payload = {
        "method": "zero_shot_transfer_openbias",
        "model": model_cfg["name"],
        "moe_ckpt": str(moe_ckpt_path),
        "threshold": threshold,
        "n_total": len(final_items),
        "n_categories": len(cats),
        "n_per_category": {cat: sum(1 for it in final_items if it.get("category") == cat) for cat in cats},
        "overall": overall,
        "category_mean": cat_mean,           # 31-cat 평균
        "per_category": per_category,
        "routing_avg_per_category": {
            cat: routing_matrix[i].tolist() for i, cat in enumerate(cats)
        },
        "cluster_names": list(cluster_names),
        "routing_accuracy": routing_accuracy,
    }
    final_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    logger.info(f"  [저장] {final_json}")

    # CSV outputs
    _write_csv_outputs(out_path, per_category, routing_matrix, cats, cluster_names)

    # PDF heatmap
    _plot_routing_heatmap(routing_matrix, cats, cluster_names,
                         out_path / "routing_heatmap_31cat.pdf")

    # Routing accuracy JSON
    (out_path / "routing_accuracy.json").write_text(
        json.dumps(routing_accuracy, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )

    # transfer_summary.md
    _write_summary_md(
        out_path / "transfer_summary.md",
        n_total=len(final_items),
        n_cats=len(cats),
        overall=overall,
        cat_mean=cat_mean,
        routing_accuracy=routing_accuracy,
        per_category=per_category,
    )

    return payload


# =============================================================
# Helpers
# =============================================================
def _compute_routing_matrix(
    items: list[dict],
    gate_weights: list[list[float]],
    cats: list[str],
) -> np.ndarray:
    n_experts = len(gate_weights[0]) if gate_weights else 4
    matrix = np.zeros((len(cats), n_experts), dtype=np.float32)
    cat_idx = {c: i for i, c in enumerate(cats)}
    counts = np.zeros(len(cats), dtype=np.int64)

    for item, w in zip(items, gate_weights):
        cat = item.get("category", "_unknown")
        if cat not in cat_idx:
            continue
        i = cat_idx[cat]
        matrix[i] += np.array(w, dtype=np.float32)
        counts[i] += 1

    for i in range(len(cats)):
        if counts[i] > 0:
            matrix[i] /= counts[i]
    return matrix


def _compute_routing_accuracy(
    cats: list[str],
    routing_matrix: np.ndarray,
    category_to_cluster: dict[str, int],
) -> dict:
    """카테고리별 dominant cluster vs ground truth 비교."""
    correct = 0
    total = 0
    unmapped = 0
    per_cat: dict[str, dict] = {}

    for i, cat in enumerate(cats):
        dom = int(np.argmax(routing_matrix[i]))
        if cat not in category_to_cluster:
            unmapped += 1
            per_cat[cat] = {
                "dominant": dom,
                "expected": None,
                "is_correct": None,
            }
            continue
        gt = category_to_cluster[cat]
        is_correct = dom == gt
        per_cat[cat] = {
            "dominant": dom,
            "expected": gt,
            "is_correct": bool(is_correct),
        }
        total += 1
        if is_correct:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "n_evaluated": total,
        "n_correct": correct,
        "n_unmapped": unmapped,
        "per_category": per_cat,
    }


def _write_csv_outputs(
    out_path: Path,
    per_category: dict[str, dict],
    routing_matrix: np.ndarray,
    cats: list[str],
    cluster_names: tuple[str, ...],
) -> None:
    cat_csv = out_path / "per_category.csv"
    with open(cat_csv, "w", encoding="utf-8") as f:
        f.write("category,n_total,n_ambig,n_disambig,accuracy_amb,accuracy_dis,bias_score_amb,bias_score_dis,false_abstention_rate,parse_fail_rate\n")
        for cat in cats:
            m = per_category.get(cat, {})
            row = [
                cat,
                m.get("n_total", 0), m.get("n_ambig", 0), m.get("n_disambig", 0),
                m.get("accuracy_amb", 0.0), m.get("accuracy_dis", 0.0),
                m.get("bias_score_amb", "") if m.get("bias_score_amb") is not None else "",
                m.get("bias_score_dis", "") if m.get("bias_score_dis") is not None else "",
                m.get("false_abstention_rate", 0.0), m.get("parse_fail_rate", 0.0),
            ]
            f.write(",".join(str(v) for v in row) + "\n")
    logger.info(f"  [저장] {cat_csv}")

    routing_csv = out_path / "cluster_routing.csv"
    with open(routing_csv, "w", encoding="utf-8") as f:
        f.write("category," + ",".join(cluster_names) + "\n")
        for i, cat in enumerate(cats):
            f.write(cat + "," + ",".join(f"{v:.4f}" for v in routing_matrix[i].tolist()) + "\n")
    logger.info(f"  [저장] {routing_csv}")


def _plot_routing_heatmap(
    matrix: np.ndarray,
    cats: list[str],
    cluster_names: tuple[str, ...],
    save_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  matplotlib 미설치 — heatmap 생략")
        return

    # 31-cat: 더 큰 figure
    fig_h = max(0.32 * len(cats) + 1.5, 4.0)
    fig, ax = plt.subplots(figsize=(0.9 * len(cluster_names) + 2.5, fig_h))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cluster_names)))
    ax.set_xticklabels(cluster_names, rotation=20, ha="right")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=8)
    ax.set_title(f"OpenBiasBench: Cluster Routing ({len(cats)} categories)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Avg gate weight")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [저장] heatmap → {save_path}")


def _write_summary_md(
    path: Path,
    n_total: int,
    n_cats: int,
    overall: dict,
    cat_mean: dict,
    routing_accuracy: dict,
    per_category: dict,
) -> None:
    """transfer_summary.md 작성 — 논문/리뷰용 narrative."""
    sorted_cats = sorted(
        per_category.items(),
        key=lambda kv: kv[1].get("accuracy_amb", 0.0),
        reverse=True,
    )
    top5 = sorted_cats[:5]
    bot5 = sorted_cats[-5:]

    lines = [
        "# OpenBiasBench Zero-shot Transfer Summary",
        "",
        f"- **Instances**: {n_total:,}",
        f"- **Categories**: {n_cats}",
        f"- **Routing accuracy** (dominant cluster vs ground truth): "
        f"{routing_accuracy['accuracy']:.4f} "
        f"({routing_accuracy['n_correct']}/{routing_accuracy['n_evaluated']})",
        f"  · Unmapped categories (no GT cluster): {routing_accuracy['n_unmapped']}",
        "",
        "## Overall (전체 instance pooled)",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| accuracy_amb | {overall.get('accuracy_amb'):.4f} |",
        f"| accuracy_dis | {overall.get('accuracy_dis'):.4f} |",
        f"| bias_score_amb | {overall.get('bias_score_amb')} |",
        f"| false_abstention_rate | {overall.get('false_abstention_rate'):.4f} |",
        "",
        "## Category-mean (31-cat 평균)",
        "",
    ]
    for k, v in cat_mean.items():
        lines.append(f"- **{k}**: {v:.4f}")
    lines.extend([
        "",
        "## Top-5 categories (acc_amb)",
        "",
        "| Category | acc_amb | acc_dis | bias_amb | n |",
        "|----------|--------:|--------:|---------:|--:|",
    ])
    for cat, m in top5:
        lines.append(
            f"| {cat} | {m.get('accuracy_amb', 0.0):.3f} | "
            f"{m.get('accuracy_dis', 0.0):.3f} | "
            f"{m.get('bias_score_amb') if m.get('bias_score_amb') is not None else 'N/A'} | "
            f"{m.get('n_total', 0)} |"
        )
    lines.extend([
        "",
        "## Bottom-5 categories (acc_amb)",
        "",
        "| Category | acc_amb | acc_dis | bias_amb | n |",
        "|----------|--------:|--------:|---------:|--:|",
    ])
    for cat, m in bot5:
        lines.append(
            f"| {cat} | {m.get('accuracy_amb', 0.0):.3f} | "
            f"{m.get('accuracy_dis', 0.0):.3f} | "
            f"{m.get('bias_score_amb') if m.get('bias_score_amb') is not None else 'N/A'} | "
            f"{m.get('n_total', 0)} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  [저장] summary → {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenBiasBench zero-shot transfer evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data/openbias")
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--moe-ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/transfer/openbias")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run(
        config_path=args.config,
        data_dir=args.data_dir,
        categories=args.categories,
        threshold=args.threshold,
        moe_ckpt=args.moe_ckpt,
        out_dir=args.out_dir,
        skip_existing=not args.force,
        max_samples=args.max_samples,
    )
    return 2 if isinstance(result, dict) and "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
