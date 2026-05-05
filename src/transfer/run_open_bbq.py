"""
Open-BBQ Zero-shot Transfer Runner.

zhaoliu0914/LLM-Bias-Benchmark (Open-DeBias 2025) 데이터를 BBQ schema로
변환한 뒤 학습된 MoE를 zero-shot 적용. 11 카테고리 (BBQ 9 + Race_x_gender,
Race_x_SES 교차 카테고리 포함).

선행 작업:
    python -m src.data.prepare_open_bbq --auto

흐름은 ImplicitBBQ runner와 동일:
    1. data/open_bbq/{cat}.jsonl 로드
    2. Stage 1 + Stage 2 + MoE + override
    3. Per-category eval + cluster routing + summary

CLI:
    python -m src.transfer.run_open_bbq                       # 기본
    python -m src.transfer.run_open_bbq --threshold 0.75
    python -m src.transfer.run_open_bbq --max-samples 30      # smoke
    python -m src.transfer.run_open_bbq --categories Age      # 일부
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

logger = logging.getLogger("open_bbq")


def _load_jsonl_dir(data_dir: Path, categories: Optional[list[str]] = None) -> list[dict]:
    """data/open_bbq/{cat}.jsonl 들을 통합 로드."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Open-BBQ 디렉토리 없음: {data_dir}\n"
            f"먼저 변환: python -m src.data.prepare_open_bbq --auto"
        )
    items: list[dict] = []
    for jf in sorted(data_dir.glob("*.jsonl")):
        if jf.name.startswith("_"):
            continue
        cat = jf.stem
        if categories and cat not in categories:
            continue
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec.setdefault("category", cat)
                    items.append(rec)
    return items


def run(
    config_path: str = "configs/default.yaml",
    data_dir: str = "data/open_bbq",
    categories: Optional[list[str]] = None,
    threshold: float = 0.5,
    moe_ckpt: Optional[str] = None,
    out_dir: str = "results/transfer/open_bbq",
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
    items = _load_jsonl_dir(Path(data_dir), categories=categories)
    if not items:
        logger.error(f"Open-BBQ data 없음: {data_dir}")
        return {"error": "data_not_found"}

    # max_samples 적용
    if max_samples is not None:
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_cat[it.get("category", "_unknown")].append(it)
        items = []
        for cat, lst in by_cat.items():
            items.extend(lst[:max_samples])
        logger.info(f"  max_samples={max_samples}로 카테고리당 제한")

    cats = sorted({it.get("category", "_unknown") for it in items})
    logger.info(f"  Loaded {len(items)} instances, {len(cats)} categories: {cats}")

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
        logger.info(f"  Stage 1 skip: {stage1_path}")
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
        logger.info(f"  Stage 2 skip: {signals_path}")
        with open(signals_path) as f:
            signals_results = [json.loads(line) for line in f if line.strip()]

    # ---- 6. Embeddings ----
    from src.models.embedding import EmbeddingExtractor, cache_embeddings

    extractor = EmbeddingExtractor(
        model_name=config.get("moe", {}).get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        device="cpu",
    )
    embeddings = cache_embeddings(items, extractor, out_path / "_embeddings.pt")
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
        logger.error(f"  MoE 체크포인트 없음")
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

    # ---- 8. 추론 + override ----
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

    # category 평균
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

    # routing matrix
    n_experts = len(gate_weights[0]) if gate_weights else 4
    matrix = np.zeros((len(cats), n_experts), dtype=np.float32)
    cat_idx = {c: i for i, c in enumerate(cats)}
    counts = np.zeros(len(cats), dtype=np.int64)
    for item, w in zip(final_items, gate_weights):
        i = cat_idx.get(item.get("category", "_unknown"), -1)
        if i < 0:
            continue
        matrix[i] += np.array(w, dtype=np.float32)
        counts[i] += 1
    for i in range(len(cats)):
        if counts[i] > 0:
            matrix[i] /= counts[i]

    cluster_names = ("Lex-Sub", "Numeric", "Cultural", "Identity")
    if n_experts != 4:
        cluster_names = tuple(f"C{i}" for i in range(n_experts))

    payload = {
        "method": "zero_shot_transfer_open_bbq",
        "model": model_cfg["name"],
        "moe_ckpt": str(moe_ckpt_path),
        "threshold": threshold,
        "n_total": len(final_items),
        "n_categories": len(cats),
        "n_per_category": {cat: int(counts[cat_idx[cat]]) for cat in cats},
        "overall": overall,
        "category_mean": cat_mean,
        "per_category": per_category,
        "routing_avg_per_category": {
            cat: matrix[i].tolist() for i, cat in enumerate(cats)
        },
        "cluster_names": list(cluster_names),
    }
    final_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    logger.info(f"  [저장] {final_json}")

    # CSV
    cat_csv = out_path / "per_category.csv"
    with open(cat_csv, "w", encoding="utf-8") as f:
        f.write("category,n_total,accuracy_amb,accuracy_dis,bias_score_amb,bias_score_dis,false_abstention_rate\n")
        for cat in cats:
            m = per_category.get(cat, {})
            row = [
                cat, m.get("n_total", 0),
                m.get("accuracy_amb", 0.0), m.get("accuracy_dis", 0.0),
                m.get("bias_score_amb") if m.get("bias_score_amb") is not None else "",
                m.get("bias_score_dis") if m.get("bias_score_dis") is not None else "",
                m.get("false_abstention_rate", 0.0),
            ]
            f.write(",".join(str(v) for v in row) + "\n")
    logger.info(f"  [저장] {cat_csv}")

    routing_csv = out_path / "cluster_routing.csv"
    with open(routing_csv, "w", encoding="utf-8") as f:
        f.write("category," + ",".join(cluster_names) + "\n")
        for i, cat in enumerate(cats):
            f.write(cat + "," + ",".join(f"{v:.4f}" for v in matrix[i].tolist()) + "\n")
    logger.info(f"  [저장] {routing_csv}")

    # heatmap
    try:
        import matplotlib.pyplot as plt
        fig_h = max(0.4 * len(cats) + 1.5, 4.0)
        fig, ax = plt.subplots(figsize=(0.9 * len(cluster_names) + 2.5, fig_h))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(cluster_names)))
        ax.set_xticklabels(cluster_names, rotation=20, ha="right")
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=8)
        ax.set_title(f"Open-BBQ: Cluster Routing ({len(cats)} categories)")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        fig.colorbar(im, ax=ax, label="Avg gate weight")
        fig.tight_layout()
        fig.savefig(out_path / "routing_heatmap.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  [저장] heatmap → {out_path / 'routing_heatmap.pdf'}")
    except ImportError:
        pass

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Open-BBQ zero-shot transfer evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data/open_bbq")
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--moe-ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/transfer/open_bbq")
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
