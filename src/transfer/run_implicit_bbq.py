"""
ImplicitBBQ Zero-shot Transfer Runner.

학습된 MoE를 ImplicitBBQ에 zero-shot 적용. 우리 메인 파이프라인 (run_pipeline)의
Stage 1-4를 ImplicitBBQ data에 그대로 돌려서 transfer 성능을 측정합니다.

흐름:
    1. ImplicitBBQ 데이터 로드 (data/implicit_bbq/)
    2. Stage 1: 4-prompt inference → primary_answer
    3. Stage 2: 7-signal extraction (s1~s7)
    4. Question embedding (sentence-transformers)
    5. 학습된 MoE 로드 (default: results/moe/main/moe_best.pt)
    6. MoE 추론 + threshold override
    7. BBQ evaluation (overall + per-category)
    8. Cluster routing 분석 (4-cluster gating weight)
    9. Heatmap PDF 생성

데이터 형식:
    data/implicit_bbq/{category}.jsonl   ← 카테고리별 JSONL 파일, OR
    data/implicit_bbq/test.parquet       ← 통합 parquet

데이터 다운로드 (참고):
    Kim et al. 2024 ImplicitBBQ. 본 모듈은 자동 다운로드를 제공하지 않으므로
    다음 중 하나를 사용해 data/implicit_bbq/에 배치하세요:
        - 논문 저자 GitHub repo
        - 자체 implicit 변형 생성 (BBQ context를 implicit cue로 paraphrase)

CLI:
    # 기본 실행 (data/implicit_bbq/ 사용)
    python -m src.transfer.run_implicit_bbq

    # 데이터 디렉토리 지정
    python -m src.transfer.run_implicit_bbq --data-dir data/my_implicit/

    # 특정 카테고리만
    python -m src.transfer.run_implicit_bbq --categories Age Gender_identity

    # threshold 조정
    python -m src.transfer.run_implicit_bbq --threshold 0.75
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("implicit_bbq")


# =============================================================
# Driver
# =============================================================
def run(
    config_path: str = "configs/default.yaml",
    data_dir: str = "data/implicit_bbq",
    categories: Optional[list[str]] = None,
    threshold: float = 0.5,
    threshold_amb: Optional[float] = None,
    threshold_dis: Optional[float] = None,
    moe_ckpt: Optional[str] = None,
    out_dir: str = "results/transfer/implicit_bbq",
    skip_existing: bool = True,
    max_samples: Optional[int] = None,
    model_key: str = "main",
) -> dict:
    """
    ImplicitBBQ zero-shot transfer 평가.

    Args:
        config_path: YAML config.
        data_dir: ImplicitBBQ 데이터 디렉토리.
        categories: 평가할 카테고리. None이면 전체.
        threshold: override 임계값 (default 0.5).
        moe_ckpt: 학습된 MoE 경로. None이면 results/moe/main/moe_best.pt.
        out_dir: 결과 저장 경로.
        max_samples: 카테고리당 최대 샘플 (smoke test용).

    Returns:
        평가 결과 dict.
    """
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
    from src.transfer.implicit_bbq import load_implicit_bbq

    try:
        items = load_implicit_bbq(data_dir=data_dir, categories=categories)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "ImplicitBBQ 데이터 형식: "
            "data/implicit_bbq/{category}.jsonl 또는 data/implicit_bbq/test.parquet"
        )
        return {"error": "data_not_found"}

    if max_samples is not None:
        # 카테고리당 max_samples로 stratified truncate (ambig/disambig 균등)
        from src.transfer._threshold_helper import stratified_sample_per_category
        items = stratified_sample_per_category(items, max_samples)

    cats = sorted({it.get("category", "_unknown") for it in items})
    logger.info(f"  Loaded {len(items)} instances, {len(cats)} categories: {cats}")

    # ---- 2. LLM 로드 ----
    from src.utils.llm_utils import LLMWrapper

    model_cfg = config["models"][model_key]
    logger.info(f"  Loading LLM: {model_cfg['name']}")
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )

    # ---- 3. Stage 1: 4-prompt inference ----
    from src.signals.inference import run_4prompt_inference

    stage1_path = out_path / "_stage1.jsonl"
    if not stage1_path.exists() or not skip_existing:
        logger.info(f"  Stage 1: 4-prompt inference on {len(items)} instances")
        t0 = time.time()
        run_4prompt_inference(
            items=items,
            llm=llm,
            output_path=stage1_path,
            max_new_tokens=model_cfg.get("max_new_tokens", 64),
            temperature=model_cfg.get("temperature", 0.0),
        )
        logger.info(f"  Stage 1 완료: {time.time() - t0:.1f}s")
    else:
        logger.info(f"  Stage 1 skip (cache exists): {stage1_path}")

    with open(stage1_path) as f:
        stage1_results = [json.loads(line) for line in f if line.strip()]

    # ---- 4. SAE 로드 (optional) ----
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
            logger.warning(f"  SAE 로드 실패 (s7 비활성화): {e}")
            sae = None

    # bias_heads 로드 (있으면)
    from src.signals.bias_head import load_bias_heads
    bias_head_indices = load_bias_heads("results/bias_heads.json")
    if bias_head_indices:
        logger.info(f"  Loaded {len(bias_head_indices)} bias heads from results/bias_heads.json")

    # ---- 5. Stage 2: 7-signal extraction ----
    from src.signals.extract_all import extract_signals_batch

    signals_path = out_path / "_signals.jsonl"
    if not signals_path.exists() or not skip_existing:
        logger.info(f"  Stage 2: 7-signal extraction")
        t0 = time.time()
        signals_results = extract_signals_batch(
            items=items,
            stage1_results=stage1_results,
            llm=llm,
            sae=sae,
            output_path=signals_path,
            n_consistency_samples=config["signals"]["s4_consistency"]["n_samples"],
            bias_head_indices=bias_head_indices,
        )
        logger.info(f"  Stage 2 완료: {time.time() - t0:.1f}s")
    else:
        logger.info(f"  Stage 2 skip (cache exists): {signals_path}")
        with open(signals_path) as f:
            signals_results = [json.loads(line) for line in f if line.strip()]

    # ---- 6. Question embeddings ----
    from src.models.embedding import EmbeddingExtractor, cache_embeddings

    emb_cache = out_path / "_embeddings.pt"
    extractor = EmbeddingExtractor(
        model_name=config.get("moe", {}).get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        device="cpu",
    )
    embeddings = cache_embeddings(items, extractor, emb_cache)
    logger.info(f"  Embeddings: {len(embeddings)} cached")

    # ---- 7. MoE 로드 ----
    from src.models.moe_aggregator import MoEAggregator, signals_dict_to_tensor

    moe_ckpt_path = Path(moe_ckpt) if moe_ckpt else None
    if moe_ckpt_path is None or not moe_ckpt_path.exists():
        candidates = []
        if model_key != "main":
            candidates += [
                f"results/v2/cross_llm/{model_key}/moe/main/moe_best.pt",
                f"results/v2/cross_llm/{model_key}/moe/{model_key}/moe_best.pt",
            ]
        candidates += [
            "results/v2/moe/main/moe_best.pt",
            "results/moe/main/moe_best.pt",
            "results/moe/main/moe_last.pt",
        ]
        for cand in candidates:
            if Path(cand).exists():
                moe_ckpt_path = Path(cand)
                break
    if moe_ckpt_path is None or not moe_ckpt_path.exists():
        logger.error(f"  MoE 체크포인트 없음. 먼저 학습 필요. (tried: {candidates})")
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

    # ---- 8. MoE 추론 + threshold override (per-condition) ----
    from src.evaluation.bbq_evaluator import evaluate_bbq
    from src.models.override import apply_per_condition_override
    from src.transfer._threshold_helper import (
        apply_composite_keys,
        make_unique_id,
        resolve_thresholds,
    )

    thresholds = resolve_thresholds(
        threshold=threshold,
        threshold_amb=threshold_amb,
        threshold_dis=threshold_dis,
        model_key=model_key,
    )

    # cross-category example_id 충돌 방지 — composite key로 통일.
    # 데이터에 collision이 있으면 helper가 warning 띄움.
    embeddings, items_by_id = apply_composite_keys(items, embeddings)

    final_preds: list[int] = []
    final_items: list[dict] = []
    p_scores: list[float] = []
    gate_weights: list[list[float]] = []
    device = next(model.parameters()).device

    with torch.inference_mode():
        for sig_rec in signals_results:
            ukey = make_unique_id(sig_rec)  # category::ex_id
            if ukey not in embeddings or ukey not in items_by_id:
                continue

            sig_t = signals_dict_to_tensor(sig_rec.get("signals", {})).unsqueeze(0).to(device)
            emb_t = embeddings[ukey].to(torch.float32).unsqueeze(0).to(device)
            out = model(sig_t, emb_t)

            p = float(out.p.item())
            gate_weights.append(out.gate_w[0].cpu().tolist())
            p_scores.append(p)

            primary = int(sig_rec.get("primary_answer", -1))
            item = items_by_id[ukey]
            override = apply_per_condition_override(
                primary_answer=primary, p_score=p, item=item, thresholds=thresholds,
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

    # ---- 10. Cluster routing ----
    routing_matrix = _compute_routing_matrix(final_items, gate_weights, cats)
    cluster_names = ("Lex-Sub", "Numeric", "Cultural", "Identity")  # default 4-cluster
    n_experts = len(gate_weights[0]) if gate_weights else 4
    if n_experts != 4:
        cluster_names = tuple(f"C{i}" for i in range(n_experts))

    # ---- 11. 저장 ----
    payload = {
        "method": "zero_shot_transfer_implicit_bbq",
        "model": model_cfg["name"],
        "moe_ckpt": str(moe_ckpt_path),
        "threshold": threshold,
        "n_total": len(final_items),
        "n_per_category": {cat: sum(1 for it in final_items if it.get("category") == cat) for cat in cats},
        "overall": overall,
        "per_category": per_category,
        "routing_avg_per_category": {
            cat: list(routing_matrix[i].tolist())
            for i, cat in enumerate(cats)
        },
        "cluster_names": list(cluster_names),
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
                         out_path / "routing_heatmap.pdf")

    # raw predictions
    preds_path = out_path / "predictions.csv"
    with open(preds_path, "w", encoding="utf-8") as f:
        f.write("example_id,category,context_condition,label,primary_answer,p_score,final_answer,overridden\n")
        for it, p, p_score, fp in zip(final_items, p_scores, p_scores, final_preds):
            ex_id = it.get("example_id", "")
            cat = it.get("category", "")
            cc = it.get("context_condition", "")
            lab = it.get("label", -1)
            primary = -1  # raw가 사라졌으므로 생략
            overridden = ""
            f.write(f"{ex_id},{cat},{cc},{lab},{primary},{p_score:.4f},{fp},{overridden}\n")

    return payload


def _compute_routing_matrix(
    items: list[dict],
    gate_weights: list[list[float]],
    cats: list[str],
) -> np.ndarray:
    """카테고리 × cluster 평균 gating weight 행렬."""
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


def _write_csv_outputs(
    out_path: Path,
    per_category: dict[str, dict],
    routing_matrix: np.ndarray,
    cats: list[str],
    cluster_names: tuple[str, ...],
) -> None:
    """per_category.csv + cluster_routing.csv 작성."""
    cat_csv = out_path / "per_category.csv"
    with open(cat_csv, "w", encoding="utf-8") as f:
        f.write("category,n_total,n_ambig,n_disambig,accuracy_amb,accuracy_dis,bias_score_amb,bias_score_dis,false_abstention_rate,parse_fail_rate\n")
        for cat, m in per_category.items():
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
    """카테고리 × cluster routing heatmap PDF."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  matplotlib 미설치 — heatmap 생략")
        return

    fig, ax = plt.subplots(figsize=(0.9 * len(cluster_names) + 2.5, 0.4 * len(cats) + 1.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cluster_names)))
    ax.set_xticklabels(cluster_names, rotation=20, ha="right")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_title("ImplicitBBQ: Cluster Routing (gating weight)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Avg gate weight")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [저장] heatmap → {save_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ImplicitBBQ zero-shot transfer evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data/implicit_bbq",
                        help="ImplicitBBQ 데이터 디렉토리")
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="threshold override τ (기본 0.5)")
    parser.add_argument("--moe-ckpt", type=str, default=None,
                        help="학습된 MoE .pt 경로 (기본: results/moe/main/moe_best.pt)")
    parser.add_argument("--out-dir", type=str, default="results/transfer/implicit_bbq")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="카테고리당 최대 샘플 (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="기존 결과 무시하고 재실행")
    parser.add_argument("--model", type=str, default="main",
                        choices=("main", "gemma", "qwen", "mistral"),
                        help="LLM model key from config['models']")
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
        model_key=args.model,
    )
    if isinstance(result, dict) and "error" in result:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
