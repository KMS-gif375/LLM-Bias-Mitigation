"""
KoBBQ (Korean BBQ) Zero-shot Cross-lingual Transfer Runner.

Source: HuggingFace `naver-ai/kobbq` (Jin et al., 2024 TACL).

핵심: 영어 BBQ로 학습된 시스템을 한국어 KoBBQ에 zero-shot 적용 → language-agnostic
generalization 입증.

KoBBQ schema:
    sample_id, label_annotation (ST/NC/TM), context (한글), question (한글),
    choices (str list repr), biased_answer (정답 텍스트), answer (정답 텍스트),
    bbq_id, bbq_category, prediction (null)

본 모듈은:
    1. HF datasets로 KoBBQ 자동 다운로드
    2. BBQ schema로 변환 (in-memory)
    3. Stage 1 + Stage 2 + MoE + override + per-category eval

NOTE: Llama-3.1-8B-Instruct는 한국어를 지원하나 학습된 bias_head/SAE는 영어
기반이라 신호 노이즈가 발생할 수 있음. 이는 cross-lingual transfer의 자연스러운
한계로 paper에 명시.

CLI:
    python -m src.transfer.run_kobbq                       # 기본
    python -m src.transfer.run_kobbq --max-samples 30      # smoke
    python -m src.transfer.run_kobbq --threshold 0.75
"""

from __future__ import annotations

import argparse
import ast
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

logger = logging.getLogger("kobbq")


def _parse_choices(choices_str: str) -> list[str]:
    """KoBBQ의 'choices'는 str list repr — ast로 안전 파싱."""
    if isinstance(choices_str, list):
        return choices_str
    try:
        result = ast.literal_eval(choices_str)
        if isinstance(result, list):
            return [str(x) for x in result]
    except (ValueError, SyntaxError):
        pass
    return []


def _parse_sample_id(sample_id: str) -> dict:
    """
    'age-001a-001-amb-bsd' →
        {category_prefix: 'age', condition: 'ambig', polarity_token: 'bsd'}
    """
    parts = sample_id.split("-")
    out = {"category_prefix": "", "condition": "", "polarity_token": ""}
    if not parts:
        return out
    out["category_prefix"] = parts[0]
    for token in parts:
        if token == "amb":
            out["condition"] = "ambig"
        elif token == "dis":
            out["condition"] = "disambig"
        elif token in ("bsd", "cnt"):
            out["polarity_token"] = token
    return out


def kobbq_to_bbq_schema(item: dict) -> Optional[dict]:
    """KoBBQ 한 record를 BBQ-style record로 변환. None이면 skip."""
    bbq_cat = item.get("bbq_category")
    if not bbq_cat:
        return None  # NC (None Category) skip

    choices = _parse_choices(item.get("choices", ""))
    if len(choices) < 3:
        return None

    answer_text = item.get("answer")
    biased_text = item.get("biased_answer")
    try:
        label = choices.index(answer_text)
    except ValueError:
        return None  # 정답 텍스트가 choices에 없음

    parsed = _parse_sample_id(item.get("sample_id", ""))
    cond = parsed["condition"] or "ambig"

    # question_polarity: bsd → neg (biased seeking direct), cnt → nonneg
    polarity = "neg" if parsed["polarity_token"] == "bsd" else "nonneg"

    # answer_info: choices[i] + group label (Unknown만 식별 가능)
    UNKNOWN_KO = ("알 수 없음", "확실하지 않다", "모름", "알수없음", "결정할 수 없음")
    answer_info = {}
    for i, ch in enumerate(choices[:3]):
        if any(uk in ch for uk in UNKNOWN_KO):
            answer_info[f"ans{i}"] = [ch, "unknown"]
        else:
            answer_info[f"ans{i}"] = [ch, "group"]

    # stereotyped_groups: biased_answer의 group label
    stereotyped_groups: list[str] = []
    if biased_text and biased_text in choices:
        idx = choices.index(biased_text)
        info = answer_info.get(f"ans{idx}", [])
        if len(info) >= 2 and info[1] != "unknown":
            stereotyped_groups = [info[0]]  # 텍스트 자체를 group label로 사용

    return {
        "example_id": item.get("sample_id"),
        "category": bbq_cat,
        "context_condition": cond,
        "question_polarity": polarity,
        "context": item.get("context", ""),
        "question": item.get("question", ""),
        "ans0": choices[0],
        "ans1": choices[1],
        "ans2": choices[2],
        "label": label,
        "answer_info": answer_info,
        "additional_metadata": {
            "stereotyped_groups": stereotyped_groups,
            "source": "kobbq",
            "language": "ko",
            "label_annotation": item.get("label_annotation"),
            "bbq_id": item.get("bbq_id"),
        },
    }


def load_kobbq_as_bbq(
    max_samples_per_category: Optional[int] = None,
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """HF에서 KoBBQ 다운로드 → BBQ schema 리스트로 변환."""
    from datasets import load_dataset
    ds = load_dataset("naver-ai/kobbq", split="test")
    logger.info(f"  KoBBQ raw: {len(ds)} entries")

    by_cat: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for raw in ds:
        bbq = kobbq_to_bbq_schema(raw)
        if bbq is None:
            skipped += 1
            continue
        if categories and bbq["category"] not in categories:
            continue
        by_cat[bbq["category"]].append(bbq)

    items: list[dict] = []
    for cat, recs in by_cat.items():
        if max_samples_per_category is not None:
            recs = recs[:max_samples_per_category]
        items.extend(recs)
        logger.info(f"  [{cat}] {len(recs)}")
    logger.info(f"  Total: {len(items)}, skipped: {skipped}")
    return items


def run(
    config_path: str = "configs/default.yaml",
    categories: Optional[list[str]] = None,
    threshold: float = 0.5,
    threshold_amb: Optional[float] = None,
    threshold_dis: Optional[float] = None,
    moe_ckpt: Optional[str] = None,
    out_dir: str = "results/transfer/kobbq",
    skip_existing: bool = True,
    max_samples: Optional[int] = None,
    model_key: str = "main",
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

    # 1. 데이터 로드
    items = load_kobbq_as_bbq(
        max_samples_per_category=max_samples, categories=categories,
    )
    if not items:
        return {"error": "no_items"}

    cats = sorted({it["category"] for it in items})

    # 2. LLM 로드
    from src.utils.llm_utils import LLMWrapper

    model_cfg = config["models"][model_key]
    logger.info(f"  Loading LLM: {model_cfg['name']}")
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )

    # 3. Stage 1
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
    with open(stage1_path) as f:
        stage1_results = [json.loads(line) for line in f if line.strip()]

    # 4. SAE + bias_heads
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
            logger.warning(f"  SAE 로드 실패: {e}")

    from src.signals.bias_head import load_bias_heads
    bias_head_indices = load_bias_heads("results/bias_heads.json")

    # 5. Stage 2
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
        with open(signals_path) as f:
            signals_results = [json.loads(line) for line in f if line.strip()]

    # 6. Embeddings
    from src.models.embedding import EmbeddingExtractor, cache_embeddings

    extractor = EmbeddingExtractor(
        model_name=config.get("moe", {}).get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        device="cpu",
    )
    embeddings = cache_embeddings(items, extractor, out_path / "_embeddings.pt")
    logger.info(f"  Embeddings: {len(embeddings)}")

    # 7. MoE 로드
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
        return {"error": "moe_checkpoint_not_found"}

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

    # 8. 추론 + override (per-condition) + 평가
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

    # cross-category example_id 충돌 방지
    embeddings, items_by_id = apply_composite_keys(items, embeddings)

    final_preds, final_items = [], []
    p_scores, gate_weights = [], []
    device = next(model.parameters()).device

    with torch.inference_mode():
        for sig_rec in signals_results:
            ukey = make_unique_id(sig_rec)
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

    overall = evaluate_bbq(final_preds, final_items)
    logger.info(
        f"  Overall: acc_amb={overall.get('accuracy_amb'):.4f} "
        f"acc_dis={overall.get('accuracy_dis'):.4f} "
        f"bias_amb={overall.get('bias_score_amb')}"
    )

    per_category = {}
    for cat in cats:
        idxs = [i for i, it in enumerate(final_items) if it.get("category") == cat]
        if not idxs:
            continue
        cat_preds = [final_preds[i] for i in idxs]
        cat_items = [final_items[i] for i in idxs]
        per_category[cat] = evaluate_bbq(cat_preds, cat_items)

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
        "method": "zero_shot_transfer_kobbq",
        "model": model_cfg["name"],
        "moe_ckpt": str(moe_ckpt_path),
        "language": "ko",
        "threshold": threshold,
        "n_total": len(final_items),
        "n_categories": len(cats),
        "n_per_category": {cat: int(counts[cat_idx[cat]]) for cat in cats},
        "overall": overall,
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

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="KoBBQ cross-lingual zero-shot transfer")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--moe-ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/transfer/kobbq")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="카테고리당 최대 샘플 수 (smoke test)")
    parser.add_argument("--force", action="store_true")
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
        categories=args.categories,
        threshold=args.threshold,
        moe_ckpt=args.moe_ckpt,
        out_dir=args.out_dir,
        skip_existing=not args.force,
        max_samples=args.max_samples,
        model_key=args.model,
    )
    return 2 if isinstance(result, dict) and "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
