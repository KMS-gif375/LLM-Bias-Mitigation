"""
SAE Layer Comparison.

현재 layer 15에서 s7 SAE feature contribution이 약함 (full data 기준 Δ=+0.0011).
다른 layer (10/12/15/18/20/22)에서 더 효과적인 SAE를 찾아 layer-wise s7 변화를
측정합니다.

알고리즘 (per layer L):
    1. SAE 로드 (release="llama_scope_lxr_8x", sae_id=f"l{L}r_8x")
    2. 모든 instance에 대해 layer L hidden state 추출 → SAE encode
       (single forward pass + output_hidden_states=True로 6 layers 한 번에 캡처)
    3. Bias feature 식별 (3 method 평균):
        - max_activation (BBQ 평균 활성도 top-N)
        - category_separability (between-category variance)
        - stereotype_correlation (stereotyped vs anti diff)
    4. s7 신호 재계산 (bias feature 평균 또는 top-k mean)
    5. 기존 s1-s6 + 새 s7 → MoE 학습 (signal_dim=7)
    6. Signal ablation: full vs s7-removed → Δ_s7 contribution
    7. BBQ 평가 (threshold search)

비용 최적화:
    - Forward pass per instance → hidden_states 6 layers 한 번에 추출
    - SAE encode는 4096 dim → 32K 선형 연산이라 ms 단위
    - 6 layers × (n_instances × 1 forward + 6 SAE encode + 3 method × identify
        + train MoE + eval) ≈ 2-4 hours on Mac MPS (single forward pass 활용)

CLI:
    # v1 (기존 2,097 signals 활용, 빠른 검증)
    python -m src.analysis.sae_layer_comparison --version v1 --layers 10,12,15,18,20,22

    # 1개 layer만 빠르게
    python -m src.analysis.sae_layer_comparison --version v1 --layers 18

    # smoke test
    python -m src.analysis.sae_layer_comparison --version v1 --layers 15 --max-samples 50

출력:
    results/{v1|v2}/sae_layers/comparison.csv
    results/{v1|v2}/sae_layers/best_layer.json
    results/{v1|v2}/sae_layers/s7_layer{L}.pt           # per-instance s7 값
    results/{v1|v2}/sae_layers/features_layer{L}.json   # 식별된 bias feature 인덱스
    results/{v1|v2}/sae_layers/comparison.pdf            # bar chart
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

logger = logging.getLogger("sae_layer_comparison")


DEFAULT_LAYERS: tuple[int, ...] = (10, 12, 15, 18, 20, 22)


# =============================================================
# Hidden state capture (6 layers in single forward)
# =============================================================
def collect_hidden_states_multi_layer(
    items: list[dict],
    llm,
    layers: list[int],
    show_progress: bool = True,
) -> dict[int, torch.Tensor]:
    """
    각 instance에 대해 지정된 layer들의 last-token hidden state를 한 번의
    forward pass로 캡처.

    Returns:
        {layer: (n_items, hidden_dim) float32 cpu tensor}
    """
    from src.utils.data_loader import format_question
    from src.signals.prompts import PROMPT_BUILDERS

    pool: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    builder = PROMPT_BUILDERS["vanilla"]

    iterator = tqdm(items, desc="Collect hiddens") if show_progress else items
    for item in iterator:
        try:
            system_msg, user_msg = builder(item)
            prompt = llm.build_chat_prompt(user_msg, system_msg)
            inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.device)

            with torch.inference_mode():
                outputs = llm.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            # outputs.hidden_states: tuple of (B, S, D), len = n_layers + 1 (embedding + per-layer)
            hs = outputs.hidden_states
            if hs is None:
                continue
            # transformers: hs[0] = embedding, hs[i] = output of layer i-1
            # 우리 layer 인덱스는 layer 0-indexed → hs[L+1]에 매핑
            for L in layers:
                idx = L + 1
                if idx >= len(hs):
                    continue
                last = hs[idx][0, -1, :].detach().to(torch.float32).cpu()
                pool[L].append(last)
        except Exception as e:
            logger.warning(f"  forward 실패 (example_id={item.get('example_id')}): {e}")
            for L in layers:
                # placeholder zero (나중에 mask)
                pool[L].append(torch.zeros(llm.model.config.hidden_size, dtype=torch.float32))

    out: dict[int, torch.Tensor] = {}
    for L in layers:
        if pool[L]:
            out[L] = torch.stack(pool[L], dim=0)
    return out


# =============================================================
# SAE encode (per layer)
# =============================================================
def encode_with_sae(
    hiddens: torch.Tensor,
    layer: int,
    release: str = "llama_scope_lxr_8x",
    device: str = "cpu",
    batch_size: int = 64,
) -> Optional[torch.Tensor]:
    """
    (n, hidden_dim) → (n, n_features) SAE feature activation.
    """
    try:
        from sae_lens import SAE
        sae = SAE.from_pretrained(
            release=release,
            sae_id=f"l{layer}r_8x",
            device=device,
        )
        if isinstance(sae, tuple):
            sae = sae[0]
        sae.eval()
    except Exception as e:
        logger.warning(f"  [layer {layer}] SAE 로드 실패: {e}")
        return None

    activations: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, hiddens.shape[0], batch_size):
            batch = hiddens[i:i + batch_size].to(device).to(
                next(sae.parameters()).dtype
            )
            try:
                feat = sae.encode(batch)
            except Exception:
                # 일부 SAELens 버전은 forward 사용
                feat = sae(batch)
            activations.append(feat.detach().to(torch.float32).cpu())

    return torch.cat(activations, dim=0)


# =============================================================
# Per-layer single-step pipeline
# =============================================================
@dataclass
class LayerResult:
    layer: int
    n_bias_features: int
    s7_delta_loss: float                    # full vs s7-removed
    full_val_loss: float
    no_s7_val_loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    bias_features: list[int] = field(default_factory=list)


def evaluate_layer(
    layer: int,
    activations: torch.Tensor,             # (n, n_features)
    records: list[dict],
    embeddings: dict,
    instances_by_id: dict,
    config: dict,
    save_dir: Path,
    top_k: int = 50,
) -> LayerResult:
    """
    1 layer에 대해 bias feature 식별 → s7 재계산 → MoE 학습 → ablation → 평가.
    """
    from src.ablation.sae_ablation import (
        identify_bias_features_max_activation,
        identify_bias_features_category_separability,
        identify_bias_features_stereotype_correlation,
    )
    from src.evaluation.bbq_evaluator import is_stereotyped_answer
    from src.models.moe_aggregator import MoEAggregator
    from src.models.trainer import SignalsDataset, TrainConfig, train_moe

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import _moe_predict_all, _infer_embed_dim  # type: ignore

    n = activations.shape[0]
    if n != len(records):
        raise ValueError(
            f"activations rows ({n}) != records ({len(records)})"
        )

    # category & stereotype labels — composite key (instances_by_id가 unique_id 사용)
    categories: list[str] = [r.get("category", "_unknown") for r in records]
    is_stereo: list[int] = []
    for r in records:
        ukey = r.get("unique_id")
        if not ukey:
            cat = r.get("category", "_unknown")
            ex_id = r.get("example_id")
            ukey = f"{cat}::{ex_id}" if ex_id is not None else None
        item = instances_by_id.get(ukey, {}) if ukey else {}
        ans_idx = int(r.get("primary_answer", -1)) if r.get("primary_answer") not in (None, "") else -1
        kind = is_stereotyped_answer(item, ans_idx) if ans_idx in (0, 1, 2) else None
        is_stereo.append(1 if kind == "stereotyped" else 0)

    # 3-method 합집합으로 bias feature 식별
    act_np = activations.numpy()
    methods_features = {
        "max": set(identify_bias_features_max_activation(act_np, top_k=top_k)),
        "cat": set(identify_bias_features_category_separability(act_np, categories, top_k=top_k)),
        "stereo": set(identify_bias_features_stereotype_correlation(act_np, is_stereo, top_k=top_k)),
    }
    bias_features = sorted(set.union(*methods_features.values()))
    logger.info(f"  [layer {layer}] bias features: max={len(methods_features['max'])} "
                f"cat={len(methods_features['cat'])} stereo={len(methods_features['stereo'])} "
                f"union={len(bias_features)}")

    # s7 재계산: bias feature 평균 활성도 (instance별)
    if bias_features:
        s7_values = act_np[:, bias_features].mean(axis=1)
    else:
        # fallback: top-k 평균
        topk_idx = np.argsort(act_np.mean(axis=0))[::-1][:top_k]
        s7_values = act_np[:, topk_idx].mean(axis=1)

    # 새 records (s7만 교체)
    new_records: list[dict] = []
    for r, s7 in zip(records, s7_values.tolist()):
        new_signals = dict(r["signals"])
        new_signals["s7_sae_feature"] = float(s7)
        new_r = dict(r)
        new_r["signals"] = new_signals
        new_records.append(new_r)

    # MoE 학습 (full) — stratified split (이전: 단순 슬라이싱 → val이 마지막
    # 카테고리에 편중되는 버그 있었음)
    from run_pipeline import _stratified_train_val_split  # type: ignore

    val_split = float(config["moe"].get("training", {}).get("val_split", 0.2))
    train_records, val_records = _stratified_train_val_split(
        new_records, val_ratio=val_split, seed=42,
    )

    train_ds = SignalsDataset(train_records, embeddings)
    val_ds = SignalsDataset(val_records, embeddings)
    embed_dim = _infer_embed_dim(embeddings, default=4096)
    moe_cfg = config["moe"]
    training_cfg = moe_cfg.get("training", {})

    layer_save = save_dir / f"layer_{layer}"
    layer_save.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
        epochs=int(training_cfg.get("epochs", 30)),
        batch_size=int(training_cfg.get("batch_size", 32)),
        lr=float(training_cfg.get("lr", 1e-3)),
        device=config["models"]["main"].get("device", "auto"),
        seed=42,
        save_dir=str(layer_save),
    )

    model_full = MoEAggregator(
        signal_dim=7, embed_dim=embed_dim,
        num_experts=int(moe_cfg.get("num_experts", 4)),
        gating_hidden=int(moe_cfg.get("gating_hidden_dim", 64)),
        expert_hidden=int(moe_cfg.get("expert_hidden_dim", 128)),
    )
    out_full = train_moe(train_ds, val_ds, model_full, train_config)
    full_val_loss = float(out_full.get("best_val_loss") or float("inf"))

    # Ablation: s7 제거 (mask to 0)
    from src.ablation.signal_ablation import MaskedSignalsDataset, SIGNAL_NAMES
    s7_idx = SIGNAL_NAMES.index("s7_sae_feature")
    train_ds_no_s7 = MaskedSignalsDataset(train_records, embeddings, mask_index=s7_idx)
    val_ds_no_s7 = MaskedSignalsDataset(val_records, embeddings, mask_index=s7_idx)

    torch.manual_seed(42)
    model_no_s7 = MoEAggregator(
        signal_dim=7, embed_dim=embed_dim,
        num_experts=int(moe_cfg.get("num_experts", 4)),
        gating_hidden=int(moe_cfg.get("gating_hidden_dim", 64)),
        expert_hidden=int(moe_cfg.get("expert_hidden_dim", 128)),
    )
    out_no_s7 = train_moe(train_ds_no_s7, val_ds_no_s7, model_no_s7, train_config)
    no_s7_val_loss = float(out_no_s7.get("best_val_loss") or float("inf"))
    delta = no_s7_val_loss - full_val_loss

    # 평가 (full model + per-condition threshold, 메인 평가와 일관)
    from src.evaluation.bbq_evaluator import evaluate_bbq
    from src.models.override import (
        apply_per_condition_override,
        search_optimal_threshold_per_condition,
    )

    val_predictions = _moe_predict_all(model_full, new_records, embeddings, instances_by_id)
    pc_search = search_optimal_threshold_per_condition(
        val_predictions,
        metric_amb="accuracy_amb",
        metric_dis="accuracy_dis",
        threshold_range=(0.05, 0.95),
        step=0.025,
    )
    thresholds = pc_search.thresholds
    final_preds: list[int] = []
    final_items: list[dict] = []
    for vp in val_predictions:
        result = apply_per_condition_override(
            primary_answer=int(vp["primary_answer"]),
            p_score=float(vp["p_score"]),
            item=vp["item"],
            thresholds=thresholds,
        )
        final_preds.append(result["final_answer"])
        final_items.append(vp["item"])

    metrics = evaluate_bbq(final_preds, final_items)
    metrics_clean = {
        k: float(v) for k, v in metrics.items()
        if v is not None and isinstance(v, (int, float))
    }
    metrics_clean["threshold_amb"] = float(thresholds["ambig"])
    metrics_clean["threshold_dis"] = float(thresholds["disambig"])

    # 저장: s7 + features
    torch.save(torch.tensor(s7_values, dtype=torch.float32), save_dir / f"s7_layer{layer}.pt")
    (save_dir / f"features_layer{layer}.json").write_text(
        json.dumps({
            "bias_features": bias_features,
            "method_features": {k: sorted(v) for k, v in methods_features.items()},
        }, indent=2),
        encoding="utf-8",
    )

    return LayerResult(
        layer=layer,
        n_bias_features=len(bias_features),
        s7_delta_loss=float(delta),
        full_val_loss=full_val_loss,
        no_s7_val_loss=no_s7_val_loss,
        metrics=metrics_clean,
        bias_features=bias_features,
    )


# =============================================================
# Driver
# =============================================================
def run(
    layers: list[int],
    config_path: str = "configs/default.yaml",
    version: str = "v1",
    out_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    top_k: int = 50,
) -> dict:
    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if version in ("v2", "smoke", "mini"):
        from src.utils.data_loader import DEFAULT_CATEGORIES_V2
        config["data"]["sampled_dir"] = {
            "v2": "data/sampled_v2",
            "smoke": "data/sampled_smoke",
            "mini": "data/sampled_mini",
        }[version]
        config["data"]["samples_per_category"] = {
            "v2": 1000, "smoke": 5, "mini": 100,
        }[version]
        config["data"]["categories"] = list(DEFAULT_CATEGORIES_V2)
        config["output"]["results_dir"] = {
            "v2": "results/v2",
            "smoke": "results/smoke_e2e",
            "mini": "results/v2_mini",
        }[version]

    if not out_dir:
        out_dir = {
            "v1": "results/sae_layers",
            "v2": "results/v2/sae_layers",
            "smoke": "results/smoke_e2e/sae_layers",
            "mini": "results/v2_mini/sae_layers",
        }[version]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import _collect_records_and_embeddings, _instances_by_id  # type: ignore
    from src.utils.llm_utils import LLMWrapper

    class _Args:
        def __init__(self):
            self.model = "main"
            self.categories = None

    args_ns = _Args()
    records, embeddings = _collect_records_and_embeddings(config, args_ns)
    if not records:
        raise RuntimeError("Stage 2 (signal_extraction) 결과 없음")

    # max_samples 적용 — 카테고리 + context_condition stratified
    # 이전 버그: records[:max_samples] → 카테고리 불균형 + ambig 편중
    if max_samples is not None and max_samples < len(records):
        from src.transfer._threshold_helper import stratified_sample_per_category
        # records를 카테고리당 max_samples//9 개로 stratified 샘플링
        n_cats = len(set(r.get("category", "_unknown") for r in records))
        per_cat = max(1, max_samples // max(n_cats, 1))
        records = stratified_sample_per_category(records, per_cat)
        logger.warning(f"  [smoke] stratified ~{max_samples}개로 제한 ({len(records)} actual)")

    instances_by_id = _instances_by_id(records, config, args_ns)
    # composite-key 사용 — record의 unique_id (없으면 ex_id+category로 fallback)
    def _ukey(r):
        u = r.get("unique_id")
        if u: return u
        cat = r.get("category", "_unknown")
        return f"{cat}::{r.get('example_id')}"
    items = [instances_by_id[_ukey(r)] for r in records if _ukey(r) in instances_by_id]
    records = [r for r in records if _ukey(r) in instances_by_id]
    logger.info(f"  records={len(records)}, embeddings={len(embeddings)}, items={len(items)}")

    # LLM 로드 (forward pass용)
    model_cfg = config["models"]["main"]
    logger.info(f"  Loading LLM: {model_cfg['name']}")
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )

    # Stage 1: 모든 layers의 hidden states 한 번에 캡처
    t0 = time.time()
    logger.info(f"  Collecting hidden states for layers {layers} ({len(items)} instances)")
    hiddens_per_layer = collect_hidden_states_multi_layer(items, llm, layers)
    logger.info(f"  Hidden state 수집 완료: {time.time() - t0:.1f}s")

    # Stage 2: per-layer SAE encode + 평가
    layer_results: list[LayerResult] = []
    sae_release = config.get("sae", {}).get("llama", {}).get("release", "llama_scope_lxr_8x")
    sae_device = "cpu"  # SAE는 메모리 절약 위해 CPU

    for L in layers:
        if L not in hiddens_per_layer:
            logger.warning(f"  [layer {L}] hidden state 없음 — skip")
            continue
        logger.info(f"\n{'='*60}\n  LAYER {L}\n{'='*60}")
        t1 = time.time()
        activations = encode_with_sae(
            hiddens_per_layer[L], L, release=sae_release, device=sae_device,
        )
        if activations is None:
            logger.warning(f"  [layer {L}] SAE encoding 실패 — skip")
            continue
        logger.info(f"  SAE encode 완료: {activations.shape} ({time.time() - t1:.1f}s)")

        try:
            res = evaluate_layer(
                layer=L,
                activations=activations,
                records=records,
                embeddings=embeddings,
                instances_by_id=instances_by_id,
                config=config,
                save_dir=out_path,
                top_k=top_k,
            )
            layer_results.append(res)
            logger.info(
                f"  [layer {L}] Δs7={res.s7_delta_loss:+.4f} "
                f"full_loss={res.full_val_loss:.4f} "
                f"acc_amb={res.metrics.get('accuracy_amb'):.4f}"
            )
        except Exception as e:
            logger.error(f"  [layer {L}] 평가 실패: {e}")

    # 결과 정리
    if not layer_results:
        raise RuntimeError("모든 layer 평가 실패")

    rows = [
        {
            "layer": r.layer,
            "n_bias_features": r.n_bias_features,
            "s7_delta_loss": r.s7_delta_loss,
            "full_val_loss": r.full_val_loss,
            "no_s7_val_loss": r.no_s7_val_loss,
            "best_threshold": r.metrics.get("best_threshold"),
            "accuracy_amb": r.metrics.get("accuracy_amb"),
            "accuracy_dis": r.metrics.get("accuracy_dis"),
            "bias_score_amb": r.metrics.get("bias_score_amb"),
            "false_abstention_rate": r.metrics.get("false_abstention_rate"),
        }
        for r in layer_results
    ]

    # CSV 저장
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(out_path / "comparison.csv", index=False)
    except ImportError:
        (out_path / "comparison.csv").write_text(
            ",".join(rows[0].keys()) + "\n" +
            "\n".join(",".join(str(v) for v in r.values()) for r in rows),
            encoding="utf-8",
        )
    logger.info(f"  [저장] comparison.csv → {out_path / 'comparison.csv'}")

    # Best layer
    best_row = max(rows, key=lambda r: (r["s7_delta_loss"] or 0))
    (out_path / "best_layer.json").write_text(
        json.dumps({
            "best_layer": int(best_row["layer"]),
            "criterion": "max s7_delta_loss",
            "row": best_row,
        }, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    logger.info(f"  Best layer: {best_row['layer']} (Δ={best_row['s7_delta_loss']:+.4f})")

    # 시각화 (matplotlib 옵션)
    try:
        import matplotlib.pyplot as plt
        layers_x = [r["layer"] for r in rows]
        deltas = [r["s7_delta_loss"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        bars = ax.bar(layers_x, deltas, color="steelblue", edgecolor="black")
        # current layer (15) 표시
        for bar, L in zip(bars, layers_x):
            if L == 15:
                bar.set_color("orange")
            if L == best_row["layer"]:
                bar.set_edgecolor("red")
                bar.set_linewidth(2.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δ val_loss (s7 contribution)")
        ax.set_title("SAE Layer Comparison (Llama-Scope l*r_8x)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for x, d in zip(layers_x, deltas):
            ax.text(x, d + (max(deltas) - min(deltas)) * 0.02, f"{d:+.4f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path / "comparison.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  [저장] comparison.pdf")
    except ImportError:
        logger.warning("  matplotlib 미설치 — plot 생략")

    return {
        "rows": rows,
        "best_layer": int(best_row["layer"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SAE Layer Comparison (Llama-Scope)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--version", type=str, default="v1", choices=("v1", "v2", "smoke", "mini"))
    parser.add_argument("--layers", type=str, default=",".join(str(L) for L in DEFAULT_LAYERS),
                        help="comma-separated layer indices")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="instance 수 제한 (smoke test)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="bias feature top-k 크기")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    layers = [int(L.strip()) for L in args.layers.split(",") if L.strip()]
    run(
        layers=layers,
        config_path=args.config,
        version=args.version,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
