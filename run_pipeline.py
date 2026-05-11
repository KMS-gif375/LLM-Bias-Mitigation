"""
End-to-End 파이프라인 통합 실행 스크립트.

전체 흐름:
    Stage 0: 데이터 샘플링       (sampling)
    Stage 1: 4-Prompt Inference  (inference)
    Stage 2: 7-Signal Extraction (signal_extraction)
    Stage 3: MoE 학습            (moe_training)
    Stage 4: Threshold Override + 평가 (evaluation)
    Stage 5: Ablation 실험       (ablation)

각 stage는 중간 결과를 results/ 하위에 저장하므로, 일부만 다시 돌릴 수 있습니다.

사용법:
    # 전체 파이프라인
    python run_pipeline.py --all

    # 특정 단계만
    python run_pipeline.py --stage signal_extraction
    python run_pipeline.py --stage moe_training
    python run_pipeline.py --stage evaluation
    python run_pipeline.py --stage ablation

    # 여러 단계 연속
    python run_pipeline.py --stage signal_extraction moe_training evaluation

    # Cross-LLM (Gemma / Qwen)
    python run_pipeline.py --cross-llm gemma
    python run_pipeline.py --cross-llm qwen --stage evaluation

    # 빠른 테스트 (카테고리당 10개, epochs=2)
    python run_pipeline.py --all --quick-test

    # 특정 카테고리만
    python run_pipeline.py --stage signal_extraction --categories Age Gender_identity
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import yaml

# .env 자동 로드 (HF_TOKEN 등). transformers/huggingface_hub가 import 되기 전에 호출.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================
# Stage registry
# =============================================================
STAGES: tuple[str, ...] = (
    "sampling",
    "inference",
    "signal_extraction",
    "moe_training",
    "evaluation",
    "ablation",
)

# stage 별칭 (shortcuts)
STAGE_ALIASES: dict[str, str] = {
    "1": "inference",
    "2": "signal_extraction",
    "3": "moe_training",
    "4": "evaluation",
    "5": "ablation",
    "0": "sampling",
    "signals": "signal_extraction",
    "train": "moe_training",
    "eval": "evaluation",
}


# =============================================================
# Logging setup
# =============================================================
def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> Path:
    """파일 + stdout 동시 로깅."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"pipeline_{int(time.time())}.log"

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(sh)

    return log_path


logger = logging.getLogger("run_pipeline")


# =============================================================
# Config helpers
# =============================================================
def load_config(path: str) -> dict:
    """YAML 설정 로드."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config 파일 없음: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_quick_test_overrides(config: dict) -> dict:
    """`--quick-test` 모드: 데이터/학습 규모 축소."""
    qt = dict(config)  # shallow ok
    qt.setdefault("data", {})
    qt["data"] = {**qt["data"], "samples_per_category": 10}

    moe = qt.get("moe", {})
    training = moe.get("training", {})
    qt.setdefault("moe", {})
    qt["moe"] = {
        **moe,
        "training": {**training, "epochs": 2, "batch_size": 8, "val_every": 1},
    }

    eval_cfg = qt.get("evaluation", {})
    qt["evaluation"] = {
        **eval_cfg,
        "bootstrap": {**eval_cfg.get("bootstrap", {}), "n_iterations": 50},
    }
    logger.warning("[quick-test] 데이터/학습 규모 축소 적용")
    return qt


def select_model_block(config: dict, model_key: str) -> dict:
    """
    `models.{key}` 블록을 반환.
    cross-llm일 경우 'gemma'/'qwen', 일반은 'main'.
    """
    if "models" not in config or model_key not in config["models"]:
        raise KeyError(f"config['models']['{model_key}'] 없음")
    return config["models"][model_key]


# =============================================================
# Stage runners
# =============================================================
def run_sampling(config: dict, args: argparse.Namespace) -> dict:
    """Stage 0: 데이터 샘플링."""
    logger.info("=" * 60)
    logger.info("[STAGE 0] 데이터 샘플링")
    logger.info("=" * 60)

    try:
        # src.utils.sampling이 있으면 호출, 없으면 안내
        from importlib import import_module
        try:
            sampling = import_module("src.utils.sampling")
        except ModuleNotFoundError:
            logger.warning(
                "  src/utils/sampling.py 미존재 — 별도 샘플링 스크립트 필요. "
                "data/sampled/ 에 카테고리별 JSONL이 있는지 확인하세요."
            )
            return {"skipped": True, "reason": "sampling module not found"}

        if hasattr(sampling, "main"):
            sampling.main()
            return {"status": "ok"}
        if hasattr(sampling, "sample_bbq"):
            sampling.sample_bbq(
                bbq_dir=config["data"]["bbq_dir"],
                output_dir=config["data"]["sampled_dir"],
                samples_per_category=config["data"].get("samples_per_category", 300),
                categories=config["data"]["categories"],
                seed=config.get("seed", 42),
            )
            return {"status": "ok"}
        logger.warning("  sampling 진입점 없음 — skip")
        return {"skipped": True}
    except Exception as e:
        logger.error(f"  sampling 실패: {e}")
        return {"error": str(e)}


def run_inference(config: dict, args: argparse.Namespace) -> dict:
    """Stage 1: 4-Prompt Inference."""
    logger.info("=" * 60)
    logger.info("[STAGE 1] 4-Prompt Inference")
    logger.info("=" * 60)

    from src.signals.inference import run_4prompt_inference
    from src.utils.data_loader import load_bbq_category
    from src.utils.llm_utils import LLMWrapper

    model_cfg = select_model_block(config, args.model)
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )

    output_dir = _stage_output_dir(config, args.model, "signals")
    categories = args.categories or config["data"]["categories"]
    n_per_cat = config["data"].get("samples_per_category", 300)

    summary: dict = {"per_category": {}}
    for category in categories:
        logger.info(f"  [{category}]")
        items = _load_items(config, category, n_per_cat)
        if not items:
            logger.warning(f"    items 없음 — skip")
            continue
        for it in items:
            it.setdefault("category", category)

        out_path = output_dir / f"{category}_stage1.jsonl"
        if args.skip_existing and out_path.exists():
            logger.info(f"    [skip-existing] {out_path}")
            summary["per_category"][category] = {"skipped": True}
            continue

        try:
            run_4prompt_inference(
                items=items,
                llm=llm,
                output_path=out_path,
                max_new_tokens=model_cfg.get("max_new_tokens", 64),
                temperature=model_cfg.get("temperature", 0.0),
            )
            summary["per_category"][category] = {"out": str(out_path), "n": len(items)}
        except Exception as e:
            logger.error(f"    실패: {e}")
            summary["per_category"][category] = {"error": str(e)}
            if args.strict:
                raise

    return summary


def run_signal_extraction(config: dict, args: argparse.Namespace) -> dict:
    """Stage 2: 7-Signal Extraction."""
    logger.info("=" * 60)
    logger.info("[STAGE 2] 7-Signal Extraction")
    logger.info("=" * 60)

    from src.signals.extract_all import extract_signals_batch
    from src.signals.sae_feature import SAEWrapper
    from src.utils.llm_utils import LLMWrapper

    model_cfg = select_model_block(config, args.model)
    llm = LLMWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device=model_cfg.get("device", "auto"),
    )

    sae = _maybe_load_sae(config, args.model, llm)

    # bias_heads.json 자동 생성 — 첫 카테고리의 stage1 결과로 contrastive 식별
    _maybe_identify_bias_heads(config, args, llm)

    signals_dir = _stage_output_dir(config, args.model, "signals")
    categories = args.categories or config["data"]["categories"]
    n_per_cat = config["data"].get("samples_per_category", 300)

    summary: dict = {"per_category": {}}
    for category in categories:
        logger.info(f"  [{category}]")
        stage1_path = signals_dir / f"{category}_stage1.jsonl"
        if not stage1_path.exists():
            logger.warning(f"    stage1 결과 없음 — Stage 1 먼저 실행 필요: {stage1_path}")
            summary["per_category"][category] = {"error": "missing_stage1"}
            continue

        with open(stage1_path, "r", encoding="utf-8") as f:
            stage1_results = [json.loads(line) for line in f if line.strip()]

        items = _load_items(config, category, n_per_cat)
        for it in items:
            it.setdefault("category", category)

        out_path = signals_dir / f"{category}_signals.jsonl"
        if args.skip_existing and out_path.exists():
            logger.info(f"    [skip-existing] {out_path}")
            summary["per_category"][category] = {"skipped": True}
            continue

        try:
            # 모델별 bias_heads 명시 전달 (Gemma/Qwen은 자체 식별된 path 사용)
            from src.signals.bias_head import bias_heads_path_for, load_bias_heads
            model_bias_heads = load_bias_heads(bias_heads_path_for(args.model))
            extract_signals_batch(
                items=items,
                stage1_results=stage1_results,
                llm=llm,
                sae=sae,
                output_path=out_path,
                n_consistency_samples=config["signals"]["s4_consistency"]["n_samples"],
                bias_head_indices=model_bias_heads,
            )
            summary["per_category"][category] = {"out": str(out_path), "n": len(items)}
        except Exception as e:
            logger.error(f"    실패: {e}")
            summary["per_category"][category] = {"error": str(e)}
            if args.strict:
                raise

    return summary


def run_moe_training(config: dict, args: argparse.Namespace) -> dict:
    """Stage 3: MoE Aggregator 학습."""
    logger.info("=" * 60)
    logger.info("[STAGE 3] MoE Aggregator Training")
    logger.info("=" * 60)

    import torch  # noqa: F401  (필요시)

    from src.models.moe_aggregator import MoEAggregator
    from src.models.trainer import SignalsDataset, TrainConfig, train_moe

    records, embeddings = _collect_records_and_embeddings(config, args)
    if not records:
        return {"error": "no signal records"}

    # 3-way split: train (학습) / val (τ tuning, early stopping) / test (최종 보고).
    # data leakage 차단을 위해 train_pipeline에서 test_records는 전혀 보지 않음.
    # 같은 seed → run_evaluation에서 동일 split 재현.
    val_ratio = float(config["moe"].get("training", {}).get("val_split", 0.15))
    test_ratio = float(config["moe"].get("training", {}).get("test_split", 0.15))
    seed = int(config.get("seed", 42))
    train_records, val_records, _test_records = _stratified_three_way_split(
        records,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # 검증 로깅: 카테고리 분포
    from collections import Counter
    train_cat = Counter(r.get("category", "_unknown") for r in train_records)
    val_cat = Counter(r.get("category", "_unknown") for r in val_records)
    logger.info(f"  train cats: {dict(train_cat)}")
    logger.info(f"  val   cats: {dict(val_cat)}")

    train_ds = SignalsDataset(train_records, embeddings)
    val_ds = SignalsDataset(val_records, embeddings)
    logger.info(f"  train={len(train_ds)}  val={len(val_ds)}  test_held_out={len(_test_records)}")

    moe_cfg = config.get("moe", {})
    training_cfg = moe_cfg.get("training", {})
    embed_dim = _infer_embed_dim(embeddings, default=4096)

    save_dir = _stage_output_dir(config, args.model, "moe")
    train_config = TrainConfig(
        epochs=int(training_cfg.get("epochs", 30)),
        batch_size=int(training_cfg.get("batch_size", 32)),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        val_every=int(training_cfg.get("val_every", 5)),
        device=config["models"][args.model].get("device", "auto"),
        seed=int(config.get("seed", 42)),
        save_dir=str(save_dir),
    )

    model = MoEAggregator(
        signal_dim=7,
        embed_dim=embed_dim,
        num_experts=int(moe_cfg.get("num_experts", 4)),
        gating_hidden=int(moe_cfg.get("gating_hidden_dim", 64)),
        expert_hidden=int(moe_cfg.get("expert_hidden_dim", 128)),
    )
    out = train_moe(train_ds, val_ds, model, train_config)
    logger.info(f"  best_val_loss={out.get('best_val_loss')} ckpt={out.get('checkpoint_path')}")
    return {
        "best_val_loss": out.get("best_val_loss"),
        "best_epoch": out.get("best_epoch"),
        "checkpoint_path": str(out.get("checkpoint_path") or ""),
    }


def run_evaluation(config: dict, args: argparse.Namespace) -> dict:
    """Stage 4: Threshold Override + BBQ 평가."""
    logger.info("=" * 60)
    logger.info("[STAGE 4] Threshold Override + Evaluation")
    logger.info("=" * 60)

    import torch

    from src.evaluation.bbq_evaluator import evaluate_bbq
    from src.models.moe_aggregator import MoEAggregator
    from src.models.override import (
        apply_per_condition_override,
        apply_threshold_override,
        apply_threshold_override_per_condition,
        risk_coverage_curve,
        search_optimal_threshold,
        search_optimal_threshold_per_condition,
    )

    records, embeddings = _collect_records_and_embeddings(config, args)
    if not records:
        return {"error": "no signal records"}

    instances_by_id = _instances_by_id(records, config, args)

    # MoE 로드 — 체크포인트의 model_config로 모델을 재구성해야 hidden dim 등이 정확히 일치.
    ckpt = _find_latest_checkpoint(_stage_output_dir(config, args.model, "moe"))
    embed_dim = _infer_embed_dim(embeddings, default=4096)
    moe_cfg = config["moe"]

    saved_state = None
    saved_model_cfg: dict = {}
    if ckpt and ckpt.exists():
        try:
            saved_state = torch.load(ckpt, map_location="cpu", weights_only=True)
            saved_model_cfg = saved_state.get("model_config", {}) or {}
        except Exception as e:
            logger.warning(f"  체크포인트 파일 읽기 실패: {e}")
            saved_state = None

    # 우선순위: 체크포인트의 model_config → config의 moe → MoEAggregator default
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
        try:
            sd = saved_state.get("model_state_dict", saved_state)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"  state_dict missing keys: {list(missing)[:5]}")
            if unexpected:
                logger.warning(f"  state_dict unexpected keys: {list(unexpected)[:5]}")
            if not missing and not unexpected:
                logger.info(f"  MoE 체크포인트 로드 완료: {ckpt}")
            else:
                logger.warning(f"  MoE 체크포인트 부분 로드: {ckpt}")
        except Exception as e:
            logger.warning(f"  체크포인트 로드 실패 — 미학습 모델로 평가: {e}")

    # 동일 split 재현: Stage 3과 같은 seed → train/val/test
    val_ratio = float(config["moe"].get("training", {}).get("val_split", 0.15))
    test_ratio = float(config["moe"].get("training", {}).get("test_split", 0.15))
    seed = int(config.get("seed", 42))
    _train_records, val_records, test_records = _stratified_three_way_split(
        records,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    logger.info(
        f"  [eval-split] train_seen={len(_train_records)} val_for_tau={len(val_records)} "
        f"test_held_out={len(test_records)}"
    )

    # MoE 추론: val (τ tuning) + test (보고)
    val_predictions = _moe_predict_all(model, val_records, embeddings, instances_by_id)
    test_predictions = _moe_predict_all(model, test_records, embeddings, instances_by_id)
    if not val_predictions or not test_predictions:
        return {"error": "no predictions"}

    # threshold search — val에서만
    tau_range = config.get("override", {}).get(
        "threshold_search", {"range": [0.3, 0.7], "step": 0.05}
    )
    pc_range_cfg = config.get("override", {}).get("threshold_search", {})
    pc_range = tuple(pc_range_cfg.get("per_condition_range", [0.05, 0.95]))
    pc_step = float(pc_range_cfg.get("per_condition_step", 0.025))

    search = search_optimal_threshold(
        val_predictions,
        threshold_range=tuple(tau_range.get("range", [0.3, 0.7])),
        step=float(tau_range.get("step", 0.05)),
    )
    logger.info(f"  [single   τ on val] best_tau={search.best_threshold} score={search.best_score:.4f}")

    pc_search = search_optimal_threshold_per_condition(
        val_predictions,
        metric_amb="accuracy_amb",
        metric_dis="accuracy_dis",
        threshold_range=pc_range,
        step=pc_step,
    )
    thresholds_by_cond = pc_search.thresholds
    logger.info(
        f"  [per-cond τ on val] tau_amb={thresholds_by_cond['ambig']:.3f} "
        f"tau_dis={thresholds_by_cond['disambig']:.3f} "
        f"combined={pc_search.combined_score:.4f}"
    )

    # 평가 1: single tau on TEST (held-out)
    final_preds_single: list[int] = []
    final_items: list[dict] = []
    for vp in test_predictions:
        result = apply_threshold_override(
            primary_answer=vp["primary_answer"],
            p_score=vp["p_score"],
            item=vp["item"],
            threshold=search.best_threshold,
        )
        final_preds_single.append(result["final_answer"])
        final_items.append(vp["item"])
    metrics_single = evaluate_bbq(final_preds_single, final_items)

    # 평가 2: per-condition tau on TEST (held-out, 메인)
    final_preds_pc: list[int] = []
    for vp in test_predictions:
        result = apply_per_condition_override(
            primary_answer=vp["primary_answer"],
            p_score=vp["p_score"],
            item=vp["item"],
            thresholds=thresholds_by_cond,
        )
        final_preds_pc.append(result["final_answer"])
    metrics_pc = evaluate_bbq(final_preds_pc, final_items)

    # 비교 요약
    logger.info(
        "  ---- Single vs Per-Condition ----\n"
        f"  single tau={search.best_threshold:.3f}: "
        f"acc_amb={metrics_single.get('accuracy_amb'):.4f} "
        f"acc_dis={metrics_single.get('accuracy_dis'):.4f} "
        f"bias_amb={metrics_single.get('bias_score_amb')}\n"
        f"  per-cond {thresholds_by_cond}: "
        f"acc_amb={metrics_pc.get('accuracy_amb'):.4f} "
        f"acc_dis={metrics_pc.get('accuracy_dis'):.4f} "
        f"bias_amb={metrics_pc.get('bias_score_amb')}"
    )

    # 결과 저장 (default = per-condition)
    eval_dir = _stage_output_dir(config, args.model, "evaluation")
    out_path = eval_dir / "final.json"
    payload = {
        # backward-compat 단일 필드들 (사용자 main repo schema 유지)
        "threshold": search.best_threshold,
        "thresholds_per_condition": thresholds_by_cond,
        "metrics": metrics_pc,                # 기본: per-condition
        "metrics_single_tau": metrics_single,
        "metrics_per_condition": metrics_pc,
        "n_predictions": len(final_preds_pc),
        # worktree 추가: per-condition search 디테일 (combined_score, grid scores)
        "per_condition_search": {
            "combined_score": pc_search.combined_score,
            "ambig_scores": {f"{k:.3f}": v for k, v in pc_search.per_condition_scores["ambig"].items()},
            "disambig_scores": {f"{k:.3f}": v for k, v in pc_search.per_condition_scores["disambig"].items()},
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"  저장: {out_path}")
    # 호환을 위해 final_preds 변수 유지
    final_preds = final_preds_pc
    metrics = metrics_pc

    # Risk-coverage curve: test set 기준 (paper figure용)
    rc = risk_coverage_curve(test_predictions)
    rc_path = eval_dir / "risk_coverage.json"
    rc_path.write_text(
        json.dumps(
            [
                {
                    "threshold": p.threshold,
                    "coverage": p.coverage,
                    "risk": p.risk,
                    "error_rate": p.error_rate,
                }
                for p in rc
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    return payload


def run_ablation(config: dict, args: argparse.Namespace) -> dict:
    """Stage 5: Ablation 실험 (signal / cluster / loco)."""
    logger.info("=" * 60)
    logger.info("[STAGE 5] Ablation 실험")
    logger.info("=" * 60)

    from src.ablation.cluster_ablation import run_cluster_ablation
    from src.ablation.loco_ablation import run_loco_ablation
    from src.ablation.signal_ablation import run_signal_ablation
    from src.models.trainer import TrainConfig

    records, embeddings = _collect_records_and_embeddings(config, args)
    if not records:
        return {"error": "no signal records"}

    # Stage 3과 동일한 3-way split 재현 (consistent data boundaries across stages)
    val_ratio = float(config["moe"].get("training", {}).get("val_split", 0.15))
    test_ratio = float(config["moe"].get("training", {}).get("test_split", 0.15))
    seed = int(config.get("seed", 42))
    train_records, val_records, _test_records = _stratified_three_way_split(
        records,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    embed_dim = _infer_embed_dim(embeddings, default=4096)
    training_cfg = config["moe"].get("training", {})
    train_config = TrainConfig(
        epochs=int(training_cfg.get("epochs", 30)),
        batch_size=int(training_cfg.get("batch_size", 32)),
        lr=float(training_cfg.get("lr", 1e-3)),
        device=config["models"][args.model].get("device", "auto"),
        seed=seed,
    )

    abl_dir = _stage_output_dir(config, args.model, "ablation")
    summary: dict = {}

    # signal ablation
    try:
        sig = run_signal_ablation(
            train_records, val_records, embeddings,
            embed_dim=embed_dim,
            train_config=train_config,
            save_dir=str(abl_dir / "signals"),
        )
        summary["signal"] = {
            "full_val_loss": sig.full.best_val_loss,
            "contributions": sig.contributions(),
        }
    except Exception as e:
        logger.error(f"  signal ablation 실패: {e}")
        summary["signal"] = {"error": str(e)}
        if args.strict:
            raise

    # cluster ablation
    try:
        cl = run_cluster_ablation(
            train_records, val_records, embeddings,
            embed_dim=embed_dim,
            train_config=train_config,
            save_dir=str(abl_dir / "cluster"),
        )
        summary["cluster"] = {
            axis: [
                {"value": r.config.value, "best_val_loss": r.best_val_loss}
                for r in results
            ]
            for axis, results in cl.by_axis.items()
        }
    except Exception as e:
        logger.error(f"  cluster ablation 실패: {e}")
        summary["cluster"] = {"error": str(e)}
        if args.strict:
            raise

    # loco ablation
    try:
        instances_by_id = _instances_by_id(records, config, args)
        loco = run_loco_ablation(
            records, embeddings, instances_by_id,
            embed_dim=embed_dim,
            train_config=train_config,
            threshold=float(config.get("override", {}).get("threshold", 0.5)),
            save_dir=str(abl_dir / "loco"),
        )
        summary["loco"] = {
            "aggregate": loco.aggregate(),
            "per_fold": {
                k: v.held_out_acc_amb for k, v in loco.per_fold.items()
            },
        }
    except Exception as e:
        logger.error(f"  loco ablation 실패: {e}")
        summary["loco"] = {"error": str(e)}
        if args.strict:
            raise

    return summary


# =============================================================
# Internal helpers
# =============================================================
def _stage_output_dir(config: dict, model_key: str, stage_name: str) -> Path:
    """results/{stage}/{model} 디렉토리 (없으면 생성)."""
    base = Path(config.get("output", {}).get("results_dir", "results"))
    out = base / stage_name / model_key
    out.mkdir(parents=True, exist_ok=True)
    return out


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _maybe_identify_bias_heads(config: dict, args: argparse.Namespace, llm) -> None:
    """
    모델별 bias_heads.json이 없으면 contrastive 분석으로 자동 식별.

    저장 경로:
        main: results/bias_heads.json (legacy)
        gemma: results/cross_llm/gemma/bias_heads.json
        qwen: results/cross_llm/qwen/bias_heads.json

    Cross-LLM 모델 (Gemma/Qwen)도 자체 attention 구조에서 bias-head를
    재식별해야 함 — Llama bias-head를 다른 모델 attention에 그대로 적용 X.
    """
    from src.signals.bias_head import (
        bias_heads_path_for,
        identify_bias_heads,
        load_bias_heads,
    )
    from src.signals.prompts import PROMPT_BUILDERS

    save_path = bias_heads_path_for(args.model)

    # 이미 캐시 있으면 skip
    cached = load_bias_heads(save_path)
    if cached:
        logger.info(f"  [bias-head:{args.model}] 캐시 로드: {len(cached)}개 head ({save_path})")
        return

    logger.info(f"  [bias-head:{args.model}] 캐시 없음 → contrastive 식별 시도 → {save_path}")

    # 첫 카테고리의 stage1 결과로 식별 시도 (전체 카테고리 통합 식별이 이상적이지만
    # 비용/시간 trade-off로 카테고리 1~2개 샘플 사용).
    signals_dir = _stage_output_dir(config, args.model, "signals")
    categories = args.categories or config["data"]["categories"]
    n_per_cat = config["data"].get("samples_per_category", 300)

    train_pool: list[dict] = []
    stage1_pool: list[dict] = []
    # 식별 비용 제한 — 카테고리당 최대 50개
    max_per_cat = min(50, n_per_cat)

    for category in categories:
        stage1_path = signals_dir / f"{category}_stage1.jsonl"
        if not stage1_path.exists():
            continue
        with open(stage1_path, "r", encoding="utf-8") as f:
            stage1_results = [json.loads(line) for line in f if line.strip()]
        items = _load_items(config, category, n_per_cat)
        for it in items:
            it.setdefault("category", category)
        train_pool.extend(items[:max_per_cat])
        stage1_pool.extend(stage1_results[:max_per_cat])
        if len(train_pool) >= 200:  # 충분한 샘플 모이면 stop
            break

    if not train_pool:
        logger.warning(
            "  [bias-head] stage1 결과 없음 — 식별 skip "
            "(Stage 1을 먼저 돌리세요. 그렇지 않으면 s5=0)"
        )
        return

    try:
        heads = identify_bias_heads(
            bbq_train_data=train_pool,
            stage1_results=stage1_pool,
            llm=llm,
            prompt_builder=PROMPT_BUILDERS["vanilla"],
            primary_prompt="vanilla",
            n_top=20,
            save_path=save_path,
            max_samples=len(train_pool),
        )
        if heads:
            logger.info(f"  [bias-head:{args.model}] {len(heads)}개 식별 → {save_path}")
        else:
            logger.warning(f"  [bias-head:{args.model}] 식별 결과 빈 리스트 (s5=0)")
    except Exception as e:
        logger.warning(f"  [bias-head:{args.model}] 식별 실패: {e} (s5=0)")


def _maybe_load_sae(config: dict, model_key: str, llm) -> Optional[object]:
    """모델 키에 맞춰 SAE 로드 (실패해도 None)."""
    sae_cfg_root = config.get("sae", {})
    if model_key == "qwen":
        return None  # Qwen은 SAE 미지원
    cfg_key = "llama" if model_key == "main" else model_key
    sae_cfg = sae_cfg_root.get(cfg_key)
    if not sae_cfg:
        return None
    try:
        from src.signals.sae_feature import SAEWrapper

        # 신규 형식 (release + sae_id) 우선, 구형식 (repo) 호환
        if "release" in sae_cfg:
            sae = SAEWrapper(
                release=sae_cfg["release"],
                sae_id=sae_cfg.get("sae_id", f"l{int(sae_cfg.get('layer', 15))}r_8x"),
                layer=int(sae_cfg.get("layer", 15)),
                device=str(getattr(llm, "device", "auto")),
            )
        else:
            # 레거시 config (repo 직접 지정) — release로 매핑 시도
            legacy_repo = sae_cfg.get("repo", "")
            logger.warning(
                f"  legacy SAE config 'repo={legacy_repo}' — release/sae_id 사용 권장"
            )
            sae = SAEWrapper(
                release=legacy_repo,
                sae_id=f"blocks.{int(sae_cfg.get('layer', 15))}.hook_resid_post",
                layer=int(sae_cfg.get("layer", 15)),
                device=str(getattr(llm, "device", "auto")),
            )

        # eager validation — lazy load 시 첫 추론에서 터지는 것을 미리 감지
        try:
            sae._load()  # type: ignore[attr-defined]
        except Exception as load_exc:
            logger.warning(f"  SAE eager load 실패 (s7 비활성화): {load_exc}")
            return None
        return sae
    except Exception as e:
        logger.warning(f"  SAE 로드 실패 (s7 비활성화): {e}")
        return None


def _load_items(config: dict, category: str, n_per_cat: int) -> list[dict]:
    """
    샘플링된 카테고리 데이터 로드.

    지원 형식 (우선순위 순):
        1. data/sampled/{category}.jsonl
        2. data/sampled/{category}/items.jsonl
        3. data/sampled/{train,val,test}.parquet — category 컬럼으로 필터
    """
    sampled_dir = Path(config["data"]["sampled_dir"])

    # JSONL 우선
    for path in (sampled_dir / f"{category}.jsonl",
                 sampled_dir / category / "items.jsonl"):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f if line.strip()]
            return items[:n_per_cat]

    # Parquet split fallback (data_loader.py --all 결과)
    parquet_files = [sampled_dir / f"{split}.parquet" for split in ("train", "val", "test")]
    if any(p.exists() for p in parquet_files):
        try:
            import pandas as pd
            dfs = [pd.read_parquet(p) for p in parquet_files if p.exists()]
            df = pd.concat(dfs, ignore_index=True)
            df = df[df["category"] == category]
            if len(df) == 0:
                logger.warning(f"  parquet에 '{category}' 카테고리 없음")
                return []
            items = df.head(n_per_cat).to_dict(orient="records")
            # parquet 정규화:
            #  1. numpy.ndarray → list
            #  2. JSON-stringified dict 컬럼 (answer_info, additional_metadata) → dict
            json_dict_cols = ("answer_info", "additional_metadata")
            for item in items:
                for k, v in list(item.items()):
                    if hasattr(v, "tolist"):
                        item[k] = v.tolist()
                    elif k in json_dict_cols and isinstance(v, str):
                        try:
                            item[k] = json.loads(v)
                        except (json.JSONDecodeError, ValueError):
                            pass
            return items
        except Exception as e:
            logger.warning(f"  parquet 로드 실패 ({category}): {e}")

    logger.warning(f"  샘플링 파일 없음: {category} (jsonl 또는 parquet split 필요)")
    return []


def _make_unique_id(category: str, ex_id) -> str:
    """
    카테고리 간 example_id 충돌을 막는 composite key.

    BBQ 원본은 example_id가 카테고리 내부에서만 unique → 카테고리 간 합치면
    충돌 (예: id=327이 Religion과 SES에 동시 등장). 이 함수가 만드는 키를
    embeddings dict / instances_by_id 등 모든 cross-category lookup에서 사용.
    """
    return f"{category}::{ex_id}"


def _collect_records_and_embeddings(config: dict, args: argparse.Namespace):
    """
    모든 카테고리의 stage2 signal record와 embedding을 모아 반환.

    각 record에 unique_id (composite key) 필드를 추가하고, embeddings dict는
    unique_id를 키로 사용하여 카테고리 간 example_id 충돌을 방지한다.

    Embedding은 results/embeddings/{model}/{category}.pt 또는
    results/signals/{model}/{category}_embeddings.pt 에서 로드.
    """
    import torch

    signals_dir = _stage_output_dir(config, args.model, "signals")
    categories = args.categories or config["data"]["categories"]

    records: list[dict] = []
    embeddings: dict = {}            # key = unique_id (composite)
    n_collision_total = 0

    for category in categories:
        sig_path = signals_dir / f"{category}_signals.jsonl"
        if not sig_path.exists():
            logger.warning(f"  [{category}] signal 파일 없음: {sig_path}")
            continue
        with open(sig_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec.setdefault("category", category)
                # composite key를 record에 추가하여 다운스트림에서 사용
                rec["unique_id"] = _make_unique_id(category, rec["example_id"])
                records.append(rec)

        # embedding 검색 (있으면 로드, 없으면 즉석 생성)
        # 캐시 파일 내부 키는 raw example_id이므로 로드 시 카테고리 prefix 추가
        emb_candidates = [
            signals_dir / f"{category}_embeddings.pt",
            Path(config.get("output", {}).get("results_dir", "results"))
            / "embeddings" / args.model / f"{category}.pt",
        ]
        loaded = False
        for emb_path in emb_candidates:
            if emb_path.exists():
                try:
                    cached = torch.load(emb_path, map_location="cpu")
                    if isinstance(cached, dict):
                        for raw_key, vec in cached.items():
                            ukey = _make_unique_id(category, raw_key)
                            if ukey in embeddings:
                                n_collision_total += 1
                            embeddings[ukey] = vec
                        loaded = True
                except Exception as e:
                    logger.warning(f"  embedding 로드 실패 ({emb_path}): {e}")
                break

        if not loaded:
            # 캐시 없음 → sentence-transformers로 즉석 생성 + 캐시
            try:
                from src.models.embedding import EmbeddingExtractor, cache_embeddings
                n_per_cat = config["data"].get("samples_per_category", 300)
                items = _load_items(config, category, n_per_cat)
                if items:
                    cache_path = signals_dir / f"{category}_embeddings.pt"
                    extractor = EmbeddingExtractor(
                        model_name=config.get("moe", {}).get(
                            "embedding_model",
                            "sentence-transformers/all-MiniLM-L6-v2",
                        ),
                        device="cpu",  # CPU로 충분, MPS는 LLM이 점유
                    )
                    new_emb = cache_embeddings(items, extractor, cache_path)
                    # 캐시 파일은 raw key로 저장됨 → 메모리에서는 unique_id로
                    for raw_key, vec in new_emb.items():
                        ukey = _make_unique_id(category, raw_key)
                        if ukey in embeddings:
                            n_collision_total += 1
                        embeddings[ukey] = vec
                    logger.info(f"  [embed] {category}: {len(new_emb)}개 신규 생성 → {cache_path}")
            except Exception as e:
                logger.warning(f"  embedding 즉석 생성 실패 ({category}): {e}")

    if n_collision_total > 0:
        # composite key 도입 후에는 충돌이 있으면 안 됨 (같은 카테고리 내부에서
        # 중복된 example_id가 있다는 뜻 → signal extraction 단계의 버그)
        logger.warning(
            f"  [embed] composite key 후에도 {n_collision_total}개 충돌 — "
            f"동일 카테고리 내부 example_id 중복 (signal extraction 검토 필요)"
        )
    logger.info(f"  records={len(records)}  embeddings={len(embeddings)}")
    return records, embeddings


def _instances_by_id(records: list[dict], config: dict, args: argparse.Namespace) -> dict[str, dict]:
    """
    Records 기반 instance lookup. 키는 unique_id (composite). record에 'item'
    키가 있으면 사용, 없으면 sampled에서 보강.

    카테고리 간 example_id 충돌 방지를 위해 unique_id를 키로 사용. 만약 record가
    legacy schema (unique_id 없음)면 example_id+category로 즉석 생성.
    """
    out: dict[str, dict] = {}
    for r in records:
        item = r.get("item")
        ukey = r.get("unique_id")
        if not ukey:
            ex_id = r.get("example_id")
            cat = r.get("category", "_unknown")
            if ex_id is None:
                continue
            ukey = _make_unique_id(cat, ex_id)
        if item:
            out[ukey] = item

    # 부족하면 sampled에서 보강
    if len(out) < len(records):
        n_per_cat = config["data"].get("samples_per_category", 300)
        for category in (args.categories or config["data"]["categories"]):
            for it in _load_items(config, category, n_per_cat):
                eid = it.get("example_id")
                if eid is None:
                    continue
                ukey = _make_unique_id(category, eid)
                if ukey not in out:
                    it.setdefault("category", category)
                    out[ukey] = it
    return out


def _stratified_train_val_split(
    records: list[dict],
    val_ratio: float,
    seed: int,
    stratify_keys: tuple[str, ...] = ("category", "context_condition"),
) -> tuple[list[dict], list[dict]]:
    """
    Stratified + shuffled train/val split.

    카테고리/맥락 비율을 보존하면서 셔플. 단순 슬라이싱 시 마지막 카테고리만
    val로 빠지는 문제를 방지.

    Args:
        records: signal record 리스트.
        val_ratio: val 비율 ∈ (0, 1).
        seed: 재현성용 시드.
        stratify_keys: 그룹 분류 기준.

    Returns:
        (train_records, val_records).
    """
    import random

    rng = random.Random(seed)

    # stratum 별로 그룹핑
    by_stratum: dict[tuple, list[dict]] = {}
    for rec in records:
        key = tuple(rec.get(k, "_unknown") for k in stratify_keys)
        by_stratum.setdefault(key, []).append(rec)

    train: list[dict] = []
    val: list[dict] = []
    for key, group in by_stratum.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) >= 2 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    logger.info(
        f"  [split] train={len(train)} val={len(val)} "
        f"(stratified by {stratify_keys}, {len(by_stratum)} strata)"
    )
    return train, val


def _stratified_three_way_split(
    records: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_keys: tuple[str, ...] = ("category", "context_condition"),
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified train/val/test 3-way split. data leakage 차단용.

    Args:
        val_ratio: val 비율 (예: 0.15).
        test_ratio: test 비율 (예: 0.15).
        train_ratio = 1 - val_ratio - test_ratio (예: 0.70).

    Returns:
        (train, val, test) 리스트. 각 stratum 내에서 셔플 후 비율대로 분할.
    """
    import random

    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1.0")

    rng = random.Random(seed)

    by_stratum: dict[tuple, list[dict]] = {}
    for rec in records:
        key = tuple(rec.get(k, "_unknown") for k in stratify_keys)
        by_stratum.setdefault(key, []).append(rec)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for key, group in by_stratum.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
        n_test = max(1, int(n * test_ratio)) if n >= 3 else 0
        val.extend(shuffled[:n_val])
        test.extend(shuffled[n_val:n_val + n_test])
        train.extend(shuffled[n_val + n_test:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    logger.info(
        f"  [3-way split] train={len(train)} val={len(val)} test={len(test)} "
        f"(stratified by {stratify_keys}, {len(by_stratum)} strata)"
    )
    return train, val, test


def _infer_embed_dim(embeddings: dict, default: int) -> int:
    if not embeddings:
        return default
    sample = next(iter(embeddings.values()))
    try:
        return int(sample.shape[-1])
    except Exception:
        return default


def _find_latest_checkpoint(moe_dir: Path) -> Optional[Path]:
    """
    체크포인트 우선순위:
        1. moe_best.pt (validation 최저 loss)
        2. best.pt (legacy)
        3. moe_last.pt (마지막 epoch)
        4. 가장 최근 .pt (mtime)
    """
    for name in ("moe_best.pt", "best.pt", "moe_last.pt"):
        path = moe_dir / name
        if path.exists():
            return path
    pts = sorted(moe_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


def _moe_predict_all(model, records, embeddings, instances_by_id) -> list[dict]:
    """
    records 전체에 대해 MoE 추론 → val_predictions 리스트.

    embeddings/instances_by_id의 키는 unique_id (composite). record에 unique_id가
    없으면 (legacy) example_id+category로 즉석 생성.
    """
    import torch

    from src.models.moe_aggregator import signals_dict_to_tensor

    device = next(model.parameters()).device
    model.eval()
    out: list[dict] = []
    with torch.inference_mode():
        for rec in records:
            ukey = rec.get("unique_id")
            if not ukey:
                ex_id = rec.get("example_id")
                cat = rec.get("category", "_unknown")
                if ex_id is None:
                    continue
                ukey = _make_unique_id(cat, ex_id)
            if ukey not in embeddings:
                continue
            sig = signals_dict_to_tensor(rec.get("signals", {})).unsqueeze(0).to(device)
            emb = embeddings[ukey].to(torch.float32).unsqueeze(0).to(device)
            res = model(sig, emb)
            p = float(res.p.item())
            primary = int(rec.get("primary_answer", -1))
            inst = instances_by_id.get(ukey, {})
            out.append({"primary_answer": primary, "p_score": p, "item": inst})
    return out


# =============================================================
# Stage dispatcher
# =============================================================
STAGE_FNS = {
    "sampling": run_sampling,
    "inference": run_inference,
    "signal_extraction": run_signal_extraction,
    "moe_training": run_moe_training,
    "evaluation": run_evaluation,
    "ablation": run_ablation,
}


def normalize_stages(stage_args: list[str], all_flag: bool) -> list[str]:
    """argparse에서 받은 stage 리스트를 정규화."""
    if all_flag:
        return list(STAGES)
    if not stage_args:
        return []
    out: list[str] = []
    for s in stage_args:
        s = STAGE_ALIASES.get(s, s)
        if s not in STAGE_FNS:
            raise ValueError(f"알 수 없는 stage: {s}. 가능: {list(STAGE_FNS)}")
        out.append(s)
    # 중복 제거 + STAGES 순서 정렬
    return [s for s in STAGES if s in set(out)]


# =============================================================
# Main
# =============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAE-Guided Multi-Signal Debiasing 통합 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--version", type=str, default="v1", choices=("v1", "v2", "smoke", "mini"),
        help="실험 버전. v1=7×300 (기본), v2=9×1000, smoke=9×5 (e2e 검증)",
    )
    parser.add_argument("--all", action="store_true", help="전체 파이프라인 실행")
    parser.add_argument(
        "--stage", "--stages",
        dest="stage",
        type=str, nargs="+", default=None,
        help=(
            "실행할 stage 이름(들). 가능: "
            f"{', '.join(STAGES)}. 별칭: 1~5, signals, train, eval"
        ),
    )
    parser.add_argument(
        "--cross-llm",
        type=str, choices=("gemma", "qwen"), default=None,
        help="Cross-LLM 모드 (--model의 alias). 지정 시 --model을 덮어씀.",
    )
    parser.add_argument(
        "--model", type=str, default="main",
        choices=("main", "gemma", "qwen", "mistral"),
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", default=None,
        help="실행할 BBQ 카테고리 (생략 시 config 전체)",
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="작은 데이터/짧은 학습으로 빠른 smoke test",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="결과 파일이 이미 있으면 해당 카테고리 skip",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="에러 발생 시 즉시 중단 (기본은 graceful continue)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # cross-llm은 model의 alias. stage 미지정 시 evaluation 기본 (학습은
    # main에서 한 모델을 transfer하는 시나리오를 가정).
    if args.cross_llm:
        args.model = args.cross_llm
        if not args.stage and not args.all:
            args.stage = ["evaluation"]

    log_path = setup_logging(args.log_dir)
    logger.info(f"로그: {log_path}")
    logger.info(f"args: {vars(args)}")

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"config 로드 실패: {e}")
        return 2

    # version 기반 데이터/결과 경로 자동 설정
    if args.version in ("v2", "smoke", "mini"):
        from src.utils.data_loader import DEFAULT_CATEGORIES_V2
        if args.version == "v2":
            config["data"]["sampled_dir"] = "data/sampled_v2"
            config["data"]["samples_per_category"] = 1000
            config["output"]["results_dir"] = "results/v2"
        elif args.version == "smoke":
            config["data"]["sampled_dir"] = "data/sampled_smoke"
            config["data"]["samples_per_category"] = 5
            config["output"]["results_dir"] = "results/smoke_e2e"
        else:  # mini
            config["data"]["sampled_dir"] = "data/sampled_mini"
            config["data"]["samples_per_category"] = 100
            config["output"]["results_dir"] = "results/v2_mini"
        config["data"]["categories"] = list(DEFAULT_CATEGORIES_V2)

        # Cross-LLM: 모델이 main이 아니면 results를 cross_llm/{model} 하위로 분리
        if args.model != "main":
            base = config["output"]["results_dir"]
            config["output"]["results_dir"] = f"{base}/cross_llm/{args.model}"

        logger.info(
            f"[{args.version}/{args.model}] sampled_dir={config['data']['sampled_dir']}, "
            f"results_dir={config['output']['results_dir']}, "
            f"{len(DEFAULT_CATEGORIES_V2)} categories × {config['data']['samples_per_category']}"
        )

    if args.quick_test:
        config = apply_quick_test_overrides(config)

    try:
        stages = normalize_stages(args.stage or [], args.all)
    except ValueError as e:
        logger.error(str(e))
        return 2

    if not stages:
        logger.error("실행할 stage 없음. --all 또는 --stage <name> 사용.")
        return 2

    logger.info(f"실행 stage: {stages}")
    logger.info(f"모델: {args.model}")

    # 각 stage 실행
    overall: dict[str, dict] = {}
    failed: list[str] = []
    for st in stages:
        fn = STAGE_FNS[st]
        t0 = time.time()
        try:
            res = fn(config, args)
            overall[st] = res or {}
            logger.info(f"  [{st}] 완료 ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.error(f"  [{st}] 실패: {e}")
            logger.debug(traceback.format_exc())
            overall[st] = {"error": str(e)}
            failed.append(st)
            if args.strict:
                logger.error("strict 모드 — 중단")
                break

    # 요약 저장
    summary_dir = Path(config.get("output", {}).get("results_dir", "results"))
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / f"pipeline_summary_{args.model}.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    if failed:
        logger.warning(f"일부 stage 실패: {failed}")
        return 1
    logger.info("[OK] 파이프라인 완료")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
