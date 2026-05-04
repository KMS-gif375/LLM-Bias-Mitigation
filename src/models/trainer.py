"""
MoE Aggregator 학습 코드.

기능:
    - SignalsDataset: 신호 + embedding + label을 (B, ...) 텐서로 변환
    - train_moe(): epoch loop, BCE + bias penalty + load balance, AdamW
    - validation 매 N epoch마다 (BBQ accuracy_amb/dis 계산)
    - best checkpoint 저장
    - wandb 옵션 로깅
    - 진행 상황 tqdm 출력
    - MPS / CUDA 자동 device

Label 정의:
    y = 1 (모델 답 == BBQ label) → "정답"
    y = 0 (모델 답 != BBQ label) → "오답"

is_ambig, is_stereotype은 bias penalty 계산용으로 미리 추출되어 데이터에 포함되어야 합니다.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.moe_aggregator import (
    MoEAggregator,
    MoEOutput,
    signals_dict_to_tensor,
    total_loss,
)

logger = logging.getLogger(__name__)


# =============================================================
# 1. Dataset
# =============================================================
class SignalsDataset(Dataset):
    """
    신호 추출 결과 + embedding을 PyTorch Dataset으로 wrapping합니다.

    각 sample = (signals, q_embed, label, is_ambig, is_stereotype).

    Args:
        signal_records: extract_signals_for_item 결과 리스트 (또는 같은 schema의 dict).
            필수 키: example_id, primary_answer, label, signals, context_condition.
            선택 키: is_stereotype (없으면 0으로 채움).
        embeddings: {example_id: torch.Tensor (embed_dim,)} 딕셔너리.
        require_all: True면 embedding 없는 record는 에러, False면 skip.
    """

    def __init__(
        self,
        signal_records: list[dict],
        embeddings: dict[str, torch.Tensor],
        require_all: bool = False,
    ) -> None:
        self.records: list[dict] = []
        skipped = 0

        for rec in signal_records:
            ex_id = rec["example_id"]
            if ex_id not in embeddings:
                if require_all:
                    raise KeyError(f"embedding 누락: {ex_id}")
                skipped += 1
                continue

            primary = rec["primary_answer"]
            true_label = rec["label"]
            label = float(primary == true_label) if primary in (0, 1, 2) else 0.0

            cond = rec.get("context_condition", "")
            is_ambig = float(cond == "ambig")
            is_stereotype = float(rec.get("is_stereotype", 0.0))

            self.records.append({
                "example_id": ex_id,                          # 보존 (LOCO 등에서 instance 매칭용)
                "signals": signals_dict_to_tensor(rec["signals"]),
                "embedding": embeddings[ex_id].to(torch.float32),
                "label": torch.tensor(label, dtype=torch.float32),
                "is_ambig": torch.tensor(is_ambig, dtype=torch.float32),
                "is_stereotype": torch.tensor(is_stereotype, dtype=torch.float32),
            })

        if skipped:
            logger.warning(f"  [SignalsDataset] {skipped}개 record skip (embedding 누락)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.records[idx]


# =============================================================
# 2. Training Config
# =============================================================
@dataclass
class TrainConfig:
    """학습 하이퍼파라미터."""

    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_every: int = 5                       # 매 N epoch마다 validation
    lambda_bias: float = 0.5                 # bias penalty 가중치
    lambda_lb: float = 0.1                   # load balance 가중치 (cluster collapse 방지)
    grad_clip: Optional[float] = 1.0         # gradient clipping
    early_stop_patience: int = 0             # 0이면 비활성화
    device: str = "auto"                     # "auto", "mps", "cuda", "cpu"
    seed: int = 42
    save_dir: Optional[str] = None           # checkpoint 저장 경로 (None이면 저장 안 함)
    wandb_enabled: bool = False
    wandb_project: str = "bbq-moe"
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = 10              # train loss 로깅 step interval


def select_device(prefer: str) -> torch.device:
    """device 자동 선택 (CUDA > MPS > CPU)."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================
# 3. Training Loop
# =============================================================
@dataclass
class EpochMetrics:
    """한 epoch의 train/val 결과."""

    epoch: int
    train_loss: float
    train_bce: float
    train_bias: float
    train_lb: float
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None          # binary classification accuracy on (correct vs incorrect)
    val_acc_amb: Optional[float] = None      # BBQ-style: ambig 맥락 정확도 (예측 + override 후)
    val_acc_dis: Optional[float] = None      # disambig 맥락 정확도
    expert_usage: Optional[list[float]] = None  # 4개 expert별 평균 사용률


def train_moe(
    train_dataset: SignalsDataset,
    val_dataset: Optional[SignalsDataset],
    model: MoEAggregator,
    config: TrainConfig = TrainConfig(),
) -> dict[str, Any]:
    """
    MoE 학습을 수행합니다.

    Args:
        train_dataset: SignalsDataset.
        val_dataset: 검증용 (None이면 train만).
        model: MoEAggregator 인스턴스.
        config: TrainConfig.

    Returns:
        {
            "history": list[EpochMetrics dict],
            "best_val_loss": float,
            "best_epoch": int,
            "checkpoint_path": str | None,
        }
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = select_device(config.device)
    logger.info(f"[Trainer] device={device}, batch_size={config.batch_size}, "
                f"epochs={config.epochs}, lr={config.lr}")
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        drop_last=False, num_workers=0,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None and len(val_dataset) > 0 else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # wandb 옵션
    wandb_run = _init_wandb(config) if config.wandb_enabled else None

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_ckpt_path: Optional[Path] = None
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        # ====== Train ======
        model.train()
        train_metrics = _run_epoch(
            model, train_loader, device, optimizer, config, training=True,
        )

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_bce": train_metrics["bce"],
            "train_bias": train_metrics["bias"],
            "train_lb": train_metrics["lb"],
        }

        # ====== Validate ======
        do_val = (val_loader is not None) and (
            epoch % config.val_every == 0 or epoch == config.epochs
        )
        if do_val:
            model.eval()
            val_metrics = _run_epoch(
                model, val_loader, device, optimizer=None, config=config, training=False,
            )
            epoch_log.update({
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "expert_usage": val_metrics["expert_usage"],
            })

            # Best checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                best_ckpt_path = _save_checkpoint(model, config, tag="best")
            else:
                patience_counter += 1

        # 로깅
        _log_epoch(epoch_log, wandb_run)
        history.append(epoch_log)

        # Early stopping
        if (
            config.early_stop_patience > 0
            and patience_counter >= config.early_stop_patience
        ):
            logger.info(f"[EarlyStop] epoch {epoch} ({patience_counter} 회 개선 없음)")
            break

    # 마지막 체크포인트도 저장
    last_ckpt = _save_checkpoint(model, config, tag="last") if config.save_dir else None

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "history": history,
        "best_val_loss": best_val_loss if best_epoch > 0 else None,
        "best_epoch": best_epoch if best_epoch > 0 else None,
        "checkpoint_path": str(best_ckpt_path) if best_ckpt_path else None,
        "last_checkpoint_path": str(last_ckpt) if last_ckpt else None,
    }


def _run_epoch(
    model: MoEAggregator,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    config: TrainConfig,
    training: bool,
) -> dict[str, Any]:
    """한 epoch의 forward/backward + metric 누적."""
    losses = {"loss": [], "bce": [], "bias": [], "lb": []}
    correct = 0
    total = 0
    expert_w_accumulator = torch.zeros(model.num_experts, dtype=torch.float32, device="cpu")

    desc = "train" if training else "val"
    pbar = tqdm(loader, desc=desc, leave=False)

    for step, batch in enumerate(pbar):
        signals = batch["signals"].to(device)
        q_embed = batch["embedding"].to(device)
        label = batch["label"].to(device)
        is_ambig = batch["is_ambig"].to(device)
        is_stereo = batch["is_stereotype"].to(device)

        if training:
            optimizer.zero_grad()
            output = model(signals, q_embed)
        else:
            with torch.inference_mode():
                output = model(signals, q_embed)

        loss_dict = total_loss(
            output=output,
            label=label,
            is_ambig=is_ambig,
            is_stereotype=is_stereo,
            lambda_bias=config.lambda_bias,
            lambda_lb=config.lambda_lb,
        )

        if training:
            loss_dict["total"].backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # accumulate
        losses["loss"].append(loss_dict["total"].item())
        losses["bce"].append(loss_dict["bce"].item())
        losses["bias"].append(loss_dict["bias"].item())
        losses["lb"].append(loss_dict["lb"].item())

        # binary acc on label vs (p > 0.5)
        pred = (output.p > 0.5).float()
        correct += (pred == label).sum().item()
        total += label.numel()
        # MPS는 float64 미지원이므로 .float() (float32) 후 cpu로 누적
        expert_w_accumulator += output.gate_w.detach().sum(dim=0).float().cpu()

        if training and (step + 1) % config.log_every_n_steps == 0:
            pbar.set_postfix({
                "loss": f"{np.mean(losses['loss'][-config.log_every_n_steps:]):.4f}",
            })

    expert_usage = (expert_w_accumulator / max(total, 1)).tolist()
    return {
        "loss": float(np.mean(losses["loss"])) if losses["loss"] else 0.0,
        "bce": float(np.mean(losses["bce"])) if losses["bce"] else 0.0,
        "bias": float(np.mean(losses["bias"])) if losses["bias"] else 0.0,
        "lb": float(np.mean(losses["lb"])) if losses["lb"] else 0.0,
        "acc": correct / total if total > 0 else 0.0,
        "expert_usage": expert_usage,
    }


def _save_checkpoint(
    model: MoEAggregator,
    config: TrainConfig,
    tag: str,
) -> Optional[Path]:
    """모델 state dict + config를 저장합니다."""
    if not config.save_dir:
        return None

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"moe_{tag}.pt"

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "signal_dim": model.signal_dim,
            "embed_dim": model.embed_dim,
            "num_experts": model.num_experts,
            "gating_hidden": model.gating_hidden,
            "expert_hidden": model.expert_hidden,
            "dropout": model.dropout,
        },
        "train_config": {
            "epochs": config.epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "lambda_bias": config.lambda_bias,
            "lambda_lb": config.lambda_lb,
            "seed": config.seed,
        },
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    device: str = "auto",
) -> MoEAggregator:
    """
    체크포인트에서 모델을 복원합니다.

    Args:
        path: checkpoint 파일 경로.
        device: device 문자열.

    Returns:
        weights가 로드된 MoEAggregator.
    """
    device_t = select_device(device)
    payload = torch.load(path, map_location=device_t, weights_only=True)

    cfg = payload["model_config"]
    model = MoEAggregator(
        signal_dim=cfg["signal_dim"],
        embed_dim=cfg["embed_dim"],
        num_experts=cfg["num_experts"],
        gating_hidden=cfg.get("gating_hidden", 128),
        expert_hidden=cfg.get("expert_hidden", 64),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device_t)
    model.eval()
    return model


def _log_epoch(epoch_log: dict, wandb_run) -> None:
    """epoch 결과를 콘솔 + wandb에 출력."""
    parts = [
        f"epoch {epoch_log['epoch']:>3}",
        f"train_loss={epoch_log['train_loss']:.4f}",
        f"bce={epoch_log['train_bce']:.4f}",
        f"bias={epoch_log['train_bias']:.4f}",
        f"lb={epoch_log['train_lb']:.4f}",
    ]
    if "val_loss" in epoch_log:
        parts += [
            f"val_loss={epoch_log['val_loss']:.4f}",
            f"val_acc={epoch_log['val_acc']:.4f}",
        ]
    if epoch_log.get("expert_usage"):
        usage_str = "/".join(f"{u:.2f}" for u in epoch_log["expert_usage"])
        parts.append(f"experts=[{usage_str}]")

    logger.info("  " + " | ".join(parts))

    if wandb_run is not None:
        wandb_run.log(epoch_log)


def _init_wandb(config: TrainConfig):
    """wandb 초기화 (옵션)."""
    try:
        import wandb
    except ImportError:
        logger.warning("[wandb] 미설치, 로깅 비활성화")
        return None

    return wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "lambda_bias": config.lambda_bias,
            "lambda_lb": config.lambda_lb,
            "seed": config.seed,
        },
    )


def save_history(history: list[dict], path: str | Path) -> None:
    """학습 history를 JSON으로 저장."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
