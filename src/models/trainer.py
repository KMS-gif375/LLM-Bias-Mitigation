"""
MoE Aggregator 학습 코드.

신호 + question embedding -> 정답 확신도(p)를 예측하도록 supervised 학습합니다.

Label 정의:
    - y = 1 (정답): primary_answer == BBQ label
    - y = 0 (오답): primary_answer != BBQ label

Loss: Binary Cross Entropy.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.models.moe_aggregator import MoEAggregator, signals_dict_to_tensor


class SignalsDataset(Dataset):
    """
    신호 추출 결과를 PyTorch Dataset으로 wrapping합니다.

    각 sample은 (signals_tensor, embedding_tensor, label_tensor) 튜플.
    """

    def __init__(
        self,
        signal_records: list[dict],
        embeddings: dict[str, torch.Tensor],
    ) -> None:
        """
        Args:
            signal_records: extract_signals_for_item 결과 리스트.
            embeddings: {example_id: embedding_tensor} 딕셔너리.
        """
        self.records = []
        for rec in signal_records:
            ex_id = rec["example_id"]
            if ex_id not in embeddings:
                continue
            self.records.append({
                "signals": signals_dict_to_tensor(rec["signals"]),
                "embedding": embeddings[ex_id],
                "label": float(rec["primary_answer"] == rec["label"]),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        return (
            rec["signals"],
            rec["embedding"],
            torch.tensor(rec["label"], dtype=torch.float32),
        )


def train_moe(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: MoEAggregator,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    save_path: Optional[Path] = None,
) -> dict:
    """
    MoE 모델을 학습합니다.

    Args:
        train_loader: 학습 DataLoader.
        val_loader: 검증 DataLoader.
        model: MoEAggregator 인스턴스.
        epochs: 학습 epoch 수.
        lr: learning rate.
        weight_decay: L2 정규화 강도.
        device: "cpu", "cuda", "mps".
        save_path: best model 저장 경로 (None이면 저장 안 함).

    Returns:
        학습 history (loss, val_loss, val_acc).
    """
    device_t = torch.device(device)
    model = model.to(device_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ===== Train =====
        model.train()
        train_losses = []
        for signals, embed, label in train_loader:
            signals = signals.to(device_t)
            embed = embed.to(device_t)
            label = label.to(device_t)

            p = model(signals, embed)
            loss = criterion(p, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ===== Validate =====
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, embed, label in val_loader:
                signals = signals.to(device_t)
                embed = embed.to(device_t)
                label = label.to(device_t)

                p = model(signals, embed)
                loss = criterion(p, label)
                val_losses.append(loss.item())

                pred = (p > 0.5).float()
                correct += (pred == label).sum().item()
                total += label.size(0)

        avg_val_loss = np.mean(val_losses)
        val_acc = correct / total if total > 0 else 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch+1:>3}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        # Save best model
        if save_path and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    return history


def predict_p(
    model: MoEAggregator,
    signals: torch.Tensor,
    embedding: torch.Tensor,
    device: str = "cpu",
) -> float:
    """
    단일 instance에 대해 p를 예측합니다 (추론용).

    Args:
        model: 학습된 MoE 모델.
        signals: (signal_dim,) 신호 벡터.
        embedding: (embed_dim,) question embedding.
        device: device.

    Returns:
        p ∈ [0, 1].
    """
    model.eval()
    device_t = torch.device(device)
    with torch.no_grad():
        s = signals.unsqueeze(0).to(device_t)
        e = embedding.unsqueeze(0).to(device_t)
        p = model(s, e)
    return p.item()
