from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Фиксирует seed для random, numpy, torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Для детерминированности (может замедлить обучение)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[utils] Seed set to {seed}")


def get_device() -> torch.device:
    """Возвращает доступное устройство: cuda, mps или cpu"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[utils] Using device: {device}")
    return device


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    """Сохраняет state_dict модели, оптимизатора и метрики"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        str(path),
    )
    print(f"[utils] Checkpoint saved → {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Загружает чекпоинт и возвращает dict с epoch и метриками"""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[utils] Checkpoint loaded ← {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


def save_metrics(path: str | Path, metrics: Dict[str, Any]) -> None:
    """Выгрузка метрик в JSON и перезаписывает файл"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[utils] Metrics saved → {path}")


def load_metrics(path: str | Path) -> Dict[str, Any]:
    """Загружает метрики из JSON"""
    with open(path, "r") as f:
        return json.load(f)


class AverageMeter:
    """
    Считает скользящее среднее и сумму

    Использовать как

        meter = AverageMeter("loss")
        for batch in loader:
            ...
            meter.update(loss_val, n=batch_size)
        print(meter)
    """

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
