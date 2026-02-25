"""
Набор оптимизаторов Adam(W) и SPAM

Предоставляет единый интерфейс для создания оптимизатора из YAML-конфига.  Поддерживаемые оптимизаторы:

  * adam  — стандартный torch.optim.AdamW
  * spam  — SPAM (Spike-Aware Adam with Momentum Reset) из spam/spam.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

# Добавляем корень проекта в sys.path для импорта spam/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spam import SPAM  # noqa: E402

def build_optimizer(
    model: nn.Module,
    cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Вход:
        model: модель, параметры которой оптимизируем
        cfg: словарь конфигурации

    Выход:
        Инициализированный torch.optim.Optimizer
    """
    name = cfg.get("name", "adam").lower()
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    betas = tuple(cfg.get("betas", [0.9, 0.999]))

    # Фильтруем параметры, weight_decay не применяем к bias и LayerNorm
    decay_params = []
    no_decay_params = []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in pname or "norm" in pname.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if name == "adam":
        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=cfg.get("eps", 1e-8),
        )
    elif name == "spam":
        optimizer = SPAM(
            param_groups,
            lr=lr,
            betas=betas,
            eps=cfg.get("eps", 1e-6),
            weight_decay=0.0,  # weight_decay уже в param_groups
            correct_bias=True,
            warmup_steps=cfg.get("spam_warmup_steps", 150),
            threshold=cfg.get("spam_threshold", 5000),
            DeltaT=cfg.get("spam_delta_t", 500),
            grad_accu_steps=cfg.get("spam_grad_accu_steps", 20),
        )
    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. Supported: 'adam', 'spam'."
        )

    print(f"[optimizer] {name.upper()} | lr={lr} | wd={weight_decay} | betas={betas}")
    return optimizer

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
) -> Optional[_LRScheduler]:
    """
    Создаёт планировщик learning rate из конфига

    Ключи:

        name: "cosine" | "none"
        T_max: int          — число эпох для cosine decay, обычно равняется числу эпох
        eta_min: float      — минимальный LR, по умолчанию default 1e-6

    Выход:
        Scheduler или None, если name == "none".
    """
    name = cfg.get("name", "cosine").lower()

    if name == "none":
        return None

    if name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("T_max", 30),
            eta_min=cfg.get("eta_min", 1e-6),
        )
        print(f"[scheduler] CosineAnnealingLR | T_max={cfg.get('T_max', 30)}")
        return scheduler

    raise ValueError(f"Unknown scheduler '{name}'. Supported: 'cosine', 'none'.")
