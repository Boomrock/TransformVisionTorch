from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from tqdm import tqdm

# Добавляем корень проекта в sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import get_dataloaders, get_mixup_fn, NUM_CLASSES
from src.model import build_model
from src.optimizer import build_optimizer, build_scheduler
from src.utils import (
    AverageMeter,
    get_device,
    save_checkpoint,
    save_metrics,
    set_seed,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_fn=None,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    Один проход по тренировочному набору
    """
    model.train()
    loss_meter = AverageMeter("train_loss")
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"[Train E{epoch}]", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # MixUp
        if mixup_fn is not None:
            images, targets_mixed = mixup_fn(images, targets)
        else:
            targets_mixed = None

        # Forward
        logits = model(images)

        if targets_mixed is not None:
            loss = criterion(logits, targets_mixed)
            # Accuracy приблизительно считается как argmax mixed labels
            _, predicted = logits.max(1)
            _, true_class = targets_mixed.max(1)
            correct += predicted.eq(true_class).sum().item()
        else:
            loss = criterion(logits, targets)
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()

        total += images.size(0)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    acc = 100.0 * correct / total if total > 0 else 0.0
    return {"loss": loss_meter.avg, "acc": acc}

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "val",
) -> Dict[str, float]:
    """
    Оценка на val / test без MixUp, стандартная кросс-энтропия
    """
    model.eval()
    loss_meter = AverageMeter(f"{phase}_loss")
    correct = 0
    total = 0

    ce_criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc=f"[{phase.capitalize()}]", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = ce_criterion(logits, targets)

        _, predicted = logits.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        loss_meter.update(loss.item(), targets.size(0))

    acc = 100.0 * correct / total if total > 0 else 0.0
    return {"loss": loss_meter.avg, "acc": acc}


def train(cfg: Dict[str, Any]) -> None:
    """
    Полный цикл обучения, настройка, train → val → save по эпохам
    """
    # --- Seed & device ---
    set_seed(cfg.get("seed", 42))
    device = get_device()

    # --- Data ---
    data_cfg = cfg.get("data", {})
    loaders = get_dataloaders(
        data_root=data_cfg.get("root", "./data"),
        splits_dir=data_cfg.get("splits_dir", "./data/splits"),
        image_size=data_cfg.get("image_size", 224),
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        seed=cfg.get("seed", 42),
    )

    # --- MixUp ---
    mixup_cfg = cfg.get("mixup", {})
    mixup_fn = get_mixup_fn(
        mixup_alpha=mixup_cfg.get("alpha", 0.8),
        cutmix_alpha=mixup_cfg.get("cutmix_alpha", 0.0),
        prob=mixup_cfg.get("prob", 1.0),
        num_classes=data_cfg.get("num_classes", NUM_CLASSES),
    )

    # SoftTarget если MixUp, иначе обычный CE
    if mixup_fn is not None:
        train_criterion = SoftTargetCrossEntropy()
        print("[train] Using SoftTargetCrossEntropy (MixUp enabled)")
    else:
        train_criterion = nn.CrossEntropyLoss()
        print("[train] Using CrossEntropyLoss (MixUp disabled)")

    # --- Model ---
    model_cfg = cfg.get("model", {})
    model = build_model(
        model_name=model_cfg.get("name", "vit_base_patch16_224"),
        pretrained=model_cfg.get("pretrained", True),
        num_classes=data_cfg.get("num_classes", NUM_CLASSES),
    )
    model = model.to(device)

    # --- Optimizer & Scheduler ---
    opt_cfg = cfg.get("optimizer", {})
    optimizer = build_optimizer(model, opt_cfg)

    sched_cfg = cfg.get("scheduler", {})
    scheduler = build_scheduler(optimizer, sched_cfg)

    # --- Training ---
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 30)
    exp_dir = Path(train_cfg.get("experiment_dir", "./experiments/default"))
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_dir.name}")
    print(f"  Epochs: {epochs}")
    print(f"  Model: {model_cfg.get('name', 'vit_base_patch16_224')}")
    print(f"  Pretrained: {model_cfg.get('pretrained', True)}")
    print(f"  Optimizer: {opt_cfg.get('name', 'adam')}")
    print(f"  MixUp alpha: {mixup_cfg.get('alpha', 0.8)}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, loaders["train"], train_criterion, optimizer,
            device, mixup_fn=mixup_fn, epoch=epoch,
        )

        # Validate
        val_metrics = evaluate(model, loaders["val"], train_criterion, device, "val")

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        # Log
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss {train_metrics['loss']:.4f} Acc {train_metrics['acc']:.2f}% | "
            f"Val Loss {val_metrics['loss']:.4f} Acc {val_metrics['acc']:.2f}% | "
            f"{elapsed:.1f}s"
        )

        # History
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        # Save best
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            save_checkpoint(
                ckpt_dir / "best_model.pt",
                model, optimizer, epoch,
                metrics={"val_acc": best_val_acc, **val_metrics},
            )

        # Save last every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            save_checkpoint(
                ckpt_dir / "last_model.pt",
                model, optimizer, epoch,
                metrics=val_metrics,
            )

    # --- Final metrics ---
    history["best_val_acc"] = best_val_acc
    save_metrics(exp_dir / "metrics.json", history)
    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    print(f"Artifacts saved to {exp_dir}")

def load_config(path: str) -> Dict[str, Any]:
    """Загружает YAML конфиг"""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train ViT on Stanford Dogs")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (e.g. configs/pretrained_adam.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
