from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import get_dataloaders, NUM_CLASSES
from src.model import build_model
from src.utils import get_device, load_checkpoint


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> Dict[str, Any]:
    model.eval()
    all_preds, all_labels = [], []

    for images, targets in tqdm(loader, desc="[Evaluate]", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    results = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }

    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  Precision (macro):  {precision_macro:.4f}")
    print(f"  Recall (macro):     {recall_macro:.4f}")
    print(f"  F1 (macro):         {f1_macro:.4f}")

    target_names = class_names or [str(i) for i in range(len(np.unique(all_labels)))]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True
    )
    results["classification_report"] = report

    cm = confusion_matrix(all_labels, all_preds)
    results["confusion_matrix"] = cm.tolist()

    return results


def save_confusion_matrix(cm: np.ndarray, path: Path, title: str = "Confusion Matrix"):
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=False, cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")


def run_evaluation(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()

    data_cfg = cfg.get("data", {})
    loaders = get_dataloaders(
        data_root=data_cfg.get("root", "./data"),
        splits_dir=data_cfg.get("splits_dir", "./data/splits"),
        image_size=data_cfg.get("image_size", 224),
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
    )

    model_cfg = cfg.get("model", {})
    was_pretrained = model_cfg.get("pretrained", True)
    model = build_model(
        model_name=model_cfg.get("name", "vit_base_patch16_224"),
        pretrained=False,
        num_classes=data_cfg.get("num_classes", NUM_CLASSES),
        use_timm=was_pretrained,
    )

    exp_dir = Path(cfg["training"]["experiment_dir"])
    ckpt_path = exp_dir / "checkpoints" / "best_model.pt"
    load_checkpoint(ckpt_path, model, device=device)
    model = model.to(device)

    test_loader = loaders["test"]
    base_dataset = test_loader.dataset
    while hasattr(base_dataset, "dataset"):
        base_dataset = base_dataset.dataset
    class_names = None
    if hasattr(base_dataset, "classes"):
        class_names = [c.split("-", 1)[-1].replace("_", " ") for c in base_dataset.classes]

    print(f"\nEvaluating: {exp_dir.name}")
    results = evaluate_model(model, test_loader, device, class_names)

    out_path = exp_dir / "test_results.json"
    with open(out_path, "w") as f:
        save_data = {k: v for k, v in results.items() if k != "confusion_matrix"}
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved -> {out_path}")

    cm = np.array(results["confusion_matrix"])
    save_confusion_matrix(cm, exp_dir / "confusion_matrix.png", title=exp_dir.name)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    run_evaluation(args.config)


if __name__ == "__main__":
    main()
