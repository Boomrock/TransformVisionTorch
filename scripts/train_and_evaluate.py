#!/usr/bin/env python3
import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SportsEmbeddingDataset
from src.model import TransformerClassifier
from src.utils import compute_metrics, plot_confusion_matrix, plot_training_history
from src.validation import collect_artifact_report, format_artifact_report


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_labels(train_labels, val_labels, test_labels):
    unique_labels = sorted(np.unique(np.concatenate([train_labels, val_labels, test_labels])))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_orig = {idx: label for label, idx in label_to_idx.items()}
    return (
        np.array([label_to_idx[label] for label in train_labels], dtype=np.int64),
        np.array([label_to_idx[label] for label in val_labels], dtype=np.int64),
        np.array([label_to_idx[label] for label in test_labels], dtype=np.int64),
        idx_to_orig,
    )


def build_class_weights(labels, num_classes, device, weighting):
    if weighting != "sqrt_inverse_frequency":
        return None, np.ones(num_classes, dtype=np.float32)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.sqrt(counts.max() / counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device), weights


def main():
    with open(PROJECT_ROOT / "configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    seed = cfg["data"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = choose_device()
    print(f"Device: {device}", flush=True)

    artifact_report = collect_artifact_report(data_dir)
    print(format_artifact_report(artifact_report), flush=True)
    if not artifact_report.embeddings_match_parquets:
        raise RuntimeError("Dataset artifacts are inconsistent; run embedding extraction first.")

    data = np.load(data_dir / "embeddings.npz")
    train_emb, train_lens = data["train_emb"], data["train_lens"]
    val_emb, val_lens = data["val_emb"], data["val_lens"]
    test_emb, test_lens = data["test_emb"], data["test_lens"]
    train_labels, val_labels, test_labels, idx_to_orig = prepare_labels(
        data["train_labels"], data["val_labels"], data["test_labels"]
    )

    with open(data_dir / "id2label.json", encoding="utf-8") as f:
        id2label_raw = json.load(f)
    class_names = [
        id2label_raw.get(str(idx_to_orig[idx]), f"Class_{idx_to_orig[idx]}")
        for idx in range(len(idx_to_orig))
    ]

    print(f"Train: {train_emb.shape}, Val: {val_emb.shape}, Test: {test_emb.shape}", flush=True)
    print(f"Classes: {class_names}", flush=True)

    max_seq_len = cfg["model"]["max_seq_len"]
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        SportsEmbeddingDataset(train_emb, train_labels, train_lens, max_seq_len=max_seq_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SportsEmbeddingDataset(val_emb, val_labels, val_lens, max_seq_len=max_seq_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        SportsEmbeddingDataset(test_emb, test_labels, test_lens, max_seq_len=max_seq_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TransformerClassifier(
        num_classes=len(class_names),
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        max_seq_len=max_seq_len,
    ).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params:,}", flush=True)

    class_weight_tensor, class_weights = build_class_weights(
        train_labels,
        len(class_names),
        device,
        cfg["training"].get("class_weighting"),
    )
    counts = np.bincount(train_labels, minlength=len(class_names)).astype(int)
    print("Class counts:", dict(zip(class_names, counts.tolist())), flush=True)
    print("Class weights:", dict(zip(class_names, [round(float(w), 4) for w in class_weights])), flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )

    def train_epoch():
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            emb = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(emb, src_key_padding_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        for batch in loader:
            emb = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(emb, src_key_padding_mask=mask)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        probs = np.array(all_probs)
        return (
            total_loss / total,
            correct / total,
            f1_score(labels, preds, average="macro", zero_division=0),
            preds,
            labels,
            probs,
        )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_macro = -1.0
    best_val_acc = -1.0
    best_epoch = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_acc = train_epoch()
        val_loss, val_acc, val_macro, _, _, _ = evaluate(val_loader)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"MacroF1: {val_macro:.4f} | LR: {lr:.6f}",
            flush=True,
        )
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> New best model saved (val_macro_f1={val_macro:.4f})", flush=True)

    fig = plot_training_history(train_losses, val_losses, train_accs, val_accs)
    fig.savefig(output_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_loss, _, _, test_preds, test_true, test_probs = evaluate(test_loader)
    metrics = compute_metrics(test_true, test_preds, test_probs, k=min(3, len(class_names)))

    print("\nTEST RESULTS", flush=True)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}", flush=True)

    report_text = classification_report(
        test_true,
        test_preds,
        target_names=class_names,
        zero_division=0,
    )
    print("\n" + report_text, flush=True)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    fig = plot_confusion_matrix(test_true, test_preds, class_names, figsize=(12, 10))
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(test_true, test_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(class_names, per_class_acc, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-class Accuracy")
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{acc:.3f}", va="center")
    plt.tight_layout()
    fig.savefig(output_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    errors = [(class_names[t], class_names[p]) for t, p in zip(test_true, test_preds) if t != p]
    error_counts = Counter(errors).most_common(10)

    results = {
        "config": cfg,
        "training_variant": "transformer_weighted_sqrt_inverse_frequency_best_macro_f1",
        "device": str(device),
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_counts": {class_names[i]: int(counts[i]) for i in range(len(class_names))},
        "class_weights": {class_names[i]: float(class_weights[i]) for i in range(len(class_names))},
        "test_metrics": {key: float(value) for key, value in metrics.items()},
        "test_loss": float(test_loss),
        "best_val_macro_f1": float(best_val_macro),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "epochs_trained": len(train_losses),
        "total_params": int(total_params),
        "train_shape": list(train_emb.shape),
        "val_shape": list(val_emb.shape),
        "test_shape": list(test_emb.shape),
        "per_class_accuracy": {class_names[i]: float(per_class_acc[i]) for i in range(len(class_names))},
        "top_errors": [
            {"true": true_cls, "predicted": pred_cls, "count": int(count)}
            for (true_cls, pred_cls), count in error_counts
        ],
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
