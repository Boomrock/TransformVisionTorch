"""
загрузка датасета, разбивка 70/15/15, MixUp
  * find_images_dir()        находит папку с изображениями
  * create_splits()          детерминированная разбивка 70/15/15 с сохранением индексов в JSON
  * get_transforms()         аугментации train/val/test
  * get_dataloaders()        DataLoader-ы + MixUp (timm.data.Mixup)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATASET_URL = "https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset"
NUM_CLASSES = 120
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CANDIDATE_PATHS = [
    "set/images/Images",   # Kaggle ZIP structure
    "set/Images",
    "images/Images",
    "Images",
]


def find_images_dir(data_root: str | Path) -> Path:
    """
    Находит папку с изображениями Stanford Dogs внутри data_root

    Проверяет несколько вариантов расположения (Kaggle ZIP, ручная
    распаковка и т.д.).  Валидация такова, что папка должна содержать больше или 100
    подпапок с породами
    """
    data_root = Path(data_root)

    for candidate in _CANDIDATE_PATHS:
        path = data_root / candidate
        if path.is_dir():
            # Проверяем что внутри есть подпапки пород
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if len(subdirs) >= 100:
                print(f"[dataset] Found images at {path} ({len(subdirs)} breed folders)")
                return path

    raise FileNotFoundError(
        f"Stanford Dogs images not found in {data_root}.\n"
        f"Checked paths: {[str(data_root / c) for c in _CANDIDATE_PATHS]}\n\n"
        f"Please download the dataset from:\n"
        f"  {DATASET_URL}\n\n"
        f"Then unzip into {data_root}/set/ so the structure is:\n"
        f"  {data_root}/set/images/Images/<breed_folders>/\n"
    )


def create_splits(
    dataset: ImageFolder,
    splits_dir: str | Path,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Стратифицированная разбивка на train / val / test

    Для каждого класса сэмплы перемешиваются с фиксированным seed и
    делятся в пропорции ratios 
    Индексы сохраняются в splits_dir/splits.json для воспроизводимости
    """
    splits_dir = Path(splits_dir)
    splits_path = splits_dir / "splits.json"

    # Если сплиты уже созданы — загружаем
    if splits_path.exists():
        print(f"[dataset] Loading existing splits from {splits_path}")
        with open(splits_path, "r") as f:
            return json.load(f)

    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    train_idx, val_idx, test_idx = [], [], []

    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # test gets the rest

        train_idx.extend(cls_indices[:n_train].tolist())
        val_idx.extend(cls_indices[n_train : n_train + n_val].tolist())
        test_idx.extend(cls_indices[n_train + n_val :].tolist())

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    # сохраняем
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(splits, f)
    print(
        f"[dataset] Splits created — "
        f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )
    return splits


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Возвращает словарь трансформов для train / val / test

    Train включает случайный ресайз-кроп, горизонтальный флип, Color Jitter и нормализацию ImageNet.
    Val/Test содержит детерминированный ресайз, центр-кроп и нормализацию
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),  # ~256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return {"train": train_tf, "val": eval_tf, "test": eval_tf}

class TransformSubset(Subset):
    """Подмножество, которое подменяет transform у базового объекта ImageFolder."""

    def __init__(self, dataset: ImageFolder, indices: List[int], transform=None):
        super().__init__(dataset, indices)
        self.custom_transform = transform

    def __getitem__(self, idx):
        original_transform = self.dataset.transform
        if self.custom_transform is not None:
            self.dataset.transform = self.custom_transform
        item = super().__getitem__(idx)
        self.dataset.transform = original_transform
        return item


def get_mixup_fn(
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 0.0,
    prob: float = 1.0,
    num_classes: int = NUM_CLASSES,
):
    """
    Возвращает функцию MixUp из timm или None, если alpha == 0

    MixUp применяется к батчу как пара mixed_images, mixed_targets = mixup_fn(images, targets).
    Targets -> soft-labels, поэтому лосс должен быть SoftTargetCrossEntropy.
    """
    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return None

    from timm.data.mixup import Mixup

    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=prob,
        switch_prob=0.5 if cutmix_alpha > 0 else 0.0,
        mode="batch",
        num_classes=num_classes,
    )

def get_dataloaders(
    data_root: str | Path,
    splits_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Полный пайплайн обработки датасета locate images → ImageFolder → splits → DataLoader.

    # 1. Locate images, уже скачаны вручную
    images_dir = find_images_dir(data_root)

    # 2. Create base dataset (если без transform, подменим в Subset)
    full_dataset = ImageFolder(root=str(images_dir))
    print(f"[dataset] Total images: {len(full_dataset)}, classes: {len(full_dataset.classes)}")

    # 3. Splits
    splits = create_splits(full_dataset, splits_dir, seed=seed)

    # 4. Transforms
    tfms = get_transforms(image_size)

    # 5. Subsets with per-split transforms
    subsets = {
        split: TransformSubset(full_dataset, indices, transform=tfms[split])
        for split, indices in splits.items()
    }

    # 6. DataLoaders
    loaders = {}
    for split, subset in subsets.items():
        loaders[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders

# CLI для быстрой проверки

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stanford Dogs dataset check")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--splits-dir", type=str, default="./data/splits")
    args = parser.parse_args()

    loaders = get_dataloaders(args.data_root, args.splits_dir)

    for name, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"  {name}: images {images.shape}, labels {labels.shape}")
