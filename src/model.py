"""
ViT через timm (заглушка)
пока содержит минимальный метод build_model() для работы пайплайна обучения.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_model(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 120,
) -> nn.Module:
    """
    Вход:
        model_name: Имя модели в timm registry
                    (например ``"vit_base_patch16_224"``, ``"vit_small_patch16_224"``).
        pretrained: Использовать ImageNet pretrained веса.
        num_classes: Число выходных классов (120 для Stanford Dogs).

    Выход:
        nn.Module — готовая к обучению модель.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    # Подсчёт параметров
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {model_name} | pretrained={pretrained}")
    print(f"[model] Total params: {total:,} | Trainable: {trainable:,}")

    return model
