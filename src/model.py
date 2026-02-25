import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, dim=768):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"Image size ({image_size}) must be divisible by patch size ({patch_size})")
        
        self.n_patches = (image_size // patch_size) ** 2
        
        # Conv2d эффективно разбивает изображение на патчи и проецирует их
        self.projection = nn.Conv2d(
            in_channels, 
            dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)          # (batch_size, dim, n_patches_h, n_patches_w)
        x = x.flatten(2)                # (batch_size, dim, n_patches)
        x = x.transpose(1, 2)           # (batch_size, n_patches, dim)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention с пред-нормализацией (Pre-LN)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x

class ViT(nn.Module):
    def __init__(
        self, 
        image_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=1000, 
        dim=768, 
        depth=12, 
        heads=12, 
        mlp_dim=3072, 
        dropout=0.1
    ):
        super().__init__()
        
        # 1. Разбиение на патчи
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        n_patches = self.patch_embedding.n_patches
        
        # 2. Токен класса [CLS]
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 3. Позиционные эмбеддинги (включая место для CLS токена)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # 4. Трансформер энкодер
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        
        # 5. Классификационная голова
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Патчи
        x = self.patch_embedding(x)  # (batch_size, n_patches, dim)
        
        # Добавляем CLS токен к батчу
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, n_patches + 1, dim)
        
        # Добавляем позиционные эмбеддинги
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Энкодер
        x = self.transformer(x)
        
        # Берем выход только для CLS токена (индекс 0)
        x = self.norm(x[:, 0])
        
        # Классификация
        return self.head(x)