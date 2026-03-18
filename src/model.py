import torch
import torch.nn as nn
import math
import timm

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, dim=768):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"Image size ({image_size}) must be divisible by patch size ({patch_size})")
        
        self.n_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, 
            dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
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
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)

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
        
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        n_patches = self.patch_embedding.n_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.norm(x[:, 0])

        return self.head(x)


def build_model(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 120,
    use_timm: bool = True,
) -> nn.Module:
    if use_timm:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        tag = "pretrained" if pretrained else "timm architecture, random weights"
        print(f"[model] {model_name} ({tag}), head -> {num_classes} classes")
    else:
        model = ViT(
            image_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
        )
        print(f"[model] Built custom ViT from scratch, {num_classes} classes")
    return model