"""Model components for LeJEPA pretraining (timm backbones)."""

import torch
import torch.nn as nn
import timm

from configs import is_vit


class MLP(nn.Module):
    """Projector MLP: Linear → LayerNorm → GELU → Linear."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LeJEPAEncoder(nn.Module):
    """timm backbone + MLP projector.

    Single-tensor forward: ``(B, C, H, W) → (emb, proj)`` where
        emb:  (B, hidden_dim) — backbone features (CLS for ViT, GAP for CNN)
        proj: (B, proj_dim)   — projected embeddings

    The training loop is responsible for handling multi-view layout
    (V_g, N) and (V_l, N) by flattening the leading dims before forward.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone_name = cfg.backbone_name
        self._is_vit = is_vit(cfg.backbone_name)

        # timm kwargs
        kwargs = dict(pretrained=False, num_classes=0)
        if self._is_vit:
            # Allow forwarding at multiple resolutions for multi-crop
            # (global=224, local=96 typically). dynamic_img_size lets a single
            # ViT handle both without re-instantiating positional embeddings.
            kwargs["dynamic_img_size"] = True
            kwargs["img_size"] = cfg.global_crop_size
            kwargs["patch_size"] = cfg.patch_size

        self.backbone = timm.create_model(cfg.backbone_name, **kwargs)
        if getattr(cfg, "grad_checkpoint", False) and hasattr(
                self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        self._hidden_dim = self.backbone.num_features
        self.projector = MLP(self._hidden_dim, cfg.proj_hidden, cfg.proj_dim)

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def forward(self, x):
        """x: (B, C, H, W) — returns (emb, proj) both shape (B, dim)."""
        emb = self.backbone(x)            # (B, hidden_dim) — pre-pooled w/ num_classes=0
        proj = self.projector(emb)        # (B, proj_dim)
        return emb, proj


class LinearProbe(nn.Module):
    """Linear classification probe on frozen backbone embeddings."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
