"""Model components for LeJEPA pretraining (timm backbones)."""

import torch
import torch.nn as nn

from configs import is_vit
from StackedLeJepa.backbone import create_backbone


class Projector(nn.Module):
    """3-layer projector MLP: Linear → LN → GELU → Linear → LN → GELU → Linear.

    Matches the authors' reference implementation (3 layers, hidden > embedding dim).
    Uses LayerNorm (not BatchNorm) so the projector is consistent across
    the two passes of pooled training.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
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

        self.backbone = create_backbone(
            model_name=cfg.backbone_name,
            crop_size=cfg.crop_size,
            patch_size=cfg.patch_size if self._is_vit else None,
            pretrained=False,
            grad_checkpoint=getattr(cfg, "grad_checkpoint", False),
        )
        self._hidden_dim = self.backbone.num_features
        self.projector = Projector(self._hidden_dim, cfg.proj_hidden, cfg.proj_dim)


    @property
    def hidden_dim(self):
        return self._hidden_dim

    def forward(self, x):
        """x: (B, C, H, W) — returns (emb, proj) both shape (B, dim)."""
        emb = self.backbone(x)            # (B, hidden_dim) — pre-pooled w/ num_classes=0
        proj = self.projector(emb)        # (B, proj_dim)
        return emb, proj


class LinearProbe(nn.Module):
    """LayerNorm + Linear probe on frozen backbone embeddings (paper Sec 6.1)."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.probe = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.probe(x)
