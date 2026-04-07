"""Loss functions and regularizer construction for LeJEPA."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from module import SIGReg
from accumulated_w1.losses import SlicedW1Loss, SlicedW2Loss, pooled_loss


def inv_loss_fn(proj: torch.Tensor, num_global_views: int) -> torch.Tensor:
    """View-invariance loss (LeJEPA Algorithm 2).

    The center is the mean of the **global views only**; the loss is the
    squared distance from each view's projection (global + local) to that
    center, averaged over all views and samples.

    Args:
        proj: (V, N, D) — first ``num_global_views`` are global, rest local.
        num_global_views: number of global views (V_g).

    Returns:
        Scalar loss.
    """
    center = proj[:num_global_views].mean(0, keepdim=True)  # (1, N, D)
    return (center - proj).square().mean()


def regularizer_loss(reg_fn, proj: torch.Tensor) -> torch.Tensor:
    """Apply a regularizer per-view and average (Paper Alg 2).

    Args:
        reg_fn: a regularizer module accepting (B, D) → scalar.
        proj: (V, N, D)

    Returns:
        Scalar loss = mean over views of reg_fn(proj[v]).
    """
    V = proj.shape[0]
    total = proj.new_zeros(())
    for v in range(V):
        total = total + reg_fn(proj[v])
    return total / V


def build_regularizer(cfg, device):
    """Build the per-view regularizer (used in standard training)."""
    if cfg.regularizer == "sigreg":
        return SIGRegPerView(knots=17, num_proj=cfg.num_proj).to(device)
    elif cfg.regularizer == "w1":
        return SlicedW1Loss(num_proj=cfg.num_proj).to(device)
    elif cfg.regularizer == "w2":
        return SlicedW2Loss(num_proj=cfg.num_proj).to(device)
    raise ValueError(f"Unknown regularizer: {cfg.regularizer}")


class SIGRegPerView(torch.nn.Module):
    """SIGReg with a (B, D) interface (adds the T=1 dim that SIGReg wants)."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.sigreg = SIGReg(knots=knots, num_proj=num_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) → SIGReg expects (T, B, D)
        return self.sigreg(x.unsqueeze(0))


def build_sigreg(cfg, device):
    """Build SIGReg module for pooled mode."""
    return SIGReg(knots=17, num_proj=cfg.num_proj).to(device)
