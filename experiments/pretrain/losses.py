"""Loss functions and regularizer construction for LeJEPA."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn as nn
from sliced_gauss_reg import SIGReg
from sliced_gauss_reg.losses import SlicedW1Loss, SlicedW2Loss


def inv_loss_fn(all_views: torch.Tensor,
                nograd_views: torch.Tensor | None = None,
                detach_centroid: bool = False) -> torch.Tensor:
    """View-invariance loss: MSE of each (grad) view to the centroid.

    The centroid is estimated from *all* available views (grad + no-grad).
    More views → lower-variance centroid → cleaner invariance signal.

    Args:
        all_views: (V_g, N, D) — views with gradients (original + aug).
        nograd_views: (V_n, N, D) or None — extra detached views used
            only to improve the centroid estimate. No gradients flow
            through these.
        detach_centroid: if True, ``stopgrad`` the centroid so each grad
            view is pulled toward a fixed target (BYOL-style one-way
            pull). If False, gradients flow through the grad views'
            contribution to the centroid (bidirectional, like VICReg).

    Returns:
        Scalar loss = mean over (v, n, d) of (view - centroid)**2,
        where the mean is taken only over the *grad* views.
    """
    if nograd_views is not None:
        all_for_centroid = torch.cat([all_views, nograd_views], dim=0)
    else:
        all_for_centroid = all_views
    anchor = all_for_centroid.mean(dim=0)  # (N, D)
    if detach_centroid:
        anchor = anchor.detach()
    return (all_views - anchor.unsqueeze(0)).square().mean()


class _CombinedReg(nn.Module):
    """Sums the outputs of multiple regularizer modules with equal weight.

    SIGReg and W1/W2 are on different scales (SIGReg statistic ~1–2 even on
    truly Gaussian samples due to the biased plug-in floor; W1 ~0.05–0.5).
    Equal-weight sum means SIGReg dominates by ~5×. Use cfg.lambd to scale
    the combined term down if needed.
    """

    def __init__(self, regs):
        super().__init__()
        self.regs = nn.ModuleList(regs)

    def forward(self, x):
        return sum(r(x) for r in self.regs)


def _build_one(mode: str, cfg, device):
    if mode == "sigreg":
        return SIGReg(knots=17, num_proj=cfg.num_proj).to(device)
    if mode == "w1":
        return SlicedW1Loss(num_proj=cfg.num_proj).to(device)
    if mode == "w2":
        return SlicedW2Loss(num_proj=cfg.num_proj).to(device)
    raise ValueError(f"Unknown regularizer: {mode}")


def parse_regularizer_modes(spec: str) -> list:
    """Parse cfg.regularizer into a list of component modes.

    "w1" -> ["w1"]; "sigreg+w1" -> ["sigreg", "w1"]. Whitespace tolerated.
    """
    return [m.strip() for m in spec.split("+") if m.strip()]


def build_regularizer(cfg, device):
    """Build the distributional regularizer.

    ``cfg.regularizer`` is either a single mode (``"sigreg"``, ``"w1"``,
    ``"w2"``) or a "+"-separated combination (``"sigreg+w1"``,
    ``"sigreg+w2"``). Combined modes sum the component losses with equal
    weight — see ``_CombinedReg`` for the scale caveat.

    All regularizers accept a leading view dim: W1/W2 sort along ``dim=-2``
    and average over all leading dims; SIGReg computes the Epps-Pulley test
    statistic per leading-dim slice and averages. So a single call on
    ``(V, B, D)`` is mathematically identical to looping over V and averaging.
    """
    modes = parse_regularizer_modes(cfg.regularizer)
    if len(modes) == 1:
        return _build_one(modes[0], cfg, device)
    return _CombinedReg([_build_one(m, cfg, device) for m in modes])


def build_sigreg(cfg, device):
    """Build SIGReg module for pooled mode (projection space)."""
    return SIGReg(knots=17, num_proj=cfg.num_proj).to(device)


def build_emb_regularizer(cfg, device):
    """Build regularizer for embedding space (same type as projection regularizer).

    Returns None if lambd_emb == 0 (disabled).
    """
    if cfg.lambd_emb <= 0:
        return None
    return build_regularizer(cfg, device)


def build_sigreg_emb(cfg, device):
    """Build SIGReg module for pooled mode (embedding space).

    Returns None if lambd_emb == 0 (disabled).
    """
    if cfg.lambd_emb <= 0:
        return None
    return SIGReg(knots=17, num_proj=cfg.num_proj).to(device)
