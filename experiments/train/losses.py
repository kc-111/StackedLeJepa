"""Distributional regularizers toward N(0, I).

All three share the same interface:
    forward(proj: Tensor of shape (..., B, D)) -> scalar

The ``needs_compensation`` flag signals whether pooled use requires
gradient-dilution compensation (multiply loss by n_total / n_live):

- SIGReg: the ×n test-statistic normalization makes per-sample gradient
  O(1), so pooling at the same λ already matches un-pooled magnitude.
  Compensation would over-signal.
- SlicedW1 / SlicedW2: sort-based, per-sample gradient O(1/n_total),
  so pooling at the same λ dilutes total gradient by n_live/n_total.
  Compensation restores parity.
"""

import math

import torch
import torch.nn as nn


def _gaussian_quantiles(n: int, device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
    """N(0, 1) quantiles at n evenly-spaced probability levels."""
    p = (torch.arange(1, n + 1, device=device, dtype=torch.float32) - 0.5) / n
    return (torch.erfinv(2 * p - 1) * math.sqrt(2)).to(dtype)


def _random_unit_directions(D: int, num_proj: int, device: torch.device,
                            dtype: torch.dtype) -> torch.Tensor:
    """Columns of the returned matrix are unit vectors in R^D."""
    A = torch.randn(D, num_proj, device=device, dtype=dtype)
    return A / A.norm(dim=0, keepdim=True)


class SIGReg(nn.Module):
    """Epps-Pulley characteristic-function test toward N(0, I).

    Weighted L2 distance between the empirical CF of random 1D projections
    and the N(0, 1) CF, evaluated at quadrature knots on [0, 3] with a
    Gaussian window. Biased plug-in estimator (paper default).
    """

    needs_compensation = False

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        D = proj.size(-1)
        n = proj.size(-2)
        A = _random_unit_directions(D, self.num_proj, proj.device, proj.dtype)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        c_bar = x_t.cos().mean(-3)
        s_bar = x_t.sin().mean(-3)
        err = (c_bar - self.phi).square() + s_bar.square()
        return ((err @ self.weights) * n).mean()


class SlicedW1(nn.Module):
    """Sliced Wasserstein-1 distance to N(0, I)."""

    needs_compensation = True

    def __init__(self, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        B, D = proj.shape[-2], proj.shape[-1]
        A = _random_unit_directions(D, self.num_proj, proj.device, proj.dtype)
        proj_sorted = torch.sort(proj @ A, dim=-2).values
        ref = _gaussian_quantiles(B, proj.device, proj.dtype)
        ref = ref.reshape(*([1] * (proj_sorted.dim() - 2)), B, 1)
        return (proj_sorted - ref).abs().mean()


class SlicedW2(nn.Module):
    """Sliced Wasserstein-2 distance to N(0, I) (root-mean-square over batch)."""

    needs_compensation = True

    def __init__(self, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        B, D = proj.shape[-2], proj.shape[-1]
        A = _random_unit_directions(D, self.num_proj, proj.device, proj.dtype)
        proj_sorted = torch.sort(proj @ A, dim=-2).values
        ref = _gaussian_quantiles(B, proj.device, proj.dtype)
        ref = ref.reshape(*([1] * (proj_sorted.dim() - 2)), B, 1)
        return ((proj_sorted - ref) ** 2).mean(-2).sqrt().mean()


REGULARIZERS = {
    "sigreg": SIGReg,
    "sigreg_raw": "stable_pretraining.methods.lejepa.SlicedEppsPulley",
    "w1": SlicedW1,
    "w2": SlicedW2,
}


def make_regularizer(name: str, num_proj: int = 1024, knots: int = 17) -> nn.Module:
    """Factory: construct a regularizer by name.

    ``sigreg_raw`` imports ``SlicedEppsPulley`` directly from
    stable_pretraining. It expects flat ``[N, D]`` input; the caller is
    responsible for reshaping ``(V, B, D) → (V*B, D)`` before calling it.
    ``needs_compensation`` is tagged onto the instance since the upstream
    class doesn't define it.
    """
    if name == "sigreg":
        return SIGReg(knots=knots, num_proj=num_proj)
    if name == "sigreg_raw":
        from stable_pretraining.methods.lejepa import SlicedEppsPulley
        reg = SlicedEppsPulley(num_slices=num_proj, n_points=knots)
        reg.needs_compensation = False
        return reg
    if name == "w1":
        return SlicedW1(num_proj=num_proj)
    if name == "w2":
        return SlicedW2(num_proj=num_proj)
    raise ValueError(f"unknown regularizer {name!r}; choose from {list(REGULARIZERS)}")
