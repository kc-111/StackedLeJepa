"""Loss functions for sliced Wasserstein regularization toward N(0, I).

All losses are model-agnostic — they operate on embedding tensors directly.

The key class is ``PooledSlicedLoss``, which collects 1D projections
over multiple sub-steps and computes the loss on the full pooled set with
dramatically better CDF resolution than per-batch losses.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def _gaussian_quantiles(n: int, device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
    """Compute N(0, 1) quantiles for n evenly-spaced probability levels.

    Computed in float32 to avoid overflow in ``erfinv`` with bfloat16,
    then cast to the requested dtype.

    Args:
        n: Number of quantile points.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tensor of shape (n,) with quantile values.
    """
    p = (torch.arange(1, n + 1, device=device, dtype=torch.float32) - 0.5) / n
    return (torch.erfinv(2 * p - 1) * math.sqrt(2)).to(dtype)


def _random_unit_directions(D: int, num_proj: int, device: torch.device,
                            dtype: torch.dtype) -> torch.Tensor:
    """Generate random unit projection directions.

    Args:
        D: Embedding dimensionality.
        num_proj: Number of directions.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tensor of shape (D, num_proj), each column is a unit vector.
    """
    A = torch.randn(D, num_proj, device=device, dtype=dtype)
    return A / A.norm(dim=0, keepdim=True)


class SlicedW1Loss(nn.Module):
    """Sliced Wasserstein-1 distance to N(0, I).

    Projects embeddings onto random unit vectors, sorts along the sample
    dimension, and compares to N(0, 1) quantiles using absolute differences.

    Supports arbitrary leading batch dimensions: ``(B, D)``, ``(V, B, D)``, etc.

    Args:
        num_proj: Number of random projection directions.
    """

    def __init__(self, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sliced W1 loss.

        Args:
            x: Embedding tensor of shape ``(..., B, D)``.

        Returns:
            Scalar loss.
        """
        B, D = x.shape[-2], x.shape[-1]
        A = _random_unit_directions(D, self.num_proj, x.device, x.dtype)
        proj_sorted = torch.sort(x @ A, dim=-2).values  # (..., B, num_proj)
        ref = _gaussian_quantiles(B, x.device, x.dtype)
        ref = ref.reshape(*([1] * (proj_sorted.dim() - 2)), B, 1)
        return (proj_sorted - ref).abs().mean()


class SlicedW2Loss(nn.Module):
    """Sliced Wasserstein-2 distance to N(0, I).

    Same as W1 but uses squared differences (root-mean-square per projection).

    Supports arbitrary leading batch dimensions: ``(B, D)``, ``(V, B, D)``, etc.

    Args:
        num_proj: Number of random projection directions.
    """

    def __init__(self, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sliced W2 loss.

        Args:
            x: Embedding tensor of shape ``(..., B, D)``.

        Returns:
            Scalar loss.
        """
        B, D = x.shape[-2], x.shape[-1]
        A = _random_unit_directions(D, self.num_proj, x.device, x.dtype)
        proj_sorted = torch.sort(x @ A, dim=-2).values  # (..., B, num_proj)
        ref = _gaussian_quantiles(B, x.device, x.dtype)
        ref = ref.reshape(*([1] * (proj_sorted.dim() - 2)), B, 1)
        return ((proj_sorted - ref) ** 2).mean(-2).sqrt().mean()


class SIGRegLoss(nn.Module):
    """Wrapper around SIGReg with a (B, D) interface.

    SIGReg expects (T, B, D) input; this wrapper adds the T=1 dimension.

    Args:
        knots: Number of quadrature knots.
        num_proj: Number of random projections.
        bias_mode: ``"biased"`` (default), ``"ustat"``, or ``"split"``. See
            ``sliced_gauss_reg.SIGReg`` for the math behind each variant.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024,
                 bias_mode: str = "biased"):
        super().__init__()
        from .sigreg import SIGReg
        self.sigreg = SIGReg(knots=knots, num_proj=num_proj,
                              bias_mode=bias_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            x: Embedding tensor of shape (B, D).

        Returns:
            Scalar loss.
        """
        return self.sigreg(x.unsqueeze(0))


def pooled_loss(live: torch.Tensor, detached: torch.Tensor,
                mode: str = "w1", num_proj: int = 1024,
                sigreg=None) -> torch.Tensor:
    """Compute regularization loss on pooled embeddings (1-large-pass pattern).

    Pools ``live`` (gradient-carrying) with ``detached`` (no gradient) and
    computes the sliced distributional loss toward N(0, I). Gradient flows
    only through ``live``.

    For W1/W2: applies gradient dilution compensation (``loss * n_total /
    n_live``). Per-sample gradient is ``O(1/n_total)``, so uncompensated
    pooled has total grad ``n_live / n_total = 1/T`` of the standard
    accumulation. Scaling by ``n_total / n_live`` restores parity with
    both standard accumulation and the big-batch oracle.

    For SIGReg: NO compensation. SIGReg's ``*n`` test-statistic
    normalization makes per-sample gradient ``O(1)`` regardless of
    ``n_total``, so uncompensated pooled SIGReg already matches standard
    SIGReg (both at total grad ``BS``) at the same lambda. Compensating
    would push pooled SIGReg to ``T·BS`` total grad, equal to the
    big-batch SIGReg magnitude — a different operating point that breaks
    drop-in lambda parity with standard accumulation. See FAIRNESS.md.

    This is the recommended API for the 2-step procedure:
        1. Forward entire dataset with ``torch.no_grad()``, get ``detached``.
        2. Forward one mini-batch with gradient, get ``live``.
        3. ``loss = pooled_loss(live, detached)``.

    Args:
        live: Embeddings with gradient, shape ``(..., B, D)``.
        detached: Detached embeddings, shape ``(..., N, D)``. Same leading
            dims as ``live``.
        mode: ``"w1"``, ``"w2"``, or ``"sigreg"``.
        num_proj: Projection directions for W1/W2 (ignored for sigreg).
        sigreg: ``SIGReg`` module instance (required when mode="sigreg").

    Returns:
        Scalar loss tensor with gradient through ``live`` only.

    Combined modes: ``"sigreg+w1"`` or ``"sigreg+w2"`` recursively call
    pooled_loss on each component and sum with equal weight. SIGReg and
    W1/W2 are on different scales — see ``experiments/pretrain/losses.py``
    ``_CombinedReg`` for the caveat.
    """
    if "+" in mode:
        parts = [m.strip() for m in mode.split("+") if m.strip()]
        return sum(pooled_loss(live, detached, mode=m, num_proj=num_proj,
                               sigreg=sigreg) for m in parts)

    all_emb = torch.cat([live, detached], dim=-2)
    n_total = all_emb.shape[-2]
    n_live = live.shape[-2]

    if mode == "sigreg":
        if sigreg is None:
            raise ValueError("sigreg module required when mode='sigreg'")
        return sigreg(all_emb)

    D = all_emb.shape[-1]
    A = _random_unit_directions(D, num_proj, all_emb.device, all_emb.dtype)
    proj_sorted = torch.sort(all_emb @ A, dim=-2).values
    ref = _gaussian_quantiles(n_total, all_emb.device, all_emb.dtype)
    ref = ref.reshape(*([1] * (proj_sorted.dim() - 2)), n_total, 1)

    if mode == "w1":
        loss = (proj_sorted - ref).abs().mean()
    elif mode == "w2":
        loss = ((proj_sorted - ref) ** 2).mean(-2).sqrt().mean()
    else:
        raise ValueError(f"mode must be 'w1', 'w2', or 'sigreg', got '{mode}'")

    return loss * (n_total / n_live)


class PooledSlicedLoss:
    """Accumulated embeddings for better sliced regularization toward N(0, I).

    Collects embeddings over multiple sub-steps and computes the loss on the
    full pooled set for dramatically better CDF resolution.

    Supports three regularizer modes:

    - ``"w1"``: Sliced Wasserstein-1 (sort + absolute difference to quantiles)
    - ``"w2"``: Sliced Wasserstein-2 (sort + squared difference)
    - ``"sigreg"``: Characteristic function test (reuses a ``SIGReg`` module)

    Two-phase API:

    - ``accum_step(embeddings)``: stores detached D-dim embeddings.
    - ``grad_step(embeddings)``: pools stored + current, computes loss.
      Gradient flows only through the current batch.

    Accepts ``(B, D)`` or ``(..., B, D)`` input (e.g. ``(V, B, D)`` for
    multi-view). All operations use ``dim=-2`` as the sample dimension,
    so views are handled vectorized with no Python loops.

    Stores full D-dim embeddings (not 1D projections). For typical
    ``proj_dim=16`` and ``num_proj=256``, this is 16x cheaper.

    Optional FIFO buffer (``fifo_size > 0``): retains stale embeddings
    from previous accumulation windows for even more samples, at the cost
    of staleness.

    Args:
        accum_steps: Number of ``accum_step`` calls per window.
        num_proj: Projection directions for W1/W2 (ignored for sigreg).
        mode: ``"w1"``, ``"w2"``, or ``"sigreg"``.
        sigreg: ``SIGReg`` module instance (required when mode="sigreg").
        fifo_size: Max stale samples from past windows (0 = disabled).

    Example::

        loss_fn = PooledSlicedLoss(accum_steps=7, num_proj=256)

        for sub in range(8):
            embeddings = encoder(batch)
            if sub < 7:
                loss_fn.accum_step(embeddings)
            else:
                loss = loss_fn.grad_step(embeddings)
                loss.backward()
                optimizer.step()
    """

    def __init__(self, accum_steps: int = 7, num_proj: int = 256,
                 mode: str = "w1", sigreg=None, fifo_size: int = 0):
        # Combined modes ("sigreg+w1", "sigreg+w2") sum the components with
        # equal weight. Components are independent: W1/W2 get gradient dilution
        # compensation; SIGReg does not. See pooled_loss for the rationale.
        parts = [m.strip() for m in mode.split("+") if m.strip()]
        for m in parts:
            if m not in ("w1", "w2", "sigreg"):
                raise ValueError(
                    f"mode must be combinations of 'w1', 'w2', 'sigreg', got {mode!r}")
        if "sigreg" in parts and sigreg is None:
            raise ValueError(f"sigreg module required when mode contains 'sigreg' (got {mode!r})")
        self.accum_steps = accum_steps
        self.num_proj = num_proj
        self.mode = mode
        self._modes = parts
        self.sigreg = sigreg
        self.fifo_size = fifo_size
        self._stored: list[torch.Tensor] = []   # current window
        self._fifo: list[torch.Tensor] = []     # past windows (stale)

    def reset(self):
        """Clear current window storage (FIFO is preserved)."""
        self._stored.clear()

    def reset_all(self):
        """Clear both current window and FIFO storage."""
        self._stored.clear()
        self._fifo.clear()

    @torch.no_grad()
    def accum_step(self, embeddings: torch.Tensor) -> None:
        """Store detached embeddings for later pooling.

        Args:
            embeddings: Tensor of shape ``(..., B, D)``.
        """
        self._stored.append(embeddings.detach())

    def grad_step(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute loss on pooled embeddings with gradient through this batch.

        Pools the current batch with all stored embeddings (and FIFO if
        enabled), computes the regularization loss, then updates FIFO
        and resets the current window.

        Args:
            embeddings: Tensor of shape ``(..., B, D)``. Gradient flows.

        Returns:
            Scalar loss tensor.
        """
        # Pool: current (grad) + stored (detached) + FIFO (detached)
        parts = [embeddings]
        if self._stored:
            parts.append(torch.cat(self._stored, dim=-2))
        if self._fifo:
            parts.append(torch.cat(self._fifo, dim=-2))
        all_emb = torch.cat(parts, dim=-2)

        n_total = all_emb.shape[-2]
        n_real = embeddings.shape[-2]

        loss = None
        sort_cache = None  # (proj_sorted, ref) shared between w1 and w2
        for m in self._modes:
            if m == "sigreg":
                # SIGReg's *n test-statistic normalization makes per-sample
                # gradient O(1) regardless of n_total. No T compensation:
                # uncompensated pooled SIGReg matches standard accumulation
                # at the same lambda. Compensating would push it to T× the
                # standard signal (= the big-batch SIGReg magnitude), which
                # is a different operating point. See FAIRNESS.md.
                term = self.sigreg(all_emb)
            else:
                if sort_cache is None:
                    D = all_emb.shape[-1]
                    A = _random_unit_directions(
                        D, self.num_proj, all_emb.device, all_emb.dtype)
                    proj_sorted = torch.sort(all_emb @ A, dim=-2).values
                    ref = _gaussian_quantiles(n_total, all_emb.device, all_emb.dtype)
                    ref = ref.reshape(
                        *([1] * (proj_sorted.dim() - 2)), n_total, 1)
                    sort_cache = (proj_sorted, ref)
                else:
                    proj_sorted, ref = sort_cache

                if m == "w1":
                    term = (proj_sorted - ref).abs().mean()
                else:  # w2
                    term = ((proj_sorted - ref) ** 2).mean(-2).sqrt().mean()

                # Sort-based gradient dilution compensation: W1/W2 per-sample
                # gradient is O(1/n_total), so uncompensated pooled has total
                # grad BS/n_total = 1/T while standard has total 1. Multiplying
                # by n_total/n_real ≈ T restores parity with both standard and
                # big-batch.
                term = term * (n_total / n_real)

            loss = term if loss is None else loss + term

        # Update FIFO with this window's embeddings
        if self.fifo_size > 0:
            window_parts = [embeddings.detach()] + self._stored
            self._fifo.append(torch.cat(window_parts, dim=-2))
            total = sum(f.shape[-2] for f in self._fifo)
            while total > self.fifo_size and len(self._fifo) > 1:
                total -= self._fifo[0].shape[-2]
                self._fifo.pop(0)

        self._stored.clear()
        return loss
