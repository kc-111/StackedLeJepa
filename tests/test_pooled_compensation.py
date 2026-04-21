"""Empirical gradient-norm verification of pooled-loss compensation policy.

Verifies the predictions from FAIRNESS.md → "Gradient dilution compensation
in pooled losses":

    For W1/W2 (per-sample grad ~ 1/n_total):
        - Uncompensated pooled grad norm  ≈  (1/T) * solo grad norm
        - Compensated   pooled grad norm  ≈  solo grad norm

    For SIGReg (per-sample grad ~ O(1) due to *n test-statistic normalization):
        - Uncompensated pooled grad norm  ≈  solo grad norm
        - Compensated   pooled grad norm  ≈  T * solo grad norm

These predictions justify why the library applies (n_total/n_real)
compensation to W1/W2 in pooled mode but NOT to SIGReg.

The test setup:
    - Tiny linear model `f(x) = x @ W` (so per-sample contribution to ∂L/∂W
      is well-defined and easy to reason about).
    - Fixed seeds so projection directions and data are reproducible across
      the solo / pooled-uncompensated / pooled-compensated branches.
    - Average gradient norm ratios over multiple trials to reduce variance
      from finite-num_proj noise.

Run:
    pytest tests/test_pooled_compensation.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "experiments" / "synthetic"
for p in (str(REPO_ROOT), str(SYNTHETIC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sliced_gauss_reg import (
    SlicedW1Loss, SlicedW2Loss, SIGRegLoss, SIGReg, PooledSlicedLoss,
)
from sliced_gauss_reg.losses import pooled_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BS = 32
T = 8
DIM = 16
NUM_PROJ = 2048    # large to keep projection-direction variance low
KNOTS = 17
N_TRIALS = 8       # average over trials to reduce sample-selection noise


def _make_model(seed: int):
    torch.manual_seed(seed)
    return torch.nn.Linear(DIM, DIM, bias=False)


def _make_data(seed: int):
    g = torch.Generator().manual_seed(seed + 999)
    return torch.randn(T * BS, DIM, generator=g)


def _grad_norm(model: torch.nn.Module) -> float:
    """L2 norm of all parameter gradients flattened together."""
    parts = [p.grad.detach().flatten() for p in model.parameters()
             if p.grad is not None]
    return torch.cat(parts).norm().item()


def _build_loss(mode: str):
    if mode == "w1":
        return SlicedW1Loss(num_proj=NUM_PROJ)
    if mode == "w2":
        return SlicedW2Loss(num_proj=NUM_PROJ)
    if mode == "sigreg":
        return SIGRegLoss(knots=KNOTS, num_proj=NUM_PROJ)
    raise ValueError(mode)


def _build_sigreg():
    return SIGReg(knots=KNOTS, num_proj=NUM_PROJ)


# ---------------------------------------------------------------------------
# Per-trial gradient norms
# ---------------------------------------------------------------------------

def _solo_grad_norm(mode: str, model_seed: int, data_seed: int,
                    proj_seed: int) -> float:
    """Gradient norm of `loss(model(x[:BS]))` — the standard non-pooled case.

    `proj_seed` controls the random projections inside the loss so that the
    solo / pooled comparison uses identical projection directions.
    """
    model = _make_model(model_seed)
    data = _make_data(data_seed)
    grad_x = data[:BS]

    torch.manual_seed(proj_seed)  # fix projection directions
    loss_fn = _build_loss(mode)

    model.zero_grad(set_to_none=True)
    out = model(grad_x)
    loss = loss_fn(out)
    loss.backward()
    return _grad_norm(model)


def _pooled_grad_norm(mode: str, model_seed: int, data_seed: int,
                      proj_seed: int, compensate: bool) -> float:
    """Gradient norm of pooled loss on (BS grad + (T-1)·BS no-grad) samples.

    The grad batch is data[:BS] (same as solo). The no-grad context is
    data[BS:T·BS]. Projection directions are seeded identically to solo.
    """
    model = _make_model(model_seed)
    data = _make_data(data_seed)
    grad_x = data[:BS]
    nograd_x = data[BS:T * BS]

    with torch.no_grad():
        nograd_emb = model(nograd_x)
    grad_emb = model(grad_x)

    all_emb = torch.cat([grad_emb, nograd_emb], dim=0)
    n_total = all_emb.shape[0]
    n_real = grad_emb.shape[0]

    torch.manual_seed(proj_seed)  # SAME projections as solo
    if mode == "sigreg":
        sigreg_mod = _build_sigreg()
        loss = sigreg_mod(all_emb.unsqueeze(0))  # (T=1, B, D)
    else:
        loss_fn = _build_loss(mode)
        loss = loss_fn(all_emb)

    if compensate:
        loss = loss * (n_total / n_real)

    model.zero_grad(set_to_none=True)
    loss.backward()
    return _grad_norm(model)


def _averaged_ratio(mode: str, compensate: bool) -> float:
    """Mean of (pooled_norm / solo_norm) across N_TRIALS independent trials."""
    ratios = []
    for trial in range(N_TRIALS):
        seeds = dict(
            model_seed=100 + trial,
            data_seed=200 + trial,
            proj_seed=300 + trial,
        )
        solo = _solo_grad_norm(mode, **seeds)
        pooled = _pooled_grad_norm(mode, compensate=compensate, **seeds)
        ratios.append(pooled / solo)
    return float(sum(ratios) / len(ratios))


# ---------------------------------------------------------------------------
# Tests: W1/W2 — uncompensated pooled is 1/T diluted, compensated matches solo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["w1", "w2"])
def test_w1w2_pooled_uncompensated_is_diluted(mode):
    """Pooled W1/W2 without × T compensation should give ≈ (1/T) of solo grad."""
    ratio = _averaged_ratio(mode, compensate=False)
    expected = 1.0 / T
    rel_err = abs(ratio - expected) / expected
    print(f"\n  [{mode}] uncompensated pooled / solo  ratio = {ratio:.4f}  "
          f"(expected ≈ {expected:.4f}, rel_err = {rel_err:.2%})")
    assert rel_err < 0.20, (
        f"{mode} uncompensated pooled grad norm ratio {ratio:.4f} not within "
        f"20% of expected 1/T = {expected:.4f}")


@pytest.mark.parametrize("mode", ["w1", "w2"])
def test_w1w2_pooled_compensated_matches_solo(mode):
    """Pooled W1/W2 with × T compensation should match solo grad norm."""
    ratio = _averaged_ratio(mode, compensate=True)
    rel_err = abs(ratio - 1.0)
    print(f"\n  [{mode}] compensated   pooled / solo  ratio = {ratio:.4f}  "
          f"(expected ≈ 1.0, rel_err = {rel_err:.2%})")
    assert rel_err < 0.20, (
        f"{mode} compensated pooled grad norm ratio {ratio:.4f} not within "
        f"20% of expected 1.0")


# ---------------------------------------------------------------------------
# Tests: SIGReg — uncompensated pooled matches solo, compensated is T× too big
# ---------------------------------------------------------------------------

def test_sigreg_pooled_uncompensated_matches_solo():
    """Pooled SIGReg WITHOUT compensation should match solo grad norm.

    This is the empirical justification for NOT compensating SIGReg in
    pooled_loss / PooledSlicedLoss. SIGReg's *n test statistic makes
    per-sample gradient O(1), so total grad scales with the number of
    grad-carrying samples, not with n_total. Both solo and pooled
    uncompensated have BS grad samples → both have ~BS total grad.
    """
    ratio = _averaged_ratio("sigreg", compensate=False)
    rel_err = abs(ratio - 1.0)
    print(f"\n  [sigreg] uncompensated pooled / solo  ratio = {ratio:.4f}  "
          f"(expected ≈ 1.0, rel_err = {rel_err:.2%})")
    assert rel_err < 0.20, (
        f"sigreg uncompensated pooled grad norm ratio {ratio:.4f} not within "
        f"20% of expected 1.0 — the no-compensation policy for SIGReg in "
        f"pooled_loss / PooledSlicedLoss is wrong!")


def test_sigreg_pooled_compensated_is_amplified():
    """Pooled SIGReg WITH × T compensation should give ≈ T × solo grad norm.

    This test exists to confirm we understand what the compensation does
    for SIGReg (it amplifies by T, NOT what we want for SIGReg). If this
    test fails, our mental model of SIGReg's gradient scaling is wrong.
    """
    ratio = _averaged_ratio("sigreg", compensate=True)
    expected = float(T)
    rel_err = abs(ratio - expected) / expected
    print(f"\n  [sigreg] compensated   pooled / solo  ratio = {ratio:.4f}  "
          f"(expected ≈ {expected:.4f}, rel_err = {rel_err:.2%})")
    assert rel_err < 0.20, (
        f"sigreg compensated pooled grad norm ratio {ratio:.4f} not within "
        f"20% of expected T = {expected:.4f}")


# ---------------------------------------------------------------------------
# Library-API tests: verify the actual behavior of PooledSlicedLoss / pooled_loss
# matches the policy (W1/W2 compensated, SIGReg not).
# ---------------------------------------------------------------------------

def _library_pooledslicedloss_grad_norm(mode: str, model_seed: int,
                                         data_seed: int) -> float:
    """Run the actual PooledSlicedLoss (production API) end-to-end."""
    model = _make_model(model_seed)
    data = _make_data(data_seed)
    grad_x = data[:BS]
    nograd_x = data[BS:T * BS]

    sigreg_mod = _build_sigreg() if mode == "sigreg" else None
    accum = PooledSlicedLoss(
        accum_steps=T - 1, num_proj=NUM_PROJ, mode=mode, sigreg=sigreg_mod)

    # T-1 no-grad sub-batches of BS samples each
    sub_size = BS
    with torch.no_grad():
        for t in range(T - 1):
            sub = nograd_x[t * sub_size:(t + 1) * sub_size]
            accum.accum_step(model(sub))

    grad_emb = model(grad_x)
    loss = accum.grad_step(grad_emb)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return _grad_norm(model)


@pytest.mark.parametrize("mode", ["w1", "w2"])
def test_library_pooledslicedloss_w1w2_matches_solo(mode):
    """The shipped PooledSlicedLoss for W1/W2 should match solo grad norm
    (because it applies the n_total/n_real compensation internally)."""
    ratios = []
    for trial in range(N_TRIALS):
        solo = _solo_grad_norm(
            mode, model_seed=400 + trial, data_seed=500 + trial,
            proj_seed=600 + trial)
        lib = _library_pooledslicedloss_grad_norm(
            mode, model_seed=400 + trial, data_seed=500 + trial)
        ratios.append(lib / solo)
    ratio = float(sum(ratios) / len(ratios))
    rel_err = abs(ratio - 1.0)
    print(f"\n  [{mode}] PooledSlicedLoss / solo  ratio = {ratio:.4f}")
    assert rel_err < 0.30, (
        f"library PooledSlicedLoss({mode}) grad norm ratio {ratio:.4f} not "
        f"within 30% of 1.0")


def test_library_pooledslicedloss_sigreg_matches_solo():
    """The shipped PooledSlicedLoss for SIGReg should match solo grad norm
    (because it does NOT apply compensation, by design)."""
    ratios = []
    for trial in range(N_TRIALS):
        solo = _solo_grad_norm(
            "sigreg", model_seed=400 + trial, data_seed=500 + trial,
            proj_seed=600 + trial)
        lib = _library_pooledslicedloss_grad_norm(
            "sigreg", model_seed=400 + trial, data_seed=500 + trial)
        ratios.append(lib / solo)
    ratio = float(sum(ratios) / len(ratios))
    rel_err = abs(ratio - 1.0)
    print(f"\n  [sigreg] PooledSlicedLoss / solo  ratio = {ratio:.4f}")
    assert rel_err < 0.30, (
        f"library PooledSlicedLoss(sigreg) grad norm ratio {ratio:.4f} not "
        f"within 30% of 1.0 — SIGReg pooled is NOT matching standard, which "
        f"breaks lambda parity")


def _library_pooled_loss_grad_norm(mode: str, model_seed: int,
                                    data_seed: int) -> float:
    """Run the standalone pooled_loss function (used by production pretrain)."""
    model = _make_model(model_seed)
    data = _make_data(data_seed)
    grad_x = data[:BS]
    nograd_x = data[BS:T * BS]

    with torch.no_grad():
        nograd_emb = model(nograd_x)
    grad_emb = model(grad_x)

    sigreg_mod = _build_sigreg() if mode == "sigreg" else None
    loss = pooled_loss(
        live=grad_emb, detached=nograd_emb,
        mode=mode, num_proj=NUM_PROJ, sigreg=sigreg_mod)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return _grad_norm(model)


@pytest.mark.parametrize("mode", ["w1", "w2", "sigreg"])
def test_library_pooled_loss_matches_solo(mode):
    """Standalone pooled_loss should match solo grad norm for all three modes
    (W1/W2 via compensation, SIGReg via no-compensation policy)."""
    ratios = []
    for trial in range(N_TRIALS):
        solo = _solo_grad_norm(
            mode, model_seed=700 + trial, data_seed=800 + trial,
            proj_seed=900 + trial)
        lib = _library_pooled_loss_grad_norm(
            mode, model_seed=700 + trial, data_seed=800 + trial)
        ratios.append(lib / solo)
    ratio = float(sum(ratios) / len(ratios))
    rel_err = abs(ratio - 1.0)
    print(f"\n  [{mode}] pooled_loss / solo  ratio = {ratio:.4f}")
    assert rel_err < 0.30, (
        f"library pooled_loss({mode}) grad norm ratio {ratio:.4f} not "
        f"within 30% of 1.0")
