"""Verify the pooled-training BN handling is mathematically consistent.

The pooled step does two encoder forward passes (a no-grad pass on (T-1)*BS
samples and a grad pass on BS samples). For the regularizer pool to be
well-defined, both passes MUST normalize their inputs with the same BN
statistics. These tests verify that property.
"""

import torch
import torch.nn as nn

from data import get_dataloaders
from models import LeJEPAEncoder
from train_loops import _BNStatCapture, inject_bn_stats


def _bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def test_capture_finds_all_bn_layers(device, small_cfg):
    encoder = LeJEPAEncoder(small_cfg).to(device).train()
    bn_layers = _bn_layers(encoder)
    assert len(bn_layers) > 0, "ResNet18 should have BN layers"

    train_ds, _, gpu_aug = get_dataloaders(small_cfg, device)
    gen = torch.Generator(device=device).manual_seed(0)
    images, _ = train_ds.sample_batch(64, gen)
    x = images.float() / 255.0
    g, _ = gpu_aug(x)
    g_flat = g.reshape(-1, *g.shape[2:])

    capture = _BNStatCapture(encoder)
    with torch.no_grad(), capture:
        encoder(g_flat)

    assert len(capture.captured) == len(bn_layers), (
        f"Captured {len(capture.captured)} stats but {len(bn_layers)} BN layers exist")

    # Each capture entry should be (mean, var) tensors with the right shape
    for layer in bn_layers:
        assert id(layer) in capture.captured
        mean, var = capture.captured[id(layer)]
        assert mean.shape == (layer.num_features,)
        assert var.shape == (layer.num_features,)
        assert torch.all(var >= 0), "Variance must be non-negative"


def test_inject_overrides_running_stats_then_restores(device, small_cfg):
    encoder = LeJEPAEncoder(small_cfg).to(device).train()
    bn_layers = _bn_layers(encoder)
    first_bn = bn_layers[0]

    # Snapshot original running stats
    orig_mean = first_bn.running_mean.detach().clone()
    orig_var = first_bn.running_var.detach().clone()

    fake_captured = {
        id(layer): (
            torch.full_like(layer.running_mean, 99.0),
            torch.full_like(layer.running_var, 7.0),
        )
        for layer in bn_layers
    }

    with inject_bn_stats(encoder, fake_captured):
        # Inside the context, running stats should equal the injected values
        assert torch.allclose(first_bn.running_mean, torch.full_like(orig_mean, 99.0))
        assert torch.allclose(first_bn.running_var, torch.full_like(orig_var, 7.0))
        assert not first_bn.training, "BN should be in eval mode during inject"

    # After the context, running stats should be restored exactly
    assert torch.allclose(first_bn.running_mean, orig_mean), \
        "running_mean should be restored after inject_bn_stats exits"
    assert torch.allclose(first_bn.running_var, orig_var), \
        "running_var should be restored after inject_bn_stats exits"
    assert first_bn.training, "BN should be back in training mode"


def test_capture_then_inject_gives_consistent_normalization(device, small_cfg):
    """The headline test: capture batch stats from a train-mode pass, then
    inject them and re-run in eval mode. The output must match (modulo float
    rounding) the train-mode output, proving both passes use IDENTICAL BN."""
    encoder = LeJEPAEncoder(small_cfg).to(device).train()

    train_ds, _, gpu_aug = get_dataloaders(small_cfg, device)
    gen = torch.Generator(device=device).manual_seed(0)
    images, _ = train_ds.sample_batch(64, gen)
    x = images.float() / 255.0
    g, _ = gpu_aug(x)
    g_flat = g.reshape(-1, *g.shape[2:])

    # Pass A: train mode, capture batch stats
    capture = _BNStatCapture(encoder)
    with torch.no_grad(), capture:
        emb_train, proj_train = encoder(g_flat)

    # Pass B: same input in eval mode with injected stats
    with inject_bn_stats(encoder, capture.captured):
        with torch.no_grad():
            emb_eval, proj_eval = encoder(g_flat)

    # The embeddings should match — both passes used the SAME normalization stats
    max_abs_diff = (emb_train.float() - emb_eval.float()).abs().max().item()
    rel_diff = ((emb_train.float() - emb_eval.float()).abs().mean() /
                emb_train.float().abs().mean()).item()

    # bf16 / float32 rounding can produce ~1e-5 absolute differences
    assert max_abs_diff < 1e-3, (
        f"Embeddings differ by {max_abs_diff}, expected near-zero "
        f"(both passes should use identical BN stats)")
    assert rel_diff < 1e-4, f"Mean relative diff {rel_diff} too large"


def test_inject_restores_even_on_exception(device, small_cfg):
    """If the body of the inject context raises, BN stats must still be restored."""
    encoder = LeJEPAEncoder(small_cfg).to(device).train()
    bn_layers = _bn_layers(encoder)
    first_bn = bn_layers[0]

    orig_mean = first_bn.running_mean.detach().clone()
    orig_var = first_bn.running_var.detach().clone()

    fake_captured = {
        id(layer): (torch.full_like(layer.running_mean, -5.0),
                    torch.full_like(layer.running_var, 3.0))
        for layer in bn_layers
    }

    try:
        with inject_bn_stats(encoder, fake_captured):
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    assert torch.allclose(first_bn.running_mean, orig_mean)
    assert torch.allclose(first_bn.running_var, orig_var)
    assert first_bn.training
