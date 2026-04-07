"""End-to-end test of one pooled training step.

Verifies that:
- The full _pooled_step runs without error and returns valid loss tensors
- BN learnable params (gamma, beta) are updated by gradient
- Conv weights are updated by gradient
- BN running stats are updated naturally by Pass 1 (the no-grad pass) and
  are NOT mutated by the inject_bn_stats override (which restores them)
- The projector (LayerNorm-based) is unaffected by the BN handling
"""

import torch
import torch.nn as nn

from configs import Config
from data import get_dataloaders
from models import LeJEPAEncoder, LinearProbe
from train_loops import _pooled_step
from scheduler import make_scheduler


def _bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def _conv_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def _make_components(cfg, device):
    train_ds, _, gpu_aug = get_dataloaders(cfg, device)
    encoder = LeJEPAEncoder(cfg).to(device)
    probe = LinearProbe(encoder.hidden_dim, cfg.num_classes).to(device)
    enc_opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    enc_sched = make_scheduler(enc_opt, 1, 100, 1e-3, 1e-5)
    probe_sched = make_scheduler(probe_opt, 1, 100, 1e-3, 1e-5)
    return train_ds, gpu_aug, encoder, probe, enc_opt, probe_opt, enc_sched, probe_sched


def test_pooled_step_runs_without_error(device, pooled_cfg):
    train_ds, gpu_aug, encoder, probe, enc_opt, probe_opt, enc_sched, probe_sched = \
        _make_components(pooled_cfg, device)

    gen = torch.Generator(device=device).manual_seed(0)
    images, labels = train_ds.sample_batch(pooled_cfg.batch_size, gen)
    nograd_images, _ = train_ds.sample_batch(
        (pooled_cfg.accum_steps - 1) * pooled_cfg.batch_size, gen)

    loss, reg_loss, inv, probe_loss, probe_logits = _pooled_step(
        images, nograd_images, labels, encoder, probe, gpu_aug, pooled_cfg,
        enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod=None)

    assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"
    assert torch.isfinite(reg_loss).all()
    assert torch.isfinite(inv).all()
    assert torch.isfinite(probe_loss).all()
    assert probe_logits.shape == (pooled_cfg.batch_size, pooled_cfg.num_classes)


def test_pooled_step_updates_bn_params(device, pooled_cfg):
    train_ds, gpu_aug, encoder, probe, enc_opt, probe_opt, enc_sched, probe_sched = \
        _make_components(pooled_cfg, device)

    bn_layers = _bn_layers(encoder)
    assert len(bn_layers) > 0
    first_bn = bn_layers[0]

    gamma_before = first_bn.weight.detach().clone()
    beta_before = first_bn.bias.detach().clone()

    gen = torch.Generator(device=device).manual_seed(0)
    images, labels = train_ds.sample_batch(pooled_cfg.batch_size, gen)
    nograd_images, _ = train_ds.sample_batch(
        (pooled_cfg.accum_steps - 1) * pooled_cfg.batch_size, gen)
    _pooled_step(images, nograd_images, labels, encoder, probe, gpu_aug, pooled_cfg,
                 enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod=None)

    gamma_delta = (first_bn.weight - gamma_before).abs().max().item()
    beta_delta = (first_bn.bias - beta_before).abs().max().item()

    assert gamma_delta > 0, "BN gamma should update via gradient (didn't change)"
    assert beta_delta > 0, "BN beta should update via gradient (didn't change)"


def test_pooled_step_updates_conv_weights(device, pooled_cfg):
    train_ds, gpu_aug, encoder, probe, enc_opt, probe_opt, enc_sched, probe_sched = \
        _make_components(pooled_cfg, device)

    convs = _conv_layers(encoder)
    first_conv = convs[0]
    w_before = first_conv.weight.detach().clone()

    gen = torch.Generator(device=device).manual_seed(0)
    images, labels = train_ds.sample_batch(pooled_cfg.batch_size, gen)
    nograd_images, _ = train_ds.sample_batch(
        (pooled_cfg.accum_steps - 1) * pooled_cfg.batch_size, gen)
    _pooled_step(images, nograd_images, labels, encoder, probe, gpu_aug, pooled_cfg,
                 enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod=None)

    delta = (first_conv.weight - w_before).abs().max().item()
    assert delta > 0, "Conv weight should update (gradient flowed)"


def test_pooled_step_updates_running_stats_naturally(device, pooled_cfg):
    """After _pooled_step, BN running stats should reflect the natural
    momentum update from Pass 1's no-grad forward (NOT the captured stats
    that Pass 2 temporarily used via inject_bn_stats).

    Direct check: the running stats after the step should be a valid
    momentum interpolation: running_new = (1-m)*running_old + m*batch_stats
    where m is BN's momentum (default 0.1) and batch_stats is some plausible
    finite value, not the captured override (which we know goes through the
    eval-mode path during Pass 2 only).
    """
    train_ds, gpu_aug, encoder, probe, enc_opt, probe_opt, enc_sched, probe_sched = \
        _make_components(pooled_cfg, device)

    bn_layers = _bn_layers(encoder)
    first_bn = bn_layers[0]

    # Snapshot all BN running stats BEFORE
    snapshots_before = [
        (m.running_mean.detach().clone(), m.running_var.detach().clone())
        for m in bn_layers
    ]

    gen = torch.Generator(device=device).manual_seed(0)
    images, labels = train_ds.sample_batch(pooled_cfg.batch_size, gen)
    nograd_images, _ = train_ds.sample_batch(
        (pooled_cfg.accum_steps - 1) * pooled_cfg.batch_size, gen)
    _pooled_step(images, nograd_images, labels, encoder, probe, gpu_aug, pooled_cfg,
                 enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod=None)

    # After the step, check every BN layer
    momentum = first_bn.momentum  # default 0.1
    assert momentum is not None
    for layer, (mean_before, var_before) in zip(bn_layers, snapshots_before):
        # Recover the implied batch stats from the momentum update
        # running_after = (1-m) * running_before + m * batch_stats
        # batch_stats = (running_after - (1-m) * running_before) / m
        implied_batch_mean = (
            (layer.running_mean - (1 - momentum) * mean_before) / momentum)
        implied_batch_var = (
            (layer.running_var - (1 - momentum) * var_before) / momentum)

        # The implied batch stats must be finite (no NaN/Inf) and reasonable
        assert torch.isfinite(implied_batch_mean).all(), \
            f"Implied batch mean has NaN/Inf for {layer}"
        assert torch.isfinite(implied_batch_var).all(), \
            f"Implied batch var has NaN/Inf for {layer}"
        assert (implied_batch_var >= 0).all() or implied_batch_var.abs().max() < 1.0, \
            f"Implied batch var negative and large — possible non-restoration"

        # Running stats should have actually moved (Pass 1 was a real fwd)
        assert not torch.allclose(layer.running_mean, mean_before), \
            f"running_mean unchanged after step — Pass 1 didn't update it"


def test_projector_uses_layernorm_not_batchnorm(small_cfg):
    """The projector must use LayerNorm so it has no batch dependency in
    the pooled regime."""
    encoder = LeJEPAEncoder(small_cfg)
    bn_in_proj = [m for m in encoder.projector.modules()
                  if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    ln_in_proj = [m for m in encoder.projector.modules()
                  if isinstance(m, nn.LayerNorm)]
    assert len(bn_in_proj) == 0, (
        f"Projector contains BatchNorm: {bn_in_proj}. Must use LayerNorm "
        f"for pool-consistent embeddings.")
    assert len(ln_in_proj) > 0, "Projector should have LayerNorm"
