"""End-to-end test of one pooled training step.

Verifies that:
- The full _pooled_step runs without error and returns valid loss tensors
- BN learnable params (gamma, beta) are updated by gradient
- Conv weights are updated by gradient
- BN running stats are updated naturally by the forward passes
- The projector (LayerNorm-based) is unaffected by normalization handling
"""

import torch
import torch.nn as nn

from configs import Config
from data import get_dataloaders
from models import LeJEPAEncoder, LinearProbe
from scheduler import make_scheduler
from train_loops import _pooled_step


def _bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def _conv_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def _make_components(cfg, device):
    train_ds, _, gpu_aug = get_dataloaders(cfg, device)
    encoder = LeJEPAEncoder(cfg).to(device)
    probe_dim = encoder.hidden_dim if cfg.probe_on_emb else cfg.proj_dim
    probe = LinearProbe(probe_dim, cfg.num_classes).to(device)
    enc_opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    enc_sched = make_scheduler(enc_opt, 1, 1e-3)
    probe_sched = make_scheduler(probe_opt, 1, 1e-3)
    return train_ds, gpu_aug, encoder, probe, enc_opt, probe_opt, enc_sched, probe_sched


def test_pooled_step_runs_without_error(device, pooled_cfg):
    train_ds, gpu_aug, encoder, probe, enc_opt, probe_opt, enc_sched, probe_sched = \
        _make_components(pooled_cfg, device)

    gen = torch.Generator(device=device).manual_seed(0)
    images, labels = train_ds.sample_batch(pooled_cfg.batch_size, gen)
    nograd_images, _ = train_ds.sample_batch(
        pooled_cfg.nograd_pool_size, gen)

    loss, reg_loss, inv, probe_loss, probe_logits, \
        window_proj, window_emb, reg_emb_loss = _pooled_step(
            images, nograd_images, labels, encoder, probe, gpu_aug, pooled_cfg,
            enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod=None)

    assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"
    assert torch.isfinite(reg_loss).all()
    assert torch.isfinite(inv).all()
    assert torch.isfinite(probe_loss).all()
    assert probe_logits.shape == (pooled_cfg.batch_size, pooled_cfg.num_classes)
    # window_proj/window_emb are only returned when single_view_reg=True (FIFO mode)
    # In default mode (single_view_reg=False), they are None because FIFO is not used
    assert window_proj is None, "window_proj should be None when single_view_reg=False"
    assert window_emb is None, "window_emb should be None when single_view_reg=False"
    # reg_emb_loss should be None when lambd_emb=0
    assert reg_emb_loss is None


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
        pooled_cfg.nograd_pool_size, gen)
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
        pooled_cfg.nograd_pool_size, gen)
    _pooled_step(images, nograd_images, labels, encoder, probe, gpu_aug, pooled_cfg,
                 enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod=None)

    delta = (first_conv.weight - w_before).abs().max().item()
    assert delta > 0, "Conv weight should update (gradient flowed)"


def test_pooled_step_updates_running_stats_naturally(device, pooled_cfg):
    """After _pooled_step, BN running stats should reflect natural
    momentum updates from the forward passes.
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
        pooled_cfg.nograd_pool_size, gen)
    _pooled_step(images, nograd_images, labels, encoder, probe, gpu_aug, pooled_cfg,
                 enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod=None)

    # After the step, running stats should be finite and have moved
    for layer, (mean_before, var_before) in zip(bn_layers, snapshots_before):
        assert torch.isfinite(layer.running_mean).all(), \
            f"running_mean has NaN/Inf for {layer}"
        assert torch.isfinite(layer.running_var).all(), \
            f"running_var has NaN/Inf for {layer}"
        assert not torch.allclose(layer.running_mean, mean_before), \
            f"running_mean unchanged after step"


def test_projector_has_normalization(small_cfg):
    """The projector should have a normalization layer."""
    encoder = LeJEPAEncoder(small_cfg)
    norm_layers = [m for m in encoder.projector.modules()
                   if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm))]
    assert len(norm_layers) > 0, "Projector should have a normalization layer"


def test_pooled_step_fifo_mode_returns_window(device, fifo_cfg):
    """In FIFO mode (single_view_reg=True), window_proj should be returned."""
    train_ds, gpu_aug, encoder, probe, enc_opt, probe_opt, enc_sched, probe_sched = \
        _make_components(fifo_cfg, device)

    gen = torch.Generator(device=device).manual_seed(0)
    images, labels = train_ds.sample_batch(fifo_cfg.batch_size, gen)
    nograd_images, _ = train_ds.sample_batch(fifo_cfg.nograd_pool_size, gen)

    loss, reg_loss, inv, probe_loss, probe_logits, \
        window_proj, window_emb, reg_emb_loss = _pooled_step(
            images, nograd_images, labels, encoder, probe, gpu_aug, fifo_cfg,
            enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod=None)

    assert torch.isfinite(loss).all()
    # In FIFO mode, window_proj should be returned (N_grad + N_nograd, D)
    assert window_proj is not None, "window_proj should be returned in FIFO mode"
    expected_samples = fifo_cfg.batch_size + fifo_cfg.nograd_pool_size
    assert window_proj.shape[0] == expected_samples
    # window_emb should be None since lambd_emb=0
    assert window_emb is None
