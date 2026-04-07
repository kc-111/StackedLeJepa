"""Training epoch functions for standard and pooled modes.

Two paths:
  - ``train_epoch_standard_inmem``: when train data is an InMemoryGPUDataset
    (no DataLoader). Steps per epoch = N // batch_size.
  - ``train_epoch_standard_loader``: when train data is a DataLoader.

Both share the same per-step computation via ``_train_step``.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from data import InMemoryGPUDataset, IMAGENET_MEAN, IMAGENET_STD
from losses import inv_loss_fn, regularizer_loss
from accumulated_w1.losses import pooled_loss


# ---------------------------------------------------------------------------
# BatchNorm consistency for the pooled 2-step procedure
# ---------------------------------------------------------------------------
#
# Problem: in pooled training, the no-grad pass and grad pass forward DIFFERENT
# batches through the encoder. If BN is in train mode for both, each pass
# normalizes with its own batch stats — the two halves of the pool are
# normalized differently, breaking the assumption that the regularizer's
# inputs are i.i.d. samples from one distribution.
#
# The fix: capture batch stats during the no-grad pass (train mode, large
# (T-1)*BS batch), then inject them as the running stats for the grad pass
# in eval mode. Both passes are normalized with the SAME stats. The natural
# running-stat update from the no-grad pass is preserved.
#
# After the grad pass, the BN running stats are restored to whatever they
# were after Pass 1 (which is what we want — that's the natural training
# update from the larger, more accurate (T-1)*BS batch).


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


class _BNStatCapture:
    """Captures batch stats from BN layers during a forward pass.

    Registers a forward hook on every BN layer in the model. Each hook
    saves the batch mean/var that the layer just used to normalize its
    input (computed from the layer's input, not output).

    Usage:
        capture = _BNStatCapture(model)
        with capture:
            model(x)  # forwards collect stats
        capture.captured  # dict[id(bn_layer)] = (mean, var)
    """

    def __init__(self, model):
        self.model = model
        self.captured: dict = {}
        self._handles: list = []

    def _make_hook(self, module):
        def hook(_mod, inputs, _output):
            x = inputs[0]
            # Compute batch stats the same way nn.BatchNorm does internally:
            # mean and biased variance over (N, *spatial) per channel.
            if x.dim() == 4:
                dims = (0, 2, 3)
            elif x.dim() == 3:
                dims = (0, 2)
            elif x.dim() == 2:
                dims = (0,)
            else:
                dims = tuple(d for d in range(x.dim()) if d != 1)
            with torch.no_grad():
                mean = x.mean(dim=dims).detach().to(module.running_mean.dtype)
                var = x.var(dim=dims, unbiased=False).detach().to(module.running_var.dtype)
            self.captured[id(module)] = (mean, var)
        return hook

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, _BN_TYPES):
                self._handles.append(m.register_forward_hook(self._make_hook(m)))
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()


@contextlib.contextmanager
def inject_bn_stats(model, captured: dict):
    """Temporarily override BN running_mean/running_var with captured stats
    and put BN in eval mode. Restores everything on exit.

    Use after _BNStatCapture has populated ``captured``: this swaps the
    captured batch stats into the BN layers so a subsequent forward pass
    in eval mode normalizes with exactly those stats.
    """
    saved: list = []
    for m in model.modules():
        if isinstance(m, _BN_TYPES) and id(m) in captured:
            saved.append((
                m,
                m.running_mean.detach().clone(),
                m.running_var.detach().clone(),
                m.training,
            ))
            new_mean, new_var = captured[id(m)]
            m.running_mean.copy_(new_mean)
            m.running_var.copy_(new_var)
            m.eval()  # use running stats for normalization
    try:
        yield
    finally:
        for m, mean, var, was_training in saved:
            m.running_mean.copy_(mean)
            m.running_var.copy_(var)
            if was_training:
                m.train()


# ---------------------------------------------------------------------------
# Multi-view encoder forward
# ---------------------------------------------------------------------------

def _encode_views(encoder, global_views, local_views):
    """Forward all global+local views through the encoder.

    Global and local views may have different spatial sizes, so they're
    forwarded in two batches and the results are concatenated along the
    view dimension (V_g first, then V_l).

    Args:
        encoder: LeJEPAEncoder.
        global_views: (V_g, N, C, gs, gs)
        local_views:  (V_l, N, C, ls, ls) or None

    Returns:
        emb:  (V, N, hidden_dim) — concat of global+local
        proj: (V, N, proj_dim)
    """
    Vg, N = global_views.shape[0], global_views.shape[1]
    g_flat = global_views.reshape(Vg * N, *global_views.shape[2:])
    g_emb, g_proj = encoder(g_flat)
    g_emb = g_emb.view(Vg, N, -1)
    g_proj = g_proj.view(Vg, N, -1)

    if local_views is None:
        return g_emb, g_proj

    Vl = local_views.shape[0]
    l_flat = local_views.reshape(Vl * N, *local_views.shape[2:])
    l_emb, l_proj = encoder(l_flat)
    l_emb = l_emb.view(Vl, N, -1)
    l_proj = l_proj.view(Vl, N, -1)

    emb = torch.cat([g_emb, l_emb], dim=0)
    proj = torch.cat([g_proj, l_proj], dim=0)
    return emb, proj


# ---------------------------------------------------------------------------
# Per-step computation (shared by both training paths)
# ---------------------------------------------------------------------------

def _train_step(images, labels, encoder, probe, reg_fn, gpu_aug, cfg,
                enc_opt, probe_opt, enc_sched, probe_sched):
    """One optimizer step. ``images`` may be uint8 (will be cast)."""
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)

    global_views, local_views = gpu_aug(images)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        emb, proj = _encode_views(encoder, global_views, local_views)
        # Center from global views only (paper Alg 2)
        reg_loss = regularizer_loss(reg_fn, proj)
        inv = inv_loss_fn(proj, num_global_views=cfg.num_global_views)

        # Probe on the global views' mean embedding (detached)
        emb_global_mean = emb[:cfg.num_global_views].mean(0)  # (N, hidden_dim)
        probe_logits = probe(emb_global_mean.detach())
        probe_loss = F.cross_entropy(probe_logits, labels)

        loss = cfg.lambd * reg_loss + (1 - cfg.lambd) * inv + probe_loss

    enc_opt.zero_grad(set_to_none=True)
    probe_opt.zero_grad(set_to_none=True)
    loss.backward()
    enc_opt.step()
    probe_opt.step()
    enc_sched.step()
    probe_sched.step()

    return loss.detach(), reg_loss.detach(), inv.detach(), probe_loss.detach(), probe_logits.detach()


def _log(epoch, step, loss, reg_loss, inv, probe_loss, probe_logits, labels, lr):
    probe_acc = (probe_logits.argmax(1) == labels).float().mean().item()
    print(f"  epoch {epoch} step {step} | "
          f"loss {loss.item():.4f} reg {reg_loss.item():.4f} "
          f"inv {inv.item():.4f} probe_loss {probe_loss.item():.4f} "
          f"probe_acc {probe_acc:.3f} lr {lr:.6f}",
          flush=True)


# ---------------------------------------------------------------------------
# Standard training — in-memory GPU dataset
# ---------------------------------------------------------------------------

def train_epoch_standard_inmem(epoch, encoder, probe, reg_fn, train_dataset,
                               gpu_aug, enc_opt, probe_opt,
                               enc_sched, probe_sched, cfg, global_step,
                               generator):
    encoder.train()
    probe.train()
    total_loss_sum = 0.0
    num_steps = len(train_dataset) // cfg.batch_size

    for step in range(num_steps):
        images, labels = train_dataset.sample_batch(cfg.batch_size, generator)
        loss, reg_loss, inv, probe_loss, probe_logits = _train_step(
            images, labels, encoder, probe, reg_fn, gpu_aug, cfg,
            enc_opt, probe_opt, enc_sched, probe_sched)

        total_loss_sum += loss.item()
        global_step += 1

        if (step + 1) % cfg.log_interval == 0:
            lr = enc_sched.get_last_lr()[0]
            _log(epoch, step + 1, loss, reg_loss, inv, probe_loss,
                 probe_logits, labels, lr)

    return total_loss_sum / max(num_steps, 1), global_step


# ---------------------------------------------------------------------------
# Standard training — DataLoader path (large datasets)
# ---------------------------------------------------------------------------

def train_epoch_standard_loader(epoch, encoder, probe, reg_fn, train_loader,
                                gpu_aug, enc_opt, probe_opt,
                                enc_sched, probe_sched, cfg, global_step):
    encoder.train()
    probe.train()
    device = next(encoder.parameters()).device
    total_loss_sum = 0.0
    num_steps = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss, reg_loss, inv, probe_loss, probe_logits = _train_step(
            images, labels, encoder, probe, reg_fn, gpu_aug, cfg,
            enc_opt, probe_opt, enc_sched, probe_sched)

        total_loss_sum += loss.item()
        num_steps += 1
        global_step += 1

        if num_steps % cfg.log_interval == 0:
            lr = enc_sched.get_last_lr()[0]
            _log(epoch, num_steps, loss, reg_loss, inv, probe_loss,
                 probe_logits, labels, lr)

    return total_loss_sum / max(num_steps, 1), global_step


# ---------------------------------------------------------------------------
# Pooled training (2-step)
# ---------------------------------------------------------------------------

def make_nograd_loader(train_dataset, nograd_bs, num_workers):
    """No-grad CDF context loader (with replacement).

    Used by the pooled training path on disk-backed datasets only.
    For in-memory datasets, draw the no-grad context directly via
    ``InMemoryGPUDataset.sample_batch``.
    """
    sampler = RandomSampler(train_dataset, replacement=True)
    return DataLoader(
        train_dataset, batch_size=nograd_bs, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0)


def _pooled_step(images, nograd_images, labels, encoder, probe, gpu_aug, cfg,
                 enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod):
    """One pooled (2-step) optimizer step with BN-consistent normalization.

    Step 1 (no-grad, train mode): forward (T-1)*BS samples through the
        encoder. BN computes batch stats from this large batch and updates
        running stats with momentum from the more accurate (T-1)*BS estimate.
        We capture the per-layer batch stats via forward hooks.

    Step 2 (grad, eval mode + injected stats): override each BN layer's
        running stats with the captured Pass-1 batch stats and switch BN
        to eval mode. Forward BS grad samples — they're normalized with
        EXACTLY the same stats Pass 1 used. ng_proj and proj are now
        consistent in the pool.

    After Step 2 the BN running stats are restored to their post-Pass-1
    values (the natural training update from the larger no-grad batch).
    """
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
    if nograd_images.dtype == torch.uint8:
        nograd_images = nograd_images.float().div_(255.0)

    global_views, local_views = gpu_aug(images)
    ng_global, ng_local = gpu_aug(nograd_images)

    # Step 1: no-grad pass through encoder in TRAIN mode → BN computes batch
    # stats from the (T-1)*BS samples and updates running stats. We capture
    # the per-layer batch stats so Step 2 can use the SAME normalization.
    capture = _BNStatCapture(encoder)
    with torch.no_grad(), capture:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, ng_proj = _encode_views(encoder, ng_global, ng_local)
    ng_proj = ng_proj.detach()

    # Step 2: grad pass with BN in EVAL mode but running stats overridden
    # to the Pass-1 batch stats. proj is normalized identically to ng_proj.
    with inject_bn_stats(encoder, capture.captured):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb, proj = _encode_views(encoder, global_views, local_views)
            reg_loss = pooled_loss(
                live=proj, detached=ng_proj,
                mode=cfg.regularizer, num_proj=cfg.num_proj,
                sigreg=sigreg_mod)
            inv = inv_loss_fn(proj, num_global_views=cfg.num_global_views)
            emb_global_mean = emb[:cfg.num_global_views].mean(0)
            probe_logits = probe(emb_global_mean.detach())
            probe_loss = F.cross_entropy(probe_logits, labels)
            loss = cfg.lambd * reg_loss + (1 - cfg.lambd) * inv + probe_loss

        enc_opt.zero_grad(set_to_none=True)
        probe_opt.zero_grad(set_to_none=True)
        loss.backward()

    # Optimizer step happens outside the inject context — by now backward is
    # done and the BN running stats are restored to their post-Pass-1 values.
    enc_opt.step()
    probe_opt.step()
    enc_sched.step()
    probe_sched.step()

    return (loss.detach(), reg_loss.detach(), inv.detach(),
            probe_loss.detach(), probe_logits.detach())


def train_epoch_pooled_inmem(epoch, encoder, probe, train_dataset,
                             gpu_aug, enc_opt, probe_opt, enc_sched, probe_sched,
                             cfg, global_step, generator, sigreg_mod=None):
    """2-step pooled training (in-memory GPU dataset path).

    Each step draws BS grad samples + (T-1)*BS no-grad samples directly
    from the cached tensor — no DataLoader, no second loader.
    """
    encoder.train()
    probe.train()
    total_loss_sum = 0.0
    nograd_bs = (cfg.accum_steps - 1) * cfg.batch_size
    num_steps = len(train_dataset) // cfg.batch_size

    for step in range(num_steps):
        images, labels = train_dataset.sample_batch(cfg.batch_size, generator)
        nograd_images, _ = train_dataset.sample_batch(nograd_bs, generator)
        loss, reg_loss, inv, probe_loss, probe_logits = _pooled_step(
            images, nograd_images, labels, encoder, probe, gpu_aug, cfg,
            enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod)

        total_loss_sum += loss.item()
        global_step += 1

        if (step + 1) % cfg.log_interval == 0:
            lr = enc_sched.get_last_lr()[0]
            _log(epoch, step + 1, loss, reg_loss, inv, probe_loss,
                 probe_logits, labels, lr)

    return total_loss_sum / max(num_steps, 1), global_step




def train_epoch_pooled_loader(epoch, encoder, probe, train_loader,
                              gpu_aug, nograd_loader, nograd_iter_state,
                              enc_opt, probe_opt, enc_sched, probe_sched, cfg,
                              global_step, sigreg_mod=None):
    """2-step pooled training (DataLoader path)."""
    encoder.train()
    probe.train()
    device = next(encoder.parameters()).device
    total_loss_sum = 0.0
    num_steps = 0

    nograd_iter = nograd_iter_state[0]

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        try:
            nograd_images, _ = next(nograd_iter)
        except StopIteration:
            nograd_iter = iter(nograd_loader)
            nograd_images, _ = next(nograd_iter)
        nograd_images = nograd_images.to(device, non_blocking=True)

        loss, reg_loss, inv, probe_loss, probe_logits = _pooled_step(
            images, nograd_images, labels, encoder, probe, gpu_aug, cfg,
            enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod)

        total_loss_sum += loss.item()
        num_steps += 1
        global_step += 1

        if num_steps % cfg.log_interval == 0:
            lr = enc_sched.get_last_lr()[0]
            _log(epoch, num_steps, loss, reg_loss, inv, probe_loss,
                 probe_logits, labels, lr)

    nograd_iter_state[0] = nograd_iter
    return total_loss_sum / max(num_steps, 1), global_step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(encoder, probe, val_loader, cfg):
    encoder.eval()
    probe.eval()
    device = next(encoder.parameters()).device

    correct = total = 0
    for images, labels in val_loader:
        if images.device != device:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        # If raw uint8 from a DataLoader, normalize on the fly
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
            mean = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
            std = IMAGENET_STD.to(device).view(1, 3, 1, 1)
            images = (images - mean) / std

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb, _ = encoder(images)
            logits = probe(emb)

        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)
