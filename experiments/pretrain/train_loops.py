"""Training epoch functions for standard and pooled modes.

Two paths:
  - ``train_epoch_standard_inmem``: when train data is an InMemoryGPUDataset
    (no DataLoader). Steps per epoch = N // batch_size.
  - ``train_epoch_standard_loader``: when train data is a DataLoader.

Both share the same per-step computation via ``_train_step``.
"""

import math
from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from data import InMemoryGPUDataset
from losses import inv_loss_fn
from sliced_gauss_reg.losses import pooled_loss


class FIFOBuffer:
    """Fixed-capacity FIFO buffer for projection tensors, tracking sample count."""

    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self._queue = deque()
        self._total_samples = 0

    def enqueue(self, proj: torch.Tensor) -> None:
        """Add projection tensor to buffer, evicting oldest if over capacity."""
        n_samples = proj.shape[-2]
        self._queue.append(proj)
        self._total_samples += n_samples
        # Dequeue oldest until under capacity
        while self._total_samples > self.max_samples and len(self._queue) > 1:
            oldest = self._queue.popleft()
            self._total_samples -= oldest.shape[-2]

    def get(self) -> torch.Tensor | None:
        """Return concatenated buffer contents, or None if empty."""
        if not self._queue:
            return None
        return torch.cat(list(self._queue), dim=-2)

    def __len__(self) -> int:
        return self._total_samples


def _probe_loss(probe_logits, labels):
    # ignore_index=-1 handles STL10's unlabeled split (label=-1). If a batch
    # happens to contain zero labeled samples, cross_entropy returns NaN under
    # reduction='mean'; fall back to a zero loss that still flows grad through
    # the probe so the optimizer step is well-defined.
    loss = F.cross_entropy(probe_logits, labels, ignore_index=-1)
    if torch.isnan(loss):
        return probe_logits.sum() * 0.0
    return loss


def _combine_loss(reg_loss, inv, probe_loss, cfg, reg_emb_loss=None):
    loss = cfg.lambd * reg_loss + (1.0 - cfg.lambd) * inv + probe_loss
    if reg_emb_loss is not None and cfg.lambd_emb > 0:
        loss = loss + cfg.lambd_emb * reg_emb_loss
    return loss


# ---------------------------------------------------------------------------
# View encoder forward
# ---------------------------------------------------------------------------

def _encode_orig_and_aug(encoder, orig, aug_views):
    """Forward the original and all augmented views through the encoder.

    ``orig`` and ``aug_views`` share the same spatial size, so they're
    concatenated into a single (1+V)*N forward.

    Args:
        encoder: LeJEPAEncoder.
        orig: (N, C, H, W)
        aug_views: (V, N, C, H, W)

    Returns:
        orig_emb:  (N, hidden_dim)
        orig_proj: (N, proj_dim)
        aug_emb:   (V, N, hidden_dim)
        aug_proj:  (V, N, proj_dim)
    """
    V, N = aug_views.shape[0], aug_views.shape[1]
    flat = torch.cat([orig, aug_views.reshape(V * N, *aug_views.shape[2:])], dim=0)
    emb_flat, proj_flat = encoder(flat)
    orig_emb, aug_emb_flat = emb_flat[:N], emb_flat[N:]
    orig_proj, aug_proj_flat = proj_flat[:N], proj_flat[N:]
    aug_emb = aug_emb_flat.view(V, N, -1)
    aug_proj = aug_proj_flat.view(V, N, -1)
    return orig_emb, orig_proj, aug_emb, aug_proj


# ---------------------------------------------------------------------------
# Per-step computation (shared by both training paths)
# ---------------------------------------------------------------------------

def _make_nograd_inv_views(images, encoder, gpu_aug, cfg):
    """Generate extra no-grad augmented views for invariance centroid.

    Applies ``cfg.num_inv_nograd_views`` independent augmentations to
    ``images``, forwards them without gradients, and returns the
    detached embeddings/projections stacked as ``(V_n, N, D)``.

    Returns ``(nograd_emb, nograd_proj)`` or ``(None, None)`` if disabled.
    """
    V_n = cfg.num_inv_nograd_views
    if V_n <= 0:
        return None, None
    parts_emb, parts_proj = [], []
    for _ in range(V_n):
        _, aug_v = gpu_aug(images)  # (1, N, C, H, W) when num_aug_views=1
        flat = aug_v.reshape(-1, *aug_v.shape[2:])
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb_i, proj_i = encoder(flat)
        # aug_v may have >1 views; reshape to (V_aug, N, D)
        V_aug, N = aug_v.shape[0], aug_v.shape[1]
        parts_emb.append(emb_i.view(V_aug, N, -1).detach())
        parts_proj.append(proj_i.view(V_aug, N, -1).detach())
    return torch.cat(parts_emb, dim=0), torch.cat(parts_proj, dim=0)


def _inv_with_nograd(grad_views, nograd_views, cfg):
    """Compute invariance loss, optionally using no-grad views for centroid."""
    return inv_loss_fn(grad_views, nograd_views=nograd_views,
                       detach_centroid=cfg.detach_inv_centroid)


def _train_step(images, labels, encoder, probe, reg_fn, gpu_aug, cfg,
                enc_opt, probe_opt, enc_sched, probe_sched, reg_emb_fn=None):
    """One optimizer step. ``images`` may be uint8 (will be cast)."""
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)

    orig, aug_views = gpu_aug(images)

    # Extra no-grad views for invariance centroid (cheap forward-only passes)
    ng_inv_emb, ng_inv_proj = _make_nograd_inv_views(
        images, encoder, gpu_aug, cfg)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        orig_emb, orig_proj, aug_emb, aug_proj = _encode_orig_and_aug(
            encoder, orig, aug_views)

        # Regularization on augmented views only (original has different distribution).
        # single_view_reg: 1 random view per image (N, D) - required for FIFO i.i.d.
        # otherwise: all V aug views (V, N, D) - V gradient signals
        if cfg.single_view_reg:
            V = aug_proj.shape[0]
            view_idx = torch.randint(V, (1,), device=aug_proj.device).item()
            reg_proj = aug_proj[view_idx]  # (N, D)
            reg_emb_selected = aug_emb[view_idx] if cfg.lambd_emb > 0 else None
        else:
            reg_proj = aug_proj  # (V, N, D)
            reg_emb_selected = aug_emb if cfg.lambd_emb > 0 else None
        reg_loss = reg_fn(reg_proj)

        # Optional embedding regularization
        reg_emb_loss = None
        if reg_emb_fn is not None and cfg.lambd_emb > 0:
            reg_emb_loss = reg_emb_fn(reg_emb_selected)

        if cfg.inv_on_emb:
            inv = _inv_with_nograd(aug_emb, ng_inv_emb, cfg)
        else:
            inv = _inv_with_nograd(aug_proj, ng_inv_proj, cfg)

        # Probe on the clean original
        probe_input = orig_emb if cfg.probe_on_emb else orig_proj
        probe_logits = probe(probe_input.detach())
        probe_loss = _probe_loss(probe_logits, labels)

        loss = _combine_loss(reg_loss, inv, probe_loss, cfg, reg_emb_loss)

    enc_opt.zero_grad(set_to_none=True)
    probe_opt.zero_grad(set_to_none=True)
    loss.backward()
    enc_opt.step()
    probe_opt.step()
    enc_sched.step()
    probe_sched.step()

    reg_emb_det = reg_emb_loss.detach() if reg_emb_loss is not None else None
    return (loss.detach(), reg_loss.detach(), inv.detach(),
            probe_loss.detach(), probe_logits.detach(), reg_emb_det)


def _log(epoch, step, loss, reg_loss, inv, probe_loss, probe_logits, labels,
         lr, cfg, reg_emb_loss=None):
    valid = labels != -1
    if valid.any():
        probe_acc = (probe_logits[valid].argmax(1) == labels[valid]).float().mean().item()
    else:
        probe_acc = float("nan")
    tag = "emb" if cfg.probe_on_emb else "proj"
    reg_emb_str = f" reg_emb {reg_emb_loss.item():.4f}" if reg_emb_loss is not None else ""
    print(f"  epoch {epoch} step {step} | "
          f"loss {loss.item():.4f} reg {reg_loss.item():.4f}{reg_emb_str} "
          f"inv {inv.item():.4f} probe_loss {probe_loss.item():.4f} "
          f"probe_acc({tag}) {probe_acc:.3f} lr {lr:.6f}",
          flush=True)


# ---------------------------------------------------------------------------
# Standard training — in-memory GPU dataset
# ---------------------------------------------------------------------------

def train_epoch_standard_inmem(epoch, encoder, probe, reg_fn, train_dataset,
                               gpu_aug, enc_opt, probe_opt,
                               enc_sched, probe_sched, cfg, global_step,
                               generator, reg_emb_fn=None):
    encoder.train()
    probe.train()
    total_loss_sum = 0.0
    num_steps = 0

    for images, labels in train_dataset.epoch_batches(cfg.batch_size, generator):
        loss, reg_loss, inv, probe_loss, probe_logits, reg_emb_loss = _train_step(
            images, labels, encoder, probe, reg_fn, gpu_aug, cfg,
            enc_opt, probe_opt, enc_sched, probe_sched, reg_emb_fn)

        total_loss_sum += loss.item()
        num_steps += 1
        global_step += 1

        if num_steps % cfg.log_interval == 0:
            lr = enc_sched.get_last_lr()[0]
            _log(epoch, num_steps, loss, reg_loss, inv, probe_loss,
                 probe_logits, labels, lr, cfg, reg_emb_loss)

    return total_loss_sum / max(num_steps, 1), global_step


# ---------------------------------------------------------------------------
# Standard training — DataLoader path (large datasets)
# ---------------------------------------------------------------------------

def train_epoch_standard_loader(epoch, encoder, probe, reg_fn, train_loader,
                                gpu_aug, enc_opt, probe_opt,
                                enc_sched, probe_sched, cfg, global_step,
                                reg_emb_fn=None):
    encoder.train()
    probe.train()
    device = next(encoder.parameters()).device
    total_loss_sum = 0.0
    num_steps = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss, reg_loss, inv, probe_loss, probe_logits, reg_emb_loss = _train_step(
            images, labels, encoder, probe, reg_fn, gpu_aug, cfg,
            enc_opt, probe_opt, enc_sched, probe_sched, reg_emb_fn)

        total_loss_sum += loss.item()

        num_steps += 1
        global_step += 1

        if num_steps % cfg.log_interval == 0:
            lr = enc_sched.get_last_lr()[0]
            _log(epoch, num_steps, loss, reg_loss, inv, probe_loss,
                 probe_logits, labels, lr, cfg, reg_emb_loss)

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
                 enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod,
                 fifo_proj=None, sigreg_emb_mod=None, fifo_emb=None):
    """One pooled (2-step) optimizer step.

    Two modes based on single_view_reg:

    1. single_view_reg=False (no FIFO): Combine grad+nograd images, apply
       augmentation once for same params. Use all V aug views → V gradient
       signals. i.i.d. within each view.

    2. single_view_reg=True (FIFO allowed): Select 1 random aug view per image.
       FIFO stores (N, D). All samples from same distribution → i.i.d.
       1 gradient signal.

    Returns loss tuple plus ``window_proj`` and ``window_emb`` for FIFO.
    """
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)

    N_grad = images.shape[0]

    # Extra no-grad views for invariance centroid
    ng_inv_emb, ng_inv_proj = _make_nograd_inv_views(
        images, encoder, gpu_aug, cfg)

    if cfg.single_view_reg:
        # === FIFO mode: 1 random view per image ===
        orig, aug_views = gpu_aug(images)
        V = aug_views.shape[0]
        view_idx = torch.randint(V, (1,), device=aug_views.device).item()

        # No-grad pool (separate augmentation is OK - we select 1 random view)
        ng_proj = None
        ng_emb = None
        if nograd_images is not None:
            if nograd_images.dtype == torch.uint8:
                nograd_images = nograd_images.float().div_(255.0)
            ng_orig, ng_aug = gpu_aug(nograd_images)
            ng_view_idx = torch.randint(V, (1,), device=ng_aug.device).item()
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, _, ng_aug_emb, ng_aug_proj = _encode_orig_and_aug(
                        encoder, ng_orig, ng_aug)
            ng_proj = ng_aug_proj[ng_view_idx].detach()  # (N_nograd, D)
            if cfg.lambd_emb > 0:
                ng_emb = ng_aug_emb[ng_view_idx].detach()

        # Build detached pool: FIFO + no-grad, all (N_i, D)
        detached_proj_parts = []
        if fifo_proj is not None:
            detached_proj_parts.append(fifo_proj)
        if ng_proj is not None:
            detached_proj_parts.append(ng_proj)

        detached_emb_parts = []
        if cfg.lambd_emb > 0:
            if fifo_emb is not None:
                detached_emb_parts.append(fifo_emb)
            if ng_emb is not None:
                detached_emb_parts.append(ng_emb)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            orig_emb, orig_proj, aug_emb, aug_proj = _encode_orig_and_aug(
                encoder, orig, aug_views)

            live_proj = aug_proj[view_idx]  # (N_grad, D)
            detached_proj = torch.cat(detached_proj_parts, dim=0) if detached_proj_parts else live_proj.new_empty(0, live_proj.shape[-1])
            reg_loss = pooled_loss(live=live_proj, detached=detached_proj,
                                   mode=cfg.regularizer, num_proj=cfg.num_proj, sigreg=sigreg_mod)

            reg_emb_loss = None
            if cfg.lambd_emb > 0 and sigreg_emb_mod is not None:
                live_emb = aug_emb[view_idx]
                detached_emb = torch.cat(detached_emb_parts, dim=0) if detached_emb_parts else live_emb.new_empty(0, live_emb.shape[-1])
                reg_emb_loss = pooled_loss(live=live_emb, detached=detached_emb,
                                           mode=cfg.regularizer, num_proj=cfg.num_proj, sigreg=sigreg_emb_mod)

            if cfg.inv_on_emb:
                inv = _inv_with_nograd(aug_emb, ng_inv_emb, cfg)
            else:
                inv = _inv_with_nograd(aug_proj, ng_inv_proj, cfg)
            probe_input = orig_emb if cfg.probe_on_emb else orig_proj
            probe_logits = probe(probe_input.detach())
            probe_loss = _probe_loss(probe_logits, labels)
            loss = _combine_loss(reg_loss, inv, probe_loss, cfg, reg_emb_loss)

        # Window for FIFO: 1 view from grad + no-grad
        window_parts = [aug_proj[view_idx].detach()]
        if ng_proj is not None:
            window_parts.append(ng_proj)
        window_proj = torch.cat(window_parts, dim=0)

        window_emb = None
        if cfg.lambd_emb > 0:
            emb_parts = [aug_emb[view_idx].detach()]
            if ng_emb is not None:
                emb_parts.append(ng_emb)
            window_emb = torch.cat(emb_parts, dim=0)

    else:
        # === No FIFO: combine images, same augmentation, all V views ===
        if nograd_images is not None:
            if nograd_images.dtype == torch.uint8:
                nograd_images = nograd_images.float().div_(255.0)
            combined = torch.cat([images, nograd_images], dim=0)
            N_nograd = nograd_images.shape[0]
        else:
            combined = images
            N_nograd = 0

        combined_orig, combined_aug = gpu_aug(combined)
        orig = combined_orig[:N_grad]
        aug_views = combined_aug[:, :N_grad]
        if N_nograd > 0:
            ng_orig = combined_orig[N_grad:]
            ng_aug = combined_aug[:, N_grad:]
        else:
            ng_orig, ng_aug = None, None

        # No-grad forward
        ng_proj = None
        ng_emb = None
        if ng_aug is not None:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, _, ng_aug_emb, ng_aug_proj = _encode_orig_and_aug(
                        encoder, ng_orig, ng_aug)
            ng_proj = ng_aug_proj.detach()  # (V, N_nograd, D)
            if cfg.lambd_emb > 0:
                ng_emb = ng_aug_emb.detach()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            orig_emb, orig_proj, aug_emb, aug_proj = _encode_orig_and_aug(
                encoder, orig, aug_views)

            # Regularize aug views only (V, N, D), pool along sample dim
            detached_proj = ng_proj if ng_proj is not None else aug_proj.new_empty(aug_proj.shape[0], 0, aug_proj.shape[-1])
            reg_loss = pooled_loss(live=aug_proj, detached=detached_proj,
                                   mode=cfg.regularizer, num_proj=cfg.num_proj, sigreg=sigreg_mod)

            reg_emb_loss = None
            if cfg.lambd_emb > 0 and sigreg_emb_mod is not None:
                detached_emb = ng_emb if ng_emb is not None else aug_emb.new_empty(aug_emb.shape[0], 0, aug_emb.shape[-1])
                reg_emb_loss = pooled_loss(live=aug_emb, detached=detached_emb,
                                           mode=cfg.regularizer, num_proj=cfg.num_proj, sigreg=sigreg_emb_mod)

            if cfg.inv_on_emb:
                inv = _inv_with_nograd(aug_emb, ng_inv_emb, cfg)
            else:
                inv = _inv_with_nograd(aug_proj, ng_inv_proj, cfg)
            probe_input = orig_emb if cfg.probe_on_emb else orig_proj
            probe_logits = probe(probe_input.detach())
            probe_loss = _probe_loss(probe_logits, labels)
            loss = _combine_loss(reg_loss, inv, probe_loss, cfg, reg_emb_loss)

        # No FIFO in this mode
        window_proj = None
        window_emb = None

    enc_opt.zero_grad(set_to_none=True)
    probe_opt.zero_grad(set_to_none=True)
    loss.backward()
    enc_opt.step()
    probe_opt.step()
    enc_sched.step()
    probe_sched.step()

    reg_emb_det = reg_emb_loss.detach() if reg_emb_loss is not None else None
    return (loss.detach(), reg_loss.detach(), inv.detach(),
            probe_loss.detach(), probe_logits.detach(),
            window_proj, window_emb, reg_emb_det)


def train_epoch_pooled_inmem(epoch, encoder, probe, train_dataset,
                             gpu_aug, enc_opt, probe_opt, enc_sched, probe_sched,
                             cfg, global_step, generator, sigreg_mod=None,
                             fifo=None, sigreg_emb_mod=None, fifo_emb=None):
    """2-step pooled training (in-memory GPU dataset path).

    Grad images are drawn without replacement (epoch_batches), cycling
    through all data each epoch. No-grad context is sampled with
    replacement via sample_batch.

    If ``cfg.fifo_size > 0``, pass ``FIFOBuffer`` instances to retain
    past window projections/embeddings across epochs.
    """
    encoder.train()
    probe.train()
    total_loss_sum = 0.0
    nograd_bs = cfg.nograd_pool_size
    num_steps = 0

    for images, labels in train_dataset.epoch_batches(cfg.batch_size, generator):
        if nograd_bs > 0:
            nograd_images, _ = train_dataset.sample_batch(nograd_bs, generator)
        else:
            nograd_images = None

        fifo_proj_data = fifo.get() if fifo else None
        fifo_emb_data = fifo_emb.get() if fifo_emb else None

        loss, reg_loss, inv, probe_loss, probe_logits, \
            window_proj, window_emb, reg_emb_loss = _pooled_step(
                images, nograd_images, labels, encoder, probe, gpu_aug, cfg,
                enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod,
                fifo_proj=fifo_proj_data, sigreg_emb_mod=sigreg_emb_mod,
                fifo_emb=fifo_emb_data)

        if fifo is not None:
            fifo.enqueue(window_proj)
        if fifo_emb is not None and window_emb is not None:
            fifo_emb.enqueue(window_emb)

        total_loss_sum += loss.item()
        num_steps += 1
        global_step += 1

        if num_steps % cfg.log_interval == 0:
            lr = enc_sched.get_last_lr()[0]
            _log(epoch, num_steps, loss, reg_loss, inv, probe_loss,
                 probe_logits, labels, lr, cfg, reg_emb_loss)

    return total_loss_sum / max(num_steps, 1), global_step




def train_epoch_pooled_loader(epoch, encoder, probe, train_loader,
                              gpu_aug, nograd_loader, nograd_iter_state,
                              enc_opt, probe_opt, enc_sched, probe_sched, cfg,
                              global_step, sigreg_mod=None, fifo=None,
                              sigreg_emb_mod=None, fifo_emb=None):
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
        if nograd_loader is not None:
            try:
                nograd_images, _ = next(nograd_iter)
            except StopIteration:
                nograd_iter = iter(nograd_loader)
                nograd_images, _ = next(nograd_iter)
            nograd_images = nograd_images.to(device, non_blocking=True)
        else:
            nograd_images = None

        fifo_proj_data = fifo.get() if fifo else None
        fifo_emb_data = fifo_emb.get() if fifo_emb else None

        loss, reg_loss, inv, probe_loss, probe_logits, \
            window_proj, window_emb, reg_emb_loss = _pooled_step(
                images, nograd_images, labels, encoder, probe, gpu_aug, cfg,
                enc_opt, probe_opt, enc_sched, probe_sched, sigreg_mod,
                fifo_proj=fifo_proj_data, sigreg_emb_mod=sigreg_emb_mod,
                fifo_emb=fifo_emb_data)

        if fifo is not None:
            fifo.enqueue(window_proj)
        if fifo_emb is not None and window_emb is not None:
            fifo_emb.enqueue(window_emb)

        total_loss_sum += loss.item()

        num_steps += 1
        global_step += 1

        if num_steps % cfg.log_interval == 0:
            lr = enc_sched.get_last_lr()[0]
            _log(epoch, num_steps, loss, reg_loss, inv, probe_loss,
                 probe_logits, labels, lr, cfg, reg_emb_loss)

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
        # If raw uint8 from a DataLoader, normalize to [-1, 1]
        if images.dtype == torch.uint8:
            images = images.float().div_(127.5).sub_(1.0)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb, proj = encoder(images)
            probe_input = emb if cfg.probe_on_emb else proj
            logits = probe(probe_input)

        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


def _effective_rank(x: torch.Tensor, eps: float = 1e-12) -> float:
    """Roy-Vetterli effective rank of an (N, D) batch.

    rank_eff = exp(H(p)), where p_i = σ_i / Σ_j σ_j and σ_i are the
    singular values of the centered batch. Equals D for an isotropic
    full-rank batch and equals 1 when all variance is in one direction.
    Cheaper-than-SVD path: eigvalsh of the D×D centered Gram matrix.
    """
    x = x.float()
    n, d = x.shape
    if n < 2 or d == 0:
        return float("nan")
    xc = x - x.mean(dim=0, keepdim=True)
    cov = xc.T @ xc / max(n - 1, 1)
    evals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    sigma = evals.sqrt()
    total = sigma.sum()
    if total <= eps:
        return float("nan")
    p = sigma / total
    p = p[p > eps]
    if p.numel() == 0:
        return float("nan")
    return float((-(p * p.log()).sum()).exp())


def _distributional_metrics(x: torch.Tensor, num_proj: int = 1024) -> dict:
    """Compare ``(N, D)`` embeddings to ``N(0, I_D)``.

    Returns sliced W1/W2 to N(0,1), per-sample mean magnitude, and
    covariance distance to identity. Computed in float32 on the same
    device as ``x``. Uses a fixed seed so the random projection set is
    consistent across calls (lets you compare runs).
    """
    x = x.float()
    n, d = x.shape
    g = torch.Generator(device=x.device).manual_seed(999)
    A = torch.randn(d, num_proj, device=x.device, generator=g)
    A = A / A.norm(dim=0, keepdim=True)
    proj_sorted = torch.sort(x @ A, dim=0).values  # (N, num_proj)
    p = (torch.arange(1, n + 1, device=x.device, dtype=x.dtype) - 0.5) / n
    ref = (torch.erfinv(2 * p - 1) * math.sqrt(2)).unsqueeze(1)
    diff = proj_sorted - ref
    w1 = diff.abs().mean().item()
    w2 = (diff.square().mean(dim=0).sqrt()).mean().item()

    mu = x.mean(dim=0)
    xc = x - mu
    cov = xc.T @ xc / max(n - 1, 1)
    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    cov_frob_rel = ((cov - eye).norm() / eye.norm()).item()
    diag = torch.diagonal(cov)
    off = cov - torch.diag(diag)
    return {
        "w1": w1,
        "w2": w2,
        "mean_norm": mu.norm().item(),
        "mean_abs_max": mu.abs().max().item(),
        "cov_frob_rel": cov_frob_rel,
        "cov_diag_mean": diag.mean().item(),
        "cov_diag_std": diag.std().item(),
        "cov_offdiag_max": off.abs().max().item(),
    }


@torch.no_grad()
def eval_distribution(encoder, val_loader, cfg, num_proj: int = 1024) -> dict:
    """Evaluate the projector output's distance to N(0, I) plus backbone
    collapse diagnostics.

    Returns the distributional metrics on the projector output (``w1``,
    ``w2``, ``mean_norm``, ``cov_frob_rel``, ``cov_diag_*``,
    ``cov_offdiag_max``) AND a small diagnostic block on the
    pre-projection backbone embedding:

    - ``proj_eff_rank``: Roy-Vetterli effective rank of the projector
      output (max = ``proj_dim``). If you're regularizing only ``proj_dim``
      dimensions to N(0, I) you want this saturated near ``proj_dim``.
    - ``emb_eff_rank``: same on the backbone embedding (max =
      ``hidden_dim``). The collapse detector — when ``proj_dim`` is small
      this should NOT collapse to ``proj_dim``.
    - ``emb_dim``, ``emb_eff_rank_ratio`` (eff rank / hidden dim).
    - ``emb_mean_norm``, ``emb_var_mean``, ``emb_var_min`` for context.
    """
    encoder.eval()
    device = next(encoder.parameters()).device

    proj_chunks, emb_chunks = [], []
    for images, _ in val_loader:
        if images.device != device:
            images = images.to(device, non_blocking=True)
        if images.dtype == torch.uint8:
            images = images.float().div_(127.5).sub_(1.0)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb, proj = encoder(images)
        proj_chunks.append(proj.float())
        emb_chunks.append(emb.float())

    proj_all = torch.cat(proj_chunks, dim=0)
    emb_all = torch.cat(emb_chunks, dim=0)

    # Projector metrics (w1, w2, mean_norm, cov_frob_rel, etc.)
    metrics = _distributional_metrics(proj_all, num_proj=num_proj)
    metrics["proj_eff_rank"] = _effective_rank(proj_all)

    # Embedding metrics (emb_w1, emb_w2, emb_cov_frob_rel, etc.)
    emb_metrics = _distributional_metrics(emb_all, num_proj=num_proj)
    for k, v in emb_metrics.items():
        metrics[f"emb_{k}"] = v
    metrics["emb_dim"] = int(emb_all.shape[1])
    metrics["emb_eff_rank"] = _effective_rank(emb_all)
    metrics["emb_eff_rank_ratio"] = (
        metrics["emb_eff_rank"] / metrics["emb_dim"]
        if metrics["emb_dim"] > 0 else float("nan"))
    return metrics
