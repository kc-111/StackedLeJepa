"""LeJEPA-style invariance SSL with non-destructive augmentations only.

Trimmed variant of ``train_pred.py`` — drops the augmentation-conditioned
predictor and runs plain invariance + reg on a *triplet* of views per
sample::

  v0  the original image, resize-only — no crop, flip, grayscale, blur
  v1  independent random crop + per-view (h, v) flips + grayscale +
      selective grid blur
  v2  independent random crop + per-view (h, v) flips + grayscale +
      selective grid blur

v1 and v2 sample their crops independently (no longer shared); v0 is
just the input resized to (image_size, image_size) with normalization,
nothing else. Per-view crops decorrelate the across-view samples for
the regularizer (better iid estimates of distance from N(0, I)) at
the cost of a larger nuisance the encoder must throw away to satisfy
invariance. v0's distribution matches the val pipeline exactly, so it
acts as a clean anchor between train and eval.

Augmentations applied to v1, v2:

  - Random resized crop, **independent per view**.
  - Horizontal + vertical flips, independently random per view.
  - Per-view grayscale with prob ``GRAYSCALE_PROB`` (3-channel output
    so backbone shapes are unchanged).
  - Iterative gaussian blur, σ = ``BLUR_SIGMA_FRAC · image_size`` so
    strength is resolution-invariant; per-view N sampled in
    BLUR_ITER_RANGE; applied on GPU against a cached σ_eff = σ·√N
    kernel table.
  - Selective grid blur: a (BLUR_GRID × BLUR_GRID) bool mask gates the
    blur per cell. With prob MASK_FULL_PROB the mask is all True (full
    uniform blur); otherwise each cell is bernoulli(MASK_CELL_PROB).
    Cell boundaries are gaussian-smoothed by σ = cell_size ·
    MASK_SMOOTH_FRAC so transitions blend rather than creating a sharp
    gradient the encoder could overfit.

Why drop the predictor — with only flip + blur strength as per-view
nuisances, invariance to them is what we actually want from the
encoder. The action-conditioned predictor in ``train_pred.py`` exists
to *retain* aug-conditioned info under destructive augs (so the encoder
doesn't collapse). Here, none of the augs is destructive enough to
threaten that, so plain invariance is fine.

Why σ is fixed not random — σ_eff = σ·√N already gives a 1-D family of
effective scales; varying σ on top of N is redundant.

Why iterative blur — keeps per-pass kernel support tight while letting
the effective σ grow as σ_eff = σ·√N (additivity of variance for a
convolution of well-resolved gaussians). We cache that closed-form
N-iterated equivalent so the GPU does one batched separable conv per
view instead of N passes per sample.

Usage:
    python experiments/train/train_pred_soft.py --dataset cifar10 \\
        --backbone resnet18 --regularizer w1 --epochs 200
"""

import argparse
import random
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import SlicedEppsPulley

sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses import make_regularizer  # noqa: E402
from train import (  # noqa: E402
    _DATASETS,
    allocate_run,
    make_backbone,
    make_projector,
)


# Iterative gaussian blur with σ scaled to image resolution.
#
# Per-pass σ is ``BLUR_SIGMA_FRAC · image_size``, so blur strength is
# resolution-invariant: σ_eff / image_size = BLUR_SIGMA_FRAC · √N
# depends only on the (sampled) N — not on the spatial size of the
# images going through. That keeps the augmentation comparable across
# CIFAR (32 px), Imagenette (128 px) and ImageNet (224 px).
#
# Per-view N is sampled in BLUR_ITER_RANGE (inclusive); N=0 is a clean
# passthrough. Use viz_pred_soft.py --sweep to read off where content
# saturates.
BLUR_SIGMA_FRAC = 0.01
BLUR_ITER_RANGE = (0, 20)

# Grid-masked selective blur for v1, v2.
#
# A (BLUR_GRID × BLUR_GRID) bool mask gates whether each cell of the
# image is blurred. With prob ``MASK_FULL_PROB`` the mask is all-True
# (full uniform blur, original behavior); otherwise each cell is drawn
# as bernoulli(``MASK_CELL_PROB``). Cell boundaries are smoothed by a
# gaussian with σ = cell_size · ``MASK_SMOOTH_FRAC`` so transitions
# blend rather than producing a sharp gradient artifact the encoder
# could overfit. Set BLUR_GRID = 1 to disable (mask is a single
# bernoulli over the whole image).
BLUR_GRID = 3
MASK_FULL_PROB = 0.5
MASK_CELL_PROB = 0.5
MASK_SMOOTH_FRAC = 0.25

# Per-view grayscale conversion. Independent of blur (so the anchor v0
# can also drop color) — forces the encoder not to rely on chromaticity.
# 3-channel output keeps tensor shapes consistent with the RGB path.
GRAYSCALE_PROB = 0.2


# ---------------------------------------------------------------------------
# Pair-augmentation dataset
# ---------------------------------------------------------------------------
# Emits a triplet {v0 (resize-only original), v1, v2}. v1, v2 each
# sample an independent random crop and apply per-view (h, v) flips +
# grayscale; they also carry blur metadata consumed on the GPU: an int
# n_iters and a (BLUR_GRID, BLUR_GRID) bool cell mask. The actual blur
# conv + mask compositing happens in forward. v0 has no augmentation —
# it's the input image resized to (image_size, image_size) and
# normalized, matching the val pipeline.

class _SoftPairDataset(Dataset):
    """Wrap a (PIL, label)-yielding source. Each sample::

        {image_v0, image_v1, image_v2,
         blur_n_v1, blur_n_v2, blur_mask_v1, blur_mask_v2, label}
    """

    def __init__(self, src, image_size, mean, std,
                 scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3)):
        self.src = src
        self.image_size = image_size
        self._mean = mean
        self._std = std
        self.scale = scale
        self.ratio = ratio

    def __len__(self):
        return len(self.src)

    def _sample_crop(self, img):
        from torchvision.transforms import RandomResizedCrop
        return RandomResizedCrop.get_params(img, scale=list(self.scale),
                                             ratio=list(self.ratio))

    def _render_original(self, img):
        """v0: resize-only. No crop, flip, grayscale, blur."""
        v = TF.resize(img, [self.image_size, self.image_size])
        v = TF.to_tensor(v)
        return TF.normalize(v, mean=self._mean, std=self._std)

    def _render_aug(self, img):
        """v1 / v2: independent crop + per-view (h, v) flips + grayscale.
        Blur is deferred to GPU."""
        i, j, h, w = self._sample_crop(img)
        v = TF.resized_crop(img, i, j, h, w,
                            [self.image_size, self.image_size])
        if random.random() < 0.5:
            v = TF.hflip(v)
        if random.random() < 0.5:
            v = TF.vflip(v)
        if random.random() < GRAYSCALE_PROB:
            v = TF.rgb_to_grayscale(v, num_output_channels=3)
        v = TF.to_tensor(v)
        return TF.normalize(v, mean=self._mean, std=self._std)

    def _sample_blur(self):
        """Return (n_iters long scalar, (G, G) bool cell mask)."""
        n_iters = random.randint(*BLUR_ITER_RANGE)
        if random.random() < MASK_FULL_PROB:
            mask = torch.ones((BLUR_GRID, BLUR_GRID), dtype=torch.bool)
        else:
            mask = (torch.rand(BLUR_GRID, BLUR_GRID) < MASK_CELL_PROB)
        return torch.tensor(n_iters, dtype=torch.long), mask

    def __getitem__(self, idx):
        img, label = self.src[idx]
        if not isinstance(img, Image.Image):
            img = TF.to_pil_image(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        v0 = self._render_original(img)
        v1 = self._render_aug(img)
        v2 = self._render_aug(img)
        n1, m1 = self._sample_blur()
        n2, m2 = self._sample_blur()
        return {
            "image_v0": v0,
            "image_v1": v1,
            "image_v2": v2,
            "blur_n_v1": n1,
            "blur_n_v2": n2,
            "blur_mask_v1": m1,
            "blur_mask_v2": m2,
            "label": int(label),
        }


# ---------------------------------------------------------------------------
# GPU iterative blur — cached N → 1D kernel table (σ scales with image dim)
# ---------------------------------------------------------------------------
#
# Iterating a well-resolved gaussian (per-pass kernel covers ≥ ±3σ) N
# times is a convolution of N gaussians, so by additivity of variance
# the resulting kernel is itself gaussian with σ_eff = σ · √N. We build
# that closed form directly: O(1) memory and runtime, no recursion. The
# only difference vs. iterating in a loop is the boundary band (single
# reflect-pad here, vs. N reflect-pads in the loop) — bulk pixels match.

def _build_blur_table_1d(n_max, sigma_frac, image_size, *,
                          device, dtype):
    """Build the (n_max + 1, K) kernel table indexed by N.

    σ is computed from ``sigma_frac · image_size`` so blur strength is
    resolution-invariant. ``image_size`` also caps K via reflect
    padding's ``pad < dim`` constraint (K ≤ 2·dim − 1) — without the
    cap, σ · √n_max can blow K past the input's spatial size. Truncating
    loses a tiny bit at the tails; we renormalize so each row remains a
    valid probability kernel.

    Index 0 along the N axis = identity (delta), so ``N=0`` is a clean
    passthrough rather than a degenerate fp ratio.
    """
    sigma = sigma_frac * image_size
    sigma_eff_max = sigma * (max(n_max, 1) ** 0.5)
    K = 2 * int(round(3 * sigma_eff_max)) + 1
    if K % 2 == 0:
        K += 1
    max_K = 2 * image_size - 1
    if max_K % 2 == 0:
        max_K -= 1
    K = min(K, max_K)
    coords = (torch.arange(K, dtype=dtype, device=device) - (K // 2))
    table = torch.zeros(n_max + 1, K, dtype=dtype, device=device)
    table[0, K // 2] = 1.0
    for n in range(1, n_max + 1):
        sig_eff = sigma * (n ** 0.5)
        g = torch.exp(-(coords ** 2) / (2 * sig_eff ** 2))
        table[n] = g / g.sum()
    return table


def _apply_separable_blur(v, k1d):
    """Per-sample separable depthwise conv. ``v`` is (B, C, H, W),
    ``k1d`` is (B, K). Reflection-pad once on each axis."""
    B, C, H, W = v.shape
    K = k1d.shape[1]
    pad = K // 2
    kh = (k1d.view(B, 1, 1, K)
              .expand(B, C, 1, K)
              .reshape(B * C, 1, 1, K))
    kv = (k1d.view(B, 1, K, 1)
              .expand(B, C, K, 1)
              .reshape(B * C, 1, K, 1))
    x = v.reshape(1, B * C, H, W)
    x = F.pad(x, [pad, pad, 0, 0], mode="reflect")
    x = F.conv2d(x, kh, groups=B * C)
    x = F.pad(x, [0, 0, pad, pad], mode="reflect")
    x = F.conv2d(x, kv, groups=B * C)
    return x.reshape(B, C, H, W)


def _apply_blur_gpu(v, n_iter, table):
    """Look up per-sample 1D kernel from ``table`` and apply. ``n_iter=0``
    selects the delta row → identity (passthrough)."""
    safe_n = n_iter.clamp(min=0, max=table.shape[0] - 1)
    return _apply_separable_blur(v, table[safe_n])


def _build_smooth_mask_table(grid, image_size, smooth_frac, *,
                              device, dtype):
    """Cache every smoothed cell mask once: (2^(G²), H, W).

    The smoothed mask depends only on (mask bits, image_size, grid,
    smooth_frac), all known statically — so we enumerate every possible
    (G × G) bool mask, upsample (nearest), smooth with σ = cell_size ·
    smooth_frac, and stash the result. At runtime we bit-pack the
    sample's mask into an int and index this table.

    Memory grows as 2^(G²) · H · W floats. Caps grid at 6 (2³⁶ masks
    would be unreasonable); 3 is the sensible default at 512 entries.
    """
    n_masks = 1 << (grid * grid)
    if grid > 6:
        raise ValueError(f"BLUR_GRID={grid} too large; cache would hold "
                         f"{n_masks} masks. Use grid ≤ 6.")

    sigma = (image_size / grid) * smooth_frac
    K = 2 * int(round(3 * sigma)) + 1
    if K % 2 == 0:
        K += 1
    max_K = 2 * image_size - 1
    if max_K % 2 == 0:
        max_K -= 1
    K = min(K, max_K)
    coords = (torch.arange(K, dtype=dtype, device=device) - (K // 2))
    smooth_k = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    smooth_k = smooth_k / smooth_k.sum()             # (K,)

    n_bits = grid * grid
    bit_idx = torch.arange(n_bits, device=device)
    masks = ((torch.arange(n_masks, device=device).unsqueeze(-1)
              >> bit_idx) & 1).bool()               # (n_masks, n_bits)
    masks = masks.reshape(n_masks, grid, grid).to(dtype)
    m = masks.unsqueeze(1)                           # (n_masks, 1, G, G)
    m = F.interpolate(m, size=(image_size, image_size), mode="nearest")
    k1d = smooth_k.unsqueeze(0).expand(n_masks, K)
    m = _apply_separable_blur(m, k1d)
    return m.squeeze(1)                              # (n_masks, H, W)


def _apply_grid_blur(v, n_iter, mask, blur_table, mask_table):
    """Selective grid blur via a cached smoothed-mask lookup.

    ``mask`` is (B, G, G) bool. Bit-pack to (B,) ids and gather from
    ``mask_table`` (built by ``_build_smooth_mask_table``). Composite
    is per-pixel: ``m · blurred + (1 − m) · v``.
    """
    B = v.shape[0]
    blurred = _apply_blur_gpu(v, n_iter, blur_table)
    G = mask.shape[-1]
    n_bits = G * G
    bit_idx = torch.arange(n_bits, device=mask.device, dtype=torch.long)
    powers = (1 << bit_idx)                          # (n_bits,)
    mask_id = (mask.reshape(B, n_bits).long() * powers).sum(dim=-1)
    m = mask_table[mask_id].unsqueeze(1).to(dtype=v.dtype)  # (B, 1, H, W)
    return m * blurred + (1.0 - m) * v


# ---------------------------------------------------------------------------
# Data: train pair-aug + standard val
# ---------------------------------------------------------------------------

def make_data(name, batch_size, num_workers=8, image_size=None,
              scale=(0.3, 1.0)):
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(_DATASETS)}")
    loaders_fn, num_classes, default_size, norm = _DATASETS[name]
    size = image_size or default_size

    train_raw, val_raw = loaders_fn()

    train_ds = _SoftPairDataset(
        src=train_raw, image_size=size,
        mean=norm["mean"], std=norm["std"],
        scale=scale,
    )
    val_tf = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((size, size)),
        transforms.ToImage(**norm),
    )
    val_ds = spt.data.FromTorchDataset(
        val_raw, names=["image", "label"], transform=val_tf)

    train_dl = DataLoader(
        dataset=train_ds, batch_size=batch_size,
        num_workers=num_workers, drop_last=True, shuffle=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        dataset=val_ds, batch_size=batch_size, num_workers=num_workers)

    return spt.data.DataModule(train=train_dl, val=val_dl), num_classes


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
#
# Train: V=3 views per sample (v0 anchor, v1/v2 grid-blurred). Apply
# selective grid blur on GPU using cached blur + smoothed-mask tables,
# backbone + projector all three views, invariance + reg on (V=3, B, D).
# Same MSE-with-margin invariance as train.py's mse path: floor =
# D·(V−1)/V, margin = inv_tol · floor, penalty zero below that.
#
# Eval: standard single-image forward (val pipeline) for OnlineProbe.

def make_forward():
    def forward(self, batch, stage):
        out = {}

        if "image_v0" in batch:
            v0 = batch["image_v0"]
            v1 = batch["image_v1"]
            v2 = batch["image_v2"]

            # Lazy-init in fp32 against the input's spatial size (σ and
            # mask resolution both scale with image dim). Rebuild blur +
            # mask tables if the spatial size changes.
            min_dim = min(v0.shape[-1], v0.shape[-2])
            blur_tab = getattr(self, "_blur_table", None)
            mask_tab = getattr(self, "_mask_table", None)
            cached_dim = getattr(self, "_blur_min_dim", None)
            if (blur_tab is None or blur_tab.device != v0.device
                    or cached_dim != min_dim):
                blur_tab = _build_blur_table_1d(
                    BLUR_ITER_RANGE[1], BLUR_SIGMA_FRAC, min_dim,
                    device=v0.device, dtype=torch.float32)
                mask_tab = _build_smooth_mask_table(
                    BLUR_GRID, min_dim, MASK_SMOOTH_FRAC,
                    device=v0.device, dtype=torch.float32)
                self._blur_table = blur_tab
                self._mask_table = mask_tab
                self._blur_min_dim = min_dim
            blur_d = blur_tab.to(dtype=v0.dtype)
            mask_d = mask_tab.to(dtype=v0.dtype)
            with torch.no_grad():
                v1 = _apply_grid_blur(
                    v1, batch["blur_n_v1"], batch["blur_mask_v1"],
                    blur_d, mask_d)
                v2 = _apply_grid_blur(
                    v2, batch["blur_n_v2"], batch["blur_mask_v2"],
                    blur_d, mask_d)

            h0 = self.backbone(v0)
            h1 = self.backbone(v1)
            h2 = self.backbone(v2)
            z0 = self.projector(h0)
            z1 = self.projector(h1)
            z2 = self.projector(h2)

            # Regularizer: per-view (V=3, B, D) by default. SlicedEppsPulley
            # only takes [N, D] so it always gets the flat pool; --flatten-reg
            # opts our regs into the same one-big-bag behavior.
            z_stack = torch.stack([z0, z1, z2], dim=0)        # (V=3, B, D)
            V, B, D = z_stack.shape
            do_flatten = self.flatten_reg or isinstance(
                self.regularizer, SlicedEppsPulley)
            if do_flatten:
                flat = z_stack.reshape(-1, D)
                reg = self.regularizer(flat)
                pool_rows = flat.shape[0]
            else:
                reg = self.regularizer(z_stack)
                pool_rows = B
            if self.regularizer.needs_compensation:
                reg = reg * (pool_rows / B)

            # Mean-target invariance with margin (matches train.py's mse path).
            # Under N(0, I) and conditional on z̄, E‖z_v − z̄‖² = D·(V−1)/V;
            # margin = inv_tol · floor zeros the penalty below that — anything
            # tighter would fight the regularizer.
            mean_z = z_stack.mean(dim=0, keepdim=True)        # (1, B, D)
            per_sample_sq = (z_stack - mean_z).square().sum(dim=-1)  # (V, B)
            # per_sample_sq = (z_stack - mean_z).abs().sum(dim=-1) # L1 norm
            prior_floor = D * (V - 1) / V
            margin = self.inv_tol * prior_floor
            inv_loss = torch.clamp(per_sample_sq - margin, min=0.0).mean() / D
            inv_margin_active = (per_sample_sq > margin).float().mean()

            loss = self.lambd * reg + (1.0 - self.lambd) * inv_loss
            out["loss"] = loss

            out["embedding"] = torch.cat([h0, h1, h2], dim=0)
            out["projection"] = torch.cat([z0, z1, z2], dim=0)
            if "label" in batch:
                lbl = batch["label"]
                out["label"] = torch.cat([lbl, lbl, lbl], dim=0)

            self.log(f"{stage}/inv_loss", inv_loss,
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/inv_margin_active", inv_margin_active,
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/reg_loss", reg,
                     on_step=False, on_epoch=True, sync_dist=True)
        else:
            emb = self.backbone(batch["image"])
            out["embedding"] = emb
            out["projection"] = self.projector(emb)
            if "label" in batch:
                out["label"] = batch["label"]

        return out

    return forward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenette",
                   choices=["cifar10", "cifar100", "imagenette", "imagenet-100"])
    p.add_argument("--image-size", type=int, default=None,
                   help="Override dataset default image size")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--regularizer", default="w1",
                   choices=["sigreg", "sigreg_raw", "w1", "w2"])
    p.add_argument("--lambd", type=float, default=0.95,
                   help="Weight on reg vs invariance.")
    p.add_argument("--inv-tol", type=float, default=0.01,
                   help="Invariance margin as a fraction of the N(0, I) "
                        "prior floor D·(V−1)/V (V=2 → D/2). 0 = strict "
                        "invariance; higher = more slack so reg has room to "
                        "spread the projections.")
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--proj-hidden", type=int, default=2048)
    p.add_argument("--num-proj", type=int, default=2048)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--flatten-reg", action="store_true",
                   help="Flatten (V=2, B, D) → (V*B, D) before reg. "
                        "Matches the reference LeJEPA impl but biases the "
                        "estimator (across-view samples are correlated).")
    p.add_argument("--scale-min", type=float, default=0.08)
    p.add_argument("--scale-max", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed. If omitted, auto-picks the smallest "
                        "nonneg int unused by prior runs with identical hparams.")
    return p


def main():
    args = build_parser().parse_args()

    if args.regularizer == "sigreg_raw":
        args.flatten_reg = True

    log_dir = Path(__file__).resolve().parent / "logs"
    run_dir, run_name, seed = allocate_run(
        log_dir, args.dataset, f"{args.backbone}_soft",
        vars(args), args.seed)
    args.seed = seed
    pl.seed_everything(seed, workers=True)
    print(f"[run] {run_name}  seed={seed}  dir={run_dir}")

    data, num_classes = make_data(
        args.dataset, args.batch_size, args.num_workers,
        image_size=args.image_size,
        scale=(args.scale_min, args.scale_max),
    )
    low_res = args.dataset.startswith("cifar")
    backbone, emb_dim = make_backbone(args.backbone, low_resolution=low_res)
    projector = make_projector(emb_dim, args.proj_dim, args.proj_hidden)
    regularizer = make_regularizer(
        args.regularizer, num_proj=args.num_proj, knots=args.knots)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=make_forward(),
        regularizer=regularizer,
        lambd=args.lambd,
        inv_tol=args.inv_tol,
        flatten_reg=args.flatten_reg,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )

    def _cls_metrics():
        return {
            "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
            "top5": torchmetrics.classification.MulticlassAccuracy(
                num_classes, top_k=5),
        }

    linear_probe_emb = spt.callbacks.OnlineProbe(
        module, name="linear_probe_emb",
        input="embedding", target="label",
        probe=nn.Linear(emb_dim, num_classes),
        loss=nn.CrossEntropyLoss(),
        metrics=_cls_metrics(),
    )
    linear_probe_proj = spt.callbacks.OnlineProbe(
        module, name="linear_probe_proj",
        input="projection", target="label",
        probe=nn.Linear(args.proj_dim, num_classes),
        loss=nn.CrossEntropyLoss(),
        metrics=_cls_metrics(),
    )
    knn_probe_emb = spt.callbacks.OnlineKNN(
        name="knn_probe_emb", input="embedding", target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes)},
        input_dim=emb_dim, k=20,
    )
    knn_probe_proj = spt.callbacks.OnlineKNN(
        name="knn_probe_proj", input="projection", target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes)},
        input_dim=args.proj_dim, k=20,
    )

    logger = CSVLogger(save_dir=str(log_dir), name=run_name, version="")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        save_last=True, save_top_k=0,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[
            linear_probe_emb, linear_probe_proj,
            knn_probe_emb, knn_probe_proj,
            ckpt_cb,
        ],
        precision="16-mixed",
        logger=logger,
    )

    from stable_pretraining.callbacks import (
        EnvironmentDumpCallback,
        LoggingCallback,
    )
    from lightning.pytorch.utilities import rank_zero_only
    from prettytable import PrettyTable
    import logging as _logging

    class EvalOnlyLoggingCallback(LoggingCallback):
        @rank_zero_only
        def on_validation_end(self, trainer, pl_module):
            pass

        @rank_zero_only
        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            table = PrettyTable()
            table.field_names = ["Metric", "Value"]
            for key in sorted(metrics):
                if not (key.startswith("eval/") or key.startswith("fit/")):
                    continue
                if key.endswith("_epoch") and key[:-len("_epoch")] in metrics:
                    continue
                table.add_row([
                    "\033[0;34;40m" + key + "\033[0m",
                    "\033[0;32;40m" + str(metrics[key].item()) + "\033[0m",
                ])
            _logging.info(f"\n{table}")

    trainer.callbacks = [
        EvalOnlyLoggingCallback() if isinstance(cb, LoggingCallback) else cb
        for cb in trainer.callbacks
        if not isinstance(cb, EnvironmentDumpCallback)
    ]

    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=seed)
    manager()


if __name__ == "__main__":
    main()
