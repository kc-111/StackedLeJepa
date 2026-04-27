"""Visualize train_pred_soft pair-augmentation outputs.

Two modes:

  * default (per-sample grid): each row is (original | view 1 | view 2)
    with the per-view N (blur iters) and flip annotated. Useful for
    sanity-checking that the augmentation pipeline produces sensible
    views and that high-N blur isn't washing samples to near-constant.

  * --sweep: one image × N grid showing the blur landscape on a single
    view (post-crop, post-flip). Makes the "kernel wider than image"
    concern concrete — at the high-N end on small images you'll see
    flat gray; that's the regime where blur destroys content rather
    than just smoothing it.

Usage:
    python experiments/train/viz_pred_soft.py --dataset cifar10 \\
        --num-samples 8 --out viz_pairs.png
    python experiments/train/viz_pred_soft.py --dataset cifar10 \\
        --sweep --out viz_sweep.png
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import _DATASETS  # noqa: E402
from train_pred_soft import (  # noqa: E402
    BLUR_GRID,
    BLUR_ITER_RANGE,
    BLUR_SIGMA_FRAC,
    MASK_SMOOTH_FRAC,
    _SoftPairDataset,
    _apply_blur_gpu,
    _apply_grid_blur,
    _build_blur_table_1d,
    _build_smooth_mask_table,
)


def _denorm(t, mean, std):
    """(C, H, W) normalized → (H, W, C) uint8 numpy."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    x = (t * s + m).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = to_pil_image(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _mask_str(mask):
    """Compact bit string for a (G, G) bool mask, row-major."""
    return "".join("1" if b else "0" for b in mask.reshape(-1).tolist())


def _render_pair_grid(args, train_raw, ds, blur_table, mask_table,
                      norm, size):
    """4 columns per row: orig | v0 (anchor, no blur) | v1 | v2."""
    indices = random.sample(range(len(train_raw)), args.num_samples)
    _, axes = plt.subplots(
        args.num_samples, 4, figsize=(10, 2.4 * args.num_samples))
    if args.num_samples == 1:
        axes = axes[None, :]

    sigma = BLUR_SIGMA_FRAC * size
    for row, idx in enumerate(indices):
        orig, label = train_raw[idx]
        orig_pil = _to_pil(orig)
        s = ds[idx]
        v0 = s["image_v0"]
        v1 = _apply_grid_blur(
            s["image_v1"].unsqueeze(0),
            s["blur_n_v1"].unsqueeze(0),
            s["blur_mask_v1"].unsqueeze(0),
            blur_table, mask_table)[0]
        v2 = _apply_grid_blur(
            s["image_v2"].unsqueeze(0),
            s["blur_n_v2"].unsqueeze(0),
            s["blur_mask_v2"].unsqueeze(0),
            blur_table, mask_table)[0]

        axes[row, 0].imshow(np.array(orig_pil))
        axes[row, 0].set_title(
            f"orig  label={label}  {orig_pil.size[0]}×{orig_pil.size[1]}",
            fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(_denorm(v0, norm["mean"], norm["std"]))
        axes[row, 1].set_title("v0 (resize-only original)", fontsize=9)
        axes[row, 1].axis("off")

        for col, (img, n, mask) in enumerate(
            [(v1, s["blur_n_v1"], s["blur_mask_v1"]),
             (v2, s["blur_n_v2"], s["blur_mask_v2"])], start=2):
            axes[row, col].imshow(_denorm(img, norm["mean"], norm["std"]))
            sig_eff = sigma * (int(n) ** 0.5)
            axes[row, col].set_title(
                f"v{col - 1}  N={int(n)}/{BLUR_ITER_RANGE[1]}  "
                f"σ_eff={sig_eff:.2f}  ({sig_eff / size:.0%})\n"
                f"mask={_mask_str(mask)} "
                f"({int(mask.sum())}/{BLUR_GRID * BLUR_GRID})",
                fontsize=8)
            axes[row, col].axis("off")

    plt.suptitle(
        f"{args.dataset}  size={size}  "
        f"scale=({args.scale_min}, {args.scale_max})  "
        f"σ_frac={BLUR_SIGMA_FRAC} (σ={sigma:.2f})  "
        f"N∈{BLUR_ITER_RANGE}  grid={BLUR_GRID}  "
        f"smooth_frac={MASK_SMOOTH_FRAC}",
        fontsize=10, y=1.0)
    plt.tight_layout()


def _render_blur_sweep(args, train_raw, ds, blur_table, _mask_table,
                        norm, size):
    """One sample × N grid showing how (un-masked) blur erodes content
    as N rises. Skips the mask path so the readout reflects the blur
    landscape only."""
    n_steps = args.sweep_steps
    ns = torch.linspace(0, BLUR_ITER_RANGE[1], n_steps).round().long()

    idx = random.sample(range(len(train_raw)), 1)[0]
    s = ds[idx]
    base = s["image_v1"]                     # (3, H, W) post-crop+flip, no blur
    sigma = BLUR_SIGMA_FRAC * size

    _, axes = plt.subplots(1, n_steps,
                            figsize=(1.8 * n_steps, 2.0))
    for c in range(n_steps):
        ni = ns[c:c + 1]
        blurred = _apply_blur_gpu(base.unsqueeze(0), ni, blur_table)[0]
        axes[c].imshow(_denorm(blurred, norm["mean"], norm["std"]))
        sig_eff = sigma * (int(ns[c]) ** 0.5)
        axes[c].set_title(
            f"N={int(ns[c])}\n"
            f"σ_eff={sig_eff:.2f} ({sig_eff / size:.0%})",
            fontsize=9)
        axes[c].axis("off")

    plt.suptitle(
        f"blur sweep — {args.dataset}  size={size}  "
        f"σ_frac={BLUR_SIGMA_FRAC} → σ={sigma:.2f}  "
        f"K_table={blur_table.shape[-1]}",
        fontsize=10, y=1.05)
    plt.tight_layout()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenet-100",
                   choices=list(_DATASETS))
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=8,
                   help="Pair-grid mode only.")
    p.add_argument("--sweep", action="store_true",
                   help="Switch to N sweep on one image.")
    p.add_argument("--sweep-steps", type=int, default=8,
                   help="Number of N values across the sweep grid.")
    p.add_argument("--scale-min", type=float, default=0.3)
    p.add_argument("--scale-max", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="viz_pred_soft.png")
    return p


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    loaders_fn, _, default_size, norm = _DATASETS[args.dataset]
    size = args.image_size or default_size
    train_raw, _ = loaders_fn()
    ds = _SoftPairDataset(
        train_raw, image_size=size,
        mean=norm["mean"], std=norm["std"],
        scale=(args.scale_min, args.scale_max),
    )
    blur_table = _build_blur_table_1d(
        BLUR_ITER_RANGE[1], BLUR_SIGMA_FRAC, size,
        device="cpu", dtype=torch.float32)
    mask_table = _build_smooth_mask_table(
        BLUR_GRID, size, MASK_SMOOTH_FRAC,
        device="cpu", dtype=torch.float32)
    print(f"[viz] dataset={args.dataset} size={size}  "
          f"σ={BLUR_SIGMA_FRAC * size:.2f} ({BLUR_SIGMA_FRAC} · {size})  "
          f"blur K={blur_table.shape[-1]} (cap={2 * size - 1})  "
          f"mask_table={tuple(mask_table.shape)}")

    if args.sweep:
        _render_blur_sweep(args, train_raw, ds, blur_table, mask_table,
                           norm, size)
    else:
        _render_pair_grid(args, train_raw, ds, blur_table, mask_table,
                          norm, size)

    out = Path(args.out).resolve()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
