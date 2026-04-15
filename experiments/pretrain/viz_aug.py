"""Visualize the LeJEPA augmentation pipeline.

Two visualizations per source image, side by side:
  1) Plain downscale and blur ops from ``data.py`` (orig, /2, /4, /8, blur,
     blur+/2, blur+/4) — useful to see how aggressive downscaling and blur
     each look on a large source image.
  2) Full ``GPUAug`` pipeline: 1 center-cropped original + N aug views.

Loads sources directly via ``_load_base_dataset`` (bypassing the cached
``InMemoryGPUDataset`` path) so any dataset can be displayed at an
arbitrary ``--full-res``, including ones too large to cache (food101).

Usage:
    python viz_aug.py --dataset food101 --full-res 256 --num-pngs 3
    python viz_aug.py --dataset pets --full-res 384 --num-samples 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import _load_base_dataset, _gaussian_blur, GPUAug


def denormalize(views: torch.Tensor) -> torch.Tensor:
    """Undo [-1, 1] normalization for display: map back to [0, 1]."""
    return ((views + 1) / 2).clamp_(0, 1)


def to_hwc(t: torch.Tensor) -> np.ndarray:
    return t.detach().permute(1, 2, 0).cpu().numpy()


def downscale(images: torch.Tensor, factor: int) -> torch.Tensor:
    """Plain bilinear downscale by an integer factor (H,W → H//f, W//f)."""
    H, W = images.shape[-2:]
    return F.interpolate(
        images, size=(H // factor, W // factor),
        mode="bilinear", align_corners=False,
    )


def load_source_images(
    dataset_name: str, data_dir: str, num_images: int,
    full_res: int, device: torch.device, offset: int = 0,
) -> torch.Tensor:
    """Load ``num_images`` images at ``full_res`` (resized + center-cropped)."""
    base = _load_base_dataset(dataset_name, data_dir, train=True)
    resize = T.Resize(full_res)
    crop = T.CenterCrop(full_res)
    arrays = []
    n = len(base)
    for i in range(num_images):
        img, _ = base[(offset + i) % n]
        if not isinstance(img, Image.Image):
            img = T.functional.to_pil_image(img)
        img = crop(resize(img.convert("RGB")))
        arrays.append(np.asarray(img))
    x = torch.from_numpy(np.stack(arrays))           # (N, H, W, 3) uint8
    x = x.permute(0, 3, 1, 2).contiguous().float() / 255.0
    return x.to(device)


def make_downscale_blur_figure(
    images: torch.Tensor, blur_sigma: float, blur_kernel: int,
    dataset_name: str,
) -> plt.Figure:
    """Grid: orig | ds/2 | ds/4 | ds/8 | blur | blur+ds/2 | blur+ds/4."""
    blurred = _gaussian_blur(images, sigma=blur_sigma,
                             kernel_size=blur_kernel, p=1.0)
    cols = [
        ("orig",       images),
        ("ds /2",      downscale(images, 2)),
        ("ds /4",      downscale(images, 4)),
        ("ds /8",      downscale(images, 8)),
        ("blur+ds/2",  downscale(blurred, 2)),
        ("blur+ds/4",  downscale(blurred, 4)),
        ("blur+ds/8",  downscale(blurred, 8)),
        ("blur+ds/16", downscale(blurred, 16)),
    ]

    N = images.shape[0]
    n_cols = len(cols)
    fig, axes = plt.subplots(N, n_cols, figsize=(2 * n_cols, 2 * N))
    if N == 1:
        axes = axes[None, :]

    for i in range(N):
        for j, (name, t) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(to_hwc(t[i].clamp(0, 1)))
            if i == 0:
                h, w = t.shape[-2], t.shape[-1]
                ax.set_title(f"{name}\n{h}×{w}", fontsize=8)
            ax.axis("off")

    fig.suptitle(
        f"{dataset_name} | plain downscale & blur "
        f"(sigma={blur_sigma}, k={blur_kernel})",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def make_blur_modes_figure(
    images: torch.Tensor, gpu_aug: GPUAug, dataset_name: str,
    num_local_samples: int = 5,
) -> plt.Figure:
    """Grid: orig | gaussian | motion (×3) | local (×N), per source image."""
    cols = [("orig", images), ("gaussian", gpu_aug._blur_gaussian(images))]
    for j in range(3):
        cols.append((f"motion {j}", gpu_aug._blur_motion(images)))
    for j in range(num_local_samples):
        cols.append((f"local {j}", gpu_aug._blur_local(images)))

    N = images.shape[0]
    n_cols = len(cols)
    fig, axes = plt.subplots(N, n_cols, figsize=(2 * n_cols, 2 * N))
    if N == 1:
        axes = axes[None, :]
    for i in range(N):
        for j, (name, t) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(to_hwc(t[i].clamp(0, 1)))
            if i == 0:
                ax.set_title(name, fontsize=8)
            ax.axis("off")
    fig.suptitle(
        f"{dataset_name} | blur modes "
        f"(motion picks angle per call, local picks rect per image)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def make_gpuaug_figure(
    images: torch.Tensor, gpu_aug: GPUAug, dataset_name: str,
) -> plt.Figure:
    """Grid: 1 center-cropped orig + V GPUAug views, per source image."""
    orig, aug_views = gpu_aug(images)
    orig_vis = denormalize(orig.float())
    aug_vis = denormalize(
        aug_views.float().reshape(-1, *aug_views.shape[2:])
    ).view(aug_views.shape)

    N = images.shape[0]
    V = aug_views.shape[0]
    cs = gpu_aug.crop_size
    n_cols = 1 + V
    fig, axes = plt.subplots(N, n_cols, figsize=(2 * n_cols, 2 * N))
    if N == 1:
        axes = axes[None, :]

    for i in range(N):
        ax = axes[i, 0]
        ax.imshow(to_hwc(orig_vis[i]))
        if i == 0:
            ax.set_title(f"orig\n{cs}²", fontsize=8)
        ax.axis("off")
        for v in range(V):
            ax = axes[i, 1 + v]
            ax.imshow(to_hwc(aug_vis[v, i]))
            if i == 0:
                ax.set_title(f"aug {v}\n{cs}²", fontsize=8)
            ax.axis("off")

    fig.suptitle(
        f"{dataset_name} | 1 orig + {V} GPUAug views @ {cs}²\n"
        f"RRC({gpu_aug.crop_scale})→jitter→gray→blur(p=0.1)→solarize(p=0.2)→flip→norm",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize LeJEPA augmentations")
    parser.add_argument("--dataset", type=str, default="food101")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of source images per png")
    parser.add_argument("--num-aug", type=int, default=4,
                        help="Number of GPUAug views per source")
    parser.add_argument("--full-res", type=int, default=256,
                        help="Source resolution to load (resize + center-crop)")
    parser.add_argument("--crop-size", type=int, default=128,
                        help="GPUAug output crop size")
    parser.add_argument("--blur-sigma", type=float, default=2.0,
                        help="Sigma for the standalone blur column")
    parser.add_argument("--blur-kernel", type=int, default=9,
                        help="Kernel size for the standalone blur column")
    parser.add_argument("--num-pngs", type=int, default=3,
                        help="Number of pngs to produce; each uses a different offset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--out-prefix", type=str, default="viz_aug")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu_aug = GPUAug(
        num_aug_views=args.num_aug,
        crop_size=args.crop_size,
    ).to(device)

    for k in range(args.num_pngs):
        offset = k * args.num_samples
        images = load_source_images(
            args.dataset, args.data_dir, args.num_samples,
            args.full_res, device, offset=offset,
        )
        print(f"[png {k+1}/{args.num_pngs}] sources: {tuple(images.shape)} "
              f"(offset={offset})")

        ds_fig = make_downscale_blur_figure(
            images, blur_sigma=args.blur_sigma,
            blur_kernel=args.blur_kernel, dataset_name=args.dataset,
        )
        ds_path = out_dir / f"{args.out_prefix}_{args.dataset}_downscale_{k:02d}.png"
        ds_fig.savefig(ds_path, dpi=120, bbox_inches="tight")
        plt.close(ds_fig)
        print(f"  saved {ds_path}")

        aug_fig = make_gpuaug_figure(images, gpu_aug, dataset_name=args.dataset)
        aug_path = out_dir / f"{args.out_prefix}_{args.dataset}_gpuaug_{k:02d}.png"
        aug_fig.savefig(aug_path, dpi=120, bbox_inches="tight")
        plt.close(aug_fig)
        print(f"  saved {aug_path}")

        blur_fig = make_blur_modes_figure(
            images, gpu_aug, dataset_name=args.dataset)
        blur_path = out_dir / f"{args.out_prefix}_{args.dataset}_blurmodes_{k:02d}.png"
        blur_fig.savefig(blur_path, dpi=120, bbox_inches="tight")
        plt.close(blur_fig)
        print(f"  saved {blur_path}")


if __name__ == "__main__":
    main()
