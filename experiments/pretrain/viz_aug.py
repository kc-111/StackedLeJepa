"""Visualize the LeJEPA augmentation pipeline.

Loads a few sample images from a dataset, runs ``MultiCropGPUAug``,
and saves a grid showing the originals next to one set of augmented
global + local views.

Usage:
    python viz_aug.py --dataset cifar100 --num-samples 4 --out viz.png
    python viz_aug.py --dataset flowers102 --out viz_flowers.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import Config
from data import (
    get_dataloaders, InMemoryGPUDataset,
    IMAGENET_MEAN, IMAGENET_STD, MultiCropGPUAug,
)


def denormalize(views: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display."""
    mean = IMAGENET_MEAN.to(views.device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(views.device).view(1, 3, 1, 1)
    return (views * std + mean).clamp_(0, 1)


def to_hwc(t: torch.Tensor) -> np.ndarray:
    return t.permute(1, 2, 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Visualize LeJEPA augmentations")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of source images to visualize")
    parser.add_argument("--num-global", type=int, default=2)
    parser.add_argument("--num-local", type=int, default=6)
    parser.add_argument("--global-size", type=int, default=224)
    parser.add_argument("--local-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="viz_aug.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a config just for data loading
    cfg = Config(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.num_samples,
        num_global_views=args.num_global,
        num_local_views=args.num_local,
        global_crop_size=args.global_size,
        local_crop_size=args.local_size,
    )
    train_source, _, _ = get_dataloaders(cfg, device)

    # Pull N images
    if isinstance(train_source, InMemoryGPUDataset):
        # Take the first N from cache (deterministic + simple)
        images = train_source.images[: args.num_samples].float() / 255.0
    else:
        # DataLoader path: grab one batch
        it = iter(train_source)
        batch_imgs, _ = next(it)
        images = batch_imgs[: args.num_samples].to(device)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

    print(f"Source images: {tuple(images.shape)}")

    # Build augmentation module
    aug = MultiCropGPUAug(
        num_global=args.num_global,
        num_local=args.num_local,
        global_size=args.global_size,
        local_size=args.local_size,
    ).to(device)

    g, l = aug(images)  # g: (Vg, N, 3, gs, gs); l: (Vl, N, 3, ls, ls) or None
    print(f"Global views: {tuple(g.shape)}")
    if l is not None:
        print(f"Local views:  {tuple(l.shape)}")

    g_vis = denormalize(g.float().reshape(-1, *g.shape[2:])).view(g.shape)
    l_vis = denormalize(l.float().reshape(-1, *l.shape[2:])).view(l.shape) if l is not None else None

    # Layout: rows = source images, columns = [original, V_g globals, V_l locals]
    N = args.num_samples
    Vg = args.num_global
    Vl = args.num_local
    n_cols = 1 + Vg + Vl

    fig, axes = plt.subplots(N, n_cols, figsize=(2 * n_cols, 2 * N))
    if N == 1:
        axes = axes[None, :]

    for i in range(N):
        # Original
        ax = axes[i, 0]
        ax.imshow(to_hwc(images[i]))
        ax.set_title(f"orig\n{tuple(images[i].shape[1:])}", fontsize=8)
        ax.axis("off")

        # Global views
        for v in range(Vg):
            ax = axes[i, 1 + v]
            ax.imshow(to_hwc(g_vis[v, i]))
            if i == 0:
                ax.set_title(f"global {v}\n{args.global_size}², blur+jitter", fontsize=8)
            ax.axis("off")

        # Local views
        for v in range(Vl):
            ax = axes[i, 1 + Vg + v]
            ax.imshow(to_hwc(l_vis[v, i]))
            if i == 0:
                ax.set_title(f"local {v}\n{args.local_size}², jitter", fontsize=8)
            ax.axis("off")

    fig.suptitle(
        f"{args.dataset} | V_g={Vg}@{args.global_size} + V_l={Vl}@{args.local_size}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
