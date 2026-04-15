"""Visualize the projector-output distribution of a trained LeJEPA encoder.

Loads a single checkpoint and produces:
  - aug_clouds_pca.png    PCA(2) of per-image augmentation clouds
  - aug_clouds_pairs.png  Raw projector dim pairs (d0,d1)/(d0,d2)/(d1,d2)/(d2,d3)
  - marginal_hists.png    Histogram of every projector dimension vs N(0,1)
  - random_dir_hists.png  Histogram along random unit directions vs N(0,1)
  - cov_spectrum.png      Sorted eigenvalues of the projector-output covariance
  - stats.json            Distributional metrics (eval_distribution + cluster stats)

All plots are computed on the **train** split (the distribution the
regularizer was actually optimized over). The latent space is the
projector output (proj_dim, default 32) — the space the w1 N(0,I)
constraint acts on.

Usage:
    python experiments/latent_viz/visualize_latents.py \
      --checkpoint runs/lejepa_v1/cifar100_convnextv2_atto_w1_pooled_bs32_seed42/final.pt \
      --out-dir experiments/latent_viz/results/atto_w1_pooled_final
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAIN_DIR = REPO_ROOT / "experiments" / "pretrain"
for p in (str(PRETRAIN_DIR), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from configs import Config
from data import (
    GPUAug, InMemoryEvalLoader, InMemoryGPUDataset, get_dataloaders,
)
from models import LeJEPAEncoder
from train_loops import eval_distribution


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path: str, data_dir: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_cfg = dict(ckpt["config"])
    saved_cfg["data_dir"] = data_dir
    cfg = Config(**saved_cfg)

    encoder = LeJEPAEncoder(cfg).to(device)
    state = ckpt["encoder"]
    # Notebook training loop saves with torch.compile, which prepends
    # "_orig_mod." to every parameter name. Strip it here.
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder, cfg, ckpt


# ---------------------------------------------------------------------------
# Latent collection
# ---------------------------------------------------------------------------

def sample_stratified_indices(train_source, num_classes, per_class, seed):
    """Pick ``num_classes`` distinct labels and ``per_class`` indices in each."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    labels_cpu = train_source.labels.cpu()
    unique = labels_cpu.unique().tolist()
    perm = torch.randperm(len(unique), generator=g)
    chosen = sorted(unique[i] for i in perm[:num_classes].tolist())
    sel_indices, sel_classes = [], []
    for c in chosen:
        class_idx = (labels_cpu == c).nonzero(as_tuple=True)[0]
        pick = class_idx[torch.randperm(len(class_idx), generator=g)[:per_class]]
        sel_indices.append(pick)
        sel_classes.extend([int(c)] * pick.numel())
    return torch.cat(sel_indices), torch.tensor(sel_classes, dtype=torch.long)


@torch.no_grad()
def collect_aug_clouds(encoder, train_source, cfg, num_aug, chunk_views,
                       device, indices):
    """Project ``num_aug`` random augmentations of the selected images.

    Returns:
        proj_aug:  (K, num_aug, proj_dim) — augmented projections
        proj_orig: (K, proj_dim)          — center-cropped projections
    """
    if not isinstance(train_source, InMemoryGPUDataset):
        raise RuntimeError("This script assumes the in-memory train source "
                           "(cifar/etc.). Got: " + type(train_source).__name__)

    src_imgs = (train_source.images[indices.to(train_source.images.device)]
                .float() / 255.0)

    chunk_views = max(1, min(chunk_views, num_aug))
    n_calls = (num_aug + chunk_views - 1) // chunk_views

    aug_module = GPUAug(
        num_aug_views=chunk_views,
        crop_size=cfg.crop_size,
        crop_scale=tuple(cfg.crop_scale),
    ).to(device)

    proj_chunks = []
    orig_proj = None
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(n_calls):
            orig, aug = aug_module(src_imgs)  # orig (N,3,cs,cs), aug (V,N,3,cs,cs)
            if orig_proj is None:
                _, orig_proj = encoder(orig)
                orig_proj = orig_proj.float()
            V, N = aug.shape[:2]
            flat = aug.reshape(V * N, *aug.shape[2:])
            _, proj = encoder(flat)
            proj = proj.float().view(V, N, -1)
            proj_chunks.append(proj)

    proj_aug = torch.cat(proj_chunks, dim=0)[:num_aug]   # (num_aug, N, D)
    proj_aug = proj_aug.transpose(0, 1).contiguous()     # (N, num_aug, D)
    return proj_aug.cpu(), orig_proj.cpu()


@torch.no_grad()
def collect_marginal(encoder, train_source, cfg, num_eval, batch_size, device):
    """Collect projector outputs over the train set (deterministic eval transform).

    Uses ``InMemoryEvalLoader`` over the same in-memory train cache, so the
    images go through the same center-crop + ImageNet normalization that
    ``viz_aug.py``'s ``orig`` path uses.
    """
    eval_loader = InMemoryEvalLoader(train_source, batch_size=batch_size)
    chunks = []
    seen = 0
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for images, _ in eval_loader:
            _, proj = encoder(images)
            chunks.append(proj.float().cpu())
            seen += images.shape[0]
            if seen >= num_eval:
                break
    proj_all = torch.cat(chunks, dim=0)[:num_eval]
    return proj_all  # (num_eval, proj_dim)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _pca_2d(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    _, _, Vh = torch.linalg.svd(xc, full_matrices=False)
    return xc @ Vh[:2].T


def _class_palette_and_marker(classes):
    """Return (color_per_image, marker_per_image, sorted unique classes)."""
    uniq = sorted(set(int(c) for c in classes.tolist()))
    cmap = plt.get_cmap("tab10", max(len(uniq), 1))
    color_of = {c: cmap(i) for i, c in enumerate(uniq)}
    seen_count = {c: 0 for c in uniq}
    markers_pool = ["o", "s", "^", "D", "P", "X", "v", "*"]
    colors, markers = [], []
    for c in classes.tolist():
        c = int(c)
        m = markers_pool[seen_count[c] % len(markers_pool)]
        seen_count[c] += 1
        colors.append(color_of[c])
        markers.append(m)
    return colors, markers, uniq, color_of


def plot_aug_clouds_pca(proj_aug, proj_orig, classes, out_path):
    """PCA(2) of all augmentation points, colored by class, marker per image."""
    K, A, D = proj_aug.shape
    flat = proj_aug.reshape(K * A, D)
    pts = _pca_2d(flat).numpy()
    pts_orig = _pca_2d(torch.cat([flat, proj_orig], dim=0))[-K:].numpy()

    colors, markers, uniq, color_of = _class_palette_and_marker(classes)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for i in range(K):
        seg = pts[i * A:(i + 1) * A]
        ax.scatter(seg[:, 0], seg[:, 1], s=8, alpha=0.5,
                   color=colors[i], marker=markers[i])
        ax.scatter(pts_orig[i, 0], pts_orig[i, 1],
                   s=110, marker="*", edgecolor="black",
                   facecolor=colors[i], linewidth=0.6)
    handles = [plt.Line2D([], [], marker="o", linestyle="",
                          markerfacecolor=color_of[c], markeredgecolor="none",
                          markersize=7, label=f"class {c}")
               for c in uniq]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title(f"Aug clouds in PCA(2) of projector output  "
                 f"(K={K} imgs × A={A} augs, D={D})  "
                 f"— color = class, marker = image-within-class")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_aug_clouds_pairs(proj_aug, proj_orig, classes, out_path):
    """Raw projector dim pairs: (d0,d1), (d0,d2), (d1,d2), (d2,d3)."""
    K, A, D = proj_aug.shape
    if D < 4:
        return
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    colors, markers, uniq, color_of = _class_palette_and_marker(classes)

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (i, j) in zip(axes.flat, pairs):
        for k in range(K):
            ax.scatter(proj_aug[k, :, i], proj_aug[k, :, j],
                       s=8, alpha=0.5, color=colors[k], marker=markers[k])
            ax.scatter(proj_orig[k, i], proj_orig[k, j],
                       s=100, marker="*", edgecolor="black",
                       facecolor=colors[k], linewidth=0.6)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.axvline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_xlabel(f"proj dim {i}")
        ax.set_ylabel(f"proj dim {j}")
        ax.set_aspect("equal", adjustable="datalim")
    handles = [plt.Line2D([], [], marker="o", linestyle="",
                          markerfacecolor=color_of[c], markeredgecolor="none",
                          markersize=7, label=f"class {c}")
               for c in uniq]
    axes[0, 1].legend(handles=handles, loc="upper right", fontsize=6,
                      framealpha=0.85)
    fig.suptitle(f"Per-image augmentation clouds in raw projector dims  "
                 f"(K={K}, A={A})  — color = class, marker = image")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_marginal_hists(proj_all, out_path):
    """Histogram of every projector dimension vs N(0,1) PDF."""
    D = proj_all.shape[1]
    n_cols = 8
    n_rows = (D + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1.7 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    xs = np.linspace(-4, 4, 200)
    pdf = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)

    arr = proj_all.numpy()
    for d in range(D):
        ax = axes[d // n_cols, d % n_cols]
        ax.hist(arr[:, d], bins=60, range=(-4, 4),
                density=True, color="#4477aa", alpha=0.75)
        ax.plot(xs, pdf, color="black", lw=1.0)
        m, s = arr[:, d].mean(), arr[:, d].std()
        ax.set_title(f"d{d}  μ={m:+.2f} σ={s:.2f}", fontsize=7)
        ax.tick_params(labelsize=6)

    for d in range(D, n_rows * n_cols):
        axes[d // n_cols, d % n_cols].axis("off")

    fig.suptitle(f"Marginal projector-dim histograms vs N(0,1)  "
                 f"(N={arr.shape[0]}, D={D})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_random_dir_hists(proj_all, num_dirs, seed, out_path):
    """Histogram of projections onto random unit directions vs N(0,1)."""
    D = proj_all.shape[1]
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(num_dirs, D, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    slices = (proj_all @ dirs.T).numpy()  # (N, num_dirs)

    n_cols = 4
    n_rows = (num_dirs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    xs = np.linspace(-4, 4, 200)
    pdf = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)

    for k in range(num_dirs):
        ax = axes[k // n_cols, k % n_cols]
        ax.hist(slices[:, k], bins=60, range=(-4, 4),
                density=True, color="#cc6677", alpha=0.75)
        ax.plot(xs, pdf, color="black", lw=1.0)
        m, s = slices[:, k].mean(), slices[:, k].std()
        ax.set_title(f"dir {k}  μ={m:+.2f} σ={s:.2f}", fontsize=7)
        ax.tick_params(labelsize=6)

    for k in range(num_dirs, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.suptitle(f"Random-direction projections vs N(0,1)  "
                 f"(N={proj_all.shape[0]}, K={num_dirs})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_cov_spectrum(proj_all, out_path):
    x = proj_all - proj_all.mean(0, keepdim=True)
    cov = x.T @ x / max(x.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov).flip(0).numpy()  # descending
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(len(eig)), eig, color="#117733", alpha=0.85)
    ax.axhline(1.0, color="black", lw=1.0,
               label="N(0,I) reference (λ=1)")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Projector-output covariance spectrum  "
                 f"(D={cov.shape[0]})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cluster stats
# ---------------------------------------------------------------------------

def cluster_stats(proj_aug: torch.Tensor) -> dict:
    """Within- vs between-image distance ratio for the projection cloud."""
    centroids = proj_aug.mean(dim=1)                               # (N, D)
    radii = (proj_aug - centroids[:, None, :]).norm(dim=-1)       # (N, A)
    intra = radii.mean().item()

    N = centroids.shape[0]
    diffs = centroids[:, None, :] - centroids[None, :, :]
    pairwise = diffs.norm(dim=-1)
    mask = ~torch.eye(N, dtype=torch.bool)
    inter = pairwise[mask].mean().item()

    return {
        "intra_cluster_radius_mean": intra,
        "inter_cluster_distance_mean": inter,
        "intra_over_inter": intra / inter if inter > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str,
                   default=str(REPO_ROOT / "data"))
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-classes", type=int, default=8,
                   help="number of classes for stratified sampling of aug-cloud")
    p.add_argument("--per-class", type=int, default=2,
                   help="images per class for the aug-cloud plot")
    p.add_argument("--num-aug", type=int, default=256,
                   help="number of augmentations per image")
    p.add_argument("--chunk-views", type=int, default=32,
                   help="how many augmented views per GPUAug call")
    p.add_argument("--num-eval", type=int, default=8192,
                   help="number of train images used for marginal/spectrum")
    p.add_argument("--num-random-dirs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=256,
                   help="forward batch size for the marginal collection")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[load] checkpoint: {args.checkpoint}")
    encoder, cfg, ckpt = load_encoder(args.checkpoint, args.data_dir, device)
    print(f"[load] dataset={cfg.dataset}  backbone={cfg.backbone_name}  "
          f"proj_dim={cfg.proj_dim}  crop_size={cfg.crop_size}")
    print(f"[load] epoch={ckpt.get('epoch', '?')}  regularizer={cfg.regularizer}  "
          f"accumulate={cfg.accumulate}")

    train_source, _val_source, _gpu_aug = get_dataloaders(cfg, device)

    indices, classes = sample_stratified_indices(
        train_source, args.num_classes, args.per_class, args.seed)
    K = len(indices)
    print(f"[samp] K={K} imgs from {args.num_classes} classes ({args.per_class}/cls), "
          f"classes={sorted(set(classes.tolist()))}")

    print(f"[aug ] K={K} imgs × A={args.num_aug} augs  "
          f"(chunk_views={args.chunk_views})")
    proj_aug, proj_orig = collect_aug_clouds(
        encoder, train_source, cfg,
        num_aug=args.num_aug,
        chunk_views=args.chunk_views,
        device=device,
        indices=indices,
    )

    print(f"[marg] collecting projector outputs over {args.num_eval} train images")
    proj_all = collect_marginal(
        encoder, train_source, cfg,
        num_eval=args.num_eval,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"[marg] proj_all shape = {tuple(proj_all.shape)}")

    print("[plot] aug_clouds_pca / aug_clouds_pairs")
    plot_aug_clouds_pca(proj_aug, proj_orig, classes,
                        out_dir / "aug_clouds_pca.png")
    plot_aug_clouds_pairs(proj_aug, proj_orig, classes,
                          out_dir / "aug_clouds_pairs.png")

    print("[plot] marginal_hists")
    plot_marginal_hists(proj_all, out_dir / "marginal_hists.png")

    print("[plot] random_dir_hists")
    plot_random_dir_hists(proj_all, args.num_random_dirs, args.seed,
                          out_dir / "random_dir_hists.png")

    print("[plot] cov_spectrum")
    plot_cov_spectrum(proj_all, out_dir / "cov_spectrum.png")

    print("[stat] eval_distribution on train split")
    train_eval_loader = InMemoryEvalLoader(train_source,
                                           batch_size=args.batch_size)
    dist_metrics = eval_distribution(encoder, train_eval_loader, cfg)

    cl_stats = cluster_stats(proj_aug)
    stats = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(ckpt.get("epoch", -1)),
        "dataset": cfg.dataset,
        "backbone": cfg.backbone_name,
        "regularizer": cfg.regularizer,
        "accumulate": bool(cfg.accumulate),
        "proj_dim": int(cfg.proj_dim),
        "num_images": int(K),
        "num_classes": int(args.num_classes),
        "per_class": int(args.per_class),
        "classes": classes.tolist(),
        "indices": indices.tolist(),
        "num_aug": int(args.num_aug),
        "num_eval": int(proj_all.shape[0]),
        "distribution": {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in dist_metrics.items()},
        "clusters": cl_stats,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] wrote artefacts to {out_dir}")


if __name__ == "__main__":
    main()
