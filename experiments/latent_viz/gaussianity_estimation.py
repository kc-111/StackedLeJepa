"""Per-image Gaussianity / variance-around-z_orig analysis for LeJEPA.

For K class-stratified train images, generates N_max augmentations of each
in the projector space and characterizes the conditional displacement
distribution

    p(z_aug | z_orig, Σ_i)   with Σ_i = E[(z_aug - z_orig)(z_aug - z_orig)ᵀ]

— i.e. the covariance is *centered on z_orig*, not on the empirical mean.
This is the matrix the SSL invariance loss ‖z_aug - z_orig‖² actually sees
(its trace is the per-image expected loss).

Per-checkpoint outputs (in ``out_dir/<tag>/``):
  sigma_spectrum.png        sorted eigenvalues of Σᵢ per image (log y, by class)
  loss_noise_vs_N.png       mean ± IQR of (1/N) Σ ‖d‖² vs N (log-log)
  sigma_error_vs_N.png      ‖Σ̂_N - Σᵢ‖_F / ‖Σᵢ‖_F vs N (log-log)
  marginals_standardized.png 4×4 hist of (z_aug-z_orig)_d / σᵢ,d vs N(0,1)
  mahalanobis_qq.png        empirical χ² QQ vs χ²(D) reference
  stats.json                per-image tr, ‖Σ‖_F, cond, eigvals, top eigvec

Cross-epoch outputs (when ≥2 ``--checkpoints`` are passed, in ``out_dir/``):
  epoch_trace.png                tr(Σᵢ) vs epoch
  epoch_normalized_dist.png      ‖Σᵢ(t)/tr − Σᵢ(T)/tr‖_F vs epoch
  epoch_top_eigvec_alignment.png |⟨v₁(t), v₁(T)⟩| vs epoch
  epoch_spectrum_overlay.png     small multiples of eigvals(Σᵢ) at each t

Usage:
    python experiments/latent_viz/gaussianity_estimation.py \
      --checkpoints \
          runs/.../epoch25.pt runs/.../epoch100.pt \
          runs/.../epoch200.pt runs/.../final.pt \
      --out-dir experiments/latent_viz/results/gaussianity_atto_w1_pooled
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
from data import GPUAug, InMemoryGPUDataset, get_dataloaders
from models import LeJEPAEncoder


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
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder, cfg, ckpt


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def sample_stratified_indices(train_source, num_classes, per_class, seed):
    """Pick ``num_classes`` distinct labels and ``per_class`` indices in each.

    Returns:
        indices:  (K,) long tensor of indices into ``train_source.images``
        classes:  (K,) long tensor of class id per selected image
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    labels_cpu = train_source.labels.cpu()
    unique = labels_cpu.unique().tolist()

    perm = torch.randperm(len(unique), generator=g)
    chosen_classes = sorted(unique[i] for i in perm[:num_classes].tolist())

    sel_indices, sel_classes = [], []
    for c in chosen_classes:
        class_idx = (labels_cpu == c).nonzero(as_tuple=True)[0]
        pick = class_idx[torch.randperm(len(class_idx), generator=g)[:per_class]]
        sel_indices.append(pick)
        sel_classes.extend([int(c)] * pick.numel())

    indices = torch.cat(sel_indices)
    classes = torch.tensor(sel_classes, dtype=torch.long)
    return indices, classes


# ---------------------------------------------------------------------------
# Latent collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def project_originals(encoder, train_source, indices, cfg, device):
    """Center-crop + ImageNet-normalize the selected images and project."""
    src_uint8 = train_source.images[indices.to(train_source.images.device)]
    src = src_uint8.float() / 255.0  # (K, 3, H, W) at crop_size+32
    cs = cfg.crop_size
    H, W = src.shape[-2:]
    top = (H - cs) // 2
    left = (W - cs) // 2
    src = src[:, :, top:top + cs, left:left + cs]

    # Use a fresh GPUAug instance only for its normalization buffers.
    aug_module = GPUAug(num_aug_views=1, crop_size=cs).to(device)
    src = (src - aug_module.mean) / aug_module.std
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, proj = encoder(src)
    return proj.float()  # (K, D)


@torch.no_grad()
def collect_aug_displacements(encoder, train_source, indices, z_orig, cfg,
                              num_aug, chunk_views, device):
    """Sample ``num_aug`` random aug projections per image, return displacements.

    Returns:
        d: (K, num_aug, D) tensor of (z_aug − z_orig) on CPU
    """
    src_imgs = (train_source.images[indices.to(train_source.images.device)]
                .float() / 255.0)

    chunk_views = max(1, min(chunk_views, num_aug))
    n_calls = (num_aug + chunk_views - 1) // chunk_views

    aug_module = GPUAug(
        num_aug_views=chunk_views,
        crop_size=cfg.crop_size,
        crop_scale=tuple(cfg.crop_scale),
    ).to(device)

    chunks = []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(n_calls):
            _, aug = aug_module(src_imgs)             # (V, K, 3, cs, cs)
            V, K = aug.shape[:2]
            flat = aug.reshape(V * K, *aug.shape[2:])
            _, proj = encoder(flat)
            proj = proj.float().view(V, K, -1)         # (V, K, D)
            chunks.append(proj)

    proj_aug = torch.cat(chunks, dim=0)[:num_aug]      # (num_aug, K, D)
    proj_aug = proj_aug.transpose(0, 1).contiguous()   # (K, num_aug, D)

    d = proj_aug - z_orig.unsqueeze(1)                 # (K, num_aug, D)
    return d.cpu()


# ---------------------------------------------------------------------------
# Per-image stats
# ---------------------------------------------------------------------------

def per_image_sigma_centered(d):
    """Σᵢ = (1/N) Σ dᵢdᵢᵀ — covariance centered on z_orig (no demean)."""
    K, N, D = d.shape
    return torch.einsum("kni,knj->kij", d, d) / N      # (K, D, D)


def sigma_diagnostics(sigma):
    """Per-image trace, Frobenius, condition number, eigvals, top eigvec."""
    K, D, _ = sigma.shape
    eye = torch.eye(D)
    out = []
    for i in range(K):
        S = sigma[i]
        eigvals, eigvecs = torch.linalg.eigh(S + 1e-12 * eye)
        eigvals = eigvals.flip(0)                      # descending
        eigvecs = eigvecs.flip(1)
        tr = float(S.diagonal().sum())
        fro = float(S.norm())
        cond = float(eigvals[0] / max(eigvals[-1].item(), 1e-12))
        out.append({
            "trace": tr,
            "frob": fro,
            "cond": cond,
            "eigvals": eigvals.tolist(),
            "top_eigvec": eigvecs[:, 0].tolist(),
        })
    return out


# ---------------------------------------------------------------------------
# Convergence sweeps
# ---------------------------------------------------------------------------

def _sample_sizes(n_max):
    sizes = []
    n = 1
    while n < n_max:
        sizes.append(n)
        n *= 2
    sizes.append(n_max)
    return sizes


def loss_noise_sweep(d, num_trials, seed):
    """For each N in {1,2,4,...,N_max}, M trials of (1/N) Σ ‖dᵢ‖² per image.

    Returns:
        sizes: list[int] of N values
        means: (K, len(sizes)) array of trial means
        stds:  (K, len(sizes)) array of trial stds
    """
    K, N_max, D = d.shape
    sq = (d ** 2).sum(dim=-1)                          # (K, N_max)
    sizes = _sample_sizes(N_max)

    g = torch.Generator().manual_seed(seed)
    means = np.zeros((K, len(sizes)))
    stds = np.zeros((K, len(sizes)))

    for si, N in enumerate(sizes):
        if N == N_max:
            est = sq.mean(dim=-1, keepdim=True)        # (K, 1) — single deterministic value
            means[:, si] = est.squeeze(-1).numpy()
            stds[:, si] = 0.0
            continue
        trials = torch.empty(K, num_trials)
        for t in range(num_trials):
            idx = torch.randint(0, N_max, (N,), generator=g)
            trials[:, t] = sq[:, idx].mean(dim=-1)
        means[:, si] = trials.mean(dim=-1).numpy()
        stds[:, si] = trials.std(dim=-1).numpy()
    return sizes, means, stds


def sigma_error_sweep(d, sigma_full, num_trials, seed):
    """For each N, M trials of ‖Σ̂_N − Σ_full‖_F / ‖Σ_full‖_F per image."""
    K, N_max, D = d.shape
    sizes = _sample_sizes(N_max)
    fro_full = sigma_full.reshape(K, -1).norm(dim=-1).clamp_min(1e-12)  # (K,)

    g = torch.Generator().manual_seed(seed + 1)
    rel_err_mean = np.zeros((K, len(sizes)))
    rel_err_std = np.zeros((K, len(sizes)))

    for si, N in enumerate(sizes):
        if N == N_max:
            sigma_hat = torch.einsum("kni,knj->kij", d, d) / N
            rel_err = ((sigma_hat - sigma_full).reshape(K, -1).norm(dim=-1)
                       / fro_full)
            rel_err_mean[:, si] = rel_err.numpy()
            rel_err_std[:, si] = 0.0
            continue
        trials = torch.empty(K, num_trials)
        for t in range(num_trials):
            idx = torch.randint(0, N_max, (N,), generator=g)
            sub = d[:, idx, :]                          # (K, N, D)
            sigma_hat = torch.einsum("kni,knj->kij", sub, sub) / N
            err = ((sigma_hat - sigma_full).reshape(K, -1).norm(dim=-1)
                   / fro_full)
            trials[:, t] = err
        rel_err_mean[:, si] = trials.mean(dim=-1).numpy()
        rel_err_std[:, si] = trials.std(dim=-1).numpy()
    return sizes, rel_err_mean, rel_err_std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def class_palette(classes):
    uniq = sorted(set(int(c) for c in classes.tolist()))
    cmap = plt.get_cmap("tab10", max(len(uniq), 1))
    return {c: cmap(i) for i, c in enumerate(uniq)}


def plot_sigma_spectrum(diag, classes, out_path):
    palette = class_palette(classes)
    fig, ax = plt.subplots(figsize=(7, 5))
    seen = set()
    for i, dd in enumerate(diag):
        c = int(classes[i])
        label = f"class {c}" if c not in seen else None
        seen.add(c)
        ax.semilogy(np.arange(len(dd["eigvals"])), dd["eigvals"],
                    color=palette[c], alpha=0.7, lw=1.2, label=label)
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel("eigenvalue (log)")
    ax.set_title(f"Per-image Σ centered on z_orig — eigenvalue spectrum  "
                 f"(K={len(diag)})")
    ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_loss_noise(sizes, means, stds, classes, out_path):
    palette = class_palette(classes)
    fig, ax = plt.subplots(figsize=(7, 5))
    K = means.shape[0]
    seen = set()
    for i in range(K):
        c = int(classes[i])
        label = f"class {c}" if c not in seen else None
        seen.add(c)
        ax.errorbar(sizes, means[i], yerr=stds[i],
                    color=palette[c], alpha=0.65, lw=1.0,
                    capsize=2, label=label)
    median = np.median(means, axis=0)
    ax.plot(sizes, median, color="black", lw=2.0, label="median over images")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (samples per image)")
    ax.set_ylabel(r"$(1/N)\,\sum\|z_{aug}-z_{orig}\|^2$  (≈ tr Σᵢ at large N)")
    ax.set_title("Per-step SSL loss-estimator vs sample budget  "
                 "(N=1 is the SSL gradient signal)")
    ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sigma_error(sizes, mean, std, classes, out_path):
    palette = class_palette(classes)
    fig, ax = plt.subplots(figsize=(7, 5))
    K = mean.shape[0]
    seen = set()
    for i in range(K):
        c = int(classes[i])
        label = f"class {c}" if c not in seen else None
        seen.add(c)
        ax.errorbar(sizes, mean[i], yerr=std[i],
                    color=palette[c], alpha=0.65, lw=1.0,
                    capsize=2, label=label)
    median = np.median(mean, axis=0)
    ax.plot(sizes, median, color="black", lw=2.0, label="median over images")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (samples per image)")
    ax.set_ylabel(r"$\|\hat\Sigma_N - \Sigma_i\|_F\,/\,\|\Sigma_i\|_F$")
    ax.set_title("Per-image covariance estimator error vs sample budget")
    ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_marginals_standardized(d, sigma, classes, out_path):
    """Per image, histogram of (z_aug-z_orig)_dim / σ_dim pooled across all dims."""
    K = d.shape[0]
    palette = class_palette(classes)
    n_cols = 4
    n_rows = (K + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    xs = np.linspace(-4, 4, 200)
    pdf = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)

    for i in range(K):
        ax = axes[i // n_cols, i % n_cols]
        std = torch.diagonal(sigma[i]).clamp_min(1e-12).sqrt()  # (D,)
        s = (d[i] / std.unsqueeze(0)).flatten().numpy()
        ax.hist(s, bins=80, range=(-4, 4), density=True,
                color=palette[int(classes[i])], alpha=0.75)
        ax.plot(xs, pdf, color="black", lw=1.0)
        ax.set_title(f"img {i}  cls {int(classes[i])}", fontsize=7)
        ax.tick_params(labelsize=6)
    for k in range(K, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.suptitle(r"Standardized displacements $(z_{aug}-z_{orig})_d/\sigma_{i,d}$"
                 " pooled over D=32 dims, vs N(0,1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_mahalanobis_qq(d, sigma, classes, out_path):
    """Empirical χ²(D) QQ plot using d²ᵢ = dᵀ Σᵢ⁻¹ d per image."""
    K, N, D = d.shape
    palette = class_palette(classes)
    fig, ax = plt.subplots(figsize=(7, 6))

    # χ²(D) reference quantiles via Monte-Carlo sampling (no scipy dependency).
    g = torch.Generator().manual_seed(0)
    n_ref = max(N, 20000)
    ref = torch.randn(n_ref, D, generator=g).pow(2).sum(dim=-1)
    ref_sorted, _ = ref.sort()
    quant_idx = ((torch.arange(N) + 0.5) / N * n_ref).long().clamp(max=n_ref - 1)
    chi2_q = ref_sorted[quant_idx].numpy()

    seen = set()
    for i in range(K):
        S = sigma[i] + 1e-6 * torch.eye(D)
        L = torch.linalg.cholesky(S)
        sol = torch.cholesky_solve(d[i].T, L)            # (D, N)
        d2 = (d[i] * sol.T).sum(dim=-1).numpy()
        d2_sorted = np.sort(d2)
        c = int(classes[i])
        label = f"class {c}" if c not in seen else None
        seen.add(c)
        ax.plot(chi2_q, d2_sorted, color=palette[c], alpha=0.7, lw=1.0,
                label=label)
    lim_max = max(chi2_q[-1], float(d2_sorted.max()))
    ax.plot([0, lim_max], [0, lim_max], color="black", lw=1.0, ls="--",
            label=r"$\chi^2(D)$ reference")
    ax.set_xlabel(r"$\chi^2(D)$ quantile")
    ax.set_ylabel(r"empirical $d^2 = (z_{aug}-z_{orig})^\top\Sigma_i^{-1}(z_{aug}-z_{orig})$")
    ax.set_title(f"Mahalanobis χ² QQ-plot per image  (D={D})")
    ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-epoch comparison plots
# ---------------------------------------------------------------------------

def plot_epoch_trace(epoch_stats, classes, out_path):
    palette = class_palette(classes)
    epochs = [s["epoch"] for s in epoch_stats]
    K = len(classes)
    fig, ax = plt.subplots(figsize=(7, 5))
    seen = set()
    for i in range(K):
        traces = [s["per_image"][i]["trace"] for s in epoch_stats]
        c = int(classes[i])
        label = f"class {c}" if c not in seen else None
        seen.add(c)
        ax.semilogy(epochs, traces, marker="o", color=palette[c],
                    alpha=0.7, lw=1.2, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\mathrm{tr}(\Sigma_i)$  (= per-image expected loss)")
    ax.set_title("Per-image trace of Σ across epochs")
    ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_epoch_normalized_dist(epoch_stats, classes, out_path):
    """‖Σᵢ(t)/tr - Σᵢ(T)/tr‖_F vs epoch — shape distance to final epoch."""
    palette = class_palette(classes)
    epochs = [s["epoch"] for s in epoch_stats]
    K = len(classes)
    final = epoch_stats[-1]["sigma"]
    final_norm = torch.stack(
        [final[i] / max(final[i].diagonal().sum().item(), 1e-12)
         for i in range(K)])

    fig, ax = plt.subplots(figsize=(7, 5))
    seen = set()
    for i in range(K):
        ds = []
        for s in epoch_stats:
            S = s["sigma"][i]
            S_norm = S / max(S.diagonal().sum().item(), 1e-12)
            ds.append(float((S_norm - final_norm[i]).norm()))
        c = int(classes[i])
        label = f"class {c}" if c not in seen else None
        seen.add(c)
        ax.plot(epochs, ds, marker="o", color=palette[c],
                alpha=0.7, lw=1.2, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|\Sigma_i(t)/\mathrm{tr} - \Sigma_i(T)/\mathrm{tr}\|_F$")
    ax.set_title("Trace-normalized shape distance of Σᵢ to final epoch")
    ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_epoch_top_eigvec_alignment(epoch_stats, classes, out_path):
    """|⟨v₁(t), v₁(T)⟩| per image vs epoch."""
    palette = class_palette(classes)
    epochs = [s["epoch"] for s in epoch_stats]
    K = len(classes)
    final_v1 = [torch.tensor(epoch_stats[-1]["per_image"][i]["top_eigvec"])
                for i in range(K)]

    fig, ax = plt.subplots(figsize=(7, 5))
    seen = set()
    for i in range(K):
        cosines = []
        for s in epoch_stats:
            v1 = torch.tensor(s["per_image"][i]["top_eigvec"])
            cosines.append(float(torch.abs(torch.dot(v1, final_v1[i]))))
        c = int(classes[i])
        label = f"class {c}" if c not in seen else None
        seen.add(c)
        ax.plot(epochs, cosines, marker="o", color=palette[c],
                alpha=0.7, lw=1.2, label=label)
    ax.axhline(1.0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$|\langle v_1^{(t)}, v_1^{(T)} \rangle|$")
    ax.set_title("Top eigenvector alignment to final epoch")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_epoch_displacement_overlay(epoch_stats, classes, out_path):
    """Per-image standardized-displacement histogram overlaid across epochs.

    Standardize each epoch's displacements by the **final epoch's** per-coord
    σ so all epochs share the same scale; if the curves overlay, the
    projector-space noise distribution that the loss sees is stable across
    training.
    """
    K = len(classes)
    n_epochs = len(epoch_stats)
    epoch_cmap = plt.get_cmap("viridis", max(n_epochs, 1))

    final_d = epoch_stats[-1]["d"]                          # (K, N, D)
    final_sigma = epoch_stats[-1]["sigma"]                  # (K, D, D)
    ref_std = torch.diagonal(final_sigma, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()

    n_cols = 4
    n_rows = (K + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    xs = np.linspace(-4, 4, 200)
    pdf = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)

    for i in range(K):
        ax = axes[i // n_cols, i % n_cols]
        for ei, s in enumerate(epoch_stats):
            d_std = (s["d"][i] / ref_std[i].unsqueeze(0)).flatten().numpy()
            ax.hist(d_std, bins=80, range=(-4, 4), density=True,
                    color=epoch_cmap(ei), alpha=0.4, histtype="stepfilled",
                    label=f"ep {s['epoch']}" if i == 0 else None)
        ax.plot(xs, pdf, color="black", lw=1.0)
        ax.set_title(f"img {i}  cls {int(classes[i])}", fontsize=7)
        ax.tick_params(labelsize=6)
    for k in range(K, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")
    axes[0, 0].legend(fontsize=6, framealpha=0.85, loc="upper right")
    fig.suptitle(r"$(z_{aug}-z_{orig})_d / \sigma_{i,d}^{(\mathrm{final})}$"
                 " pooled over D dims, overlaid across epochs", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_epoch_loss_distribution(epoch_stats, classes, out_path):
    """Per-image distribution of the per-step loss ‖d‖² overlaid across epochs.

    This is exactly the per-step gradient signal magnitude an SSL training
    step at this epoch would see. If overlaid curves coincide, one slow-
    refreshed noise model carries the same loss signal across all epochs.
    """
    K = len(classes)
    n_epochs = len(epoch_stats)
    epoch_cmap = plt.get_cmap("viridis", max(n_epochs, 1))

    n_cols = 4
    n_rows = (K + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows),
                             sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    for i in range(K):
        ax = axes[i // n_cols, i % n_cols]
        for ei, s in enumerate(epoch_stats):
            sq = (s["d"][i] ** 2).sum(dim=-1).numpy()
            ax.hist(sq, bins=60, density=True,
                    color=epoch_cmap(ei), alpha=0.45, histtype="stepfilled",
                    label=f"ep {s['epoch']}" if i == 0 else None)
        ax.set_title(f"img {i}  cls {int(classes[i])}", fontsize=7)
        ax.tick_params(labelsize=6)
    for k in range(K, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")
    axes[0, 0].legend(fontsize=6, framealpha=0.85, loc="upper right")
    fig.suptitle(r"per-step SSL loss $\|z_{aug}-z_{orig}\|^2$ "
                 "distribution overlaid across epochs", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_epoch_sigma_pair_distance(epoch_stats, classes, out_path):
    """T×T heatmap of trace-normalized ‖Σᵢ(t₁) − Σᵢ(t₂)‖_F averaged over images."""
    T = len(epoch_stats)
    K = len(classes)
    epochs = [s["epoch"] for s in epoch_stats]

    norm_sigmas = []
    for s in epoch_stats:
        per_img = []
        for i in range(K):
            S = s["sigma"][i]
            per_img.append(S / max(S.diagonal().sum().item(), 1e-12))
        norm_sigmas.append(torch.stack(per_img))   # (K, D, D)

    M = np.zeros((T, T))
    for a in range(T):
        for b in range(T):
            diff = (norm_sigmas[a] - norm_sigmas[b]).reshape(K, -1)
            M[a, b] = float(diff.norm(dim=-1).mean())

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(M, cmap="magma", origin="lower")
    ax.set_xticks(range(T)); ax.set_xticklabels([str(e) for e in epochs])
    ax.set_yticks(range(T)); ax.set_yticklabels([str(e) for e in epochs])
    ax.set_xlabel("epoch"); ax.set_ylabel("epoch")
    ax.set_title(r"mean over images of $\|\Sigma_i(t_1)/\mathrm{tr} - \Sigma_i(t_2)/\mathrm{tr}\|_F$")
    for a in range(T):
        for b in range(T):
            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center",
                    color="white" if M[a, b] > M.mean() else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_epoch_spectrum_overlay(epoch_stats, classes, out_path):
    """K small subplots, eigenvalue spectrum at each epoch overlaid per image."""
    K = len(classes)
    palette = class_palette(classes)
    n_cols = 4
    n_rows = (K + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    n_epochs = len(epoch_stats)
    epoch_cmap = plt.get_cmap("viridis", max(n_epochs, 1))

    for i in range(K):
        ax = axes[i // n_cols, i % n_cols]
        for ei, s in enumerate(epoch_stats):
            eig = s["per_image"][i]["eigvals"]
            ax.semilogy(np.arange(len(eig)), eig,
                        color=epoch_cmap(ei), alpha=0.85, lw=1.0,
                        label=f"ep {s['epoch']}" if i == 0 else None)
        ax.set_title(f"img {i}  cls {int(classes[i])}", fontsize=7,
                     color=palette[int(classes[i])])
        ax.tick_params(labelsize=6)
    for k in range(K, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")
    axes[0, 0].legend(fontsize=6, framealpha=0.85, loc="lower left")
    fig.suptitle("Σᵢ eigenvalue spectrum across epochs (per image)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-checkpoint pipeline
# ---------------------------------------------------------------------------

def run_one_checkpoint(checkpoint_path, args, indices, classes, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, cfg, ckpt = load_encoder(checkpoint_path, args.data_dir, device)
    print(f"  [load] {checkpoint_path}  epoch={ckpt.get('epoch', '?')}  "
          f"backbone={cfg.backbone_name}  proj_dim={cfg.proj_dim}")

    train_source, _, _ = get_dataloaders(cfg, device)

    z_orig = project_originals(encoder, train_source, indices, cfg, device)
    print(f"  [orig] z_orig shape = {tuple(z_orig.shape)}")

    print(f"  [aug ] generating {args.num_aug} augs × K={len(indices)} images")
    d = collect_aug_displacements(
        encoder, train_source, indices, z_orig, cfg,
        num_aug=args.num_aug, chunk_views=args.chunk_views, device=device,
    )

    sigma = per_image_sigma_centered(d)               # (K, D, D)
    diag = sigma_diagnostics(sigma)

    print("  [plot] sigma_spectrum / marginals / mahalanobis")
    plot_sigma_spectrum(diag, classes, out_dir / "sigma_spectrum.png")
    plot_marginals_standardized(d, sigma, classes,
                                out_dir / "marginals_standardized.png")
    plot_mahalanobis_qq(d, sigma, classes, out_dir / "mahalanobis_qq.png")

    print("  [sweep] loss-noise convergence")
    sizes, ln_mean, ln_std = loss_noise_sweep(d, args.num_trials, args.seed)
    plot_loss_noise(sizes, ln_mean, ln_std, classes,
                    out_dir / "loss_noise_vs_N.png")

    print("  [sweep] sigma matrix convergence")
    sizes, se_mean, se_std = sigma_error_sweep(d, sigma, args.num_trials, args.seed)
    plot_sigma_error(sizes, se_mean, se_std, classes,
                     out_dir / "sigma_error_vs_N.png")

    stats = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "backbone": cfg.backbone_name,
        "proj_dim": int(cfg.proj_dim),
        "num_images": int(len(indices)),
        "num_aug": int(args.num_aug),
        "classes": classes.tolist(),
        "indices": indices.tolist(),
        "sample_sizes": sizes,
        "loss_noise_mean": ln_mean.tolist(),
        "loss_noise_std": ln_std.tolist(),
        "sigma_error_mean": se_mean.tolist(),
        "sigma_error_std": se_std.tolist(),
        "per_image": diag,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "sigma": sigma,
        "per_image": diag,
        "d": d,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="one or more checkpoint paths")
    p.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data"))
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-classes", type=int, default=8)
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--num-aug", type=int, default=2048)
    p.add_argument("--chunk-views", type=int, default=32)
    p.add_argument("--num-trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pick stratified sample once using the first checkpoint's dataset
    # (the dataset must be identical across all checkpoints — same dataset,
    # same data_dir → same in-memory ordering).
    print(f"[init] loading first checkpoint to set up dataset")
    encoder0, cfg0, _ = load_encoder(args.checkpoints[0], args.data_dir, device)
    train_source0, _, _ = get_dataloaders(cfg0, device)
    indices, classes = sample_stratified_indices(
        train_source0, args.num_classes, args.per_class, args.seed)
    K = len(indices)
    print(f"[init] selected K={K} images from {args.num_classes} classes "
          f"({args.per_class}/class), classes={sorted(set(classes.tolist()))}")
    del encoder0, train_source0

    epoch_results = []
    for ckpt_path in args.checkpoints:
        tag = Path(ckpt_path).stem
        print(f"[run ] {tag}")
        epoch_dir = out_dir / tag
        result = run_one_checkpoint(
            ckpt_path, args, indices, classes, device, epoch_dir)
        epoch_results.append(result)

    if len(epoch_results) >= 2:
        epoch_results.sort(key=lambda r: r["epoch"])
        print("[xepoch] cross-epoch comparison")
        plot_epoch_trace(epoch_results, classes, out_dir / "epoch_trace.png")
        plot_epoch_normalized_dist(epoch_results, classes,
                                   out_dir / "epoch_normalized_dist.png")
        plot_epoch_top_eigvec_alignment(
            epoch_results, classes, out_dir / "epoch_top_eigvec_alignment.png")
        plot_epoch_spectrum_overlay(
            epoch_results, classes, out_dir / "epoch_spectrum_overlay.png")
        plot_epoch_displacement_overlay(
            epoch_results, classes, out_dir / "epoch_displacement_overlay.png")
        plot_epoch_loss_distribution(
            epoch_results, classes, out_dir / "epoch_loss_distribution.png")
        plot_epoch_sigma_pair_distance(
            epoch_results, classes, out_dir / "epoch_sigma_pair_distance.png")

        cross = {
            "checkpoints": [r["checkpoint"] for r in epoch_results],
            "epochs": [r["epoch"] for r in epoch_results],
            "classes": classes.tolist(),
            "indices": indices.tolist(),
        }
        with open(out_dir / "cross_epoch_stats.json", "w") as f:
            json.dump(cross, f, indent=2)

    print(f"[done] artefacts in {out_dir}")


if __name__ == "__main__":
    main()
