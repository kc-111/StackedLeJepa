"""Invariance dynamics in 2D — MLP backbone, full batch, view-count study.

Three by three: each column is a V-configuration, each row is a loss/target
variant. All panels share MLP init, data, and step count so differences are
entirely due to the loss formulation / view setup.

Columns (views per image):
    V=2  all grad                     — LeJEPA baseline
    V=4  all grad                     — more views, more pairs for invariance
    V=4  (2 grad / 2 detached)        — extra_view_factor = 1 analog: extras
                                        feed invariance through the mean but
                                        contribute no direct gradient term

Rows (loss + stop-grad):
    MSE                                — SG on z̄ is a no-op (Σ(z_v - z̄) = 0)
    MAE, no SG on z̄                   — centered by R̄; dynamics differ from MSE
    MAE, SG on z̄                     — chain through z̄ cut; non-trivial for MAE

Setup:
    - M = 16 "images" on a 4×4 grid so initial z_v clouds are clearly spread
    - Residual MLP (z = x + net(x)) with near-zero final layer → at init
      z ≈ x, so the initial spread matches the data spread (prior viz had
      near-zero outputs because of the MLP's small init, which looked wrong).
    - Full batch every step (this viz is *not* about mini-batching — that's
      a different question best shown with well-separated blobs and sample
      sampling; we can build a second viz for it).

Run:
    python experiments/synthetic/invariance_viz_mlp.py
"""

import math
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model: residual MLP so outputs inherit input spread at init
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)   # near-identity at init
            self.net[-1].bias.zero_()

    def forward(self, x):
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Data: M images on a grid of well-separated blobs
# ---------------------------------------------------------------------------

def _rotations(V: int) -> torch.Tensor:
    thetas = torch.linspace(0, 2 * math.pi, V + 1)[:V]
    Rs = torch.zeros(V, 2, 2)
    Rs[:, 0, 0] = thetas.cos();  Rs[:, 0, 1] = -thetas.sin()
    Rs[:, 1, 0] = thetas.sin();  Rs[:, 1, 1] = thetas.cos()
    return Rs


def _images(M: int) -> torch.Tensor:
    side = int(math.ceil(math.sqrt(M)))
    grid = torch.linspace(-2.2, 2.2, side)
    xs, ys = torch.meshgrid(grid, grid, indexing="xy")
    pts = torch.stack([xs.flatten(), ys.flatten()], dim=1)[:M]
    torch.manual_seed(1)
    return pts + 0.08 * torch.randn(M, 2)


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------

def run(M: int, V: int, n_grad: int, loss_type: str, stop_grad_mean: bool,
        steps: int, lr: float, mlp_seed: int, margin: float = None):
    """If ``margin`` is not None, the loss becomes ``max(inv - margin, 0)``:
    gradient is identical to the plain loss while inv > margin, and zero
    below — training plateaus at `inv ≈ margin` instead of collapsing to 0.
    """
    torch.manual_seed(mlp_seed)
    mlp = ResidualMLP(hidden=32)
    opt = torch.optim.SGD(mlp.parameters(), lr=lr)

    Rs = _rotations(V)
    X = _images(M)

    def snapshot():
        with torch.no_grad():
            aug = torch.einsum("vij,mj->vmi", Rs, X)
            return mlp(aug.reshape(-1, 2)).reshape(V, M, 2).numpy()

    z_traj, losses = [snapshot()], [0.0]
    for _ in range(steps):
        aug = torch.einsum("vij,mj->vmi", Rs, X)
        z = mlp(aug.reshape(-1, 2)).reshape(V, M, 2)

        # Detach the last (V - n_grad) views so they feed z̄ without
        # contributing their own gradient term to dL/dθ.
        if n_grad < V:
            z_grad = z[:n_grad]
            z_det = z[n_grad:].detach()
            z = torch.cat([z_grad, z_det], dim=0)

        # mean = z.mean(dim=0, keepdim=True)
        # if stop_grad_mean:
        #     mean = mean.detach()
        # diff = z - mean
        # ALL pairwise differences in the first dim
        diff = (z[None, :] - z[:, None, :])
        raw = diff.pow(2).mean() if loss_type == "mse" else diff.abs().mean()
        loss = (raw - margin).clamp(min=0) if margin is not None else raw

        opt.zero_grad(); loss.backward(); opt.step()
        z_traj.append(snapshot()); losses.append(raw.item())  # log raw inv for comparison
        print(losses[-1])

    return np.array(z_traj), np.array(losses)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

M = 16
STEPS = 400
LR = 0.03
MLP_SEED = 7
TRAIL = 25
FPS = 20

# 2 rows × 3 cols: no-margin vs margin, across three view configs
EPS = 0.3  # margin threshold — training stops pulling once inv <= EPS

VARIANTS = [
    # Row 1: no margin (baseline collapse)
    ("MSE  V=2, no margin",                    dict(V=2, n_grad=2, loss_type="mse", stop_grad_mean=False, margin=None)),
    ("MSE  V=4, no margin",                    dict(V=4, n_grad=4, loss_type="mse", stop_grad_mean=False, margin=None)),
    ("MSE  V=4, 2 grad / 2 det, no margin",    dict(V=4, n_grad=2, loss_type="mse", stop_grad_mean=False, margin=None)),
    # Row 2: margin max(inv - ε, 0), same three view configs
    (f"MSE  V=2, margin ε={EPS}",              dict(V=2, n_grad=2, loss_type="mse", stop_grad_mean=False, margin=EPS)),
    (f"MSE  V=4, margin ε={EPS}",              dict(V=4, n_grad=4, loss_type="mse", stop_grad_mean=False, margin=EPS)),
    (f"MSE  V=4, 2 grad / 2 det, margin ε={EPS}", dict(V=4, n_grad=2, loss_type="mse", stop_grad_mean=False, margin=EPS)),
]


def main():
    results = [run(M=M, steps=STEPS, lr=LR, mlp_seed=MLP_SEED, **kw)
               for _, kw in VARIANTS]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    cmap = plt.get_cmap("tab20")
    per_axis = []

    for ax, (title, kw), (z_traj, _) in zip(axes, VARIANTS, results):
        V = z_traj.shape[1]
        n_grad = kw["n_grad"]
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.3)
        ax.set_title(title, fontsize=10)

        # Initial positions — hollow rings, persistent reference
        for m in range(M):
            for v in range(V):
                color = cmap(m % 20)
                ax.plot(z_traj[0, v, m, 0], z_traj[0, v, m, 1], "o",
                        mfc="none", mec=color, ms=4, alpha=0.4, zorder=1)

        lines = [[None] * M for _ in range(V)]
        scatters = [[None] * M for _ in range(V)]
        for v in range(V):
            is_det = v >= n_grad
            for m in range(M):
                color = cmap(m % 20)
                (ln,) = ax.plot([], [], color=color,
                                alpha=0.35 if is_det else 0.55,
                                linewidth=0.7,
                                linestyle="--" if is_det else "-", zorder=2)
                sc = ax.scatter([], [], color=color, s=28, zorder=3,
                                edgecolors="black" if is_det else "white",
                                linewidths=0.5)
                lines[v][m] = ln; scatters[v][m] = sc

        (cline,) = ax.plot([], [], color="red", linewidth=1.0,
                           linestyle=":", alpha=0.7, zorder=4)
        cscat = ax.scatter([], [], color="red", s=150, marker="*", zorder=5,
                           edgecolors="black", linewidths=0.6)
        text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=8,
                       va="top",
                       bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))
        per_axis.append((lines, scatters, cline, cscat, text, z_traj))

    fig.suptitle(
        f"Invariance: plain vs. margin max(inv - ε, 0), ε = {EPS}.  "
        f"M={M} grid blobs, full batch, residual MLP. "
        "Top row: plain MSE (collapse). Bottom row: margin MSE (plateau at inv ≈ ε).",
        fontsize=11,
    )
    plt.tight_layout()

    def update(frame):
        arts = []
        for (lines, scatters, cline, cscat, text, z_traj), \
                (_, _), (_, losses) in zip(per_axis, VARIANTS, results):
            V = z_traj.shape[1]
            lo = max(0, frame - TRAIL)
            for v in range(V):
                for m in range(M):
                    seg = z_traj[lo:frame + 1, v, m, :]
                    lines[v][m].set_data(seg[:, 0], seg[:, 1])
                    scatters[v][m].set_offsets(z_traj[frame, v, m, :])

            # Centroid: mean of all M*V output points
            current = z_traj[frame].reshape(-1, 2).mean(axis=0)
            history = z_traj[lo:frame + 1].reshape(
                z_traj[lo:frame + 1].shape[0], -1, 2).mean(axis=1)
            cline.set_data(history[:, 0], history[:, 1])
            cscat.set_offsets(current)

            text.set_text(
                f"step={frame}  loss={losses[frame]:.4f}\n"
                f"centroid=({current[0]:+.2f}, {current[1]:+.2f})"
            )
            for v in range(V):
                for m in range(M):
                    arts.extend([lines[v][m], scatters[v][m]])
            arts.extend([cline, cscat, text])
        return arts

    ani = animation.FuncAnimation(
        fig, update, frames=STEPS + 1, interval=1000 // FPS, blit=False)
    out = Path(__file__).resolve().parent / "invariance_viz_mlp.gif"
    ani.save(str(out), writer="pillow", fps=FPS)
    print(f"Saved: {out}")

    print("\nFinal collapse centers:")
    for (title, _), (z_traj, _) in zip(VARIANTS, results):
        c = z_traj[-1].reshape(-1, 2).mean(axis=0)
        print(f"  {title:40s}  ({c[0]:+.3f}, {c[1]:+.3f})")


if __name__ == "__main__":
    main()
