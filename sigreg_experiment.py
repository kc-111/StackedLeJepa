"""
SIGReg Isotropic Gaussian Convergence Experiment

Demonstrates that optimizing the SIGReg loss pushes an arbitrary
non-Gaussian distribution toward an isotropic Gaussian N(0, I).

Pipeline:
    Source points (K, N) -> Fixed random projection (K, M) -> Learned MLP (K, D=N) -> SIGReg loss

Example commands:
    # N=3, uniform square, full-batch
    python sigreg_experiment.py --distribution uniform_square --input-dim 3 --num-points 4096 --steps 5000 --save-dir sigreg_results_3d

    # N=4 (default), blobs, full-batch
    python sigreg_experiment.py --distribution blobs --num-points 4096 --steps 5000 --save-dir sigreg_results_4d

    # N=4, ring, mini-batch with snapshots
    python sigreg_experiment.py --distribution ring --num-points 8192 --steps 10000 --batch-mode mini --batch-size 512 --snapshot-interval 1000 --save-dir sigreg_results_mini
"""

import os
import sys
import math
import argparse

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module import SIGReg


class DeepMLP(nn.Module):
    """Multi-layer MLP for the experiment (more expressive than module.MLP)."""

    def __init__(self, input_dim, hidden_dim, output_dim, depth=3):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(in_d, hidden_dim), nn.GELU()]
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_ring(K, N, radius=3.0):
    """Points uniformly on a hypersphere shell of fixed radius."""
    z = torch.randn(K, N)
    z = z / z.norm(dim=1, keepdim=True)
    return z * radius


def generate_uniform_square(K, N, half_width=3.0):
    """Uniform in [-half_width, half_width]^N."""
    return torch.rand(K, N) * 2 * half_width - half_width


def generate_blobs(K, N, num_blobs=4, spread=4.0, blob_std=0.3):
    """Mixture of well-separated Gaussian clusters."""
    centers = torch.zeros(num_blobs, N)
    angles = torch.linspace(0, 2 * math.pi, num_blobs + 1)[:num_blobs]
    centers[:, 0] = spread * angles.cos()
    if N >= 2:
        centers[:, 1] = spread * angles.sin()
    assignments = torch.randint(0, num_blobs, (K,))
    return centers[assignments] + torch.randn(K, N) * blob_std


def generate_spiral(K, N, turns=2.0, noise=0.1):
    """Archimedean spiral (2D structure, zero-padded for N > 2)."""
    t = torch.linspace(0, turns * 2 * math.pi, K) + torch.randn(K) * noise
    r = torch.linspace(0.5, 3.0, K)
    points = torch.zeros(K, N)
    points[:, 0] = r * t.cos()
    if N >= 2:
        points[:, 1] = r * t.sin()
    return points


GENERATORS = {
    "ring": generate_ring,
    "uniform_square": generate_uniform_square,
    "blobs": generate_blobs,
    "spiral": generate_spiral,
}


def generate_data(name, K, N):
    return GENERATORS[name](K, N)


# ---------------------------------------------------------------------------
# Fixed random projection
# ---------------------------------------------------------------------------

def make_fixed_projection(N, M, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(N, M, generator=gen)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_loss_curve(losses, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("SIGReg Loss")
    ax.set_title("SIGReg Loss During Training")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _auto_lim(vals_x, vals_y, margin=0.15):
    """Compute equal-aspect axis limits with margin, centered on the data."""
    lo_x, hi_x = vals_x.min(), vals_x.max()
    lo_y, hi_y = vals_y.min(), vals_y.max()
    span = max(hi_x - lo_x, hi_y - lo_y)
    if span < 1e-6:
        span = 1.0
    pad = span * margin
    cx, cy = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
    half = span / 2 + pad
    return cx - half, cx + half, cy - half, cy + half


def _pairwise_scatter(axes_row, data_np, pairs, color, row_label):
    """Draw pairwise scatter plots on a row of axes."""
    for col, (i, j) in enumerate(pairs):
        ax = axes_row[col]
        ax.scatter(data_np[:, i], data_np[:, j], s=2, alpha=0.3, c=color)
        ax.set_xlabel(f"dim {i}")
        ax.set_ylabel(f"dim {j}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        x0, x1, y0, y1 = _auto_lim(data_np[:, i], data_np[:, j])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        # Reference circles (1-sigma, 2-sigma)
        for r in [1, 2]:
            circle = plt.Circle((0, 0), r, fill=False, ls="--", color="gray", alpha=0.5)
            ax.add_patch(circle)
        if col == 0:
            ax.set_title(f"{row_label}\n({i},{j})", fontsize=10)
        else:
            ax.set_title(f"({i},{j})", fontsize=10)


def _marginal_hist(axes_row, data_np, color, start_col, row_label):
    """Draw 1D marginal histograms with N(0,1) overlay."""
    D = data_np.shape[1]
    xs = np.linspace(-4, 4, 200)
    pdf = np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi)
    for d in range(D):
        ax = axes_row[start_col + d]
        ax.hist(data_np[:, d], bins=50, density=True, alpha=0.7, color=color)
        ax.plot(xs, pdf, "k--", alpha=0.5, label="N(0,1)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        if d == 0:
            ax.set_title(f"{row_label}\ndim {d}", fontsize=10)
        else:
            ax.set_title(f"dim {d}", fontsize=10)


def plot_distributions(source, initial, final, save_path):
    from itertools import combinations

    D = initial.shape[1]
    initial_np = initial.numpy()
    final_np = final.numpy()

    pairs = list(combinations(range(D), 2))
    n_pairs = len(pairs)
    total_cols = n_pairs + D  # pairwise scatters + 1D marginals

    # 2 rows: before training, after training
    fig, axes = plt.subplots(2, total_cols, figsize=(4 * total_cols, 8), squeeze=False)

    _pairwise_scatter(axes[0], initial_np, pairs, "tab:orange", "Before")
    _marginal_hist(axes[0], initial_np, "tab:orange", n_pairs, "Before")

    _pairwise_scatter(axes[1], final_np, pairs, "tab:green", "After")
    _marginal_hist(axes[1], final_np, "tab:green", n_pairs, "After")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_marginal_diagnostics(final, save_path):
    final_np = final.numpy()
    D = final.shape[1]
    cols = min(D, 4)
    rows = max(1, (D + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    xs = np.linspace(-4, 4, 200)
    pdf = np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi)
    for d in range(D):
        r, c = divmod(d, cols)
        ax = axes[r, c]
        ax.hist(final_np[:, d], bins=50, density=True, alpha=0.7, color="tab:green", label=f"dim {d}")
        ax.plot(xs, pdf, "k--", alpha=0.7, label="N(0,1)")
        ax.set_title(f"Dimension {d}")
        ax.legend(fontsize=8)
    # Hide unused axes
    for d in range(D, rows * cols):
        r, c = divmod(d, cols)
        axes[r, c].set_visible(False)
    fig.suptitle("Marginal Diagnostics (Final Output)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_evolution(snapshots, save_path):
    from itertools import combinations

    steps_sorted = sorted(snapshots.keys())
    n = len(steps_sorted)
    if n == 0:
        return
    D = snapshots[steps_sorted[0]].shape[1]
    pairs = list(combinations(range(D), 2))

    # One row per pair, one column per snapshot step
    n_pairs = len(pairs)
    fig, axes = plt.subplots(
        n_pairs, n, figsize=(4 * n, 4 * n_pairs), squeeze=False
    )

    for col, step in enumerate(steps_sorted):
        data = snapshots[step].numpy()
        for row, (i, j) in enumerate(pairs):
            ax = axes[row, col]
            ax.scatter(data[:, i], data[:, j], s=2, alpha=0.3, c="tab:blue")
            for radius in [1, 2]:
                circle = plt.Circle(
                    (0, 0), radius, fill=False, ls="--", color="gray", alpha=0.5
                )
                ax.add_patch(circle)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            x0, x1, y0, y1 = _auto_lim(data[:, i], data[:, j])
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            if row == 0:
                ax.set_title(f"Step {step}")
            if col == 0:
                ax.set_ylabel(f"({i},{j})")

    fig.suptitle("Distribution Evolution", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args, device):
    torch.manual_seed(args.seed)

    if args.proj_dim < args.input_dim:
        raise ValueError(
            f"proj_dim M={args.proj_dim} < input_dim N={args.input_dim}. "
            f"The projection would lose information. Set M >= N."
        )

    # 1. Generate data
    data = generate_data(args.distribution, args.num_points, args.input_dim)  # (K, N)

    # 2. Fixed random projection
    W = make_fixed_projection(args.input_dim, args.proj_dim, seed=args.seed + 1).to(device)
    projected = (data.to(device) @ W)  # (K, M)

    # 3. Build MLP and SIGReg (output dim = N, the true data dimensionality)
    output_dim = args.input_dim
    mlp = DeepMLP(
        input_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        depth=args.depth,
    ).to(device)
    sigreg = SIGReg(knots=args.knots, num_proj=args.num_proj).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    # 4. Initial snapshot
    mlp.eval()
    with torch.no_grad():
        initial_output = mlp(projected).cpu()

    # 5. Training loop
    losses = []
    snapshots = {}

    for step in range(1, args.steps + 1):
        mlp.train()

        if args.batch_mode == "full":
            batch = projected
        else:
            idx = torch.randint(0, args.num_points, (args.batch_size,), device=device)
            batch = projected[idx]

        output = mlp(batch)  # (B, D)
        # SIGReg expects (T, B, D) — use T=1
        loss = sigreg(output.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % args.log_interval == 0:
            print(f"Step {step:5d}/{args.steps} | SIGReg loss: {loss.item():.6f}")

        if args.snapshot_interval > 0 and step % args.snapshot_interval == 0:
            mlp.eval()
            with torch.no_grad():
                snapshots[step] = mlp(projected).cpu()

    # 6. Final output
    mlp.eval()
    with torch.no_grad():
        final_output = mlp(projected).cpu()

    return losses, initial_output, final_output, data.cpu(), snapshots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Demonstrate SIGReg convergence to isotropic Gaussian"
    )

    # Data
    parser.add_argument("--distribution", type=str, default="ring",
                        choices=list(GENERATORS.keys()))
    parser.add_argument("--num-points", "-K", type=int, default=1024)
    parser.add_argument("--input-dim", "-N", type=int, default=4)

    # Fixed projection (M > N inflates dimensionality; NN learns to recover N-dim representation)
    parser.add_argument("--proj-dim", "-M", type=int, default=8,
                        help="Dimension after fixed random projection (should be > N)")

    # Learned MLP (output dim = N, the true data dimensionality)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3,
                        help="Number of hidden layers in the MLP")

    # SIGReg
    parser.add_argument("--knots", type=int, default=17)
    parser.add_argument("--num-proj", type=int, default=1024)

    # Training
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-mode", type=str, default="full",
                        choices=["full", "mini"])
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size (only used with --batch-mode mini)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="sigreg_results")
    parser.add_argument("--snapshot-interval", type=int, default=0,
                        help="Save distribution snapshots every N steps (0=off)")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Config: dist={args.distribution}, K={args.num_points}, "
          f"N={args.input_dim}, M={args.proj_dim}, D(=N)={args.input_dim}, "
          f"batch_mode={args.batch_mode}, device={device}")

    losses, initial, final, source, snapshots = train(args, device)

    # Plots
    plot_loss_curve(losses, os.path.join(args.save_dir, "loss_curve.png"))
    plot_distributions(source, initial, final, os.path.join(args.save_dir, "distributions.png"))
    plot_marginal_diagnostics(final, os.path.join(args.save_dir, "marginal_diagnostics.png"))
    if snapshots:
        plot_evolution(snapshots, os.path.join(args.save_dir, "evolution.png"))

    # Summary statistics
    final_np = final.numpy()
    mean = final_np.mean(axis=0)
    std = final_np.std(axis=0)
    cov = np.cov(final_np, rowvar=False)

    print("\n===== Summary =====")
    print(f"Distribution: {args.distribution}")
    print(f"Points: {args.num_points}, Data dim (N): {args.input_dim}, "
          f"Proj dim (M): {args.proj_dim}, Output dim (=N): {args.input_dim}")
    print(f"Final SIGReg loss: {losses[-1]:.6f}")
    print(f"\nOutput statistics (target: N(0, I_{args.input_dim})):")
    print(f"  Mean:     {mean}")
    print(f"  Std:      {std}")
    print(f"  Cov diag: {np.diag(cov)}")
    if args.input_dim > 1:
        off_diag = cov - np.diag(np.diag(cov))
        print(f"  Cov off-diag max abs: {np.abs(off_diag).max():.4f}")
    print(f"\nPlots saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
