"""
Sliced Wasserstein-1 Isotropic Gaussian Convergence Experiment

Replaces SIGReg with a Sliced 1D Wasserstein (Earth Mover's) distance to N(0,1).
Same pipeline as sigreg_experiment.py — only the loss changes.

Pipeline:
    Source points (K, N) -> Fixed random projection (K, M) -> Learned MLP (K, D=N) -> Sliced W1 loss

Example commands:
    # N=3, uniform square, full-batch
    python w1_experiment.py --distribution uniform_square --input-dim 3 --num-points 4096 --steps 5000 --save-dir w1_results_3d

    # N=4 (default), blobs, full-batch
    python w1_experiment.py --distribution blobs --num-points 4096 --steps 5000 --save-dir w1_results_4d

    # N=4, ring, mini-batch with snapshots
    python w1_experiment.py --distribution ring --num-points 8192 --steps 10000 --batch-mode mini --batch-size 512 --snapshot-interval 1000 --save-dir w1_results_mini
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
from sigreg_experiment import (
    DeepMLP,
    generate_data,
    GENERATORS,
    make_fixed_projection,
    resolve_device,
    plot_distributions,
    plot_marginal_diagnostics,
    plot_evolution,
)


# ---------------------------------------------------------------------------
# Sliced Wasserstein-1 loss
# ---------------------------------------------------------------------------

class SlicedW1Gaussian(nn.Module):
    """Sliced 1D Wasserstein distance to N(0, I).

    Projects D-dim data onto random unit vectors, then computes the
    closed-form 1D W1 between the sorted projections and N(0,1) quantiles.
    """

    def __init__(self, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        self._cached_n = 0

    def _get_ref_quantiles(self, n, device):
        """N(0,1) quantiles for batch size n, cached."""
        if n != self._cached_n:
            p = (torch.arange(1, n + 1, device=device, dtype=torch.float32) - 0.5) / n
            self._ref = torch.erfinv(2 * p - 1) * math.sqrt(2)
            self._cached_n = n
        return self._ref

    def forward(self, x):
        """
        x: (B, D)
        Returns: scalar W1 loss averaged over projections.
        """
        B, D = x.shape

        # Random unit projections
        A = torch.randn(D, self.num_proj, device=x.device)
        A = A / A.norm(dim=0, keepdim=True)

        # Project and sort along batch dim
        proj = x @ A  # (B, num_proj)
        proj_sorted = torch.sort(proj, dim=0).values

        # Reference N(0,1) quantiles
        ref = self._get_ref_quantiles(B, x.device)  # (B,)

        # W1 per projection: mean|sorted - quantile|
        w1 = (proj_sorted - ref.unsqueeze(1)).abs().mean(dim=0)  # (num_proj,)
        return w1.mean()


# ---------------------------------------------------------------------------
# Visualization (loss curve with W1 labels)
# ---------------------------------------------------------------------------

def plot_loss_curve(losses, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Sliced W1 Loss")
    ax.set_title("Sliced W1 Distance to N(0,1) During Training")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
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
    data = generate_data(args.distribution, args.num_points, args.input_dim)

    # 2. Fixed random projection
    W = make_fixed_projection(args.input_dim, args.proj_dim, seed=args.seed + 1).to(device)
    projected = (data.to(device) @ W)

    # 3. Build MLP and W1 loss (output dim = N)
    output_dim = args.input_dim
    mlp = DeepMLP(
        input_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        depth=args.depth,
    ).to(device)
    w1_loss = SlicedW1Gaussian(num_proj=args.num_proj).to(device)
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

        output = mlp(batch)
        loss = w1_loss(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % args.log_interval == 0:
            print(f"Step {step:5d}/{args.steps} | W1 loss: {loss.item():.6f}")

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
        description="Demonstrate Sliced W1 convergence to isotropic Gaussian"
    )

    # Data
    parser.add_argument("--distribution", type=str, default="ring",
                        choices=list(GENERATORS.keys()))
    parser.add_argument("--num-points", "-K", type=int, default=1024)
    parser.add_argument("--input-dim", "-N", type=int, default=4)

    # Fixed projection
    parser.add_argument("--proj-dim", "-M", type=int, default=8,
                        help="Dimension after fixed random projection (should be > N)")

    # Learned MLP (output dim = N)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3,
                        help="Number of hidden layers in the MLP")

    # W1 loss
    parser.add_argument("--num-proj", type=int, default=1024,
                        help="Number of random projection directions for sliced W1")

    # Training
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-mode", type=str, default="full",
                        choices=["full", "mini"])
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size (only used with --batch-mode mini)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="w1_results")
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
    print(f"Final W1 loss: {losses[-1]:.6f}")
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
