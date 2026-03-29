"""
Sweep comparison: SIGReg vs Sliced W1 across dimensions and distributions.

Runs both methods for each (distribution, dimension, seed) combination,
aggregates over seeds, and produces comparison plots with error bars.

Example commands:
    # Default sweep: 3 seeds, 3 distributions, N=2..6
    python sweep_experiment.py

    # Quick test: fewer seeds and dims
    python sweep_experiment.py --num-seeds 2 --dims 3 4 --distributions uniform_square blobs

    # Full sweep with more seeds
    python sweep_experiment.py --num-seeds 5 --steps 5000
"""

import os
import sys
import json
import time
import argparse
from argparse import Namespace
from collections import defaultdict

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigreg_experiment import train as sigreg_train, resolve_device
from w1_experiment import train as w1_train


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sliced_w1_to_gaussian(data_np, num_proj=4096):
    """Sliced W1 distance between empirical data and N(0, I).

    Projects onto random directions, sorts, compares to N(0,1) quantiles.
    Uses many projections for a stable estimate.
    """
    from scipy.stats import norm

    B, D = data_np.shape
    # Random unit projections
    A = np.random.randn(D, num_proj)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)

    proj = data_np @ A  # (B, num_proj)
    proj_sorted = np.sort(proj, axis=0)

    # N(0,1) quantiles
    p = (np.arange(1, B + 1) - 0.5) / B
    ref = norm.ppf(p)  # (B,)

    w1 = np.abs(proj_sorted - ref[:, None]).mean(axis=0)  # (num_proj,)
    return float(w1.mean())


def bures_w2_to_gaussian(data_np):
    """Bures-Wasserstein W2 distance between empirical distribution and N(0, I).

    Assumes Gaussian approximation:
        W2^2 = ||mu||^2 + Tr(Sigma) + D - 2 * Tr(Sigma^{1/2})
    where Sigma^{1/2} via eigenvalues: Tr(Sigma^{1/2}) = sum(sqrt(eigenvalues)).
    """
    D = data_np.shape[1]
    mean = data_np.mean(axis=0)
    cov = np.cov(data_np, rowvar=False)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)  # numerical safety

    w2_sq = (np.dot(mean, mean)
             + np.trace(cov) + D
             - 2.0 * np.sum(np.sqrt(eigvals)))
    return float(np.sqrt(max(w2_sq, 0)))



def compute_metrics(final, losses):
    """Compute isotropy metrics from final output and loss curve."""
    final_np = final.numpy()
    D = final_np.shape[1]
    mean = final_np.mean(axis=0)
    std = final_np.std(axis=0)
    cov = np.cov(final_np, rowvar=False)

    diag = np.diag(cov)
    off_diag = cov - np.diag(diag)

    tail = max(1, len(losses) // 10)
    loss_tail = losses[-tail:]

    return {
        "final_loss": losses[-1],
        "loss_std_tail": float(np.std(loss_tail)),
        "mean_abs": float(np.abs(mean).max()),
        "std_mean": float(std.mean()),
        "cov_diag_err": float(np.abs(diag - 1.0).max()),
        "cov_offdiag_max": float(np.abs(off_diag).max()) if D > 1 else 0.0,
        "sliced_w1": sliced_w1_to_gaussian(final_np),
        "bures_w2": bures_w2_to_gaussian(final_np),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

METHODS = {"sigreg": sigreg_train, "w1": w1_train}


def make_args(base_args, input_dim, distribution, seed, method):
    """Create a Namespace for a single run."""
    ns = Namespace(
        distribution=distribution,
        num_points=base_args.num_points,
        input_dim=input_dim,
        proj_dim=max(base_args.proj_dim, input_dim + 4),
        hidden_dim=base_args.hidden_dim,
        depth=base_args.depth,
        num_proj=base_args.num_proj,
        steps=base_args.steps,
        lr=base_args.lr,
        batch_mode=base_args.batch_mode,
        batch_size=base_args.batch_size,
        seed=seed,
        save_dir="",
        snapshot_interval=0,
        log_interval=base_args.log_interval,
    )
    if method == "sigreg":
        ns.knots = base_args.knots
    return ns


def run_sweep(base_args, device):
    dims = base_args.dims
    distributions = base_args.distributions
    seeds = [base_args.base_seed + i for i in range(base_args.num_seeds)]

    total = len(distributions) * len(dims) * len(METHODS) * len(seeds)
    run_idx = 0

    # all_runs[dist][method][N] = list of {metrics, losses} over seeds
    all_runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dist in distributions:
        for N in dims:
            for method_name, train_fn in METHODS.items():
                for seed in seeds:
                    run_idx += 1
                    print(f"[{run_idx}/{total}] dist={dist}, method={method_name}, "
                          f"N={N}, seed={seed}")
                    args = make_args(base_args, N, dist, seed, method_name)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    t0 = time.perf_counter()
                    losses, _, final, _, _ = train_fn(args, device)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    elapsed = time.perf_counter() - t0
                    metrics = compute_metrics(final, losses)
                    metrics["wall_time_s"] = elapsed

                    all_runs[dist][method_name][N].append({
                        "metrics": metrics,
                        "losses": losses,
                        "seed": seed,
                    })
                    print(f"  loss={metrics['final_loss']:.6f}, "
                          f"diag_err={metrics['cov_diag_err']:.4f}, "
                          f"offdiag={metrics['cov_offdiag_max']:.4f}, "
                          f"time={elapsed:.1f}s")

    return all_runs


def aggregate(all_runs, metric_key):
    """Return {dist: {method: {N: (mean, std)}}} for a given metric."""
    agg = {}
    for dist in all_runs:
        agg[dist] = {}
        for method in all_runs[dist]:
            agg[dist][method] = {}
            for N in all_runs[dist][method]:
                vals = [r["metrics"][metric_key] for r in all_runs[dist][method][N]]
                agg[dist][method][N] = (float(np.mean(vals)), float(np.std(vals)))
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METHOD_COLORS = {"sigreg": "tab:blue", "w1": "tab:red"}
METHOD_LABELS = {"sigreg": "SIGReg", "w1": "Sliced W1"}


def plot_convergence(all_runs, dims, save_path):
    """Loss curves: one row per distribution, one col per dimension.
    Plots mean loss with shaded std across seeds."""
    distributions = list(all_runs.keys())
    n_dist = len(distributions)
    n_dims = len(dims)

    fig, axes = plt.subplots(n_dist, n_dims,
                             figsize=(4 * n_dims, 3.5 * n_dist), squeeze=False)

    for row, dist in enumerate(distributions):
        for col, N in enumerate(dims):
            ax = axes[row, col]
            for method, color in METHOD_COLORS.items():
                runs = all_runs[dist].get(method, {}).get(N, [])
                if not runs:
                    continue
                # Stack loss curves: (num_seeds, steps)
                all_losses = np.array([r["losses"] for r in runs])
                mean_loss = all_losses.mean(axis=0)
                std_loss = all_losses.std(axis=0)
                steps = np.arange(1, len(mean_loss) + 1)

                ax.plot(steps, mean_loss, color=color, alpha=0.9,
                        label=METHOD_LABELS[method], linewidth=0.8)
                ax.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss,
                                color=color, alpha=0.15)

            ax.set_yscale("log")
            ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(f"N = {N}", fontsize=10)
            if col == 0:
                ax.set_ylabel(dist, fontsize=10)
            if row == n_dist - 1:
                ax.set_xlabel("Step")
            if row == 0 and col == n_dims - 1:
                ax.legend(fontsize=7)

    fig.suptitle("Convergence: SIGReg vs Sliced W1 (mean ± std over seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(all_runs, dims, save_path):
    """Bar charts with error bars: one row per metric, one col per distribution."""
    metric_defs = [
        ("sliced_w1", "Sliced W1 to N(0,I)"),
        ("bures_w2", "Bures W2 to N(0,I)"),
        ("cov_diag_err", "Max |Cov_diag - 1|"),
        ("cov_offdiag_max", "Max |Cov_offdiag|"),
        ("loss_std_tail", "Loss Std (last 10%)"),
    ]

    distributions = list(all_runs.keys())
    n_dist = len(distributions)
    n_metrics = len(metric_defs)

    fig, axes = plt.subplots(n_metrics, n_dist,
                             figsize=(5 * n_dist, 3.5 * n_metrics), squeeze=False)

    x = np.arange(len(dims))
    width = 0.35

    for mrow, (mkey, mlabel) in enumerate(metric_defs):
        agg = aggregate(all_runs, mkey)
        for dcol, dist in enumerate(distributions):
            ax = axes[mrow, dcol]
            for i, (method, color) in enumerate(METHOD_COLORS.items()):
                means = [agg[dist].get(method, {}).get(N, (0, 0))[0] for N in dims]
                stds = [agg[dist].get(method, {}).get(N, (0, 0))[1] for N in dims]
                offset = -width / 2 + i * width
                ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                       label=METHOD_LABELS[method], color=color, alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels([f"N={d}" for d in dims], fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")

            if mrow == 0:
                ax.set_title(dist, fontsize=10)
            if dcol == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            if mrow == 0 and dcol == n_dist - 1:
                ax.legend(fontsize=7)

    fig.suptitle("Isotropy Metrics (mean ± std over seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_timing(all_runs, dims, save_path):
    """Wall-clock time comparison: one subplot per distribution."""
    distributions = list(all_runs.keys())
    n_dist = len(distributions)
    fig, axes = plt.subplots(1, n_dist, figsize=(5 * n_dist, 4), squeeze=False)

    x = np.arange(len(dims))
    width = 0.35

    for dcol, dist in enumerate(distributions):
        ax = axes[0, dcol]
        agg = aggregate(all_runs, "wall_time_s")
        for i, (method, color) in enumerate(METHOD_COLORS.items()):
            means = [agg[dist].get(method, {}).get(N, (0, 0))[0] for N in dims]
            stds = [agg[dist].get(method, {}).get(N, (0, 0))[1] for N in dims]
            offset = -width / 2 + i * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   label=METHOD_LABELS[method], color=color, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"N={d}" for d in dims], fontsize=8)
        ax.set_title(dist, fontsize=10)
        ax.set_ylabel("Wall time (s)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Training Time (mean ± std over seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_results_table(all_runs, dims):
    """Print aggregated results table."""
    distributions = list(all_runs.keys())
    metric_keys = ["sliced_w1", "bures_w2", "cov_diag_err",
                    "cov_offdiag_max", "loss_std_tail", "wall_time_s"]

    header = (f"{'Dist':<16} {'Method':<10} {'N':>3}  "
              f"{'SW1↓':>14} {'BW2↓':>14} "
              f"{'Diag Err':>14} {'Offdiag':>14} {'Stability':>14} {'Time (s)':>12}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for dist in distributions:
        for N in dims:
            for method in ["sigreg", "w1"]:
                runs = all_runs[dist].get(method, {}).get(N, [])
                if not runs:
                    continue
                vals = {k: [r["metrics"][k] for r in runs] for k in metric_keys}
                parts = []
                for k in metric_keys:
                    m, s = np.mean(vals[k]), np.std(vals[k])
                    if k == "wall_time_s":
                        parts.append(f"{m:.1f}±{s:.1f}")
                    else:
                        parts.append(f"{m:.4f}±{s:.4f}")
                print(f"{dist:<16} {method.upper():<10} {N:>3}  "
                      f"{'  '.join(parts)}")
        print("-" * len(header))
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep comparison of SIGReg vs Sliced W1 across dims/distributions"
    )

    # Sweep axes
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--distributions", type=str, nargs="+",
                        default=["uniform_square", "blobs", "ring"],
                        choices=["ring", "uniform_square", "blobs", "spiral"])
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Number of random seeds per configuration")
    parser.add_argument("--base-seed", type=int, default=42)

    # Data
    parser.add_argument("--num-points", "-K", type=int, default=4096)
    parser.add_argument("--proj-dim", "-M", type=int, default=8)

    # MLP
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)

    # Loss
    parser.add_argument("--knots", type=int, default=17)
    parser.add_argument("--num-proj", type=int, default=1024)

    # Training
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-mode", type=str, default="full",
                        choices=["full", "mini"])
    parser.add_argument("--batch-size", type=int, default=256)

    # Misc
    parser.add_argument("--save-dir", type=str, default="sweep_results")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    total_runs = len(args.distributions) * len(args.dims) * 2 * args.num_seeds
    print(f"Sweep: dims={args.dims}, dists={args.distributions}, "
          f"seeds={args.num_seeds}, total_runs={total_runs}, device={device}")

    all_runs = run_sweep(args, device)

    # Plots
    plot_convergence(all_runs, args.dims,
                     os.path.join(args.save_dir, "convergence.png"))
    plot_metric_comparison(all_runs, args.dims,
                           os.path.join(args.save_dir, "isotropy_metrics.png"))
    plot_timing(all_runs, args.dims,
                os.path.join(args.save_dir, "timing.png"))

    # Table
    print_results_table(all_runs, args.dims)

    # Save aggregated metrics to JSON
    json_out = {}
    for dist in all_runs:
        for method in all_runs[dist]:
            for N in all_runs[dist][method]:
                key = f"{dist}_{method}_N{N}"
                runs = all_runs[dist][method][N]
                metric_keys = list(runs[0]["metrics"].keys())
                agg = {}
                for mk in metric_keys:
                    vals = [r["metrics"][mk] for r in runs]
                    agg[mk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                json_out[key] = {
                    "dist": dist, "method": method, "N": N,
                    "num_seeds": len(runs), "metrics": agg,
                }
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"\nPlots and metrics saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
