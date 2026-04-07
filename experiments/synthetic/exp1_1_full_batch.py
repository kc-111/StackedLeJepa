"""
Experiment 1.1: Full Batch Baseline — Which Loss Is Best?

Compares SIGReg, Sliced W1, and Sliced W2 regularization toward N(0, I)
when batch size is not a constraint (full-batch training).

Sweeps all 5 synthetic generators, projection dimensions M={8, 32},
and 3 seeds per configuration (90 runs total).

Pipeline:
    Source (K, d) -> Fixed projection (K, M) -> MLP (K, d) -> Loss

Run:
    python experiments/synthetic/exp1_1_full_batch.py
    python experiments/synthetic/exp1_1_full_batch.py --distributions blobs ring --proj-dims 8 --n-seeds 1 --steps 500
"""

import os
import sys
import json
import time
import argparse

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.accumulated_w1 import (
    SlicedW1Loss, SlicedW2Loss, SIGRegLoss,
    DeepMLP, generate_data, make_fixed_projection, GENERATORS,
    eval_w1, evaluate_full,
)


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def make_loss_fn(loss_mode, num_proj, knots, device):
    if loss_mode == "w1":
        return SlicedW1Loss(num_proj=num_proj).to(device)
    elif loss_mode == "w2":
        return SlicedW2Loss(num_proj=num_proj).to(device)
    elif loss_mode == "sigreg":
        return SIGRegLoss(knots=knots, num_proj=num_proj).to(device)
    raise ValueError(f"Unknown loss_mode: {loss_mode}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(seed, args, loss_mode, proj_dim, distribution, device):
    """Train one run: full-batch, single loss type."""
    D = args.input_dim

    torch.manual_seed(seed)
    data = generate_data(distribution, args.num_points, D)
    W = make_fixed_projection(D, proj_dim, seed=seed + 100).to(device)
    projected = data.to(device) @ W

    torch.manual_seed(seed + 200)
    mlp = DeepMLP(proj_dim, args.hidden_dim, D, depth=args.depth).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    loss_fn = make_loss_fn(loss_mode, args.num_proj, args.knots, device)

    # Colors from source angle
    src_np = data.numpy()
    colors = np.arctan2(src_np[:, 1] if D > 1 else np.zeros(len(src_np)),
                        src_np[:, 0])
    colors = (colors + np.pi) / (2 * np.pi)

    mlp.eval()
    with torch.no_grad():
        initial_out = mlp(projected).cpu().numpy()
    mlp.train()

    eval_steps, eval_w1s = [], []

    for step in range(1, args.steps + 1):
        mlp.train()
        output = mlp(projected)
        loss = loss_fn(output)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_interval == 0:
            mlp.eval()
            with torch.no_grad():
                full_out = mlp(projected).cpu().numpy()
                w1_val = eval_w1(full_out)
                eval_steps.append(step)
                eval_w1s.append(w1_val)
            mlp.train()

    mlp.eval()
    with torch.no_grad():
        final_out = mlp(projected).cpu().numpy()
        final_metrics = evaluate_full(final_out)

    return eval_steps, eval_w1s, final_out, initial_out, src_np, colors, final_metrics, mlp.state_dict()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METHOD_COLORS = {"w1": "tab:blue", "w2": "tab:green", "sigreg": "tab:orange"}
METHOD_LABELS = {"w1": "Sliced W1", "w2": "Sliced W2", "sigreg": "SIGReg"}


def plot_convergence(all_runs, dist, M, save_path):
    """Convergence plot for one generator x M: 3 method curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for loss_mode, color in METHOD_COLORS.items():
        runs = all_runs[dist][loss_mode][M]
        # Filter runs that have convergence history (not loaded from checkpoint)
        valid = [(s, w) for s, w, *_ in runs.values() if len(s) > 0]
        if not valid:
            continue
        all_w1 = np.array([w for _, w in valid])
        steps_arr = valid[0][0]
        mean = all_w1.mean(axis=0)
        std = all_w1.std(axis=0)
        ax.plot(steps_arr, mean, label=METHOD_LABELS[loss_mode],
                color=color, linewidth=2)
        ax.fill_between(steps_arr, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Sliced W1 to N(0,1)", fontsize=11)
    ax.set_title(f"{dist}, M={M}, full batch", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_point_flow(src_np, initial_out, final_out, colors, D, save_path, title):
    """Source -> initial -> final scatter plot."""
    pairs = [(i, j) for i in range(min(D, 4)) for j in range(i + 1, min(D, 4))]
    n_cols = max(len(pairs), 1)
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    stages = [("Source", src_np), ("Initial", initial_out), ("Final", final_out)]
    for row, (label, data) in enumerate(stages):
        for col, (di, dj) in enumerate(pairs):
            ax = axes[row, col]
            ax.scatter(data[:, di], data[:, dj], s=3, alpha=0.5,
                       c=colors, cmap="hsv")
            for r in [1, 2]:
                ax.add_patch(plt.Circle((0, 0), r, fill=False, color="grey",
                                        linestyle="--", linewidth=0.8, alpha=0.5))
            ax.set_aspect("equal")
            lim = max(5, np.abs(data[:, [di, dj]]).max() * 1.2)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(f"dim {di} vs {dj}", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 1.1: Full batch baseline — W1 vs W2 vs SIGReg"
    )
    p.add_argument("--distributions", nargs="+",
                    default=list(GENERATORS.keys()),
                    choices=list(GENERATORS.keys()))
    p.add_argument("--input-dim", type=int, default=4)
    p.add_argument("--proj-dims", type=int, nargs="+", default=[8, 32])
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--save-dir", default="results/exp1_1_full_batch")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=1,
                    help="Total parallel workers (for sharding)")
    p.add_argument("--worker-id", type=int, default=0,
                    help="This worker's index (0..num_workers-1)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    plots_dir = os.path.join(args.save_dir, "plots")
    summary_dir = os.path.join(args.save_dir, "summary")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    seeds = [42 + i for i in range(args.n_seeds)]
    loss_modes = ["w1", "w2", "sigreg"]

    total = (len(args.distributions) * len(args.proj_dims)
             * len(loss_modes) * len(seeds))
    print(f"Exp 1.1: Full batch baseline")
    print(f"  distributions={args.distributions}")
    print(f"  proj_dims={args.proj_dims}, loss_modes={loss_modes}")
    print(f"  seeds={seeds}, total_runs={total}, device={device}")
    print()

    # Build flat config list for sharding
    configs = []
    for dist in args.distributions:
        for loss_mode in loss_modes:
            for M in args.proj_dims:
                for seed in seeds:
                    configs.append((dist, loss_mode, M, seed))

    # Run configs assigned to this worker
    all_runs = {}
    for dist in args.distributions:
        all_runs[dist] = {}
        for loss_mode in loss_modes:
            all_runs[dist][loss_mode] = {}
            for M in args.proj_dims:
                all_runs[dist][loss_mode][M] = {}

    for cfg_idx, (dist, loss_mode, M, seed) in enumerate(configs):
        model_path = os.path.join(
            args.save_dir, f"model_{dist}_{loss_mode}_M{M}_{seed}.pt")

        if os.path.exists(model_path):
            print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, loss={loss_mode}, "
                  f"M={M}, seed={seed} SKIP (exists)")
            state = torch.load(model_path, map_location=device,
                               weights_only=True)
            torch.manual_seed(seed)
            data = generate_data(dist, args.num_points, args.input_dim)
            W = make_fixed_projection(args.input_dim, M, seed=seed + 100).to(device)
            proj = data.to(device) @ W
            torch.manual_seed(seed + 200)
            mlp = DeepMLP(M, args.hidden_dim, args.input_dim, depth=args.depth).to(device)
            mlp.load_state_dict(state)
            mlp.eval()
            with torch.no_grad():
                final_out = mlp(proj).cpu().numpy()
            metrics = evaluate_full(final_out)
            all_runs[dist][loss_mode][M][seed] = (
                [], [], final_out, final_out, data.numpy(),
                np.zeros(len(data)), metrics, state)
            continue

        # Shard: only this worker trains new configs
        if cfg_idx % args.num_workers != args.worker_id:
            continue

        print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, loss={loss_mode}, "
              f"M={M}, seed={seed}", end=" ", flush=True)
        t0 = time.perf_counter()
        result = train_one(seed, args, loss_mode, M, dist, device)
        elapsed = time.perf_counter() - t0
        all_runs[dist][loss_mode][M][seed] = result
        print(f"W1={result[6]['w1']:.4f}  time={elapsed:.1f}s")

        torch.save(result[7], model_path)

    # --- Plots ---
    for dist in args.distributions:
        for M in args.proj_dims:
            plot_convergence(
                all_runs, dist, M,
                os.path.join(plots_dir, f"convergence_{dist}_M{M}.png"))

            # Point flow for first seed of each method
            for loss_mode in loss_modes:
                first_seed = seeds[0]
                _, _, final, initial, src, colors, metrics, _ = \
                    all_runs[dist][loss_mode][M][first_seed]
                plot_point_flow(
                    src, initial, final, colors, args.input_dim,
                    os.path.join(plots_dir,
                                 f"point_flow_{dist}_{loss_mode}_M{M}.png"),
                    f"{dist}, {METHOD_LABELS[loss_mode]}, M={M} "
                    f"(W1={metrics['w1']:.4f})")

    # --- Summary ---
    summary_lines = [
        f"Exp 1.1: Full Batch Baseline",
        f"d={args.input_dim}, K={args.num_points}, steps={args.steps}, "
        f"seeds={len(seeds)}",
        "",
        f"{'Distribution':<18} {'Loss':<10} {'M':>3}  "
        f"{'W1':>14} {'W2':>14} {'Cov Frob':>14} {'Offdiag':>14}",
        "-" * 80,
    ]

    json_out = {}
    for dist in args.distributions:
        for M in args.proj_dims:
            for loss_mode in loss_modes:
                runs = all_runs[dist][loss_mode][M]
                metrics_list = [r[6] for r in runs.values()]
                key = f"{dist}_{loss_mode}_M{M}"
                agg = {}
                for mk in metrics_list[0]:
                    if isinstance(metrics_list[0][mk], (int, float)):
                        vals = [m[mk] for m in metrics_list]
                        agg[mk] = {"mean": float(np.mean(vals)),
                                    "std": float(np.std(vals))}
                json_out[key] = agg

                w1m = agg["w1"]["mean"]
                w1s = agg["w1"]["std"]
                w2m = agg["w2"]["mean"]
                w2s = agg["w2"]["std"]
                cfm = agg["cov_frob"]["mean"]
                cfs = agg["cov_frob"]["std"]
                odm = agg["cov_offdiag_max"]["mean"]
                ods = agg["cov_offdiag_max"]["std"]
                summary_lines.append(
                    f"{dist:<18} {loss_mode:<10} {M:>3}  "
                    f"{w1m:.4f}+/-{w1s:.4f}  "
                    f"{w2m:.4f}+/-{w2s:.4f}  "
                    f"{cfm:.4f}+/-{cfs:.4f}  "
                    f"{odm:.4f}+/-{ods:.4f}")
            summary_lines.append("")

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    with open(os.path.join(summary_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(summary_dir, "metrics.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"\nSaved to {args.save_dir}/")


if __name__ == "__main__":
    main()
