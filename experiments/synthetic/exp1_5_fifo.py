"""
Experiment 1.5: FIFO Buffer — Fresh vs Stale CDF Samples

Sweeps T_cur (current-window accumulation steps) and T_fifo (number of
retained past windows) to explore the tradeoff between CDF resolution
and staleness.

Total CDF samples per update = (1 + T_fifo) * T_cur * BS
  - T_cur * BS from current window (fresh, 1 batch has gradient)
  - T_fifo * T_cur * BS from FIFO (stale, all detached)

The FIFO stores complete past windows. Each past window has T_cur * BS
embeddings computed from an older model state. As T_fifo grows, CDF
resolution improves but staleness increases.

Run:
    python experiments/synthetic/exp1_5_fifo.py
    python experiments/synthetic/exp1_5_fifo.py --distributions blobs --t-cur 4 --t-fifo 0 2 4 --n-seeds 1 --steps 500
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.accumulated_w1 import (
    SlicedW1Loss, AccumulatedSlicedLoss,
    DeepMLP, generate_data, make_fixed_projection, GENERATORS,
    eval_w1, evaluate_full,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _setup(seed, args, distribution, device):
    D = args.input_dim
    torch.manual_seed(seed)
    data = generate_data(distribution, args.num_points, D)
    W = make_fixed_projection(D, args.proj_dim, seed=seed + 100).to(device)
    projected = data.to(device) @ W

    src_np = data.numpy()
    colors = np.arctan2(src_np[:, 1] if D > 1 else np.zeros(len(src_np)),
                        src_np[:, 0])
    colors = (colors + np.pi) / (2 * np.pi)

    torch.manual_seed(seed + 200)
    mlp = DeepMLP(args.proj_dim, args.hidden_dim, D, depth=args.depth).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    mlp.eval()
    with torch.no_grad():
        initial_out = mlp(projected).cpu().numpy()
    mlp.train()

    return projected, mlp, opt, src_np, colors, initial_out


def train_one(seed, args, distribution, T_cur, T_fifo, device):
    """Train with pooled accumulation + FIFO buffer.

    Args:
        T_cur: Current-window steps (T_cur-1 no-grad + 1 grad).
        T_fifo: Number of past windows to retain in FIFO.
            fifo_size = T_fifo * T_cur * BS to hold exactly T_fifo windows.
            T_fifo=0 means no FIFO (same as exp1_2 pooled).
    """
    projected, mlp, opt, src_np, colors, initial_out = \
        _setup(seed, args, distribution, device)
    K = args.num_points
    BS = args.batch_size

    fifo_size = T_fifo * T_cur * BS if T_fifo > 0 else 0

    accum_loss = AccumulatedSlicedLoss(
        accum_steps=max(T_cur - 1, 0), num_proj=args.num_proj,
        mode="w1", fifo_size=fifo_size)

    eval_steps, eval_w1s = [], []

    for step in range(1, args.steps + 1):
        mlp.train()

        # T_cur - 1 no-grad forward passes
        with torch.no_grad():
            for _ in range(T_cur - 1):
                idx = torch.randint(0, K, (BS,), device=projected.device)
                accum_loss.accum_step(mlp(projected[idx]))

        # 1 gradient step
        idx = torch.randint(0, K, (BS,), device=projected.device)
        loss = accum_loss.grad_step(mlp(projected[idx]))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_interval == 0:
            mlp.eval()
            with torch.no_grad():
                w1_val = eval_w1(mlp(projected).cpu().numpy())
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

def plot_fifo_heatmap(all_runs, dist, T_cur_values, T_fifo_values, seeds,
                      save_path):
    """Heatmap: final W1 for each (T_cur, T_fifo) combination."""
    grid = np.zeros((len(T_fifo_values), len(T_cur_values)))

    for i, T_fifo in enumerate(T_fifo_values):
        for j, T_cur in enumerate(T_cur_values):
            vals = []
            for seed in seeds:
                result = all_runs.get((dist, T_cur, T_fifo, seed))
                if result is not None:
                    vals.append(result[6]["w1"])
            grid[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", cmap="viridis_r",
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    ax.set_xticks(range(len(T_cur_values)))
    ax.set_xticklabels([str(t) for t in T_cur_values])
    ax.set_yticks(range(len(T_fifo_values)))
    ax.set_yticklabels([str(t) for t in T_fifo_values])
    ax.set_xlabel("T_cur (current window steps)")
    ax.set_ylabel("T_fifo (retained past windows)")

    BS = 8  # will be from args
    for i in range(len(T_fifo_values)):
        for j in range(len(T_cur_values)):
            val = grid[i, j]
            total = (1 + T_fifo_values[i]) * T_cur_values[j] * BS
            color = "white" if val > np.nanmedian(grid) else "black"
            ax.text(j, i, f"{val:.3f}\n({total})",
                    ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Final W1")
    ax.set_title(f"{dist}: W1 by (T_cur, T_fifo)\n"
                 f"numbers in cells = W1 (total CDF samples)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fifo_lines(all_runs, dist, T_cur_values, T_fifo_values, seeds,
                    save_path):
    """Line plot: W1 vs T_fifo for each T_cur."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(T_cur_values)))

    for j, T_cur in enumerate(T_cur_values):
        means, stds = [], []
        for T_fifo in T_fifo_values:
            vals = [all_runs[(dist, T_cur, T_fifo, s)][6]["w1"]
                    for s in seeds
                    if (dist, T_cur, T_fifo, s) in all_runs]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)
        ax.errorbar(T_fifo_values, means, yerr=stds, fmt="o-",
                     color=colors[j], label=f"T_cur={T_cur}",
                     linewidth=2, markersize=6, capsize=3)

    ax.set_xlabel("T_fifo (retained past windows)", fontsize=12)
    ax.set_ylabel("Final Sliced W1 to N(0,1)", fontsize=11)
    ax.set_title(f"{dist}: Effect of FIFO depth", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 1.5: FIFO Buffer — Fresh vs Stale CDF Samples"
    )
    p.add_argument("--distributions", nargs="+",
                    default=list(GENERATORS.keys()),
                    choices=list(GENERATORS.keys()))
    p.add_argument("--input-dim", type=int, default=4)
    p.add_argument("--proj-dim", type=int, default=32)
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--t-cur", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="Current-window accumulation steps")
    p.add_argument("--t-fifo", type=int, nargs="+", default=[0, 1, 2, 4, 8],
                    help="Number of past windows to retain (0=no FIFO)")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--save-dir", default="results/exp1_5_fifo")
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
    T_cur_values = sorted(args.t_cur)
    T_fifo_values = sorted(args.t_fifo)

    total = (len(args.distributions) * len(T_cur_values)
             * len(T_fifo_values) * len(seeds))
    print(f"Exp 1.5: FIFO Buffer")
    print(f"  distributions={args.distributions}")
    print(f"  T_cur={T_cur_values}, T_fifo={T_fifo_values}")
    print(f"  batch_size={args.batch_size}, seeds={seeds}")
    print(f"  total_runs={total}, device={device}")
    print()

    # Build flat config list for sharding
    configs = []
    for dist in args.distributions:
        for T_cur in T_cur_values:
            for T_fifo in T_fifo_values:
                for seed in seeds:
                    configs.append((dist, T_cur, T_fifo, seed))

    all_runs = {}
    for cfg_idx, (dist, T_cur, T_fifo, seed) in enumerate(configs):
        total_cdf = (1 + T_fifo) * T_cur * args.batch_size
        model_path = os.path.join(
            args.save_dir,
            f"model_{dist}_Tc{T_cur}_Tf{T_fifo}_{seed}.pt")

        if os.path.exists(model_path):
            print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, T_cur={T_cur}, "
                  f"T_fifo={T_fifo}, seed={seed} SKIP (exists)")
            state = torch.load(model_path, map_location=device,
                               weights_only=True)
            proj, mlp, _, src_np, colors, initial_out = \
                _setup(seed, args, dist, device)
            mlp.load_state_dict(state)
            mlp.eval()
            with torch.no_grad():
                final_out = mlp(proj).cpu().numpy()
            metrics = evaluate_full(final_out)
            all_runs[(dist, T_cur, T_fifo, seed)] = (
                [], [], final_out, initial_out, src_np,
                colors, metrics, state)
            continue

        if cfg_idx % args.num_workers != args.worker_id:
            continue

        print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, T_cur={T_cur}, "
              f"T_fifo={T_fifo}, CDF={total_cdf}, seed={seed}",
              end=" ", flush=True)
        t0 = time.perf_counter()
        result = train_one(
            seed, args, dist, T_cur, T_fifo, device)
        elapsed = time.perf_counter() - t0
        all_runs[(dist, T_cur, T_fifo, seed)] = result
        print(f"W1={result[6]['w1']:.4f}  time={elapsed:.1f}s")

        torch.save(result[7], model_path)

    # --- Plots ---
    for dist in args.distributions:
        plot_fifo_heatmap(
            all_runs, dist, T_cur_values, T_fifo_values, seeds,
            os.path.join(plots_dir, f"fifo_heatmap_{dist}.png"))
        plot_fifo_lines(
            all_runs, dist, T_cur_values, T_fifo_values, seeds,
            os.path.join(plots_dir, f"fifo_lines_{dist}.png"))

    # --- Summary ---
    summary_lines = [
        f"Exp 1.5: FIFO Buffer",
        f"d={args.input_dim}, M={args.proj_dim}, BS={args.batch_size}, "
        f"steps={args.steps}, seeds={len(seeds)}",
        "",
        f"{'Distribution':<18} {'T_cur':>5} {'T_fifo':>6} {'CDF':>6}  {'W1':>14}",
        "-" * 60,
    ]

    json_out = {}
    for dist in args.distributions:
        for T_cur in T_cur_values:
            for T_fifo in T_fifo_values:
                total_cdf = (1 + T_fifo) * T_cur * args.batch_size
                vals = []
                for seed in seeds:
                    result = all_runs.get((dist, T_cur, T_fifo, seed))
                    if result is not None:
                        vals.append(result[6])
                if not vals:
                    continue

                key = f"{dist}_Tc{T_cur}_Tf{T_fifo}"
                agg = {}
                for mk in vals[0]:
                    if isinstance(vals[0][mk], (int, float)):
                        v = [m[mk] for m in vals]
                        agg[mk] = {"mean": float(np.mean(v)),
                                    "std": float(np.std(v))}
                json_out[key] = {"T_cur": T_cur, "T_fifo": T_fifo,
                                 "total_cdf": total_cdf, **agg}

                w1m = agg["w1"]["mean"]
                w1s = agg["w1"]["std"]
                summary_lines.append(
                    f"{dist:<18} {T_cur:>5} {T_fifo:>6} {total_cdf:>6}  "
                    f"{w1m:.4f}+/-{w1s:.4f}")
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
