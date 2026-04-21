"""
Experiment 1.4: Timing Comparison — SIGReg vs Sliced W1 vs Sliced W2

Measures forward-pass cost across batch sizes and dimensions.
No training — pure timing benchmark.

Run:
    python experiments/synthetic/exp1_4_timing.py
    python experiments/synthetic/exp1_4_timing.py --batch-sizes 256 1024 4096 --dimensions 4 16 64
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
from matplotlib.ticker import ScalarFormatter

SYNTHETIC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SYNTHETIC_DIR)

from sliced_gauss_reg import SlicedW1Loss, SlicedW2Loss, SIGRegLoss


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_forward(loss_fn, x, warmup=5, repeats=50):
    """Time the forward pass, returning (median_ms, std_ms)."""
    for _ in range(warmup):
        loss_fn(x)

    if x.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_fn(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times)), float(np.std(times))


def time_forward_backward(loss_fn, x, warmup=5, repeats=50):
    """Time forward + backward pass, returning (median_ms, std_ms)."""
    x = x.detach().requires_grad_(True)

    for _ in range(warmup):
        loss = loss_fn(x)
        loss.backward()
        x.grad = None

    if x.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = loss_fn(x)
        loss.backward()
        if x.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        x.grad = None

    return float(np.median(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METHOD_COLORS = {"sigreg": "tab:orange", "w1": "tab:blue", "w2": "tab:green"}
METHOD_LABELS = {"sigreg": "SIGReg", "w1": "Sliced W1", "w2": "Sliced W2"}


def fmt_tick(ax, axis="x"):
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.yaxis.set_major_formatter(fmt)


def plot_heatmap(ratio_grid, row_labels, col_labels, title, save_path,
                 vmin=0.5, vmax=3.0):
    """Plot annotated heatmap of time ratios."""
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(ratio_grid, aspect="auto", cmap="RdBu_r",
                   vmin=vmin, vmax=max(vmax, np.percentile(ratio_grid, 95)))
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([str(b) for b in col_labels], rotation=45)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([str(d) for d in row_labels])
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Dimension D")
    ax.set_title(title)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = ratio_grid[i, j]
            color = "white" if val > 2.0 or val < 0.7 else "black"
            ax.text(j, i, f"{val:.2f}x", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_sample_cost(results, batch_sizes, fixed_dims, save_path):
    """Per-sample cost (time/B) at fixed D slices."""
    B_arr = np.array(batch_sizes, dtype=float)
    n_cols = len(fixed_dims)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)

    for col, D in enumerate(fixed_dims):
        ax = axes[0, col]
        for method, color in METHOD_COLORS.items():
            if D in results[method]:
                times = results[method][D]
                ax.plot(batch_sizes, times / B_arr, "o-", color=color,
                        label=METHOD_LABELS[method], markersize=3)
        ax.set_xlabel("Batch size B")
        ax.set_ylabel("Time / B (ms per sample)")
        ax.set_title(f"D = {D}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels([str(b) for b in batch_sizes], rotation=45, fontsize=7)
        fmt_tick(ax, "x")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-sample cost vs B", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 1.4: Timing — SIGReg vs W1 vs W2"
    )
    p.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--dimensions", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--save-dir", default="results/exp1_4_timing")
    p.add_argument("--device", default="auto")
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

    print(f"Exp 1.4: Timing Comparison")
    print(f"  batch_sizes={args.batch_sizes}")
    print(f"  dimensions={args.dimensions}")
    print(f"  device={device}")
    print()

    methods = {
        "sigreg": lambda: SIGRegLoss(knots=args.knots, num_proj=args.num_proj).to(device),
        "w1": lambda: SlicedW1Loss(num_proj=args.num_proj).to(device),
        "w2": lambda: SlicedW2Loss(num_proj=args.num_proj).to(device),
    }

    # Collect timing: grids[method] = (len(dims), len(batches)) array
    # fwd/fwdbwd _grids = median, _std_grids = std from repeats
    fwd_grids = {m: np.zeros((len(args.dimensions), len(args.batch_sizes)))
                 for m in methods}
    fwdbwd_grids = {m: np.zeros((len(args.dimensions), len(args.batch_sizes)))
                    for m in methods}
    fwd_std_grids = {m: np.zeros((len(args.dimensions), len(args.batch_sizes)))
                     for m in methods}
    fwdbwd_std_grids = {m: np.zeros((len(args.dimensions), len(args.batch_sizes)))
                        for m in methods}
    fwd_results = {m: {} for m in methods}
    fwdbwd_results = {m: {} for m in methods}

    total = len(args.dimensions) * len(args.batch_sizes) * len(methods)
    run_idx = 0

    for i, D in enumerate(args.dimensions):
        for method_name, make_fn in methods.items():
            loss_fn = make_fn()
            fwd_times, fwdbwd_times = [], []
            for j, B in enumerate(args.batch_sizes):
                run_idx += 1
                x = torch.randn(B, D, device=device)

                # Forward only
                with torch.no_grad():
                    fwd_med, fwd_std = time_forward(
                        loss_fn, x, args.warmup, args.repeats)
                fwd_grids[method_name][i, j] = fwd_med
                fwd_std_grids[method_name][i, j] = fwd_std
                fwd_times.append(fwd_med)

                # Forward + backward
                fwdbwd_med, fwdbwd_std = time_forward_backward(
                    loss_fn, x, args.warmup, args.repeats)
                fwdbwd_grids[method_name][i, j] = fwdbwd_med
                fwdbwd_std_grids[method_name][i, j] = fwdbwd_std
                fwdbwd_times.append(fwdbwd_med)

                print(f"[{run_idx}/{total}] {method_name:>7s} D={D:>3d} B={B:>5d}  "
                      f"fwd={fwd_med:.3f}ms  fwd+bwd={fwdbwd_med:.3f}ms  "
                      f"bwd={fwdbwd_med - fwd_med:.3f}ms")

            fwd_results[method_name][D] = np.array(fwd_times)
            fwdbwd_results[method_name][D] = np.array(fwdbwd_times)

    # --- Heatmaps (forward only) ---
    pairs = [
        ("sigreg", "w1", "SIGReg / W1 forward time ratio\n(>1 = SIGReg slower)",
         "heatmap_fwd_sigreg_w1.png"),
        ("sigreg", "w2", "SIGReg / W2 forward time ratio\n(>1 = SIGReg slower)",
         "heatmap_fwd_sigreg_w2.png"),
        ("w1", "w2", "W1 / W2 forward time ratio\n(>1 = W1 slower)",
         "heatmap_fwd_w1_w2.png"),
    ]
    for m1, m2, title, fname in pairs:
        ratio = fwd_grids[m1] / np.maximum(fwd_grids[m2], 1e-9)
        plot_heatmap(ratio, args.dimensions, args.batch_sizes, title,
                     os.path.join(plots_dir, fname))

    # --- Heatmaps (forward + backward) ---
    pairs_fb = [
        ("sigreg", "w1", "SIGReg / W1 fwd+bwd time ratio\n(>1 = SIGReg slower)",
         "heatmap_fwdbwd_sigreg_w1.png"),
        ("sigreg", "w2", "SIGReg / W2 fwd+bwd time ratio\n(>1 = SIGReg slower)",
         "heatmap_fwdbwd_sigreg_w2.png"),
        ("w1", "w2", "W1 / W2 fwd+bwd time ratio\n(>1 = W1 slower)",
         "heatmap_fwdbwd_w1_w2.png"),
    ]
    for m1, m2, title, fname in pairs_fb:
        ratio = fwdbwd_grids[m1] / np.maximum(fwdbwd_grids[m2], 1e-9)
        plot_heatmap(ratio, args.dimensions, args.batch_sizes, title,
                     os.path.join(plots_dir, fname))

    # --- Per-sample cost (both forward and fwd+bwd) ---
    fixed_dims = [d for d in [4, 64, 256] if d in args.dimensions]
    if fixed_dims:
        plot_per_sample_cost(fwd_results, args.batch_sizes, fixed_dims,
                             os.path.join(plots_dir, "per_sample_cost_fwd.png"))
        plot_per_sample_cost(fwdbwd_results, args.batch_sizes, fixed_dims,
                             os.path.join(plots_dir, "per_sample_cost_fwdbwd.png"))

    # --- Save raw data ---
    json_out = {}
    for method_name in methods:
        for i, D in enumerate(args.dimensions):
            for j, B in enumerate(args.batch_sizes):
                key = f"{method_name}_D{D}_B{B}"
                json_out[key] = {
                    "method": method_name, "D": D, "B": B,
                    "fwd_ms": float(fwd_grids[method_name][i, j]),
                    "fwd_std_ms": float(fwd_std_grids[method_name][i, j]),
                    "fwdbwd_ms": float(fwdbwd_grids[method_name][i, j]),
                    "fwdbwd_std_ms": float(fwdbwd_std_grids[method_name][i, j]),
                    "bwd_ms": float(fwdbwd_grids[method_name][i, j]
                                    - fwd_grids[method_name][i, j]),
                }

    with open(os.path.join(summary_dir, "timing_results.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    # --- Summary table ---
    for phase, grids in [("Forward only", fwd_grids),
                         ("Forward + Backward", fwdbwd_grids)]:
        print(f"\n===== {phase} (median ms) =====")
        header = f"{'D':>4} {'B':>6}"
        for m in methods:
            header += f" | {METHOD_LABELS[m]:>10s}"
        print(header)
        print("-" * len(header))
        for i, D in enumerate(args.dimensions):
            for j, B in enumerate(args.batch_sizes):
                line = f"{D:>4} {B:>6}"
                for m in methods:
                    line += f" | {grids[m][i, j]:>10.3f}"
                print(line)

    print(f"\nSaved to {args.save_dir}/")


if __name__ == "__main__":
    main()
