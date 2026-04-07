"""
Experiment 1.2: Standard vs Pooled Accumulation

Shows that pooled accumulation breaks the finite-sample bias floor while
standard gradient accumulation (averaging independent losses) does not.

6 methods: {W1, W2, SIGReg} x {standard, pooled}
Sweep: T={1,2,4,8,16,32}, M={8,32}, all 5 generators, 3 seeds.

Standard: T independent losses averaged (T backward passes).
Pooled (2-step): T-1 no-grad forward passes + 1 grad forward pass,
    pool all T*BS embeddings, compute loss, 1 backward pass.

Pipeline:
    Source (K, d) -> Fixed projection (K, M) -> MLP (K, d) -> Loss

Run:
    python experiments/synthetic/exp1_2_accumulation.py
    python experiments/synthetic/exp1_2_accumulation.py --distributions blobs --proj-dims 8 --accum-steps 1 4 8 --n-seeds 1 --steps 500
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
    SlicedW1Loss, SlicedW2Loss, SIGRegLoss, AccumulatedSlicedLoss,
    DeepMLP, generate_data, make_fixed_projection, GENERATORS,
    eval_w1, evaluate_full,
)
from module import SIGReg


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

def _setup(seed, args, proj_dim, distribution, device):
    """Shared setup for both training modes."""
    D = args.input_dim
    torch.manual_seed(seed)
    data = generate_data(distribution, args.num_points, D)
    W = make_fixed_projection(D, proj_dim, seed=seed + 100).to(device)
    projected = data.to(device) @ W

    src_np = data.numpy()
    colors = np.arctan2(src_np[:, 1] if D > 1 else np.zeros(len(src_np)),
                        src_np[:, 0])
    colors = (colors + np.pi) / (2 * np.pi)

    torch.manual_seed(seed + 200)
    mlp = DeepMLP(proj_dim, args.hidden_dim, D, depth=args.depth).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    mlp.eval()
    with torch.no_grad():
        initial_out = mlp(projected).cpu().numpy()
    mlp.train()

    return projected, mlp, opt, src_np, colors, initial_out


def train_one_standard(seed, args, loss_mode, proj_dim, distribution, T, device):
    """Standard gradient accumulation: T independent losses averaged."""
    projected, mlp, opt, src_np, colors, initial_out = \
        _setup(seed, args, proj_dim, distribution, device)
    K = args.num_points
    BS = args.batch_size
    loss_fn = make_loss_fn(loss_mode, args.num_proj, args.knots, device)

    eval_steps, eval_w1s = [], []

    for step in range(1, args.steps + 1):
        mlp.train()
        opt.zero_grad()
        for _ in range(T):
            idx = torch.randint(0, K, (BS,), device=projected.device)
            loss = loss_fn(mlp(projected[idx])) / T
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


def train_one_pooled(seed, args, loss_mode, proj_dim, distribution, T, device):
    """Pooled accumulation: T-1 no-grad forward passes + 1 grad pass.

    Two-step procedure:
        1. Run T-1 forward passes with torch.no_grad(), collect detached embeddings
        2. Run 1 forward pass with gradient, pool all T*BS embeddings, compute loss
    Only 1 backward pass per update — cheaper than T-step grad accumulation.
    """
    projected, mlp, opt, src_np, colors, initial_out = \
        _setup(seed, args, proj_dim, distribution, device)
    K = args.num_points
    BS = args.batch_size

    sigreg_mod = None
    if loss_mode == "sigreg":
        sigreg_mod = SIGReg(knots=args.knots, num_proj=args.num_proj).to(device)

    accum_loss = AccumulatedSlicedLoss(
        accum_steps=max(T - 1, 0), num_proj=args.num_proj,
        mode=loss_mode, sigreg=sigreg_mod)

    eval_steps, eval_w1s = [], []

    for step in range(1, args.steps + 1):
        mlp.train()

        # T-1 no-grad forward passes (collect CDF context, no activation storage)
        with torch.no_grad():
            for _ in range(T - 1):
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

METHOD_STYLES = {
    "w1_standard":     {"color": "tab:blue",   "ls": "--", "label": "W1 standard"},
    "w1_pooled":       {"color": "tab:blue",   "ls": "-",  "label": "W1 pooled"},
    "w2_standard":     {"color": "tab:green",  "ls": "--", "label": "W2 standard"},
    "w2_pooled":       {"color": "tab:green",  "ls": "-",  "label": "W2 pooled"},
    "sigreg_standard": {"color": "tab:orange", "ls": "--", "label": "SIGReg standard"},
    "sigreg_pooled":   {"color": "tab:orange", "ls": "-",  "label": "SIGReg pooled"},
}


def plot_bias_floor(all_runs, dist, M, T_values, seeds, save_path):
    """Key figure: final eval_w1 vs T for all 6 methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_key, style in METHOD_STYLES.items():
        means, stds = [], []
        for T in T_values:
            vals = []
            for seed in seeds:
                result = all_runs.get((dist, M, method_key, T, seed))
                if result is not None:
                    vals.append(result[6]["w1"])  # final eval_w1 from metrics
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0.0)

        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(T_values, means, yerr=stds, fmt="o",
                     color=style["color"], linestyle=style["ls"],
                     label=style["label"], linewidth=2, markersize=5,
                     capsize=3)

    ax.set_xlabel("Accumulation Steps T", fontsize=12)
    ax.set_ylabel("Final Sliced W1 to N(0,1)", fontsize=11)
    ax.set_title(f"{dist}, M={M}, micro_bs={8}", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(T_values)
    ax.set_xticklabels([str(t) for t in T_values])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 1.2: Standard vs Pooled Accumulation"
    )
    p.add_argument("--distributions", nargs="+",
                    default=list(GENERATORS.keys()),
                    choices=list(GENERATORS.keys()))
    p.add_argument("--input-dim", type=int, default=4)
    p.add_argument("--proj-dims", type=int, nargs="+", default=[8, 32])
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--save-dir", default="results/exp1_2_accumulation")
    p.add_argument("--num-workers", type=int, default=1,
                    help="Total parallel workers (for sharding)")
    p.add_argument("--worker-id", type=int, default=0,
                    help="This worker's index (0..num_workers-1)")
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
    seeds = [42 + i for i in range(args.n_seeds)]
    loss_modes = ["w1", "w2", "sigreg"]
    accum_modes = ["standard", "pooled"]
    T_values = sorted(args.accum_steps)

    # Count runs: T=1 only needs 1 run per loss (standard==pooled), T>1 needs 2
    n_T1 = len(loss_modes)  # 3
    n_Tgt1 = len([T for T in T_values if T > 1]) * len(loss_modes) * 2  # T>1 * 3 * 2
    runs_per_combo = n_T1 + n_Tgt1
    total = len(args.distributions) * len(args.proj_dims) * len(seeds) * runs_per_combo

    print(f"Exp 1.2: Standard vs Pooled Accumulation")
    print(f"  distributions={args.distributions}")
    print(f"  proj_dims={args.proj_dims}, T_values={T_values}")
    print(f"  batch_size={args.batch_size}, seeds={seeds}")
    print(f"  total_runs~={total}, device={device}")
    print()

    # Build flat config list (excluding T=1 pooled duplicates)
    configs = []
    for dist in args.distributions:
        for M in args.proj_dims:
            for T in T_values:
                for loss_mode in loss_modes:
                    for accum_mode in accum_modes:
                        if T == 1 and accum_mode == "pooled":
                            continue  # handled via copy below
                        method_key = f"{loss_mode}_{accum_mode}"
                        for seed in seeds:
                            configs.append((dist, M, T, loss_mode, accum_mode, method_key, seed))

    # Run configs assigned to this worker
    all_runs = {}
    for cfg_idx, (dist, M, T, loss_mode, accum_mode, method_key, seed) in enumerate(configs):
        model_path = os.path.join(
            args.save_dir,
            f"model_{dist}_{method_key}_M{M}_T{T}_{seed}.pt")

        if os.path.exists(model_path):
            print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, M={M}, T={T}, "
                  f"{method_key}, seed={seed} SKIP (exists)")
            state = torch.load(model_path, map_location=device,
                               weights_only=True)
            proj, mlp, _, src_np, colors, initial_out = \
                _setup(seed, args, M, dist, device)
            mlp.load_state_dict(state)
            mlp.eval()
            with torch.no_grad():
                final_out = mlp(proj).cpu().numpy()
            metrics = evaluate_full(final_out)
            all_runs[(dist, M, method_key, T, seed)] = (
                [], [], final_out, initial_out, src_np,
                colors, metrics, state)
            continue

        # Shard: only this worker trains new configs
        if cfg_idx % args.num_workers != args.worker_id:
            continue

        print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, M={M}, T={T}, "
              f"{method_key}, seed={seed}",
              end=" ", flush=True)
        t0 = time.perf_counter()

        if accum_mode == "standard":
            result = train_one_standard(
                seed, args, loss_mode, M, dist, T, device)
        else:
            result = train_one_pooled(
                seed, args, loss_mode, M, dist, T, device)

        elapsed = time.perf_counter() - t0
        all_runs[(dist, M, method_key, T, seed)] = result
        print(f"W1={result[6]['w1']:.4f}  time={elapsed:.1f}s")

        torch.save(result[7], model_path)

    # T=1 copies: pooled == standard
    for dist in args.distributions:
        for M in args.proj_dims:
            if 1 in T_values:
                for loss_mode in loss_modes:
                    std_key = (dist, M, f"{loss_mode}_standard", 1)
                    pool_key = (dist, M, f"{loss_mode}_pooled", 1)
                    for seed in seeds:
                        sk = (*std_key, seed)
                        pk = (*pool_key, seed)
                        if sk in all_runs:
                            all_runs[pk] = all_runs[sk]

    # --- Plots ---
    for dist in args.distributions:
        for M in args.proj_dims:
            plot_bias_floor(
                all_runs, dist, M, T_values, seeds,
                os.path.join(plots_dir, f"bias_floor_{dist}_M{M}.png"))

    # --- Summary ---
    summary_lines = [
        f"Exp 1.2: Standard vs Pooled Accumulation",
        f"d={args.input_dim}, K={args.num_points}, micro_bs={args.batch_size}, "
        f"steps={args.steps}, seeds={len(seeds)}",
        "",
        f"{'Distribution':<18} {'Method':<20} {'M':>3} {'T':>3}  {'W1':>14}",
        "-" * 70,
    ]

    json_out = {}
    for dist in args.distributions:
        for M in args.proj_dims:
            for method_key in METHOD_STYLES:
                for T in T_values:
                    vals = []
                    for seed in seeds:
                        result = all_runs.get((dist, M, method_key, T, seed))
                        if result is not None:
                            vals.append(result[6])  # final_metrics dict
                    if not vals:
                        continue

                    key = f"{dist}_{method_key}_M{M}_T{T}"
                    agg = {}
                    for mk in vals[0]:
                        if isinstance(vals[0][mk], (int, float)):
                            v = [m[mk] for m in vals]
                            agg[mk] = {"mean": float(np.mean(v)),
                                        "std": float(np.std(v))}
                    json_out[key] = agg

                    w1m = agg["w1"]["mean"]
                    w1s = agg["w1"]["std"]
                    summary_lines.append(
                        f"{dist:<18} {method_key:<20} {M:>3} {T:>3}  "
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
