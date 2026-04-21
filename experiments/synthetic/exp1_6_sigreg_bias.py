"""
Experiment 1.6: SIGReg Bias Variants — Biased vs U-stat vs Sample-Split

Tests whether U-statistic debiasing or a sample-split estimator of |φ_θ(t)|²
changes the SIGReg loss floor, by sweeping the full-batch size K and
comparing the three variants.

Background: the plug-in estimator |φ̂(t)|² has a positive bias of
(1 − |φ_θ(t)|²)/n that vanishes only as n → ∞. After SIGReg's n-rescaling
this becomes a (1 − |φ_θ|²) offset per frequency. This offset depends on
the model state through φ_θ, so its gradient is not zero — debiasing may
or may not move the optimization in practice.

Variants (see ``experiments/synthetic/sliced_gauss_reg/sigreg.py``):
    biased — current SIGReg: |φ̂(t) − φ_N(t)|²
    ustat  — U-statistic: |φ_θ|² ≈ (n/(n-1))|φ̂|² − 1/(n-1)
    split  — sample split: |φ_θ|² ≈ Re(φ̂_A · φ̂_B*) on disjoint halves

Pipeline (full batch):
    Source (K, d) -> Fixed projection (K, M) -> MLP (K, d) -> SIGReg variant

The bias scales as 1/n. With K ≳ 10³ all three variants should give
indistinguishable results; the interesting regime is K ≲ 256.

Run:
    python experiments/synthetic/exp1_6_sigreg_bias.py
    python experiments/synthetic/exp1_6_sigreg_bias.py --distributions blobs --Ks 64 256 1024 --n-seeds 1 --steps 1000
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

SYNTHETIC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SYNTHETIC_DIR)

from sliced_gauss_reg import (
    SIGRegLoss,
    DeepMLP, generate_data, make_fixed_projection, GENERATORS,
    eval_w1, evaluate_full,
)


VARIANTS = ["biased", "ustat", "split"]
VARIANT_COLORS = {"biased": "tab:orange", "ustat": "tab:purple", "split": "tab:cyan"}
VARIANT_LABELS = {"biased": "SIGReg (biased)",
                  "ustat":  "SIGReg (U-stat)",
                  "split":  "SIGReg (split)"}


# ---------------------------------------------------------------------------
# Training (full batch)
# ---------------------------------------------------------------------------

def train_one(seed, args, K, variant, distribution, device):
    """Full-batch training with one SIGReg variant."""
    D = args.input_dim
    M = args.proj_dim

    torch.manual_seed(seed)
    data = generate_data(distribution, K, D)
    W = make_fixed_projection(D, M, seed=seed + 100).to(device)
    projected = data.to(device) @ W

    torch.manual_seed(seed + 200)
    mlp = DeepMLP(M, args.hidden_dim, D, depth=args.depth).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    loss_fn = SIGRegLoss(knots=args.knots, num_proj=args.num_proj,
                          bias_mode=variant).to(device)

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
                w1_val = eval_w1(mlp(projected).cpu().numpy())
                print(f"  step={step:5d}  loss={loss.item():+.4f}  W1={w1_val:.4f}")
            mlp.train()

    mlp.eval()
    with torch.no_grad():
        final_out = mlp(projected).cpu().numpy()
        final_metrics = evaluate_full(final_out)

    return final_metrics, mlp.state_dict()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_w1_vs_K(all_runs, dist, K_values, seeds, save_path, M):
    """Final W1 vs full-batch K, one curve per variant. Median across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant in VARIANTS:
        meds, lows, highs = [], [], []
        for K in K_values:
            vals = []
            for seed in seeds:
                r = all_runs.get((dist, K, variant, seed))
                if r is not None:
                    vals.append(r[0]["w1"])
            if vals:
                meds.append(np.median(vals))
                # Min/max as light error bars when seeds are few
                lows.append(np.median(vals) - np.min(vals))
                highs.append(np.max(vals) - np.median(vals))
            else:
                meds.append(np.nan)
                lows.append(0.0)
                highs.append(0.0)
        meds = np.array(meds)
        ax.errorbar(K_values, meds, yerr=[lows, highs], fmt="o-",
                     color=VARIANT_COLORS[variant],
                     label=VARIANT_LABELS[variant],
                     linewidth=2, markersize=6, capsize=3)

    ax.set_xlabel("Full-batch K", fontsize=12)
    ax.set_ylabel("Final Sliced W1 to N(0,1)  [median over seeds]", fontsize=11)
    ax.set_title(f"{dist}: SIGReg variants vs K  (M={M})", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_values)
    ax.set_xticklabels([str(k) for k in K_values])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 1.6: SIGReg bias variants — biased vs U-stat vs split"
    )
    p.add_argument("--distributions", nargs="+",
                    default=list(GENERATORS.keys()),
                    choices=list(GENERATORS.keys()))
    p.add_argument("--input-dim", type=int, default=4)
    p.add_argument("--proj-dim", type=int, default=8)
    p.add_argument("--Ks", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024, 2048, 4096, 16384],
                    help="Full-batch sizes to sweep")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-dir", default="results/exp1_6_sigreg_bias")
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
    K_values = sorted(args.Ks)

    total = len(args.distributions) * len(K_values) * len(VARIANTS) * len(seeds)
    print(f"Exp 1.6: SIGReg bias variants")
    print(f"  distributions={args.distributions}")
    print(f"  Ks={K_values}, variants={VARIANTS}")
    print(f"  d={args.input_dim}, M={args.proj_dim}, steps={args.steps}")
    print(f"  seeds={seeds}, total_runs={total}, device={device}")
    print()

    configs = []
    for dist in args.distributions:
        for K in K_values:
            for variant in VARIANTS:
                for seed in seeds:
                    configs.append((dist, K, variant, seed))

    all_runs = {}
    for cfg_idx, (dist, K, variant, seed) in enumerate(configs):
        model_path = os.path.join(
            args.save_dir, f"model_{dist}_{variant}_K{K}_{seed}.pt")

        if os.path.exists(model_path):
            print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, K={K}, "
                  f"variant={variant}, seed={seed} SKIP (exists)")
            state = torch.load(model_path, map_location=device, weights_only=True)
            torch.manual_seed(seed)
            data = generate_data(dist, K, args.input_dim)
            W = make_fixed_projection(args.input_dim, args.proj_dim, seed=seed + 100).to(device)
            proj = data.to(device) @ W
            torch.manual_seed(seed + 200)
            mlp = DeepMLP(args.proj_dim, args.hidden_dim, args.input_dim, depth=args.depth).to(device)
            mlp.load_state_dict(state)
            mlp.eval()
            with torch.no_grad():
                final_out = mlp(proj).cpu().numpy()
            metrics = evaluate_full(final_out)
            all_runs[(dist, K, variant, seed)] = (metrics, state)
            continue

        if cfg_idx % args.num_workers != args.worker_id:
            continue

        print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, K={K}, "
              f"variant={variant}, seed={seed}", end=" ", flush=True)
        t0 = time.perf_counter()
        metrics, state = train_one(seed, args, K, variant, dist, device)
        elapsed = time.perf_counter() - t0
        all_runs[(dist, K, variant, seed)] = (metrics, state)
        print(f"W1={metrics['w1']:.4f}  time={elapsed:.1f}s")

        torch.save(state, model_path)

    # --- Plots ---
    for dist in args.distributions:
        plot_w1_vs_K(
            all_runs, dist, K_values, seeds,
            os.path.join(plots_dir, f"w1_vs_K_{dist}.png"),
            M=args.proj_dim)

    # --- Summary ---
    summary_lines = [
        f"Exp 1.6: SIGReg Bias Variants",
        f"d={args.input_dim}, M={args.proj_dim}, steps={args.steps}, "
        f"seeds={len(seeds)}",
        f"Aggregation: median across seeds (min/max as error bars)",
        "",
        f"{'Distribution':<18} {'Variant':<10} {'K':>6}  "
        f"{'W1 median':>12} {'W1 min':>10} {'W1 max':>10}",
        "-" * 75,
    ]

    json_out = {}
    for dist in args.distributions:
        for K in K_values:
            for variant in VARIANTS:
                vals = []
                for seed in seeds:
                    r = all_runs.get((dist, K, variant, seed))
                    if r is not None:
                        vals.append(r[0])
                if not vals:
                    continue

                key = f"{dist}_{variant}_K{K}"
                agg = {}
                for mk in vals[0]:
                    if isinstance(vals[0][mk], (int, float)):
                        v = [m[mk] for m in vals]
                        agg[mk] = {"median": float(np.median(v)),
                                    "min": float(np.min(v)),
                                    "max": float(np.max(v))}
                json_out[key] = agg

                w1med = agg["w1"]["median"]
                w1min = agg["w1"]["min"]
                w1max = agg["w1"]["max"]
                summary_lines.append(
                    f"{dist:<18} {variant:<10} {K:>6}  "
                    f"{w1med:>12.4f} {w1min:>10.4f} {w1max:>10.4f}")
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
