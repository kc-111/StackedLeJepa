"""
Experiment 1.8: Effect of latent dimension on convergence

Sweeps the source/output latent dimension D ∈ {4, 8, 16, 32, 64} with fixed
batch_size=32, T=8 pooled accumulation, fixed projection_dim=256, and tracks
sliced W1 to N(0,1), mean magnitude, and covariance distance to identity each
epoch. Compares 4 loss methods: w1, w2, sigreg, sigreg+w1.

Pipeline:
    Source (K, D) -> Fixed projection (K, 256) -> MLP (K, D) -> Loss

Per the user's request: 3 seeds × 5 dims × 4 methods. The MLP output (and
loss-input) dimension D is what we vary; the fixed source-to-MLP projection
width is held at 256 across the sweep.

Run:
    python experiments/synthetic/exp1_8_dim_sweep.py
    python experiments/synthetic/exp1_8_dim_sweep.py --dims 4 16 64 --epochs 50
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

from src.sliced_gauss_reg import (
    SIGReg, PooledSlicedLoss, DeepMLP, generate_data, make_fixed_projection,
    epoch_iter, evaluate_full_gpu,
)


# ---------------------------------------------------------------------------
# Setup + training
# ---------------------------------------------------------------------------

def _setup(seed, args, D, distribution, device):
    torch.manual_seed(seed)
    data = generate_data(distribution, args.num_points, D)
    W = make_fixed_projection(D, args.proj_dim, seed=seed + 100).to(device)
    projected = data.to(device) @ W

    torch.manual_seed(seed + 200)
    mlp = DeepMLP(args.proj_dim, args.hidden_dim, D, depth=args.depth).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    return projected, mlp, opt


def train_pooled(seed, args, loss_mode, D, distribution, device):
    """Pooled accumulation, evaluating distributional metrics each epoch."""
    projected, mlp, opt = _setup(seed, args, D, distribution, device)
    K = args.num_points
    BS = args.batch_size
    T = args.accum_steps
    chunk = T * BS
    if chunk > K:
        raise ValueError(f"T*BS={chunk} exceeds dataset K={K}")

    sigreg_mod = None
    if "sigreg" in loss_mode:
        sigreg_mod = SIGReg(knots=args.knots, num_proj=args.num_proj).to(device)
    accum_loss = PooledSlicedLoss(
        accum_steps=max(T - 1, 0), num_proj=args.num_proj,
        mode=loss_mode, sigreg=sigreg_mod)
    epoch_gen = torch.Generator(device=device).manual_seed(seed + 400)

    history = {"epoch": [], "w1": [], "w2": [], "mean_mse": [],
               "cov_frob": [], "cov_offdiag_max": []}

    # Epoch 0 baseline (untrained MLP)
    mlp.eval()
    with torch.no_grad():
        out0 = mlp(projected)
        m0 = evaluate_full_gpu(out0, num_proj=args.eval_num_proj)
    del out0
    history["epoch"].append(0)
    for k in ("w1", "w2", "mean_mse", "cov_frob", "cov_offdiag_max"):
        history[k].append(float(m0[k]))
    mlp.train()

    for epoch in range(1, args.epochs + 1):
        for big_chunk in epoch_iter(K, chunk, device, epoch_gen):
            mlp.train()
            with torch.no_grad():
                for t in range(T - 1):
                    sub = big_chunk[t * BS:(t + 1) * BS]
                    accum_loss.accum_step(mlp(projected[sub]))
            sub = big_chunk[(T - 1) * BS:T * BS]
            loss = accum_loss.grad_step(mlp(projected[sub]))
            opt.zero_grad()
            loss.backward()
            opt.step()

        mlp.eval()
        with torch.no_grad():
            out = mlp(projected)
            m = evaluate_full_gpu(out, num_proj=args.eval_num_proj)
        del out
        history["epoch"].append(epoch)
        for k in ("w1", "w2", "mean_mse", "cov_frob", "cov_offdiag_max"):
            history[k].append(float(m[k]))
        mlp.train()

    return history, mlp.state_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Exp 1.8: latent dim sweep")
    p.add_argument("--distributions", nargs="+",
                   default=["blobs"], choices=["blobs"])
    p.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32, 64],
                   help="Source/output latent dims D to sweep.")
    p.add_argument("--proj-dim", type=int, default=256,
                   help="Fixed source-to-MLP projection width (held constant).")
    p.add_argument("--num-points", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-proj", type=int, default=1024,
                   help="Random sliced directions for the training loss.")
    p.add_argument("--eval-num-proj", type=int, default=1024,
                   help="Random sliced directions for the per-epoch eval.")
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--loss-modes", nargs="+",
                   default=["w1", "w2", "sigreg", "sigreg+w1"],
                   help="4 default methods. Combined modes (sigreg+w1, etc.) "
                        "sum the components with equal weight.")
    p.add_argument("--save-dir", default="results/exp1_8_dim_sweep")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    plots_dir = os.path.join(args.save_dir, "plots")
    summary_dir = os.path.join(args.save_dir, "summary")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    seeds = [42 + i for i in range(args.n_seeds)]
    dims = sorted(args.dims)
    loss_modes = args.loss_modes

    print(f"Exp 1.8: Latent dim sweep")
    print(f"  distributions={args.distributions}")
    print(f"  dims={dims}, proj_dim={args.proj_dim}")
    print(f"  batch_size={args.batch_size}, T={args.accum_steps}, "
          f"num_proj={args.num_proj}, epochs={args.epochs}")
    print(f"  loss_modes={loss_modes}, seeds={seeds}, device={device}")
    print()

    configs = []
    for dist in args.distributions:
        for D in dims:
            for loss_mode in loss_modes:
                for seed in seeds:
                    configs.append((dist, D, loss_mode, seed))

    histories = {}
    for cfg_idx, (dist, D, loss_mode, seed) in enumerate(configs):
        slug = loss_mode.replace("+", "-")
        history_path = os.path.join(
            args.save_dir,
            f"history_{dist}_{slug}_D{D}_seed{seed}.json")
        model_path = os.path.join(
            args.save_dir,
            f"model_{dist}_{slug}_D{D}_seed{seed}.pt")

        if os.path.exists(history_path):
            with open(history_path) as f:
                histories[(dist, D, loss_mode, seed)] = json.load(f)
            print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, D={D}, "
                  f"{loss_mode}, seed={seed} SKIP (exists)")
            continue

        if cfg_idx % args.num_workers != args.worker_id:
            continue

        print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, D={D}, "
              f"{loss_mode}, seed={seed}", end=" ", flush=True)
        t0 = time.perf_counter()
        history, state = train_pooled(seed, args, loss_mode, D, dist, device)
        elapsed = time.perf_counter() - t0
        histories[(dist, D, loss_mode, seed)] = history
        print(f"final w1={history['w1'][-1]:.4f}  cov={history['cov_frob'][-1]:.4f}  "
              f"|μ|²={history['mean_mse'][-1]:.4f}  time={elapsed:.1f}s")

        torch.save(state, model_path)
        with open(history_path, "w") as f:
            json.dump(history, f)

    # ---------- Aggregate per (dist, D, loss_mode) using median ----------
    aggregated = {}
    for dist in args.distributions:
        for D in dims:
            for loss_mode in loss_modes:
                seed_hists = [histories.get((dist, D, loss_mode, s)) for s in seeds]
                seed_hists = [h for h in seed_hists if h is not None]
                if not seed_hists:
                    continue
                agg = {"epoch": list(seed_hists[0]["epoch"])}
                for k in ("w1", "w2", "mean_mse", "cov_frob", "cov_offdiag_max"):
                    arr = np.array([h[k] for h in seed_hists])  # (n_seeds, E)
                    agg[k + "_median"] = np.median(arr, axis=0).tolist()
                    agg[k + "_p25"] = np.percentile(arr, 25, axis=0).tolist()
                    agg[k + "_p75"] = np.percentile(arr, 75, axis=0).tolist()
                aggregated[f"{dist}|{loss_mode}|D{D}"] = agg

    with open(os.path.join(summary_dir, "aggregated.json"), "w") as f:
        json.dump(aggregated, f, indent=2)

    # ---------- Summary text ----------
    lines = [
        f"Exp 1.8: latent dim sweep",
        f"  proj_dim={args.proj_dim}, BS={args.batch_size}, T={args.accum_steps}, "
        f"num_proj={args.num_proj}, epochs={args.epochs}, n_seeds={args.n_seeds}",
        "",
        f"{'Distribution':<14} {'Loss':<14} {'D':>4}  "
        f"{'W1 (median[p25,p75])':<28} {'mean_mse':>10} {'cov_frob':>10}",
        "-" * 90,
    ]
    for dist in args.distributions:
        for loss_mode in loss_modes:
            for D in dims:
                key = f"{dist}|{loss_mode}|D{D}"
                if key not in aggregated:
                    continue
                a = aggregated[key]
                w1m = a["w1_median"][-1]
                w1lo = a["w1_p25"][-1]
                w1hi = a["w1_p75"][-1]
                mm = a["mean_mse_median"][-1]
                cv = a["cov_frob_median"][-1]
                lines.append(
                    f"{dist:<14} {loss_mode:<14} {D:>4}  "
                    f"{w1m:.4f}[{w1lo:.4f},{w1hi:.4f}]   "
                    f"{mm:>10.4f} {cv:>10.4f}")
            lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(summary_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    # ---------- Plots ----------
    # One figure per (distribution, loss_mode); 4 panels (W1, mean_mse,
    # cov_frob, cov_offdiag_max) with one line per D + IQR shading.
    cmap = plt.get_cmap("viridis")
    for dist in args.distributions:
        for loss_mode in loss_modes:
            fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
            any_data = False
            for di, D in enumerate(dims):
                key = f"{dist}|{loss_mode}|D{D}"
                if key not in aggregated:
                    continue
                any_data = True
                a = aggregated[key]
                color = cmap(di / max(len(dims) - 1, 1))
                ep = a["epoch"]
                for ax, k in zip(axes,
                                  ("w1", "mean_mse", "cov_frob", "cov_offdiag_max")):
                    ax.plot(ep, a[k + "_median"], color=color,
                            label=f"D={D}", linewidth=2)
                    ax.fill_between(ep, a[k + "_p25"], a[k + "_p75"],
                                    color=color, alpha=0.15)
            if not any_data:
                plt.close(fig)
                continue
            for ax, title, ylabel in [
                (axes[0], "Sliced W1 to N(0,1)", "W1"),
                (axes[1], "||mean(emb)||²", "mean_mse"),
                (axes[2], "||cov - I||_F / ||I||_F", "cov_frob"),
                (axes[3], "max |off-diag(cov)|", "cov_offdiag_max"),
            ]:
                ax.set_xlabel("Epoch")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            fig.suptitle(f"Exp 1.8: dim sweep — {dist}, {loss_mode} "
                         f"(BS={args.batch_size}, T={args.accum_steps}, "
                         f"proj_dim={args.proj_dim})")
            plt.tight_layout()
            slug = loss_mode.replace("+", "-")
            plot_path = os.path.join(plots_dir, f"{dist}_{slug}.png")
            plt.savefig(plot_path, dpi=110)
            plt.close()

    # Cross-method comparison: one figure per distribution, one column per
    # metric, one row per D. Lines = methods.
    method_colors = {
        "w1":         "tab:blue",
        "w2":         "tab:green",
        "sigreg":     "tab:orange",
        "sigreg+w1":  "tab:red",
        "sigreg+w2":  "tab:purple",
    }
    metrics = [("w1", "Sliced W1"), ("mean_mse", "||mean||²"),
               ("cov_frob", "||cov-I||_F / ||I||_F")]
    for dist in args.distributions:
        fig, axes = plt.subplots(len(dims), len(metrics),
                                  figsize=(5 * len(metrics), 3 * len(dims)),
                                  squeeze=False)
        for ri, D in enumerate(dims):
            for ci, (mkey, mtitle) in enumerate(metrics):
                ax = axes[ri][ci]
                for loss_mode in loss_modes:
                    key = f"{dist}|{loss_mode}|D{D}"
                    if key not in aggregated:
                        continue
                    a = aggregated[key]
                    c = method_colors.get(loss_mode, "gray")
                    ep = a["epoch"]
                    ax.plot(ep, a[mkey + "_median"], color=c,
                            label=loss_mode, linewidth=2)
                    ax.fill_between(ep, a[mkey + "_p25"], a[mkey + "_p75"],
                                    color=c, alpha=0.15)
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                if ri == 0:
                    ax.set_title(mtitle)
                if ci == 0:
                    ax.set_ylabel(f"D={D}")
                if ri == len(dims) - 1:
                    ax.set_xlabel("Epoch")
                if ri == 0 and ci == len(metrics) - 1:
                    ax.legend(fontsize=8, loc="best")
        fig.suptitle(f"Exp 1.8: methods × dims — {dist} "
                     f"(BS={args.batch_size}, T={args.accum_steps}, "
                     f"proj_dim={args.proj_dim})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"compare_{dist}.png"), dpi=110)
        plt.close()

    print(f"\nSaved to {args.save_dir}/")


if __name__ == "__main__":
    main()
