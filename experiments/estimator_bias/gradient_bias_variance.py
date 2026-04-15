"""
Per-sample gradient bias & variance of W1 and Epps-Pulley vs N

Companion to bias_variance_scaling.py — that script studies the
*value* of each estimator; this one studies the *gradient* on a single
"anchor" sample, which is what would actually flow through a network
during training.

Setup
-----
For each (loss, distribution P, anchor value y*, batch size N) we draw
many fresh batches of N-1 samples from P, insert y*, and record the
per-sample gradient at y*:

    g_n^(s) = ∂L / ∂y*  computed on  (x_1^(s), …, x_{N-1}^(s), y*)

Across S seeds we estimate:
    bias_n  = | median_s g_n^(s) − g*  |    (g* ≈ median at the largest N)
    std_n   =   std_s g_n^(s)
    snr_n   = | median_s g_n^(s) | / std_s

Closed-form per-anchor gradients (used directly — no autograd needed)
---------------------------------------------------------------------
For
    L_W1   = (1/n) Σ_j | x_(j) − Phi^{-1}((j − 1/2)/n) |
    L_TEP  = (1/n) Σ_{j,k} exp(−(x_j − x_k)^2/2)
             − √2 Σ_j exp(−x_j^2/4) + n/√3
    L_In   = L_TEP / n

we have

    ∂L_W1/∂y*  = (1/n) · sign( y* − Phi^{-1}( (rank(y*) − 1/2) / n ) )
    ∂L_TEP/∂y* = −(2/n) Σ_k (y* − x_k) exp(−(y* − x_k)^2/2)
                 + (√2/2) y* exp(−y*^2/4)
    ∂L_In/∂y*  = (1/n) · ∂L_TEP/∂y*

(For W1 the gradient flows through torch.sort: only the rank-of-y*
slot contributes.  The k = anchor term in the T_EP sum is 0 because
diff = 0, so we can sum over the N-1 "other" samples only.)

Both forms are O(N) per (seed, anchor, loss), and they vectorise across
seeds with no Python loop, so we can run thousands of replicates in
seconds.

Distributions are imported from bias_variance_scaling.py.

Run
---
    python experiments/estimator_bias/gradient_bias_variance.py
    python experiments/estimator_bias/gradient_bias_variance.py \
        --Ns 32 128 512 2048 --n-seeds 4000 \
        --anchors -1.0 0.0 1.0
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the standardized 1-D distributions from the value script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bias_variance_scaling import DISTRIBUTIONS  # noqa: E402


# ---------------------------------------------------------------------------
# Closed-form anchor gradients (batched over seeds)
# ---------------------------------------------------------------------------

def w1_grad_at_anchor(rest: torch.Tensor, anchor: float) -> torch.Tensor:
    """
    rest:   (S, N-1) tensor — each row is a fresh "context" batch from P.
    anchor: scalar y*.
    Returns (S,) tensor of ∂L_W1/∂y* per replicate.
    """
    n = rest.shape[1] + 1
    rank = (rest < anchor).sum(dim=1) + 1            # 1..n, position of y*
    p = (rank.to(rest.dtype) - 0.5) / n
    target = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
    return torch.sign(anchor - target) / n


def tep_grad_at_anchor(rest: torch.Tensor, anchor: float) -> torch.Tensor:
    """∂L_TEP/∂y*, evaluated row-wise on (S, N-1) batches."""
    n = rest.shape[1] + 1
    diff = anchor - rest                              # (S, N-1)
    s = (diff * torch.exp(-0.5 * diff * diff)).sum(dim=1)   # (S,)
    return (-(2.0 / n) * s
            + (math.sqrt(2.0) / 2.0) * anchor * math.exp(-0.25 * anchor * anchor))


def in_grad_at_anchor(rest: torch.Tensor, anchor: float) -> torch.Tensor:
    """∂L_In/∂y* = (1/N) ∂L_TEP/∂y*."""
    n = rest.shape[1] + 1
    return tep_grad_at_anchor(rest, anchor) / n


GRAD_FNS = {
    "w1":  w1_grad_at_anchor,
    "tep": tep_grad_at_anchor,
    "in":  in_grad_at_anchor,
}
LOSS_LABELS = {
    "w1":  r"$\hat W_1$",
    "tep": r"$T_{\rm EP}$",
    "in":  r"$I_n=T_{\rm EP}/N$",
}


# ---------------------------------------------------------------------------
# Optional autograd cross-check (sanity)
# ---------------------------------------------------------------------------

def _autograd_check(device):
    """Verify the closed-form anchor gradients against torch autograd."""
    rng = np.random.default_rng(0)
    N = 64
    rest_np = rng.standard_normal((1, N - 1)).astype(np.float64)
    rest = torch.from_numpy(rest_np).to(device)

    for anchor_val in (-0.7, 0.0, 1.3):
        # closed form
        g_w1_cf  = w1_grad_at_anchor(rest, anchor_val).item()
        g_tep_cf = tep_grad_at_anchor(rest, anchor_val).item()

        # autograd reference
        anchor = torch.tensor([anchor_val], dtype=torch.float64,
                              device=device, requires_grad=True)
        x = torch.cat([rest[0], anchor])

        # W1
        n = x.shape[0]
        p = (torch.arange(1, n + 1, dtype=x.dtype, device=device) - 0.5) / n
        target = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
        sorted_x, _ = torch.sort(x)
        L_w1 = (sorted_x - target).abs().mean()
        g_w1_auto = torch.autograd.grad(L_w1, anchor, retain_graph=False)[0].item()

        # T_EP
        anchor2 = torch.tensor([anchor_val], dtype=torch.float64,
                               device=device, requires_grad=True)
        x2 = torch.cat([rest[0], anchor2])
        diff2 = x2.unsqueeze(0) - x2.unsqueeze(1)
        L_tep = (torch.exp(-0.5 * diff2 * diff2).sum() / n
                 - math.sqrt(2.0) * torch.exp(-0.25 * x2 * x2).sum()
                 + n / math.sqrt(3.0))
        g_tep_auto = torch.autograd.grad(L_tep, anchor2)[0].item()

        ok_w1  = abs(g_w1_cf  - g_w1_auto)  < 1e-9
        ok_tep = abs(g_tep_cf - g_tep_auto) < 1e-9
        flag   = "OK" if (ok_w1 and ok_tep) else "MISMATCH"
        print(f"  autograd check  y*={anchor_val:+.2f}  "
              f"W1: cf={g_w1_cf:+.6e} auto={g_w1_auto:+.6e}  "
              f"TEP: cf={g_tep_cf:+.6e} auto={g_tep_auto:+.6e}  [{flag}]")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args, device):
    Ns = sorted(args.Ns)
    seed_master = np.random.default_rng(args.master_seed)

    results = {}                       # (loss, dist, anchor, N) -> (S,) np.ndarray
    total_cfg = len(args.distributions) * len(Ns)
    cfg = 0
    t_start = time.perf_counter()

    for dist in args.distributions:
        sampler = DISTRIBUTIONS[dist]
        for N in Ns:
            # Pre-draw the (S, N-1) "rest" batches once and share them
            # across the 3 losses and the anchor list.
            rest_np = np.empty((args.n_seeds, N - 1), dtype=np.float32)
            for s in range(args.n_seeds):
                seed = int(seed_master.integers(0, 2 ** 31 - 1))
                rng = np.random.default_rng(seed)
                rest_np[s] = sampler(N - 1, rng)
            rest = torch.from_numpy(rest_np).to(device)

            for anchor in args.anchors:
                for loss_name, grad_fn in GRAD_FNS.items():
                    g = grad_fn(rest, anchor).cpu().numpy()
                    results[(loss_name, dist, anchor, N)] = g

            cfg += 1
            elapsed = time.perf_counter() - t_start
            # Show one summary line per (dist, N): grad std at the first anchor.
            a0 = args.anchors[0]
            print(f"[{cfg:>3d}/{total_cfg}] {dist:<18} N={N:>5}  "
                  f"std∂W1 @y*={a0:+.1f}={np.std(results[('w1',  dist, a0, N)], ddof=1):.2e}  "
                  f"std∂TEP@y*={a0:+.1f}={np.std(results[('tep', dist, a0, N)], ddof=1):.2e}  "
                  f"({elapsed:5.1f}s)")
    return results, Ns


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _color_cycle(names):
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(names)}


def plot_grad_for_loss(results, Ns, dists, anchors, loss_name, save_path):
    """One figure per loss: rows = anchors, columns = (median grad | std | SNR)."""
    nrow = len(anchors)
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 3.6 * nrow), squeeze=False)
    colors = _color_cycle(dists)

    for ai, anchor in enumerate(anchors):
        ax_med, ax_std, ax_snr = axes[ai]
        for dist in dists:
            arrs = [results[(loss_name, dist, anchor, N)] for N in Ns]
            meds = np.array([np.median(a) for a in arrs])
            q25  = np.array([np.quantile(a, 0.25) for a in arrs])
            q75  = np.array([np.quantile(a, 0.75) for a in arrs])
            stds = np.array([np.std(a, ddof=1) for a in arrs])
            snr  = np.abs(meds) / np.maximum(stds, 1e-30)
            c = colors[dist]
            ax_med.plot(Ns, meds, "o-", color=c, label=dist, linewidth=1.5)
            ax_med.fill_between(Ns, q25, q75, color=c, alpha=0.15)
            ax_std.plot(Ns, stds, "o-", color=c, label=dist, linewidth=1.5)
            ax_snr.plot(Ns, snr,  "o-", color=c, label=dist, linewidth=1.5)

        ax_med.axhline(0, color="k", alpha=0.3, linestyle=":")
        for ax in (ax_med, ax_std, ax_snr):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("N (batch size)")
            ax.grid(alpha=0.3)
        ax_std.set_yscale("log")
        ax_snr.set_yscale("log")

        ax_med.set_ylabel(rf"med $\partial${LOSS_LABELS[loss_name]}$/\partial y^\star$")
        ax_med.set_title(f"median grad   y*={anchor:+.2f}")
        ax_std.set_ylabel("std")
        ax_std.set_title(f"std grad   y*={anchor:+.2f}")
        ax_snr.set_ylabel("|median| / std")
        ax_snr.set_title(f"signal/noise   y*={anchor:+.2f}")

    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Per-sample gradient bias & variance: {LOSS_LABELS[loss_name]}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_grad_summary(results, Ns, dists, anchor, save_path):
    """Side-by-side gradient std vs N for the three losses at one anchor."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = _color_cycle(dists)

    for li, loss_name in enumerate(["w1", "tep", "in"]):
        ax = axes[li]
        for dist in dists:
            stds = np.array([np.std(results[(loss_name, dist, anchor, N)], ddof=1)
                             for N in Ns])
            ax.plot(Ns, stds, "o-", color=colors[dist], label=dist, linewidth=1.5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N (batch size)")
        ax.set_ylabel(r"std $\partial L/\partial y^\star$")
        ax.set_title(f"{LOSS_LABELS[loss_name]}: gradient std")
        ax.grid(alpha=0.3)

    Nmin, Nmax = Ns[0], Ns[-1]
    ref_x = np.array([Nmin, Nmax], dtype=float)
    # Anchor reference lines to the topmost std value at Nmin so the slope
    # comparison is visible regardless of absolute scale.
    def _anchor_line(ax, exponent, label):
        sample_stds = [np.std(results[(loss_name, d, anchor, Nmin)], ddof=1)
                       for d in dists]
        y0 = max(sample_stds)
        ax.plot(ref_x, y0 * (ref_x / Nmin) ** exponent,
                "k--", alpha=0.5, label=label)

    loss_name = "w1";  _anchor_line(axes[0], -1.0, r"$1/N$")
    loss_name = "tep"; _anchor_line(axes[1], -0.5, r"$1/\sqrt{N}$")
    loss_name = "in";  _anchor_line(axes[2], -1.5, r"$1/N^{3/2}$")

    for ax in axes:
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Gradient noise scaling at  y*={anchor:+.2f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-sample gradient bias/variance of W1, T_EP, and T_EP/N "
                    "at fixed anchor points")
    p.add_argument("--distributions", nargs="+",
                   default=list(DISTRIBUTIONS.keys()),
                   choices=list(DISTRIBUTIONS.keys()))
    p.add_argument("--anchors", type=float, nargs="+",
                   default=[-1.5, -0.5, 0.5, 1.5])
    p.add_argument("--Ns", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--n-seeds", type=int, default=2000,
                   help="Replicates per (distribution, N). Closed-form O(N) "
                        "gradients let us crank this up.")
    p.add_argument("--master-seed", type=int, default=0)
    p.add_argument("--save-dir", default="results/estimator_grad_bias")
    p.add_argument("--device", default="auto")
    p.add_argument("--check-autograd", action="store_true",
                   help="Verify closed-form gradients against torch autograd "
                        "and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.check_autograd:
        print("Sanity check: closed form vs torch autograd")
        _autograd_check(device)
        return

    os.makedirs(args.save_dir, exist_ok=True)
    plots_dir = os.path.join(args.save_dir, "plots")
    summary_dir = os.path.join(args.save_dir, "summary")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    print("Per-sample gradient bias/variance scaling")
    print(f"  distributions = {args.distributions}")
    print(f"  anchors       = {args.anchors}")
    print(f"  Ns            = {sorted(args.Ns)}")
    print(f"  n_seeds       = {args.n_seeds}")
    print(f"  device        = {device}")
    print()
    _autograd_check(device)
    print()

    results, Ns = run_sweep(args, device)

    # ---- per-loss plots ----
    for loss_name in GRAD_FNS:
        plot_grad_for_loss(results, Ns, args.distributions, args.anchors,
                           loss_name,
                           os.path.join(plots_dir, f"grad_{loss_name}.png"))

    # ---- cross-loss summary at each anchor ----
    for anchor in args.anchors:
        tag = f"{anchor:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        plot_grad_summary(results, Ns, args.distributions, anchor,
                          os.path.join(plots_dir, f"summary_y{tag}.png"))

    # ---- summary ----
    summary_json = {
        "config": {
            "distributions": args.distributions,
            "anchors": args.anchors,
            "Ns": Ns,
            "n_seeds": args.n_seeds,
            "master_seed": args.master_seed,
        },
        "stats": {},
    }
    lines = [
        "Per-sample gradient bias/variance at fixed anchor points",
        f"distributions={args.distributions}",
        f"anchors={args.anchors}, Ns={Ns}, n_seeds={args.n_seeds}",
        "Aggregation: median over seeds (IQR / std).",
        "",
        f"{'loss':<5} {'distribution':<18} {'y*':>6} {'N':>6}  "
        f"{'med grad':>14} {'std grad':>14} {'|med|/std':>10}",
        "-" * 86,
    ]
    for loss_name in GRAD_FNS:
        for dist in args.distributions:
            for anchor in args.anchors:
                for N in Ns:
                    g = results[(loss_name, dist, anchor, N)]
                    med = float(np.median(g))
                    std = float(np.std(g, ddof=1))
                    snr = abs(med) / max(std, 1e-30)
                    summary_json["stats"][
                        f"{loss_name}_{dist}_y{anchor:+.2f}_N{N}"] = {
                        "median": med, "std": std, "snr": snr,
                    }
                    lines.append(
                        f"{loss_name:<5} {dist:<18} {anchor:>+6.2f} {N:>6}  "
                        f"{med:>+14.4e} {std:>14.4e} {snr:>10.3f}")
            lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(summary_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(summary_dir, "metrics.json"), "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved to {args.save_dir}/")


if __name__ == "__main__":
    main()
