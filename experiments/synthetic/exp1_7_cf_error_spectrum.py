"""
Experiment 1.7: CF Error Spectrum — Where Does W1 Win vs SIGReg?

Both losses regularize toward N(0, I) via 1D projections (Cramer-Wold),
but use different 1D metrics: W1 matches quantiles, SIGReg matches the
characteristic function weighted by exp(-t^2/2).

This script trains both losses to convergence, then measures the CF
error |phi_hat(t) - phi_N(t)|^2 as a function of frequency t on the
converged outputs (averaged over 1D projections).

SIGReg's Gaussian weight exp(-t^2/2) makes it a low-pass filter:
strong at low frequencies (moments, overall shape), blind to high
frequencies (individual sample placement). W1 treats all quantile
levels equally, so it captures all frequencies.

Expected result:
    - SIGReg has lower CF error at low t (what it explicitly optimizes)
    - W1 has lower CF error at high t (fine structure SIGReg can't see)
    - There is a crossover frequency t* where W1 starts winning

Also plots the SIGReg weighting exp(-t^2/2) alongside the error curves
so you can see exactly which frequencies SIGReg cares about.

Run:
    python experiments/synthetic/exp1_7_cf_error_spectrum.py
    python experiments/synthetic/exp1_7_cf_error_spectrum.py --K 8192 --steps 8000
"""

import os
import sys
import time
import json
import argparse

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNTHETIC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SYNTHETIC_DIR)

from sliced_gauss_reg import (
    SlicedW1Loss, SIGRegLoss,
    DeepMLP, generate_data, make_fixed_projection, GENERATORS,
    eval_w1,
)


# ---------------------------------------------------------------------------
# Loss factory & training
# ---------------------------------------------------------------------------

class _SumLoss(torch.nn.Module):
    """Sums the outputs of two losses with equal weight (sigreg+w1).

    SIGReg sits at ~1–2 (biased plug-in floor) and W1 at ~0.05–0.2, so the
    sum is dominated by SIGReg by ~5–10×. The point of measuring this here
    is the spectrum, not the absolute value.
    """

    def __init__(self, *losses):
        super().__init__()
        self.losses = torch.nn.ModuleList(losses)

    def forward(self, x):
        return sum(L(x) for L in self.losses)


def make_loss_fn(mode, num_proj, knots, device):
    if mode == "w1":
        return SlicedW1Loss(num_proj=num_proj).to(device)
    elif mode == "sigreg":
        return SIGRegLoss(knots=knots, num_proj=num_proj,
                          bias_mode="biased").to(device)
    elif mode == "sigreg+w1":
        return _SumLoss(
            SlicedW1Loss(num_proj=num_proj).to(device),
            SIGRegLoss(knots=knots, num_proj=num_proj,
                       bias_mode="biased").to(device),
        )
    raise ValueError(mode)


def train(mlp, projected, loss_fn, steps, lr, eval_interval):
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    eval_steps, eval_w1s = [], []
    for step in range(1, steps + 1):
        mlp.train()
        loss = loss_fn(mlp(projected))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_interval == 0:
            mlp.eval()
            with torch.no_grad():
                eval_steps.append(step)
                eval_w1s.append(eval_w1(mlp(projected).cpu().numpy()))
    return eval_steps, eval_w1s


# ---------------------------------------------------------------------------
# CF error spectrum
# ---------------------------------------------------------------------------

def cf_error_spectrum(output_np, num_proj=1024, t_max=8.0, n_freq=200):
    """Measure |phi_hat(t) - phi_N(t)|^2 vs frequency t.

    Averages over num_proj random 1D projections using a fixed seed
    (same as eval_w1) for reproducibility.

    Returns:
        freqs: (n_freq,) array of frequencies.
        err_sq: (n_freq,) array of mean squared CF error.
    """
    x = torch.tensor(output_np, dtype=torch.float32)
    K, D = x.shape
    torch.manual_seed(999)
    A = torch.randn(D, num_proj)
    A = A / A.norm(dim=0, keepdim=True)
    proj = x @ A                                    # (K, num_proj)

    freqs = torch.linspace(0, t_max, n_freq)
    phi_target = torch.exp(-0.5 * freqs * freqs)    # N(0,1) CF

    err_sq_sum = torch.zeros(n_freq)
    chunk = 64
    for start in range(0, num_proj, chunk):
        end = min(start + chunk, num_proj)
        p = proj[:, start:end]                      # (K, c)
        x_t = p.unsqueeze(2) * freqs                # (K, c, F)
        c_bar = x_t.cos().mean(dim=0)               # (c, F)
        s_bar = x_t.sin().mean(dim=0)
        err = (c_bar - phi_target).square() + s_bar.square()
        err_sq_sum += err.sum(dim=0)

    return freqs.numpy(), (err_sq_sum / num_proj).numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"w1": "tab:blue", "sigreg": "tab:orange", "sigreg+w1": "tab:red"}
LABELS = {"w1": "Sliced W1", "sigreg": "SIGReg", "sigreg+w1": "SIGReg + W1"}


def plot_spectrum(spectra, save_path):
    """Two-panel plot: CF error spectrum + SIGReg weight overlay."""
    fig, (ax_raw, ax_weighted) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: raw CF error vs frequency ---
    for lm, (freqs, err_sq) in spectra.items():
        ax_raw.semilogy(freqs, err_sq, color=COLORS[lm],
                        linewidth=2, label=LABELS[lm])

    # Mark the SIGReg quadrature range [0, 3] and its weight
    ax_raw.axvspan(0, 3, alpha=0.07, color="orange")
    ax_raw.annotate("SIGReg\nquadrature\nrange", xy=(1.5, ax_raw.get_ylim()[0]),
                    fontsize=9, ha="center", va="bottom", color="tab:orange",
                    alpha=0.7)
    ax_raw.set_xlabel("frequency  t")
    ax_raw.set_ylabel(r"$|\hat\varphi(t) - \varphi_N(t)|^2$")
    ax_raw.set_title("CF error spectrum at convergence")
    ax_raw.legend(fontsize=11)
    ax_raw.grid(alpha=0.3)

    # --- Panel 2: error * SIGReg weight (what SIGReg actually sees) ---
    for lm, (freqs, err_sq) in spectra.items():
        w = np.exp(-0.5 * freqs * freqs)
        ax_weighted.semilogy(freqs, err_sq * w, color=COLORS[lm],
                             linewidth=2, label=LABELS[lm])
    # Also show the weight function itself
    f = spectra[list(spectra.keys())[0]][0]
    w = np.exp(-0.5 * f * f)
    ax2 = ax_weighted.twinx()
    ax2.plot(f, w, "k--", alpha=0.3, linewidth=1)
    ax2.set_ylabel(r"$e^{-t^2/2}$  (SIGReg weight)", alpha=0.4)
    ax2.set_ylim(0, 1.1)

    ax_weighted.set_xlabel("frequency  t")
    ax_weighted.set_ylabel(r"$|\hat\varphi - \varphi_N|^2 \cdot e^{-t^2/2}$")
    ax_weighted.set_title("What SIGReg sees (error × its weight)")
    ax_weighted.legend(fontsize=11)
    ax_weighted.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_convergence(all_curves, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for lm, curves in all_curves.items():
        all_w1 = np.array([w for _, w in curves.values()])
        steps = list(curves.values())[0][0]
        med = np.median(all_w1, axis=0)
        for _, w1s in curves.values():
            ax.plot(steps, w1s, color=COLORS[lm], alpha=0.3, lw=1)
        ax.plot(steps, med, color=COLORS[lm], linewidth=2.5,
                label=f"{LABELS[lm]}  (floor={med[-5:].mean():.4f})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Sliced W1 to N(0, 1)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Full-batch convergence")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 1.7: CF error spectrum — where W1 wins vs SIGReg")
    p.add_argument("--distribution", default="blobs",
                   choices=list(GENERATORS.keys()))
    p.add_argument("--input-dim", type=int, default=4)
    p.add_argument("--proj-dim", type=int, default=8)
    p.add_argument("--K", type=int, default=8192)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--t-max", type=float, default=8.0)
    p.add_argument("--n-freq", type=int, default=200)
    p.add_argument("--save-dir", default="results/exp1_7_cf_error_spectrum")
    p.add_argument("--device", default="auto")
    return p.parse_args()


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

    loss_modes = ["w1", "sigreg", "sigreg+w1"]
    seeds = [42 + i for i in range(args.n_seeds)]

    print(f"Exp 1.7: CF error spectrum")
    print(f"  distribution={args.distribution}, K={args.K}, steps={args.steps}")
    print(f"  seeds={seeds}, device={device}")
    print()

    all_curves = {lm: {} for lm in loss_modes}
    all_spectra = {lm: [] for lm in loss_modes}

    for seed in seeds:
        torch.manual_seed(seed)
        data = generate_data(args.distribution, args.K, args.input_dim)
        W = make_fixed_projection(args.input_dim, args.proj_dim,
                                  seed=seed + 100).to(device)
        projected = data.to(device) @ W

        for lm in loss_modes:
            loss_fn = make_loss_fn(lm, args.num_proj, args.knots, device)
            torch.manual_seed(seed + 200)
            mlp = DeepMLP(args.proj_dim, args.hidden_dim, args.input_dim,
                          depth=args.depth).to(device)

            print(f"  [{lm}] seed={seed} training ...", end=" ", flush=True)
            t0 = time.perf_counter()
            eval_steps, eval_w1s = train(
                mlp, projected, loss_fn, args.steps, args.lr,
                args.eval_interval)
            elapsed = time.perf_counter() - t0
            floor = float(np.median(eval_w1s[-max(1, len(eval_w1s) // 10):]))
            print(f"W1_floor={floor:.4f}  ({elapsed:.1f}s)")
            all_curves[lm][seed] = (eval_steps, eval_w1s)

            mlp.eval()
            with torch.no_grad():
                out_np = mlp(projected).cpu().numpy()
            freqs, err_sq = cf_error_spectrum(
                out_np, num_proj=args.num_proj,
                t_max=args.t_max, n_freq=args.n_freq)
            all_spectra[lm].append(err_sq)
            del mlp, loss_fn

    # Median spectrum across seeds
    spectra = {}
    for lm in loss_modes:
        spectra[lm] = (freqs, np.median(np.stack(all_spectra[lm]), axis=0))

    # --- Plots ---
    plot_spectrum(spectra, os.path.join(plots_dir, "cf_error_spectrum.png"))
    plot_convergence(all_curves, os.path.join(plots_dir, "convergence.png"))

    # --- Find crossover frequency ---
    # Skip t=0 neighbourhood (both errors are ~0 there, noisy ratio)
    f_w1, err_w1 = spectra["w1"]
    f_sg, err_sg = spectra["sigreg"]
    diff = err_w1 - err_sg  # positive where SIGReg is better
    # Look for the first crossing where diff goes from positive to negative
    # (i.e. W1 starts winning), skipping the first few points near t=0
    start_idx = max(1, int(0.5 / (f_w1[1] - f_w1[0])))  # skip t < 0.5
    crossings = np.where(np.diff(np.sign(diff[start_idx:])))[0]
    if len(crossings) > 0:
        crossover_t = float(f_w1[start_idx + crossings[0]])
    else:
        crossover_t = float("nan")

    # --- Summary ---
    lines = [
        "Exp 1.7: CF Error Spectrum — W1 vs SIGReg",
        f"K={args.K}, steps={args.steps}, seeds={len(seeds)}",
        "",
        f"Crossover frequency t* ~ {crossover_t:.2f}",
        f"  t < t*: SIGReg has lower CF error (low-frequency structure)",
        f"  t > t*: W1 has lower CF error (high-frequency / fine placement)",
        f"  SIGReg weight exp(-t*^2/2) = {np.exp(-0.5*crossover_t**2):.4f}",
        "",
        "W1 evaluation floors:",
    ]
    for lm in loss_modes:
        floors = []
        for seed, (steps, w1s) in all_curves[lm].items():
            floors.append(float(np.median(w1s[-max(1, len(w1s) // 10):])))
        lines.append(f"  {LABELS[lm]:<12} {np.median(floors):.4f}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(summary_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(summary_dir, "metrics.json"), "w") as f:
        json.dump({"crossover_t": crossover_t,
                    "sigreg_weight_at_crossover": float(np.exp(-0.5 * crossover_t**2))},
                  f, indent=2)
    print(f"\nSaved to {args.save_dir}/")


if __name__ == "__main__":
    main()
