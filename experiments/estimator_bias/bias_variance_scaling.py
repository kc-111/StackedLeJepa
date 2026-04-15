"""
Finite-sample bias & variance of two normality estimators vs N

For samples X_n drawn from a 1-D distribution P, this script estimates two
quantities that measure how far P is from N(0, 1):

  1) W1(F_n, Phi)   — the 1-D Wasserstein-1 distance to N(0, 1).
  2) T_EP(X_n)       — the Epps-Pulley test statistic for normality.

Both are computed for many sample sizes N and many seeds, so we can
quantify how the *bias* med(S_n) - S* and the *spread* (std/IQR) scale
with N for several base distributions.

Definitions
-----------
1-D Wasserstein-1 to N(0, 1) (closed form via sorted samples):

    W1_hat(x) = (1/n) * sum_j | x_(j) - Phi^{-1}((j - 1/2)/n) |

  • Reference S*(P) = W1(P, N(0,1)).  For P = N(0,1) this is 0; for the
    other distributions we estimate it once with a Monte-Carlo run at
    N_ref samples.  Even at large N_ref this is itself slightly biased,
    so we call it the "reference" rather than the "true" value.

Epps-Pulley statistic (Epps & Pulley, Biometrika 1983):

    T_EP(x) = (1/n) sum_{j,k} exp( -(x_j - x_k)^2 / 2 )
              - sqrt(2)   sum_j exp( -x_j^2 / 4 )
              + n / sqrt(3)

  • Equivalent to  n * integral |phi_hat_n(t) - phi_N(t)|^2 *
    exp(-t^2/2)/sqrt(2*pi) dt, the L2 distance between empirical and
    target characteristic functions weighted by a Gaussian kernel.
  • Under N(0, 1) it has a non-degenerate limit distribution and does
    NOT converge to 0; under H_1 it grows linearly with n.
  • We also report the per-sample integral I_n := T_EP / n, which is
    the actual L2 CF distance and converges to 0 like 1/n under H_0.

Distributions
-------------
All standardized to mean 0 and variance 1 so the deviation from N(0,1)
is purely a shape difference:

    standard_normal       N(0, 1)                       — H_0
    shifted_normal        N(0.5, 1)                     — wrong mean
    scaled_normal         N(0, 1.5^2)                   — wrong variance
    laplace               Laplace, var=1                — heavier tails
    student_t5            Student-t(5), var=1           — heavy tails
    uniform               Uniform, var=1                — light tails
    mixture_gauss         (1/2) N(-1, 1/4) + (1/2) N(1, 1/4)  — bimodal
    skewnormal            alpha = 4, std-normalized     — skew

Run
---
    python experiments/estimator_bias/bias_variance_scaling.py
    python experiments/estimator_bias/bias_variance_scaling.py \
        --Ns 32 64 128 256 1024 4096 --n-seeds 256
"""

import os
import json
import math
import time
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1-D distributions, all standardized to mean 0 / variance 1
# ---------------------------------------------------------------------------

def sample_standard_normal(n, rng):
    return rng.standard_normal(n)

def sample_shifted_normal(n, rng):
    return rng.standard_normal(n) + 0.5

def sample_scaled_normal(n, rng):
    return 1.5 * rng.standard_normal(n)

def sample_laplace(n, rng):
    # Laplace(0, b) has variance 2 b^2 -> b = 1/sqrt(2) for unit variance.
    return rng.laplace(0.0, 1.0 / math.sqrt(2.0), n)

def sample_student_t5(n, rng):
    # Student-t(nu) has variance nu/(nu-2) -> divide by sqrt(5/3) for unit var.
    df = 5
    return rng.standard_t(df, n) / math.sqrt(df / (df - 2.0))

def sample_uniform(n, rng):
    # Uniform(-a, a) has variance a^2/3 -> a = sqrt(3) for unit variance.
    return rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), n)

def sample_mixture_gauss(n, rng):
    # 0.5 N(-1, 0.25) + 0.5 N(1, 0.25); mean=0, var = 0.25 + 1 = 1.25.
    z = rng.standard_normal(n) * 0.5
    s = rng.choice([-1.0, 1.0], n)
    return (z + s) / math.sqrt(1.25)

def sample_skewnormal(n, rng, alpha=4.0):
    # Azzalini construction: X = delta |U0| + sqrt(1 - delta^2) U1.
    # Then standardize using the closed-form mean/variance.
    delta = alpha / math.sqrt(1.0 + alpha * alpha)
    u0 = rng.standard_normal(n)
    u1 = rng.standard_normal(n)
    x = delta * np.abs(u0) + math.sqrt(1.0 - delta * delta) * u1
    mu = math.sqrt(2.0 / math.pi) * delta
    var = 1.0 - 2.0 * delta * delta / math.pi
    return (x - mu) / math.sqrt(var)


DISTRIBUTIONS = {
    "standard_normal": sample_standard_normal,
    "shifted_normal":  sample_shifted_normal,
    "scaled_normal":   sample_scaled_normal,
    "laplace":         sample_laplace,
    "student_t5":      sample_student_t5,
    "uniform":         sample_uniform,
    "mixture_gauss":   sample_mixture_gauss,
    "skewnormal":      sample_skewnormal,
}


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def w1_to_standard_normal(x: np.ndarray) -> float:
    """Closed-form 1-D W_1(F_n, N(0,1)) using sorted samples and inverse CDF.

        W1_hat = (1/n) sum_j | x_(j) - Phi^{-1}((j - 1/2)/n) |
    """
    n = x.size
    p = (np.arange(1, n + 1) - 0.5) / n
    # torch.erfinv accepts a tensor; convert via numpy <-> torch.
    ref = math.sqrt(2.0) * torch.erfinv(
        torch.from_numpy(2.0 * p - 1.0)).numpy()
    return float(np.mean(np.abs(np.sort(x) - ref)))


def epps_pulley_torch(x: torch.Tensor, chunk: int = 2048) -> float:
    """Epps-Pulley statistic, computed with chunked O(n^2) sums.

    The double sum is split into row blocks of size `chunk` to keep peak
    memory at O(chunk * n) instead of O(n^2).
    """
    n = x.shape[0]
    s1 = torch.zeros((), device=x.device, dtype=x.dtype)
    for i in range(0, n, chunk):
        xi = x[i:i + chunk]                          # (c,)
        diff = xi.unsqueeze(1) - x.unsqueeze(0)      # (c, n)
        s1 = s1 + torch.exp(-0.5 * diff * diff).sum()
    s1 = s1 / n
    s2 = math.sqrt(2.0) * torch.exp(-0.25 * x * x).sum()
    return float(s1 - s2 + n / math.sqrt(3.0))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def estimate_w1_reference(sampler, n_ref, n_avg, base_seed=999):
    """Monte-Carlo reference for W_1(P, N(0, 1)).

    Each replicate draws n_ref samples and computes the closed-form W_1
    estimator; we return the median across replicates.  This is itself
    biased (the W_1 estimator has bias O(N^{-1/2})) but at n_ref ~ 1e5
    the bias is small relative to the differences between distributions.
    """
    vals = []
    for r in range(n_avg):
        rng = np.random.default_rng(base_seed + r)
        x = sampler(n_ref, rng)
        vals.append(w1_to_standard_normal(x))
    return float(np.median(vals))


def run_sweep(args, device):
    Ns = sorted(args.Ns)
    seed_master = np.random.default_rng(args.master_seed)

    # 1) Reference W_1 for each distribution.
    print("Estimating reference W_1 values via Monte Carlo ...")
    refs_w1 = {}
    for name in args.distributions:
        if name == "standard_normal":
            refs_w1[name] = 0.0
            print(f"  {name:<18} W1* = 0.000000  (analytic, H_0)")
            continue
        ref = estimate_w1_reference(
            DISTRIBUTIONS[name], n_ref=args.n_ref, n_avg=args.n_ref_avg)
        refs_w1[name] = ref
        print(f"  {name:<18} W1* ~ {ref:.6f}  "
              f"(MC: {args.n_ref_avg} x {args.n_ref})")
    print()

    # 2) Sweep N x distribution x seed.
    results = {}
    total = len(args.distributions) * len(Ns) * args.n_seeds
    counter = 0
    t_start = time.perf_counter()
    for name in args.distributions:
        sampler = DISTRIBUTIONS[name]
        for N in Ns:
            w1_arr = np.empty(args.n_seeds)
            tep_arr = np.full(args.n_seeds, np.nan)
            for s in range(args.n_seeds):
                seed = int(seed_master.integers(0, 2 ** 31 - 1))
                rng = np.random.default_rng(seed)
                x_np = sampler(N, rng).astype(np.float64)
                w1_arr[s] = w1_to_standard_normal(x_np)
                if N <= args.epps_max_n:
                    x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
                    tep_arr[s] = epps_pulley_torch(x_t)
                counter += 1
            elapsed = time.perf_counter() - t_start
            tep_str = (f"{np.nanmedian(tep_arr):+9.3f}"
                       if N <= args.epps_max_n else "    skip")
            print(f"[{counter:>5d}/{total}] {name:<18} N={N:>6}  "
                  f"W1 med={np.median(w1_arr):.4f}  "
                  f"T_EP med={tep_str}  ({elapsed:5.1f}s)")
            results[(name, N)] = {"w1": w1_arr, "tep": tep_arr}

    return results, refs_w1, Ns


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _color_cycle(names):
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(names)}


def plot_w1(results, refs_w1, Ns, dists, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_val, ax_bias, ax_std = axes
    colors = _color_cycle(dists)

    for name in dists:
        meds = np.array([np.median(results[(name, N)]["w1"]) for N in Ns])
        q25  = np.array([np.quantile(results[(name, N)]["w1"], 0.25) for N in Ns])
        q75  = np.array([np.quantile(results[(name, N)]["w1"], 0.75) for N in Ns])
        stds = np.array([np.std(results[(name, N)]["w1"], ddof=1) for N in Ns])
        ref  = refs_w1[name]
        c = colors[name]

        ax_val.plot(Ns, meds, "o-", color=c, label=name, linewidth=1.6)
        ax_val.fill_between(Ns, q25, q75, color=c, alpha=0.15)
        ax_val.axhline(ref, color=c, linestyle=":", alpha=0.6)

        ax_bias.plot(Ns, np.abs(meds - ref), "o-", color=c,
                     label=name, linewidth=1.6)
        ax_std.plot(Ns, stds, "o-", color=c, label=name, linewidth=1.6)

    # 1/sqrt(N) reference line
    Nmin, Nmax = Ns[0], Ns[-1]
    ref_x = np.array([Nmin, Nmax], dtype=float)
    ref_y = 1.0 / np.sqrt(ref_x)
    for ax in (ax_bias, ax_std):
        ax.plot(ref_x, ref_y, "k--", alpha=0.5, label=r"$1/\sqrt{N}$")

    ax_val.set_title(r"$\hat W_1(F_n,\,\mathcal{N}(0,1))$  vs $N$")
    ax_val.set_xlabel("N (samples)")
    ax_val.set_ylabel("median (IQR band) — dotted = reference")

    ax_bias.set_title(r"$|\,\mathrm{med}\,\hat W_1 - W_1^\star|$  (bias)")
    ax_bias.set_xlabel("N (samples)")
    ax_bias.set_ylabel("absolute bias")
    ax_bias.set_yscale("log")

    ax_std.set_title(r"$\mathrm{std}(\hat W_1)$  across seeds")
    ax_std.set_xlabel("N (samples)")
    ax_std.set_ylabel("std")
    ax_std.set_yscale("log")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    ax_val.legend(fontsize=8, ncol=2)
    ax_bias.legend(fontsize=8)
    fig.suptitle("1-D Wasserstein-1 estimator: bias & variance vs N",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_epps_pulley(results, Ns, dists, epps_max_n, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_val, ax_int, ax_std = axes
    colors = _color_cycle(dists)

    valid_Ns = [N for N in Ns if N <= epps_max_n]
    if not valid_Ns:
        plt.close(fig)
        return

    for name in dists:
        meds = np.array([np.nanmedian(results[(name, N)]["tep"]) for N in valid_Ns])
        q25  = np.array([np.nanquantile(results[(name, N)]["tep"], 0.25) for N in valid_Ns])
        q75  = np.array([np.nanquantile(results[(name, N)]["tep"], 0.75) for N in valid_Ns])
        stds = np.array([np.nanstd(results[(name, N)]["tep"], ddof=1) for N in valid_Ns])
        per_sample = meds / np.array(valid_Ns, dtype=float)
        c = colors[name]

        ax_val.plot(valid_Ns, meds, "o-", color=c, label=name, linewidth=1.6)
        ax_val.fill_between(valid_Ns, q25, q75, color=c, alpha=0.15)

        ax_int.plot(valid_Ns, per_sample, "o-", color=c,
                    label=name, linewidth=1.6)

        ax_std.plot(valid_Ns, stds, "o-", color=c, label=name, linewidth=1.6)

    # Reference 1/N line on the per-sample integral plot.
    ref_x = np.array([valid_Ns[0], valid_Ns[-1]], dtype=float)
    ax_int.plot(ref_x, 1.0 / ref_x, "k--", alpha=0.5, label=r"$1/N$")

    ax_val.set_title(r"$T_{\rm EP}$  vs $N$")
    ax_val.set_xlabel("N (samples)")
    ax_val.set_ylabel("median (IQR band)")

    ax_int.set_title(r"$\hat I_n = T_{\rm EP}/N$  (per-sample CF L$^2$)")
    ax_int.set_xlabel("N (samples)")
    ax_int.set_ylabel(r"$\hat I_n$")
    ax_int.set_yscale("symlog", linthresh=1e-3)

    ax_std.set_title(r"$\mathrm{std}(T_{\rm EP})$  across seeds")
    ax_std.set_xlabel("N (samples)")
    ax_std.set_ylabel("std")
    ax_std.set_yscale("log")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    ax_val.legend(fontsize=8, ncol=2)
    ax_int.legend(fontsize=8)
    fig.suptitle("Epps-Pulley statistic: scaling with N",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Finite-sample bias/variance of W1 and Epps-Pulley vs N"
    )
    p.add_argument("--distributions", nargs="+",
                   default=list(DISTRIBUTIONS.keys()),
                   choices=list(DISTRIBUTIONS.keys()))
    p.add_argument("--Ns", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--n-seeds", type=int, default=100,
                   help="Replicates per (distribution, N).")
    p.add_argument("--n-ref", type=int, default=200_000,
                   help="N for the Monte Carlo W1(P, N(0,1)) reference.")
    p.add_argument("--n-ref-avg", type=int, default=8,
                   help="Replicates of the reference estimate to median.")
    p.add_argument("--epps-max-n", type=int, default=4096,
                   help="Skip the O(n^2) Epps-Pulley sum above this N.")
    p.add_argument("--master-seed", type=int, default=0)
    p.add_argument("--save-dir", default="results/estimator_bias")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    plots_dir = os.path.join(args.save_dir, "plots")
    summary_dir = os.path.join(args.save_dir, "summary")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    print("Estimator bias/variance scaling")
    print(f"  distributions = {args.distributions}")
    print(f"  Ns            = {sorted(args.Ns)}")
    print(f"  n_seeds       = {args.n_seeds}")
    print(f"  device        = {device}")
    print()

    results, refs_w1, Ns = run_sweep(args, device)

    # ---- plots ----
    plot_w1(results, refs_w1, Ns, args.distributions,
            os.path.join(plots_dir, "w1_bias_variance.png"))
    plot_epps_pulley(results, Ns, args.distributions, args.epps_max_n,
                     os.path.join(plots_dir, "epps_pulley_bias_variance.png"))

    # ---- summary tables ----
    summary_json = {
        "config": {
            "distributions": args.distributions,
            "Ns": Ns,
            "n_seeds": args.n_seeds,
            "n_ref": args.n_ref,
            "n_ref_avg": args.n_ref_avg,
            "epps_max_n": args.epps_max_n,
            "master_seed": args.master_seed,
        },
        "ref_w1": refs_w1,
        "stats": {},
    }
    lines = [
        "Finite-sample bias/variance of W1 and Epps-Pulley estimators",
        f"distributions={args.distributions}",
        f"Ns={Ns}, n_seeds={args.n_seeds}",
        "Aggregation: median over seeds (IQR band on plots).",
        "",
        f"{'distribution':<18} {'N':>6}  "
        f"{'W1 med':>10} {'W1 std':>10} {'W1 ref':>10} {'W1 bias':>11}  "
        f"{'TEP med':>10} {'TEP std':>10}",
        "-" * 102,
    ]
    for name in args.distributions:
        for N in Ns:
            w1 = results[(name, N)]["w1"]
            tep = results[(name, N)]["tep"]
            w1_med = float(np.median(w1))
            w1_std = float(np.std(w1, ddof=1))
            ref = refs_w1[name]
            bias = w1_med - ref
            tep_med = float(np.nanmedian(tep))
            tep_std = float(np.nanstd(tep, ddof=1))
            summary_json["stats"][f"{name}_N{N}"] = {
                "w1_median": w1_med, "w1_std": w1_std, "w1_ref": ref,
                "w1_bias": bias,
                "tep_median": tep_med, "tep_std": tep_std,
            }
            lines.append(
                f"{name:<18} {N:>6}  "
                f"{w1_med:>10.4f} {w1_std:>10.4f} {ref:>10.4f} "
                f"{bias:>+11.4f}  "
                f"{tep_med:>+10.3f} {tep_std:>10.3f}")
        lines.append("")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(summary_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(summary_dir, "metrics.json"), "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved to {args.save_dir}/")


if __name__ == "__main__":
    main()
