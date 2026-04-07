"""
Aggregate plots across all synthetic experiments.

Reads metrics.json / timing_results.json and generates clean figures
aggregated across distributions and seeds with IQR error bars.

Run:
    python experiments/synthetic/aggregate_plots.py
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KNOWN_DISTS = ["blobs", "diagonal_cross", "uniform_square", "ring", "spiral"]

# Global plot style
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": False,
})
SUBPLOT_SIZE = (4, 4)  # inches per subplot


def load_json(results_root, exp_name, filename="metrics.json"):
    for subdir in ["summary", ""]:
        path = os.path.join(results_root, exp_name, subdir, filename)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def parse_key(key):
    for dist in sorted(KNOWN_DISTS, key=len, reverse=True):
        if key.startswith(dist + "_"):
            return dist, key[len(dist) + 1:]
    return None, key


def median_iqr(values):
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return np.nan, np.nan, np.nan
    return np.median(values), np.percentile(values, 25), np.percentile(values, 75)


def iqr_yerr(meds, q25s, q75s):
    lo = [m - q if not np.isnan(q) else 0 for m, q in zip(meds, q25s)]
    hi = [q - m if not np.isnan(q) else 0 for m, q in zip(meds, q75s)]
    return [lo, hi]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_exp1_1(data):
    r = defaultdict(list)
    for key, val in data.items():
        dist, rem = parse_key(key)
        if dist is None or "w1" not in val:
            continue
        m_pos = rem.rfind("_M")
        r[(rem[:m_pos], int(rem[m_pos + 2:]))].append(val["w1"]["mean"])
    return r


def parse_exp1_2(data):
    r = defaultdict(list)
    for key, val in data.items():
        dist, rem = parse_key(key)
        if dist is None or "w1" not in val:
            continue
        t_pos = rem.rfind("_T")
        m_pos = rem.rfind("_M", 0, t_pos)
        if t_pos < 0 or m_pos < 0:
            continue
        method = rem[:m_pos]
        M, T = int(rem[m_pos + 2:t_pos]), int(rem[t_pos + 2:])
        for suf in ["_standard", "_pooled"]:
            if method.endswith(suf):
                r[(method[:-len(suf)], suf[1:], M, T)].append(val["w1"]["mean"])
                break
    return r


def parse_exp1_3(data):
    r = defaultdict(list)
    for key, val in data.items():
        dist, rem = parse_key(key)
        if dist is None or "w1" not in val:
            continue
        d_pos = rem.rfind("_D")
        if d_pos < 0:
            continue
        r[(rem[:d_pos], int(rem[d_pos + 2:]))].append(val["w1"]["mean"])
    return r


def parse_exp1_5(data):
    """Collect W1 across distributions for each (T_cur, T_fifo)."""
    r = defaultdict(list)
    for key, val in data.items():
        dist, rem = parse_key(key)
        if dist is None or "w1" not in val:
            continue
        T_cur = val.get("T_cur")
        T_fifo = val.get("T_fifo")
        if T_cur is not None and T_fifo is not None:
            r[(T_cur, T_fifo)].append(val["w1"]["mean"])
    return r


# ---------------------------------------------------------------------------
# Exp 1.1 + 1.2: Accumulation with full-batch oracle
# ---------------------------------------------------------------------------

def plot_accumulation(exp1_1_data, exp1_2_data, out_dir):
    if exp1_2_data is None:
        print("  accumulation: no data")
        return

    r2 = parse_exp1_2(exp1_2_data)
    r1 = parse_exp1_1(exp1_1_data) if exp1_1_data else {}

    losses = sorted(set(k[0] for k in r2))
    Ms = sorted(set(k[2] for k in r2))
    Ts = sorted(set(k[3] for k in r2))

    colors = {
        "w1_standard": "tab:blue", "w1_pooled": "dodgerblue",
        "w2_standard": "tab:green", "w2_pooled": "limegreen",
        "sigreg_standard": "tab:orange", "sigreg_pooled": "gold",
    }
    labels = {"w1": "W1", "w2": "W2", "sigreg": "SIGReg"}

    for M in Ms:
        fig, ax = plt.subplots(figsize=SUBPLOT_SIZE)

        for loss in losses:
            for mode in ["standard", "pooled"]:
                meds, q25s, q75s = [], [], []
                for T in Ts:
                    med, q25, q75 = median_iqr(r2.get((loss, mode, M, T), []))
                    meds.append(med); q25s.append(q25); q75s.append(q75)
                ls = "--" if mode == "standard" else "-"
                lw = 1.2 if mode == "standard" else 2
                ax.errorbar(Ts, meds, yerr=iqr_yerr(meds, q25s, q75s),
                            color=colors[f"{loss}_{mode}"], linestyle=ls,
                            linewidth=lw, capsize=2, label=f"{labels[loss]} {mode}")

        # Full-batch oracle lines
        if r1:
            for loss in losses:
                vals = r1.get((loss, M), [])
                if vals:
                    med, q25, q75 = median_iqr(vals)
                    ax.axhline(med, color=colors[f"{loss}_pooled"],
                               linestyle=":", alpha=0.5, linewidth=1)

        ax.set_xlabel("Accumulation steps $T$")
        ax.set_ylabel("W1 to $\\mathcal{N}(0, I)$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts])
        ax.legend(ncol=2, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"accumulation_M{M}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Advantage plot
    for M in Ms:
        fig, ax = plt.subplots(figsize=SUBPLOT_SIZE)
        for loss in losses:
            meds, q25s, q75s = [], [], []
            for T in Ts:
                sv = r2.get((loss, "standard", M, T), [])
                pv = r2.get((loss, "pooled", M, T), [])
                if sv and pv:
                    pct = [(s - p) / s * 100 for s, p in
                           zip(sorted(sv), sorted(pv)) if s > 0]
                    med, q25, q75 = median_iqr(pct)
                else:
                    med, q25, q75 = np.nan, np.nan, np.nan
                meds.append(med); q25s.append(q25); q75s.append(q75)
            ax.errorbar(Ts, meds, yerr=iqr_yerr(meds, q25s, q75s),
                        color=colors[f"{loss}_pooled"],
                        linewidth=2, capsize=2, label=labels[loss])

        ax.axhline(0, color="black", linestyle=":", alpha=0.3)
        ax.set_xlabel("Accumulation steps $T$")
        ax.set_ylabel("% improvement (pooled over standard)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts])
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"accumulation_advantage_M{M}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  accumulation: {2 * len(Ms)} plots")


# ---------------------------------------------------------------------------
# Exp 1.3: DDP simulation
# ---------------------------------------------------------------------------

def plot_ddp(data, out_dir):
    if data is None:
        print("  ddp: no data")
        return

    r3 = parse_exp1_3(data)
    D_values = sorted(set(k[1] for k in r3))

    mode_cfg = {
        "local_sigreg":         ("tab:orange",  "--"),
        "global_sigreg":        ("darkorange",  "-."),
        "pooled_sigreg":        ("tab:red",     "-"),
        "pooled_global_sigreg": ("darkred",     "-"),
        "local_w1":             ("tab:blue",    "--"),
        "global_w1":            ("royalblue",   "-."),
        "pooled_w1":            ("tab:green",   "-"),
        "pooled_global_w1":     ("darkgreen",   "-"),
    }

    for group_name, mode_list in [
        ("sigreg", [m for m in mode_cfg if "sigreg" in m]),
        ("w1", [m for m in mode_cfg if "w1" in m]),
    ]:
        fig, ax = plt.subplots(figsize=SUBPLOT_SIZE)
        for mode in mode_list:
            if mode not in mode_cfg:
                continue
            color, ls = mode_cfg[mode]
            meds, q25s, q75s = [], [], []
            for D in D_values:
                med, q25, q75 = median_iqr(r3.get((mode, D), []))
                meds.append(med); q25s.append(q25); q75s.append(q75)
            ax.errorbar(D_values, meds, yerr=iqr_yerr(meds, q25s, q75s),
                        color=color, linestyle=ls, linewidth=2, capsize=2,
                        label=mode.replace("_", " "))

        ax.set_xlabel("Simulated devices $D$")
        ax.set_ylabel("W1 to $\\mathcal{N}(0, I)$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(D_values)
        ax.set_xticklabels([str(d) for d in D_values])
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"ddp_{group_name}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Advantage over local
    for group_name, baseline, tests in [
        ("sigreg", "local_sigreg", [
            ("global_sigreg", "global", "darkorange"),
            ("pooled_sigreg", "pooled local", "tab:red"),
            ("pooled_global_sigreg", "pooled + global", "darkred"),
        ]),
        ("w1", "local_w1", [
            ("global_w1", "global", "royalblue"),
            ("pooled_w1", "pooled local", "tab:green"),
            ("pooled_global_w1", "pooled + global", "darkgreen"),
        ]),
    ]:
        fig, ax = plt.subplots(figsize=SUBPLOT_SIZE)
        for test_mode, label, color in tests:
            meds, q25s, q75s = [], [], []
            for D in D_values:
                bv = r3.get((baseline, D), [])
                tv = r3.get((test_mode, D), [])
                if bv and tv:
                    pct = [(b - t) / b * 100 for b, t in
                           zip(sorted(bv), sorted(tv)) if b > 0]
                    med, q25, q75 = median_iqr(pct)
                else:
                    med, q25, q75 = np.nan, np.nan, np.nan
                meds.append(med); q25s.append(q25); q75s.append(q75)
            ax.errorbar(D_values, meds, yerr=iqr_yerr(meds, q25s, q75s),
                        color=color, linewidth=2, capsize=2, label=label)

        ax.axhline(0, color="black", linestyle=":", alpha=0.3)
        ax.set_xlabel("Simulated devices $D$")
        ax.set_ylabel("% improvement over local")
        ax.set_xscale("log", base=2)
        ax.set_xticks(D_values)
        ax.set_xticklabels([str(d) for d in D_values])
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"ddp_advantage_{group_name}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  ddp: 4 plots")


# ---------------------------------------------------------------------------
# Exp 1.4: Timing
# ---------------------------------------------------------------------------

def plot_timing(data, out_dir):
    if data is None:
        print("  timing: no data")
        return

    # Collect into grids: timing[method][phase] = {D: {B: (median, std)}}
    timing = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for key, val in data.items():
        m = val["method"]
        D, B = val["D"], val["B"]
        fwd_std = val.get("fwd_std_ms", 0)
        fwdbwd_std = val.get("fwdbwd_std_ms", 0)
        timing[m]["fwd"][D][B] = (val["fwd_ms"], fwd_std)
        timing[m]["fwdbwd"][D][B] = (val["fwdbwd_ms"], fwdbwd_std)

    methods = sorted(timing.keys())
    Ds = sorted(set(D for m in timing for D in timing[m]["fwd"]))
    Bs = sorted(set(B for m in timing for D in timing[m]["fwd"]
                    for B in timing[m]["fwd"][D]))

    colors = {"sigreg": "tab:orange", "w1": "tab:blue", "w2": "tab:green"}
    labels = {"sigreg": "SIGReg", "w1": "Sliced W1", "w2": "Sliced W2"}

    fixed_Ds = [d for d in [4, 16, 64, 256] if d in Ds]

    for phase, phase_label in [("fwd", "Forward"), ("fwdbwd", "Forward + Backward")]:
        fig, axes = plt.subplots(1, len(fixed_Ds),
                                  figsize=(SUBPLOT_SIZE[0] * len(fixed_Ds), SUBPLOT_SIZE[1]))
        if len(fixed_Ds) == 1:
            axes = [axes]
        for ax, D in zip(axes, fixed_Ds):
            for m in methods:
                meds = [timing[m][phase].get(D, {}).get(B, (np.nan, 0))[0]
                        for B in Bs]
                stds = [timing[m][phase].get(D, {}).get(B, (np.nan, 0))[1]
                        for B in Bs]
                ax.errorbar(Bs, meds, yerr=stds, color=colors.get(m, "grey"),
                            linewidth=2, capsize=2, label=labels.get(m, m))
            ax.set_xlabel("Batch size $B$")
            if ax == axes[0]:
                ax.set_ylabel(f"{phase_label} time (ms)")
            ax.set_title(f"$D = {D}$")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"timing_{phase}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Ratio: SIGReg / W1 for fwd+bwd
    fig, axes = plt.subplots(1, len(fixed_Ds),
                              figsize=(SUBPLOT_SIZE[0] * len(fixed_Ds), SUBPLOT_SIZE[1]))
    if len(fixed_Ds) == 1:
        axes = [axes]
    for ax, D in zip(axes, fixed_Ds):
        for m2, label, color in [("w1", "SIGReg / W1", "tab:blue"),
                                  ("w2", "SIGReg / W2", "tab:green")]:
            ratios = []
            for B in Bs:
                s = timing["sigreg"]["fwdbwd"].get(D, {}).get(B, (np.nan, 0))[0]
                w = timing[m2]["fwdbwd"].get(D, {}).get(B, (np.nan, 0))[0]
                ratios.append(s / w if w > 0 else np.nan)
            ax.plot(Bs, ratios, color=color, linewidth=2, label=label)
        ax.axhline(1, color="black", linestyle=":", alpha=0.3)
        ax.set_xlabel("Batch size $B$")
        if ax == axes[0]:
            ax.set_ylabel("Time ratio (fwd+bwd)")
        ax.set_title(f"$D = {D}$")
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "timing_ratio.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  timing: 3 plots")


# ---------------------------------------------------------------------------
# Exp 1.5: FIFO buffer
# ---------------------------------------------------------------------------

def plot_fifo(data, out_dir):
    if data is None:
        print("  fifo: no data")
        return

    r5 = parse_exp1_5(data)
    T_curs = sorted(set(k[0] for k in r5))
    T_fifos = sorted(set(k[1] for k in r5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(T_curs)))

    # W1 vs FIFO depth, one line per T_cur
    fig, ax = plt.subplots(figsize=SUBPLOT_SIZE)
    for j, T_cur in enumerate(T_curs):
        meds, q25s, q75s = [], [], []
        for T_fifo in T_fifos:
            med, q25, q75 = median_iqr(r5.get((T_cur, T_fifo), []))
            meds.append(med); q25s.append(q25); q75s.append(q75)
        ax.errorbar(T_fifos, meds, yerr=iqr_yerr(meds, q25s, q75s),
                    color=colors[j], linewidth=2, capsize=2,
                    label=f"$T_{{cur}}={T_cur}$")
    ax.set_xlabel("FIFO depth $T_{fifo}$")
    ax.set_ylabel("W1 to $\\mathcal{N}(0, I)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fifo_depth.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # W1 vs T_cur, one line per T_fifo (accumulation benefit at each FIFO depth)
    colors_fifo = plt.cm.magma(np.linspace(0.2, 0.85, len(T_fifos)))
    fig, ax = plt.subplots(figsize=SUBPLOT_SIZE)
    for j, T_fifo in enumerate(T_fifos):
        meds, q25s, q75s = [], [], []
        for T_cur in T_curs:
            med, q25, q75 = median_iqr(r5.get((T_cur, T_fifo), []))
            meds.append(med); q25s.append(q25); q75s.append(q75)
        ax.errorbar(T_curs, meds, yerr=iqr_yerr(meds, q25s, q75s),
                    color=colors_fifo[j], linewidth=2, capsize=2,
                    label=f"$T_{{fifo}}={T_fifo}$")
    ax.set_xlabel("Accumulation steps $T_{cur}$")
    ax.set_ylabel("W1 to $\\mathcal{N}(0, I)$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(T_curs)
    ax.set_xticklabels([str(t) for t in T_curs])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fifo_accumulation.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  fifo: 2 plots")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-dir", default="results/aggregate_plots")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading...")
    exp1_1 = load_json(args.results_root, "exp1_1_full_batch")
    exp1_2 = load_json(args.results_root, "exp1_2_accumulation")
    exp1_3 = load_json(args.results_root, "exp1_3_ddp_sim")
    exp1_4 = load_json(args.results_root, "exp1_4_timing", "timing_results.json")
    exp1_5 = load_json(args.results_root, "exp1_5_fifo")

    print("Plotting...")
    plot_accumulation(exp1_1, exp1_2, args.out_dir)
    plot_ddp(exp1_3, args.out_dir)
    plot_timing(exp1_4, args.out_dir)
    plot_fifo(exp1_5, args.out_dir)

    print(f"Done. Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
