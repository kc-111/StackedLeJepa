"""Generate plots from compute_cost benchmark results.

Reads the markdown tables saved by saturation_curve.py and pooled_overhead.py
and produces matplotlib figures.

Outputs:
  saturation_curve.png      — GPU saturation curve (forward time vs n_imgs)
  pooled_main.png           — main: std vs pool_nograd, line plot with error bars
  pooled_main_memory.png    — main: peak memory, line plot
  pooled_supplementary.png  — supplementary: includes pool_nograd_chunked
  pooled_ratio.png          — pool_nograd / std ratio across BS

Run:
    python experiments/compute_cost/plot_results.py
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style: no grid lines, clean
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


METHOD_COLORS = {
    "std": "#2c7bb6",
    "pool_nograd": "#1a9641",
    "pool_chunked": "#fdae61",
}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_saturation_md(path: Path):
    rows = []
    in_table = False
    for line in path.read_text().splitlines():
        if line.startswith("| n_imgs"):
            in_table = True
            continue
        if line.startswith("|---"):
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 4:
                continue
            n = int(cells[0])
            ms = float(cells[1].split()[0])
            ms_per_img = float(cells[2].split()[0])
            thru = float(cells[3].split("/")[0])
            rows.append((n, ms, ms_per_img, thru))
    return rows


def parse_pooled_md(path: Path):
    """Parse the timing + memory tables from a pooled_overhead markdown file.

    Returns (rows, mem_rows) where:
      rows: list of dicts mapping column header → (mean_ms, std_ms) tuple
      mem_rows: dict mapping bs → {column header: GB}
    """
    text = path.read_text()
    rows = []
    match = re.search(r"## Results — timing.*?\n(\|.*?\n(?:\|.*?\n)+)", text, re.DOTALL)
    if not match:
        return [], {}
    table = match.group(1)
    lines = table.strip().splitlines()
    if len(lines) < 3:
        return [], {}
    header = [c.strip() for c in lines[0].split("|")[1:-1]]
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) != len(header):
            continue
        row = {}
        for h, c in zip(header, cells):
            if h == "BS":
                row["bs"] = int(c)
            elif "OOM" in c:
                row[h] = (float("nan"), float("nan"))
            else:
                m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", c)
                if m:
                    row[h] = (float(m.group(1)), float(m.group(2)))
        rows.append(row)

    memory_rows = {}
    match = re.search(r"## Results — peak GPU memory.*?\n(\|.*?\n(?:\|.*?\n)+)", text, re.DOTALL)
    if match:
        table = match.group(1)
        lines = table.strip().splitlines()
        header = [c.strip() for c in lines[0].split("|")[1:-1]]
        for line in lines[2:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) != len(header):
                continue
            bs = int(cells[0])
            memory_rows[bs] = {}
            for h, c in zip(header[1:], cells[1:]):
                if "OOM" in c:
                    memory_rows[bs][h] = float("nan")
                    continue
                try:
                    memory_rows[bs][h] = float(c.split()[0])
                except (ValueError, IndexError):
                    pass
    return rows, memory_rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _arch_label(path: Path) -> str:
    s = path.stem.replace("pooled_overhead_NVIDIA_GeForce_RTX_4090_", "").replace("_T8", "")
    s = s.replace("128_multicrop", "128 multicrop").replace("128", "@128²")
    return s


def plot_saturation(saturation_files, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.cm.tab10.colors
    for i, sat_file in enumerate(saturation_files):
        rows = parse_saturation_md(sat_file)
        if not rows:
            continue
        ns = [r[0] for r in rows]
        ms = [r[1] for r in rows]
        thru = [r[3] for r in rows]
        label = sat_file.stem.replace("saturation_NVIDIA_GeForce_RTX_4090_", "")
        c = colors[i % len(colors)]
        axes[0].loglog(ns, ms, "o-", color=c, label=label, markersize=5)
        axes[1].semilogx(ns, thru, "o-", color=c, label=label, markersize=5)

    axes[0].set_xlabel("# images forwarded")
    axes[0].set_ylabel("forward time (ms)")
    axes[0].set_title("GPU saturation curve — wall time")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)

    axes[1].set_xlabel("# images forwarded")
    axes[1].set_ylabel("throughput (imgs / sec)")
    axes[1].set_title("GPU saturation curve — throughput")
    axes[1].legend(fontsize=8, loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_pooled_main(pooled_files, out_path: Path):
    """Main plot: std vs pool_nograd, line + error bars, one panel per backbone."""
    n_files = len(pooled_files)
    fig, axes = plt.subplots(1, n_files, figsize=(5.5 * n_files, 4.5),
                              squeeze=False)

    for ax_idx, pooled_file in enumerate(pooled_files):
        ax = axes[0, ax_idx]
        rows, _ = parse_pooled_md(pooled_file)
        if not rows:
            continue
        bs_values = [r["bs"] for r in rows]

        for method, label, color in [
            ("std", "std", METHOD_COLORS["std"]),
            ("pool_nograd", "pool_nograd", METHOD_COLORS["pool_nograd"]),
        ]:
            means = [r.get(method, (float("nan"), 0))[0] for r in rows]
            stds = [r.get(method, (float("nan"), 0))[1] for r in rows]
            ax.errorbar(bs_values, means, yerr=stds, marker="o",
                        markersize=7, linewidth=1.8, capsize=4, capthick=1.2,
                        color=color, label=label)

        ax.set_xscale("log", base=2)
        ax.set_xticks(bs_values)
        ax.set_xticklabels([str(b) for b in bs_values])
        ax.set_xlabel("batch size (per gradient step)")
        ax.set_ylabel("step time (ms)")
        ax.set_title(_arch_label(pooled_file), fontsize=11)
        ax.legend(loc="upper left", frameon=False)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_pooled_main_memory(pooled_files, out_path: Path):
    """Main plot: peak memory, line per method, one panel per backbone."""
    n_files = len(pooled_files)
    fig, axes = plt.subplots(1, n_files, figsize=(5.5 * n_files, 4.5),
                              squeeze=False)

    for ax_idx, pooled_file in enumerate(pooled_files):
        ax = axes[0, ax_idx]
        _, mem_rows = parse_pooled_md(pooled_file)
        if not mem_rows:
            continue
        bs_values = sorted(mem_rows.keys())

        for method, label, color in [
            ("std", "std", METHOD_COLORS["std"]),
            ("pool_nograd", "pool_nograd", METHOD_COLORS["pool_nograd"]),
        ]:
            mems = [mem_rows[bs].get(method, float("nan")) for bs in bs_values]
            ax.plot(bs_values, mems, marker="o", markersize=7, linewidth=1.8,
                    color=color, label=label)

        ax.set_xscale("log", base=2)
        ax.set_xticks(bs_values)
        ax.set_xticklabels([str(b) for b in bs_values])
        ax.set_xlabel("batch size (per gradient step)")
        ax.set_ylabel("peak GPU memory (GB)")
        ax.set_title(_arch_label(pooled_file), fontsize=11)
        ax.legend(loc="upper left", frameon=False)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_pooled_supplementary(pooled_files, out_path: Path):
    """Supplementary: include pool_nograd_chunked too. Shows when chunking
    helps (large BS, ResNet) and when it hurts (small BS, ViT)."""
    n_files = len(pooled_files)
    fig, axes = plt.subplots(2, n_files, figsize=(5.5 * n_files, 8.5),
                              squeeze=False)

    for ax_idx, pooled_file in enumerate(pooled_files):
        ax_top = axes[0, ax_idx]
        ax_bot = axes[1, ax_idx]
        rows, mem_rows = parse_pooled_md(pooled_file)
        if not rows:
            continue
        bs_values = [r["bs"] for r in rows]

        method_specs = [
            ("std", "std", METHOD_COLORS["std"]),
            ("pool_nograd", "pool_nograd", METHOD_COLORS["pool_nograd"]),
            ("pool_chunked", "pool_nograd_chunked", METHOD_COLORS["pool_chunked"]),
        ]

        for method, label, color in method_specs:
            means = [r.get(method, (float("nan"), 0))[0] for r in rows]
            stds = [r.get(method, (float("nan"), 0))[1] for r in rows]
            ax_top.errorbar(bs_values, means, yerr=stds, marker="o",
                            markersize=6, linewidth=1.6, capsize=4, capthick=1,
                            color=color, label=label)

        for method, label, color in method_specs:
            mems = [mem_rows.get(bs, {}).get(method, float("nan")) for bs in bs_values]
            ax_bot.plot(bs_values, mems, marker="o", markersize=6, linewidth=1.6,
                        color=color, label=label)

        ax_top.set_xscale("log", base=2)
        ax_top.set_xticks(bs_values)
        ax_top.set_xticklabels([str(b) for b in bs_values])
        ax_top.set_ylabel("step time (ms)")
        ax_top.set_title(_arch_label(pooled_file), fontsize=11)
        ax_top.legend(loc="upper left", frameon=False, fontsize=9)
        ax_top.set_ylim(bottom=0)

        ax_bot.set_xscale("log", base=2)
        ax_bot.set_xticks(bs_values)
        ax_bot.set_xticklabels([str(b) for b in bs_values])
        ax_bot.set_xlabel("batch size (per gradient step)")
        ax_bot.set_ylabel("peak GPU memory (GB)")
        ax_bot.legend(loc="upper left", frameon=False, fontsize=9)
        ax_bot.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_pooled_ratio(pooled_files, out_path: Path):
    """Pool / std ratio across BS for each backbone (production variant only)."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    markers = ["o", "s", "^", "D"]

    for f_idx, pooled_file in enumerate(pooled_files):
        rows, _ = parse_pooled_md(pooled_file)
        if not rows:
            continue
        bs_values = [r["bs"] for r in rows]
        label = _arch_label(pooled_file)

        means_pool = [r.get("pool_nograd", (float("nan"), 0))[0] for r in rows]
        means_std = [r.get("std", (float("nan"), 0))[0] for r in rows]
        ratios = [p / s if s else float("nan") for p, s in zip(means_pool, means_std)]
        ax.plot(bs_values, ratios,
                marker=markers[f_idx % len(markers)],
                linestyle="-",
                label=label, markersize=8, linewidth=2)

    ax.axhline(y=1.0, color="k", linestyle=":", alpha=0.5,
               label="standard baseline")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size (BS)")
    ax.set_ylabel("pool_nograd time / std time")
    ax.set_title("Pooled training cost ratio vs standard (T=8)")
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.set_ylim(bottom=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",
                        default=str(Path(__file__).parent / "results"))
    parser.add_argument("--out-dir",
                        default=str(Path(__file__).parent / "plots"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saturation_files = sorted(results_dir.glob("saturation_*.md"))
    pooled_files = sorted(results_dir.glob("pooled_overhead_*.md"))

    if saturation_files:
        plot_saturation(saturation_files, out_dir / "saturation_curve.png")
    if pooled_files:
        plot_pooled_main(pooled_files, out_dir / "pooled_main.png")
        plot_pooled_main_memory(pooled_files, out_dir / "pooled_main_memory.png")
        plot_pooled_supplementary(pooled_files, out_dir / "pooled_supplementary.png")
        plot_pooled_ratio(pooled_files, out_dir / "pooled_ratio.png")


if __name__ == "__main__":
    main()
