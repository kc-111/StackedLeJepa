#!/usr/bin/env python
"""Demo: accumulated pooled-sort W1 vs baseline vs standard grad accum.

Shows the advantage of computing W1 on pooled exact embeddings (from a
frozen encoder over N sub-steps) versus standard approaches on two
distributions: blobs (clustered) and hypercube_vertices (X pattern).

Run:
    python experiments/synthetic/demo_pooled_sliced.py
"""

import copy
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sliced_gauss_reg import (
    PooledSlicedLoss,
    DeepMLP,
    SlicedW1Loss,
    eval_w1,
    evaluate_full,
    generate_data,
    make_fixed_projection,
)

# ── Config ──────────────────────────────────────────────────────────────────
D = 4                   # embedding dimension
K = 1024                # dataset size
BS = 8                  # batch size per forward pass
PROJ_DIM = 32            # fixed projection dimension
NUM_PROJ = 256          # projection directions for loss
ACCUM_N = 8             # accumulation steps
WARMUP_UPDATES = 20000  # warmup parameter updates (plain W1)
FINETUNE_UPDATES = 20000 # finetune parameter updates (comparison phase)
LR = 1e-3
EVAL_EVERY = 100       # eval interval (in parameter updates)
DISTRIBUTIONS = ["blobs", "diagonal_cross"]
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "demo_accum_results")


def run_baseline(mlp_state, projected, steps, eval_every, batch_size=BS):
    """Plain W1 on batch_size samples, update every step."""
    mlp = DeepMLP(PROJ_DIM, 128, D, depth=2)
    mlp.load_state_dict(copy.deepcopy(mlp_state))
    opt = torch.optim.Adam(mlp.parameters(), lr=LR)
    loss_fn = SlicedW1Loss(num_proj=NUM_PROJ)
    history = []

    for step in range(1, steps + 1):
        mlp.train()
        idx = torch.randint(0, K, (batch_size,))
        loss = loss_fn(mlp(projected[idx]))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0:
            mlp.eval()
            with torch.no_grad():
                history.append((step, eval_w1(mlp(projected).numpy())))

    mlp.eval()
    with torch.no_grad():
        final = mlp(projected).cpu().numpy()
    return history, final


def run_std_accum(mlp_state, projected, steps, eval_every, N):
    """Standard gradient accumulation: average N independent W1 losses."""
    mlp = DeepMLP(PROJ_DIM, 128, D, depth=2)
    mlp.load_state_dict(copy.deepcopy(mlp_state))
    opt = torch.optim.Adam(mlp.parameters(), lr=LR)
    loss_fn = SlicedW1Loss(num_proj=NUM_PROJ)
    history = []

    for step in range(1, steps + 1):
        mlp.train()
        opt.zero_grad()
        for sub in range(N):
            idx = torch.randint(0, K, (BS,))
            loss = loss_fn(mlp(projected[idx])) / N
            loss.backward()
        opt.step()

        if step % eval_every == 0:
            mlp.eval()
            with torch.no_grad():
                history.append((step, eval_w1(mlp(projected).numpy())))

    mlp.eval()
    with torch.no_grad():
        final = mlp(projected).cpu().numpy()
    return history, final


def run_accumulated(mlp_state, projected, steps, eval_every, N):
    """Accumulated pooled sort: one W1 on N×BS exact embeddings."""
    mlp = DeepMLP(PROJ_DIM, 128, D, depth=2)
    mlp.load_state_dict(copy.deepcopy(mlp_state))
    opt = torch.optim.Adam(mlp.parameters(), lr=LR)
    accum_loss = PooledSlicedLoss(accum_steps=N - 1, num_proj=NUM_PROJ,
                                       mode="w1")
    history = []

    for step in range(1, steps + 1):
        mlp.train()
        # N-1 accumulation steps (no_grad handled internally)
        for sub in range(N - 1):
            idx = torch.randint(0, K, (BS,))
            accum_loss.accum_step(mlp(projected[idx]))

        # 1 gradient step
        idx = torch.randint(0, K, (BS,))
        loss = accum_loss.grad_step(mlp(projected[idx]))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0:
            mlp.eval()
            with torch.no_grad():
                history.append((step, eval_w1(mlp(projected).numpy())))

    mlp.eval()
    with torch.no_grad():
        final = mlp(projected).cpu().numpy()
    return history, final


def plot_convergence(warmup_hist, results, save_path, title):
    """Plot W1 convergence for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ws, wv = zip(*warmup_hist)
    ax.plot(ws, wv, color="grey", linewidth=1.5, label="warmup")
    ax.axvline(WARMUP_UPDATES, color="black", linestyle=":", alpha=0.5)

    cmap = ["tab:blue", "black", "tab:orange", "tab:red"]
    for idx, (name, (hist, _)) in enumerate(results.items()):
        hs, hv = zip(*hist)
        hs = [h + WARMUP_UPDATES for h in hs]
        ls = "--" if "Large" in name else "-"
        ax.plot(hs, hv, color=cmap[idx % len(cmap)], linewidth=2,
                linestyle=ls, label=name)

    ax.set_xlabel("Parameter Updates", fontsize=12)
    ax.set_ylabel("Sliced W1 to N(0,1)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_point_flow(source, initial, final, colors, save_path, title):
    """Plot source → initial → final with consistent coloring."""
    pairs = [(i, j) for i in range(min(D, 4)) for j in range(i + 1, min(D, 4))]
    n_cols = max(len(pairs), 1)
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, (label, data) in enumerate(
            [("Source", source), ("Initial", initial), ("Final", final)]):
        for col, (di, dj) in enumerate(pairs):
            ax = axes[row, col]
            ax.scatter(data[:, di], data[:, dj], s=3, alpha=0.5,
                       c=colors, cmap="hsv", vmin=0, vmax=1)
            for r in [1, 2]:
                ax.add_patch(plt.Circle((0, 0), r, fill=False, color="grey",
                                        linestyle="--", linewidth=0.8, alpha=0.5))
            ax.set_aspect("equal")
            lim = max(5, np.abs(data[:, [di, dj]]).max() * 1.2)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(f"dim {di} vs {dj}", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=11)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    summary_lines = []

    for dist_name in DISTRIBUTIONS:
        print(f"=== {dist_name} ===")

        # Generate data
        torch.manual_seed(42)
        data = generate_data(dist_name, K, D)
        W = make_fixed_projection(D, PROJ_DIM, seed=0)
        projected = data @ W

        # Point colors from source angle
        src_np = data.numpy()
        angles = np.arctan2(src_np[:, 1] if D > 1 else np.zeros(K), src_np[:, 0])
        pt_colors = (angles + np.pi) / (2 * np.pi)

        # Warmup (shared checkpoint)
        torch.manual_seed(123)
        mlp = DeepMLP(PROJ_DIM, 128, D, depth=2)
        opt = torch.optim.Adam(mlp.parameters(), lr=LR)
        loss_fn = SlicedW1Loss(num_proj=NUM_PROJ)
        warmup_hist = []

        print("  Warmup...", end=" ", flush=True)
        for step in range(1, WARMUP_UPDATES + 1):
            mlp.train()
            idx = torch.randint(0, K, (BS,))
            loss = loss_fn(mlp(projected[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % EVAL_EVERY == 0:
                mlp.eval()
                with torch.no_grad():
                    warmup_hist.append((step, eval_w1(mlp(projected).numpy())))

        ckpt_w1 = warmup_hist[-1][1]
        ckpt = copy.deepcopy(mlp.state_dict())
        mlp.eval()
        with torch.no_grad():
            initial_out = mlp(projected).cpu().numpy()
        print(f"W1={ckpt_w1:.4f}")

        # Run all methods
        methods = {}
        timings = {}
        large_bs = BS * ACCUM_N  # e.g. 8*8=64
        for name, fn, n_fwd_per_update in [
            (f"Baseline (BS={BS})",
             lambda s: run_baseline(s, projected, FINETUNE_UPDATES, EVAL_EVERY, BS), 1),
            (f"Large batch (BS={large_bs})",
             lambda s: run_baseline(s, projected, FINETUNE_UPDATES, EVAL_EVERY, large_bs), 1),
            (f"Std grad accum (N={ACCUM_N})",
             lambda s: run_std_accum(s, projected, FINETUNE_UPDATES, EVAL_EVERY, ACCUM_N), ACCUM_N),
            (f"Pooled sort (N={ACCUM_N})",
             lambda s: run_accumulated(s, projected, FINETUNE_UPDATES, EVAL_EVERY, ACCUM_N), ACCUM_N),
        ]:
            t0 = time.time()
            hist, final = fn(ckpt)
            elapsed = time.time() - t0
            final_w1 = hist[-1][1]
            methods[name] = (hist, final)
            total_fwd = FINETUNE_UPDATES * n_fwd_per_update
            ms_per_update = elapsed / FINETUNE_UPDATES * 1000
            timings[name] = elapsed
            print(f"  {name:<30s} W1={final_w1:.4f}  "
                  f"time={elapsed:.1f}s  fwd={total_fwd}  "
                  f"ms/update={ms_per_update:.2f}")

        # Convergence plot
        plot_convergence(
            warmup_hist, methods,
            os.path.join(SAVE_DIR, f"convergence_{dist_name}.png"),
            f"{dist_name}, D={D}, BS={BS}, N={ACCUM_N}",
        )

        # Point flow for best method
        for name, (hist, final) in methods.items():
            safe_name = name.lower().replace(" ", "_")
            plot_point_flow(
                src_np, initial_out, final, pt_colors,
                os.path.join(SAVE_DIR, f"point_flow_{dist_name}_{safe_name}.png"),
                f"{name} — {dist_name} (W1={hist[-1][1]:.4f})",
            )

        # Summary
        summary_lines.append(f"\n{dist_name} (checkpoint W1={ckpt_w1:.4f}):")
        for name, (hist, _) in methods.items():
            w1 = hist[-1][1]
            diff = w1 - ckpt_w1
            t = timings[name]
            summary_lines.append(
                f"  {name:<30s} W1={w1:.4f}  diff={diff:+.4f}  "
                f"time={t:.1f}s")
        print()

    # Save summary
    summary = "\n".join(summary_lines)
    print(summary)
    with open(os.path.join(SAVE_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
