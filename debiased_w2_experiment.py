"""
Debiased Sliced W2: Convergence comparison

Full batch vs Pseudo r=4 (ForgettingNIW) — how long does pseudo take to
reach full-batch quality?

Run:
    python debiased_w2_experiment.py
"""

import os
import sys
import math
import argparse

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigreg_experiment import (
    DeepMLP,
    generate_data,
    GENERATORS,
    make_fixed_projection,
    resolve_device,
)
from src.niw_tracking.estimators import ForgettingNIWEstimator


def eval_w1(x: np.ndarray, num_proj: int = 1024) -> float:
    """Sliced W1 between samples and N(0,1) on full dataset."""
    t = torch.tensor(x, dtype=torch.float32)
    n, d = t.shape
    torch.manual_seed(999)
    A = torch.randn(d, num_proj)
    A = A / A.norm(dim=0, keepdim=True)
    proj_sorted = torch.sort(t @ A, dim=0).values
    p = (torch.arange(1, n + 1, dtype=torch.float32) - 0.5) / n
    ref = torch.erfinv(2 * p - 1) * math.sqrt(2)
    return (proj_sorted - ref.unsqueeze(1)).abs().mean(0).mean().item()


def train_one(seed, args, use_pseudo, device):
    """Train one run."""
    D = args.input_dim

    torch.manual_seed(seed)
    data = generate_data(args.distribution, args.num_points, D)
    W = make_fixed_projection(D, args.proj_dim, seed=seed + 100).to(device)
    projected = data.to(device) @ W

    torch.manual_seed(seed + 200)
    mlp = DeepMLP(args.proj_dim, args.hidden_dim, D, depth=args.depth).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    # Compute colors from source data (angle in dims 0,1 for smooth transition)
    src_np = data.numpy()
    colors = np.arctan2(src_np[:, 1] if D > 1 else np.zeros(len(src_np)),
                        src_np[:, 0])  # angle in [-pi, pi]
    colors = (colors + np.pi) / (2 * np.pi)  # normalize to [0, 1]

    # Initial output before training
    mlp.eval()
    with torch.no_grad():
        initial_out = mlp(projected).cpu().numpy()
    mlp.train()

    niw = None
    if use_pseudo:
        # Start with alpha=0.9 (aggressive forgetting), will anneal to 0.99
        niw = ForgettingNIWEstimator(
            dim=D, alpha=0.9,
            alpha_schedule=lambda step, total: 0.9 + 0.09 * min(step / (total * 0.5), 1.0),
            total_steps=args.steps,
            device=device,
        )

    eval_steps, eval_w1s = [], []

    for step in range(1, args.steps + 1):
        mlp.train()
        if not use_pseudo:
            # Full batch
            batch_proj = projected
        else:
            idx = torch.randint(0, args.num_points, (args.batch_size,), device=device)
            batch_proj = projected[idx]

        output = mlp(batch_proj)
        B = output.shape[0]
        D = output.shape[1]

        A = torch.randn(D, args.num_proj, device=device)
        A = A / A.norm(dim=0, keepdim=True)

        if use_pseudo:
            # 1) Update running estimate with current batch (detached for NIW state)
            niw.update(output.detach())

            # 2) Also compute differentiable running stats for moment loss
            #    (gradient through current batch only, history detached)
            beta_ema = 1.0 - niw._effective_alpha()
            if not hasattr(niw, '_diff_T1'):
                niw._diff_T1 = torch.zeros(D, dtype=torch.float32, device=device)
                niw._diff_T2 = torch.zeros(D, D, dtype=torch.float32, device=device)
                niw._diff_w = 0.0
            niw._diff_w = (1 - beta_ema) * niw._diff_w + beta_ema * B
            # These have gradient through `output`
            T1_diff = ((1 - beta_ema) * niw._diff_T1.detach()
                       + beta_ema * output.sum(0))
            T2_diff = ((1 - beta_ema) * niw._diff_T2.detach()
                       + beta_ema * (output.T @ output))
            mu_diff = T1_diff / niw._diff_w
            cov_diff = T2_diff / niw._diff_w - torch.outer(mu_diff, mu_diff)
            # Save detached for next step
            niw._diff_T1 = T1_diff.detach()
            niw._diff_T2 = T2_diff.detach()

            # 3) Project real batch: s_i = w_k^T z_i (has gradient)
            proj_real = output @ A  # (B, P)

            # 4) Project detached running stats for pseudo-samples
            mu_run = niw.get_mean().float().detach()
            cov_run = niw.get_cov().float().detach()
            mu_r = A.T @ mu_run
            var_r = ((A.T @ cov_run) * A.T).sum(1).clamp(min=1e-8)
            sigma_r = var_r.sqrt()

            # 5) Draw m 1D pseudo-samples — no gradient
            m = int(args.pseudo_ratio * B)
            if niw.step > 5 and m > 0:
                pseudo_1d = (mu_r.unsqueeze(0)
                             + sigma_r.unsqueeze(0)
                             * torch.randn(m, args.num_proj, device=device))
                proj_all = torch.cat([proj_real, pseudo_1d.detach()], 0)
            else:
                proj_all = proj_real

            n_tot = proj_all.shape[0]
            proj_sorted = torch.sort(proj_all, dim=0).values

            # 6) Moment correction loss from differentiable running stats
            #    L1 + L2: constant push (L1) + proportional push (L2)
            cov_err = cov_diff - torch.eye(D, device=device)
            moment_l1 = mu_diff.abs().sum() + cov_err.abs().sum()
            moment_l2 = mu_diff.pow(2).sum() + cov_err.pow(2).sum()
            moment_loss = moment_l1 + moment_l2
        else:
            proj_real = output @ A
            proj_sorted = torch.sort(proj_real, dim=0).values
            n_tot = B
            moment_loss = 0.0

        p = (torch.arange(1, n_tot + 1, device=device, dtype=torch.float32) - 0.5) / n_tot
        ref = torch.erfinv(2 * p - 1) * math.sqrt(2)
        diff = proj_sorted - ref.unsqueeze(1)
        w1_loss = diff.abs().mean(0).mean()
        w2_loss = (diff ** 2).mean(0).sqrt().mean()

        # loss = w1_loss + args.w2_weight * w2_loss + args.moment_weight * moment_loss
        loss = w2_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_interval == 0:
            mlp.eval()
            with torch.no_grad():
                full_out = mlp(projected).cpu().numpy()
                w2_val = eval_w1(full_out)
                eval_steps.append(step)
                eval_w1s.append(w2_val)

                # Diagnostics
                mu = full_out.mean(axis=0)
                cov = np.cov(full_out, rowvar=False)
                D_out = full_out.shape[1]
                mse_mean = np.sum(mu ** 2)
                frob_cov = np.linalg.norm(cov - np.eye(D_out), 'fro') / np.linalg.norm(np.eye(D_out), 'fro')
                diag = np.diag(cov)
                off = cov - np.diag(diag)
                print(f"    step={step:6d}  W1={w2_val:.4f}  "
                      f"||mu||²={mse_mean:.4f}  "
                      f"cov_frob={frob_cov:.4f}  "
                      f"diag=[{', '.join(f'{d:.3f}' for d in diag)}]  "
                      f"off_max={np.abs(off).max():.4f}  "
                      f"w1={w1_loss.item():.4f}  w2={w2_loss.item():.4f}" +
                      (f"  moment={moment_loss.item():.4f}" if use_pseudo else ""))
            mlp.train()

    mlp.eval()
    with torch.no_grad():
        final_out = mlp(projected).cpu().numpy()
    return eval_steps, eval_w1s, final_out, initial_out, src_np, colors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distribution", default="blobs", choices=list(GENERATORS.keys()))
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--input-dim", type=int, default=4)
    p.add_argument("--proj-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-proj", type=int, default=2048)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--pseudo-ratio", type=float, default=4.0)
    p.add_argument("--moment-weight", type=float, default=0.0)
    p.add_argument("--w2-weight", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--save-dir", default="debiased_w2_results")
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device)

    print(f"dist={args.distribution}, D={args.input_dim}, batch={args.batch_size}, "
          f"steps={args.steps}, seeds={args.n_seeds}")
    print(f"ForgettingNIW alpha=0.9->0.99, pseudo_ratio=1->{args.pseudo_ratio} (scheduled)")
    print()

    methods = [
        # ("Full batch", False),
        (f"Pseudo r={args.pseudo_ratio:.0f} (NIW, batch={args.batch_size})", True),
    ]

    all_runs = {}
    for label, use_pseudo in methods:
        all_runs[label] = {}
        for si in range(args.n_seeds):
            seed = 42 + si
            print(f"  {label}, seed={seed}...", end=" ", flush=True)
            steps, w1s, final, initial, src, pt_colors = train_one(
                seed, args, use_pseudo, device)
            all_runs[label][seed] = (steps, w1s, final, initial, src, pt_colors)
            print(f"W1={w1s[-1]:.4f}")

    # === Convergence plot ===
    line_colors = ["black", "tab:red"]
    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (label, _) in enumerate(methods):
        runs = all_runs[label]
        all_w1 = np.array([w1s for _, w1s, _, _, _, _ in runs.values()])
        steps_arr = list(runs.values())[0][0]
        mean = all_w1.mean(axis=0)
        std = all_w1.std(axis=0)
        c = line_colors[idx]
        ax.plot(steps_arr, mean, label=label, color=c, linewidth=2)
        ax.fill_between(steps_arr, mean - std, mean + std, color=c, alpha=0.1)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Sliced W1 to N(0,1)\n(full dataset eval)", fontsize=11)
    ax.set_title(f"{args.distribution}, D={args.input_dim}, "
                 f"ForgettingNIW, {args.n_seeds} seeds", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_dir, "convergence.png"), dpi=150)
    plt.close(fig)

    # === Colored distributions: source → initial → final ===
    D = args.input_dim
    pairs = [(i, j) for i in range(min(D, 4)) for j in range(i + 1, min(D, 4))]
    n_cols = max(len(pairs), 1)

    # Use first seed
    first_label = methods[0][0]
    _, w1s, final, initial, src, pt_colors = all_runs[first_label][42]

    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    stages = [("Source", src), ("Initial (before training)", initial), ("Final", final)]
    for row, (stage_label, data) in enumerate(stages):
        for col, (di, dj) in enumerate(pairs):
            ax = axes[row, col]
            ax.scatter(data[:, di], data[:, dj], s=3, alpha=0.5,
                       c=pt_colors, cmap="hsv")
            for r in [1, 2]:
                ax.add_patch(plt.Circle((0, 0), r, fill=False, color="grey",
                                        linestyle="--", linewidth=0.8, alpha=0.5))
            ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
            ax.grid(alpha=0.2)
            if row == 0: ax.set_title(f"dim {di} vs {dj}", fontsize=9)
            if col == 0: ax.set_ylabel(stage_label, fontsize=10)

    fig.suptitle(f"Point Flow: {args.distribution} — colored by initial angle\n"
                 f"(final W1={w1s[-1]:.4f})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.save_dir, "point_flow.png"), dpi=150)
    plt.close(fig)

    # Summary
    lines = []
    for label, _ in methods:
        all_w1 = np.array([w1s for _, w1s, _, _, _, _ in all_runs[label].values()])
        tail = all_w1[:, -10:].mean(axis=1)
        last = all_w1[:, -1]
        lines.append(f"{label:<35} tail={tail.mean():.4f}±{tail.std():.4f}  "
                     f"last={last.mean():.4f}±{last.std():.4f}")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(args.save_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved to {args.save_dir}/")


if __name__ == "__main__":
    main()
