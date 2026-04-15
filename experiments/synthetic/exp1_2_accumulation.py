"""
Experiment 1.2: Standard vs Pooled vs Big-Batch Accumulation

Shows that pooled accumulation breaks the finite-sample bias floor while
standard gradient accumulation (averaging independent losses) does not, and
that pooled closely tracks the big-batch oracle at a fraction of the memory.

9 methods: {W1, W2, SIGReg} x {standard, pooled, bigbatch}
Sweep: T={1,2,4,8,16,32}, M={8,32}, all 5 generators, 3 seeds.

Standard: T independent losses averaged (T backward passes).
Pooled (2-step): T-1 no-grad forward passes + 1 grad forward pass,
    pool all T*BS embeddings, compute loss, 1 backward pass.
Big-batch (oracle): single forward through T*BS samples with grad,
    one loss on the union, 1 backward through all T*BS. This is the
    "infinite VRAM" reference that pooled approximates.

Pipeline:
    Source (K, d) -> Fixed projection (K, M) -> MLP (K, d) -> Loss

Fairness: Option B (total-data-matched). Each opt step consumes T*BS samples
for all three modes, so an epoch is K // (T*BS) opt steps. See
experiments/synthetic/FAIRNESS.md.

Run:
    python experiments/synthetic/exp1_2_accumulation.py
    python experiments/synthetic/exp1_2_accumulation.py --distributions blobs --proj-dims 8 --accum-steps 1 4 8 --n-seeds 1 --epochs 5
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
    SlicedW1Loss, SlicedW2Loss, SIGRegLoss, SIGReg, PooledSlicedLoss,
    DeepMLP, generate_data, make_fixed_projection, epoch_iter, GENERATORS,
    eval_w1, evaluate_full,
)


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
    """Standard gradient accumulation: T independent losses averaged.

    Option B (total-data-matched): each opt step consumes T*BS samples (T grad
    sub-batches of size BS). Epoch = K // (T*BS) opt steps. Different T's see
    different opt-steps-per-epoch but the same total data per epoch.
    """
    projected, mlp, opt, src_np, colors, initial_out = \
        _setup(seed, args, proj_dim, distribution, device)
    K = args.num_points
    BS = args.batch_size
    chunk = T * BS
    if chunk > K:
        raise ValueError(f"T*BS={chunk} exceeds dataset K={K}")
    loss_fn = make_loss_fn(loss_mode, args.num_proj, args.knots, device)
    epoch_gen = torch.Generator(device=device).manual_seed(seed + 400)

    eval_steps, eval_w1s = [], []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for big_chunk in epoch_iter(K, chunk, device, epoch_gen):
            mlp.train()
            opt.zero_grad()
            for t in range(T):
                sub = big_chunk[t * BS:(t + 1) * BS]
                loss = loss_fn(mlp(projected[sub])) / T
                loss.backward()
            opt.step()
            global_step += 1

        mlp.eval()
        with torch.no_grad():
            w1_val = eval_w1(mlp(projected).cpu().numpy())
            eval_steps.append(global_step)
            eval_w1s.append(w1_val)
            print(f"  epoch={epoch:3d}  step={global_step:6d}  W1={w1_val:.4f}")
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

    Option B (total-data-matched): each opt step consumes T*BS samples
    ((T-1)*BS no-grad + BS grad). Epoch = K // (T*BS) opt steps — exactly
    matched to the standard path with the same T.
    """
    projected, mlp, opt, src_np, colors, initial_out = \
        _setup(seed, args, proj_dim, distribution, device)
    K = args.num_points
    BS = args.batch_size
    chunk = T * BS
    if chunk > K:
        raise ValueError(f"T*BS={chunk} exceeds dataset K={K}")

    sigreg_mod = None
    if loss_mode == "sigreg":
        sigreg_mod = SIGReg(knots=args.knots, num_proj=args.num_proj).to(device)

    accum_loss = PooledSlicedLoss(
        accum_steps=max(T - 1, 0), num_proj=args.num_proj,
        mode=loss_mode, sigreg=sigreg_mod)
    epoch_gen = torch.Generator(device=device).manual_seed(seed + 400)

    eval_steps, eval_w1s = [], []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for big_chunk in epoch_iter(K, chunk, device, epoch_gen):
            mlp.train()

            # T-1 no-grad sub-batches (CDF context, no activation storage)
            with torch.no_grad():
                for t in range(T - 1):
                    sub = big_chunk[t * BS:(t + 1) * BS]
                    accum_loss.accum_step(mlp(projected[sub]))

            # 1 gradient sub-batch (the last BS-slice of the chunk)
            sub = big_chunk[(T - 1) * BS:T * BS]
            loss = accum_loss.grad_step(mlp(projected[sub]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1

        mlp.eval()
        with torch.no_grad():
            w1_val = eval_w1(mlp(projected).cpu().numpy())
            eval_steps.append(global_step)
            eval_w1s.append(w1_val)
            print(f"  epoch={epoch:3d}  step={global_step:6d}  W1={w1_val:.4f}")
        mlp.train()

    mlp.eval()
    with torch.no_grad():
        final_out = mlp(projected).cpu().numpy()
        final_metrics = evaluate_full(final_out)

    return eval_steps, eval_w1s, final_out, initial_out, src_np, colors, final_metrics, mlp.state_dict()


def train_one_bigbatch(seed, args, loss_mode, proj_dim, distribution, T, device):
    """Big-batch (oracle): single forward+backward through T*BS samples.

    All T*BS samples carry gradient through one forward pass, one loss on
    the union, one backward pass. This is the "infinite VRAM" reference
    that pooled approximates: same data layout as pooled (one CDF over
    T*BS embeddings) but no detaching.

    Option B (total-data-matched): each opt step consumes T*BS samples in
    a single grad-carrying forward pass. Epoch = K // (T*BS) opt steps —
    matched to standard and pooled with the same T.
    """
    projected, mlp, opt, src_np, colors, initial_out = \
        _setup(seed, args, proj_dim, distribution, device)
    K = args.num_points
    BS = args.batch_size
    chunk = T * BS
    if chunk > K:
        raise ValueError(f"T*BS={chunk} exceeds dataset K={K}")
    loss_fn = make_loss_fn(loss_mode, args.num_proj, args.knots, device)
    epoch_gen = torch.Generator(device=device).manual_seed(seed + 400)

    eval_steps, eval_w1s = [], []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for big_chunk in epoch_iter(K, chunk, device, epoch_gen):
            mlp.train()
            opt.zero_grad()
            loss = loss_fn(mlp(projected[big_chunk]))
            loss.backward()
            opt.step()
            global_step += 1

        mlp.eval()
        with torch.no_grad():
            w1_val = eval_w1(mlp(projected).cpu().numpy())
            eval_steps.append(global_step)
            eval_w1s.append(w1_val)
            print(f"  epoch={epoch:3d}  step={global_step:6d}  W1={w1_val:.4f}")
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
    "w1_bigbatch":     {"color": "tab:blue",   "ls": ":",  "label": "W1 big-batch"},
    "w2_standard":     {"color": "tab:green",  "ls": "--", "label": "W2 standard"},
    "w2_pooled":       {"color": "tab:green",  "ls": "-",  "label": "W2 pooled"},
    "w2_bigbatch":     {"color": "tab:green",  "ls": ":",  "label": "W2 big-batch"},
    "sigreg_standard": {"color": "tab:orange", "ls": "--", "label": "SIGReg standard"},
    "sigreg_pooled":   {"color": "tab:orange", "ls": "-",  "label": "SIGReg pooled"},
    "sigreg_bigbatch": {"color": "tab:orange", "ls": ":",  "label": "SIGReg big-batch"},
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 1.2: Standard vs Pooled vs Big-Batch Accumulation"
    )
    p.add_argument("--distributions", nargs="+",
                    default=list(GENERATORS.keys()),
                    choices=list(GENERATORS.keys()))
    p.add_argument("--input-dim", type=int, default=4)
    p.add_argument("--proj-dims", type=int, nargs="+", default=[8])
    p.add_argument("--num-points", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--epochs", type=int, default=80,
                    help="Total-data-matched epochs (Option B): each epoch is "
                         "K // (T*BS) opt steps. See FAIRNESS.md.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--n-seeds", type=int, default=3)
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
    accum_modes = ["standard", "pooled", "bigbatch"]
    T_values = sorted(args.accum_steps)

    # Count runs: T=1 collapses all 3 modes (standard == pooled == bigbatch),
    # so only 1 run per loss is needed; T>1 needs all 3.
    n_T1 = len(loss_modes)  # 3
    n_Tgt1 = len([T for T in T_values if T > 1]) * len(loss_modes) * len(accum_modes)
    runs_per_combo = n_T1 + n_Tgt1
    total = len(args.distributions) * len(args.proj_dims) * len(seeds) * runs_per_combo

    print(f"Exp 1.2: Standard vs Pooled vs Big-Batch Accumulation")
    print(f"  distributions={args.distributions}")
    print(f"  proj_dims={args.proj_dims}, T_values={T_values}")
    print(f"  batch_size={args.batch_size}, seeds={seeds}")
    print(f"  total_runs~={total}, device={device}")
    print()

    # Build flat config list (excluding T=1 pooled/bigbatch duplicates)
    configs = []
    for dist in args.distributions:
        for M in args.proj_dims:
            for T in T_values:
                for loss_mode in loss_modes:
                    for accum_mode in accum_modes:
                        if T == 1 and accum_mode != "standard":
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
        elif accum_mode == "pooled":
            result = train_one_pooled(
                seed, args, loss_mode, M, dist, T, device)
        else:  # bigbatch
            result = train_one_bigbatch(
                seed, args, loss_mode, M, dist, T, device)

        elapsed = time.perf_counter() - t0
        all_runs[(dist, M, method_key, T, seed)] = result
        print(f"W1={result[6]['w1']:.4f}  time={elapsed:.1f}s")

        torch.save(result[7], model_path)

    # T=1 copies: pooled == standard == bigbatch (all reduce to a single
    # forward+backward through BS samples)
    for dist in args.distributions:
        for M in args.proj_dims:
            if 1 in T_values:
                for loss_mode in loss_modes:
                    std_key = (dist, M, f"{loss_mode}_standard", 1)
                    for variant in ("pooled", "bigbatch"):
                        var_key = (dist, M, f"{loss_mode}_{variant}", 1)
                        for seed in seeds:
                            sk = (*std_key, seed)
                            vk = (*var_key, seed)
                            if sk in all_runs:
                                all_runs[vk] = all_runs[sk]

    # --- Summary ---
    summary_lines = [
        f"Exp 1.2: Standard vs Pooled vs Big-Batch Accumulation",
        f"d={args.input_dim}, K={args.num_points}, micro_bs={args.batch_size}, "
        f"epochs={args.epochs} (Option B), seeds={len(seeds)}",
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
