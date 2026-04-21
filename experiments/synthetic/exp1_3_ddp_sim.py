"""
Experiment 1.3: DDP Simulation — Local Shards vs Pooled

Simulates multi-GPU DDP by splitting the batch into D shards on a single
device. Shows that local-shard-only regularization degrades with more
devices, while global operations and pooled accumulation do not.

8 modes:
    local_sigreg         — SIGReg on each shard independently, average gradients
    global_sigreg        — all-reduce ECF across shards (= SIGReg on full batch)
    pooled_sigreg        — T no-grad + 1 grad per device, local pooled SIGReg
    pooled_global_sigreg — local accumulation + all-gather grad embeddings, SIGReg on union
    local_w1             — W1 on each shard independently, average gradients
    global_w1            — differentiable all-gather, sort all embeddings locally
    pooled_w1            — T no-grad + 1 grad per device, local pooled W1
    pooled_global_w1     — local accumulation + all-gather grad embeddings, W1 on union

Fairness: Option A (grad-compute-matched). Each opt step consumes
D*BS_per_device grad samples; pooled no-grad context is "free". An epoch is
K // (D*BS_per_device) opt steps. See experiments/synthetic/FAIRNESS.md.

Run:
    python experiments/synthetic/exp1_3_ddp_sim.py
    python experiments/synthetic/exp1_3_ddp_sim.py --distributions blobs --num-devices 1 4 8 --n-seeds 1 --epochs 5
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
    SlicedW1Loss, SIGRegLoss, SIGReg, PooledSlicedLoss,
    DeepMLP, generate_data, make_fixed_projection, epoch_iter, GENERATORS,
    eval_w1, evaluate_full,
)
from sliced_gauss_reg.losses import _random_unit_directions, _gaussian_quantiles


DDP_MODES = [
    "local_sigreg", "global_sigreg", "pooled_sigreg", "pooled_global_sigreg",
    "local_w1", "global_w1", "pooled_w1", "pooled_global_w1",
]

MODE_STYLES = {
    "local_sigreg":         {"color": "tab:orange", "ls": "--", "marker": "v",
                             "label": "SIGReg local"},
    "global_sigreg":        {"color": "tab:orange", "ls": "-",  "marker": "^",
                             "label": "SIGReg global (all-reduce ECF)"},
    "pooled_sigreg":        {"color": "tab:orange", "ls": "-",  "marker": "D",
                             "label": "SIGReg pooled local"},
    "pooled_global_sigreg": {"color": "tab:orange", "ls": "-",  "marker": "p",
                             "label": "SIGReg pooled + all-reduce"},
    "local_w1":             {"color": "tab:blue",   "ls": "--", "marker": "v",
                             "label": "W1 local"},
    "global_w1":            {"color": "tab:blue",   "ls": "-",  "marker": "^",
                             "label": "W1 global (all-gather)"},
    "pooled_w1":            {"color": "tab:red",    "ls": "-",  "marker": "D",
                             "label": "W1 pooled local"},
    "pooled_global_w1":     {"color": "tab:green",  "ls": "-",  "marker": "p",
                             "label": "W1 pooled + all-gather"},
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _setup(seed, args, distribution, device):
    D_dim = args.input_dim
    torch.manual_seed(seed)
    data = generate_data(distribution, args.num_points, D_dim)
    W = make_fixed_projection(D_dim, args.proj_dim, seed=seed + 100).to(device)
    projected = data.to(device) @ W

    src_np = data.numpy()
    colors = np.arctan2(src_np[:, 1] if D_dim > 1 else np.zeros(len(src_np)),
                        src_np[:, 0])
    colors = (colors + np.pi) / (2 * np.pi)

    torch.manual_seed(seed + 200)
    mlp = DeepMLP(args.proj_dim, args.hidden_dim, D_dim, depth=args.depth).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    mlp.eval()
    with torch.no_grad():
        initial_out = mlp(projected).cpu().numpy()
    mlp.train()

    return projected, mlp, opt, src_np, colors, initial_out


def train_one(seed, args, ddp_mode, distribution, num_devices, device):
    """Train one run with simulated DDP.

    Option A (grad-compute-matched): each opt step consumes total_bs =
    D * shard_size grad samples drawn from the epoch permutation. Pooled
    modes additionally draw (T_accum-1) no-grad batches per device per step,
    but those are "free context" and not counted toward the epoch.
    """
    projected, mlp, opt, src_np, colors, initial_out = \
        _setup(seed, args, distribution, device)
    D_dim = args.input_dim
    shard_size = args.batch_size  # per-device batch size
    total_bs = shard_size * num_devices
    K = args.num_points
    T_accum = args.accum_steps  # local accumulation steps per device
    if total_bs > K:
        raise ValueError(f"total_bs={total_bs} exceeds dataset K={K}")

    w1_fn = SlicedW1Loss(num_proj=args.num_proj).to(device)
    sigreg_fn = SIGRegLoss(knots=args.knots, num_proj=args.num_proj).to(device)
    sigreg_mod = SIGReg(knots=args.knots, num_proj=args.num_proj).to(device)

    # For pooled modes, each simulated device has its own accumulator
    # Use separate RNG per device to ensure different no-grad batches
    device_rngs = [torch.Generator(device=device).manual_seed(seed + 300 + d)
                   for d in range(num_devices)]
    epoch_gen = torch.Generator(device=device).manual_seed(seed + 400)

    eval_steps, eval_w1s = [], []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
      for grad_chunk in epoch_iter(K, total_bs, device, epoch_gen):
        mlp.train()

        # Split the epoch chunk into D per-device shards (= effective batch).
        idx = grad_chunk
        shards = [idx[i * shard_size:(i + 1) * shard_size]
                  for i in range(num_devices)]

        if ddp_mode == "local_sigreg":
            # Each device computes SIGReg on its shard, average gradients
            opt.zero_grad()
            for shard_idx in shards:
                shard_loss = sigreg_fn(mlp(projected[shard_idx])) / num_devices
                shard_loss.backward()
            opt.step()

        elif ddp_mode == "global_sigreg":
            # All-reduce ECF: equivalent to SIGReg on full batch.
            # All samples contribute gradient through the all-reduced cos/sin.
            # Divide by D to match local_sigreg gradient scale: SIGReg's
            # *n statistic amplifies with the all-gathered sample count.
            output = mlp(projected[idx])
            loss = sigreg_fn(output) / num_devices
            opt.zero_grad()
            loss.backward()
            opt.step()

        elif ddp_mode == "pooled_sigreg":
            # Each device pools locally with PooledSlicedLoss (sigreg).
            # No communication between devices.
            opt.zero_grad()
            for d in range(num_devices):
                accum = PooledSlicedLoss(
                    accum_steps=T_accum - 1, num_proj=args.num_proj,
                    mode="sigreg", sigreg=sigreg_mod)
                with torch.no_grad():
                    for _ in range(T_accum - 1):
                        local_idx = torch.randint(
                            0, K, (shard_size,), device=device,
                            generator=device_rngs[d])
                        accum.accum_step(mlp(projected[local_idx]))
                local_idx = torch.randint(
                    0, K, (shard_size,), device=device,
                    generator=device_rngs[d])
                local_loss = accum.grad_step(mlp(projected[local_idx])) / num_devices
                local_loss.backward()
            opt.step()

        elif ddp_mode == "pooled_global_sigreg":
            # Each device accumulates locally, then all-reduce the ECF
            # from all devices' grad batches + local accumulated.
            # Since SIGReg's ECF is additive, this is equivalent to
            # computing SIGReg on the union of all embeddings.
            opt.zero_grad()

            # Phase 1: each device accumulates locally
            device_accumulated = []
            with torch.no_grad():
                for d in range(num_devices):
                    acc = []
                    for _ in range(T_accum - 1):
                        local_idx = torch.randint(
                            0, K, (shard_size,), device=device,
                            generator=device_rngs[d])
                        acc.append(mlp(projected[local_idx]).detach())
                    device_accumulated.append(acc)

            # Phase 2: each device does 1 grad forward pass
            grad_embeddings = []
            for d in range(num_devices):
                local_idx = torch.randint(
                    0, K, (shard_size,), device=device,
                    generator=device_rngs[d])
                grad_embeddings.append(mlp(projected[local_idx]))

            # Phase 3: all-gather grad embeddings
            all_grad = torch.cat(grad_embeddings, dim=0)

            # Phase 4: each device's view = all_grad + local_acc → SIGReg
            # Divide by D² (D for DDP averaging, D for all-gather
            # amplification) to match local/pooled gradient scale.
            total_loss = torch.tensor(0.0, device=device)
            for d in range(num_devices):
                if device_accumulated[d]:
                    local_acc = torch.cat(device_accumulated[d], dim=0)
                    device_emb = torch.cat([all_grad, local_acc], dim=0)
                else:
                    device_emb = all_grad
                device_loss = sigreg_mod(device_emb) / (num_devices * num_devices)
                total_loss = total_loss + device_loss

            total_loss.backward()
            opt.step()

        elif ddp_mode == "local_w1":
            # Each device computes W1 on its shard, average gradients
            opt.zero_grad()
            for shard_idx in shards:
                shard_loss = w1_fn(mlp(projected[shard_idx])) / num_devices
                shard_loss.backward()
            opt.step()

        elif ddp_mode == "global_w1":
            # Differentiable all-gather: each device sends embeddings,
            # all sort locally. All samples carry gradient.
            output = mlp(projected[idx])
            loss = w1_fn(output)
            opt.zero_grad()
            loss.backward()
            opt.step()

        elif ddp_mode == "pooled_w1":
            # Each device does T_accum-1 no-grad passes + 1 grad pass
            # on its own shard, pooled locally. No communication.
            # Each device's CDF = T_accum * shard_size
            opt.zero_grad()
            for d in range(num_devices):
                # Accumulate locally for this device
                detached = []
                with torch.no_grad():
                    for _ in range(T_accum - 1):
                        local_idx = torch.randint(
                            0, K, (shard_size,), device=device,
                            generator=device_rngs[d])
                        detached.append(mlp(projected[local_idx]).detach())

                # Grad batch for this device
                local_idx = torch.randint(
                    0, K, (shard_size,), device=device,
                    generator=device_rngs[d])
                live = mlp(projected[local_idx])

                # Pool and compute W1 locally
                all_emb = torch.cat([live] + detached, dim=0)
                n_total = all_emb.shape[0]
                A = _random_unit_directions(D_dim, args.num_proj, device, all_emb.dtype)
                proj_sorted = torch.sort(all_emb @ A, dim=0).values
                ref = _gaussian_quantiles(n_total, device, all_emb.dtype).unsqueeze(1)
                local_loss = (proj_sorted - ref).abs().mean()
                # Gradient dilution + device averaging
                local_loss = local_loss * (n_total / shard_size) / num_devices
                local_loss.backward()

            opt.step()

        elif ddp_mode == "pooled_global_w1":
            # Each device accumulates locally (no-grad), then at grad step
            # all devices all-gather their grad embeddings.
            # Each device's CDF = its own local_accumulated (detached)
            #                    + all D devices' grad embeddings (with grad)
            # Simulate all D devices, average their gradients (= DDP all-reduce).
            opt.zero_grad()

            # Phase 1: each device accumulates T_accum-1 no-grad batches
            device_accumulated = []  # list of D lists of detached tensors
            with torch.no_grad():
                for d in range(num_devices):
                    acc = []
                    for _ in range(T_accum - 1):
                        local_idx = torch.randint(
                            0, K, (shard_size,), device=device,
                            generator=device_rngs[d])
                        acc.append(mlp(projected[local_idx]).detach())
                    device_accumulated.append(acc)

            # Phase 2: each device does 1 grad forward pass
            grad_embeddings = []
            for d in range(num_devices):
                local_idx = torch.randint(
                    0, K, (shard_size,), device=device,
                    generator=device_rngs[d])
                grad_embeddings.append(mlp(projected[local_idx]))

            # Phase 3: all-gather grad embeddings (differentiable)
            all_grad = torch.cat(grad_embeddings, dim=0)  # (D*shard_size,) all with grad

            # Phase 4: each device computes loss from its own view.
            # Sum losses then backward once (= DDP gradient all-reduce).
            total_loss = torch.tensor(0.0, device=device)
            for d in range(num_devices):
                if device_accumulated[d]:
                    local_acc = torch.cat(device_accumulated[d], dim=0)
                    device_emb = torch.cat([all_grad, local_acc], dim=0)
                else:
                    device_emb = all_grad

                n_total = device_emb.shape[0]
                n_grad = all_grad.shape[0]
                A = _random_unit_directions(D_dim, args.num_proj, device, device_emb.dtype)
                proj_sorted = torch.sort(device_emb @ A, dim=0).values
                ref = _gaussian_quantiles(n_total, device, device_emb.dtype).unsqueeze(1)
                device_loss = (proj_sorted - ref).abs().mean()
                # Compensation + DDP averaging
                total_loss = total_loss + device_loss * (n_total / n_grad) / num_devices

            total_loss.backward()
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

def plot_ddp_degradation(all_runs, dist, D_values, seeds, save_path):
    """Key figure: final eval_w1 vs num_devices for all modes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for mode, style in MODE_STYLES.items():
        means, stds = [], []
        for D in D_values:
            vals = []
            for seed in seeds:
                result = all_runs.get((dist, mode, D, seed))
                if result is not None:
                    vals.append(result[6]["w1"])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0.0)

        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(D_values, means, yerr=stds,
                     fmt=style["marker"],
                     color=style["color"], linestyle=style["ls"],
                     label=style["label"], linewidth=2, markersize=6,
                     capsize=3)

    ax.set_xlabel("Simulated Devices D", fontsize=12)
    ax.set_ylabel("Final Sliced W1 to N(0,1)", fontsize=11)
    ax.set_title(f"{dist}, BS/device={64}, M=8", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(D_values)
    ax.set_xticklabels([str(d) for d in D_values])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 1.3: DDP Simulation — Local Shards vs Pooled"
    )
    p.add_argument("--distributions", nargs="+",
                    default=list(GENERATORS.keys()),
                    choices=list(GENERATORS.keys()))
    p.add_argument("--input-dim", type=int, default=4)
    p.add_argument("--proj-dim", type=int, default=8)
    p.add_argument("--num-points", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=32,
                    help="Batch size per device (total = D * batch_size)")
    p.add_argument("--accum-steps", type=int, default=8,
                    help="T: local accumulation steps per device (for pooled modes)")
    p.add_argument("--num-devices", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--epochs", type=int, default=80,
                    help="Grad-compute-matched epochs (Option A): each epoch is "
                         "K // (D*BS_per_device) opt steps. See FAIRNESS.md.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--save-dir", default="results/exp1_3_ddp_sim")
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
    D_values = sorted(args.num_devices)

    # At D=1, local==global and pooled_local==pooled_global (no one to communicate with).
    # Only these pairs are redundant; local != pooled even at D=1.
    D1_EQUIVALENCES = {
        "global_sigreg": "local_sigreg",
        "global_w1": "local_w1",
        "pooled_global_sigreg": "pooled_sigreg",
        "pooled_global_w1": "pooled_w1",
    }

    total = len(args.distributions) * len(D_values) * len(DDP_MODES) * len(seeds)
    if 1 in D_values:
        total -= len(args.distributions) * len(D1_EQUIVALENCES) * len(seeds)

    print(f"Exp 1.3: DDP Simulation")
    print(f"  distributions={args.distributions}")
    print(f"  num_devices={D_values}, batch_size_per_device={args.batch_size}")
    print(f"  accum_steps={args.accum_steps} (for pooled modes)")
    print(f"  seeds={seeds}, total_runs~={total}, device={device}")
    print()

    # Build flat config list for sharding
    # At D=1, skip modes that are equivalent to another mode
    configs = []
    for dist in args.distributions:
        for D in D_values:
            for mode in DDP_MODES:
                if D == 1 and mode in D1_EQUIVALENCES:
                    continue  # handled via copy below
                for seed in seeds:
                    configs.append((dist, mode, D, seed))

    # Run configs assigned to this worker
    all_runs = {}
    for cfg_idx, (dist, mode, D, seed) in enumerate(configs):
        model_path = os.path.join(
            args.save_dir, f"model_{dist}_{mode}_D{D}_{seed}.pt")

        if os.path.exists(model_path):
            print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, D={D}, "
                  f"mode={mode}, seed={seed} SKIP (exists)")
            state = torch.load(model_path, map_location=device,
                               weights_only=True)
            proj, mlp, _, src_np, colors, initial_out = \
                _setup(seed, args, dist, device)
            mlp.load_state_dict(state)
            mlp.eval()
            with torch.no_grad():
                final_out = mlp(proj).cpu().numpy()
            metrics = evaluate_full(final_out)
            all_runs[(dist, mode, D, seed)] = (
                [], [], final_out, initial_out, src_np,
                colors, metrics, state)
            continue

        # Shard: only this worker trains new configs
        if cfg_idx % args.num_workers != args.worker_id:
            continue

        print(f"[{cfg_idx+1}/{len(configs)}] dist={dist}, D={D}, "
              f"mode={mode}, seed={seed}", end=" ", flush=True)
        t0 = time.perf_counter()
        result = train_one(seed, args, mode, dist, D, device)
        elapsed = time.perf_counter() - t0
        all_runs[(dist, mode, D, seed)] = result
        print(f"W1={result[6]['w1']:.4f}  time={elapsed:.1f}s")

        torch.save(result[7], model_path)

    # D=1 copies: global == local, pooled_global == pooled_local
    if 1 in D_values:
        for dist in args.distributions:
            for mode, source in D1_EQUIVALENCES.items():
                for seed in seeds:
                    src_key = (dist, source, 1, seed)
                    if src_key in all_runs:
                        all_runs[(dist, mode, 1, seed)] = all_runs[src_key]

    # --- Plots ---
    for dist in args.distributions:
        plot_ddp_degradation(
            all_runs, dist, D_values, seeds,
            os.path.join(plots_dir, f"ddp_degradation_{dist}.png"))

    # --- Summary ---
    summary_lines = [
        f"Exp 1.3: DDP Simulation",
        f"d={args.input_dim}, M={args.proj_dim}, BS_per_device={args.batch_size}, "
        f"accum_steps={args.accum_steps}, "
        f"K={args.num_points}, epochs={args.epochs} (Option A), seeds={len(seeds)}",
        "",
        f"{'Distribution':<18} {'Mode':<22} {'D':>3}  {'W1':>14}",
        "-" * 65,
    ]

    json_out = {}
    for dist in args.distributions:
        for mode in DDP_MODES:
            for D in D_values:
                vals = []
                for seed in seeds:
                    result = all_runs.get((dist, mode, D, seed))
                    if result is not None:
                        vals.append(result[6])
                if not vals:
                    continue

                key = f"{dist}_{mode}_D{D}"
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
                    f"{dist:<18} {mode:<22} {D:>3}  "
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
