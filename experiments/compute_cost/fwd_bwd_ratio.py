"""Measure cost of training steps with varying no-grad invariance views.

Benchmarks the *actual* training step cost (augmentation + encoder forward +
backward + optimizer) for different combinations of:
  - N grad views (num_aug_views)
  - M no-grad invariance views (num_inv_nograd_views)

This directly answers: "how much overhead do M extra no-grad views add to
a real step, and is it cheaper than adding more grad views?"

Output columns:
  - base: standard step with N grad views, M=0 nograd
  - nograd_M: step with N grad views + M nograd inv views
  - grad_N: step with N+M grad views (the "oracle" cost of using all views
    with gradients)

Usage:
    python fwd_bwd_ratio.py
    python fwd_bwd_ratio.py --backbone convnextv2_pico --batch-size 32
    python fwd_bwd_ratio.py --grad-views 1 2 --nograd-views 0 2 4 8
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAIN_DIR = Path(__file__).resolve().parents[1] / "pretrain"
for p in (str(PRETRAIN_DIR), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from configs import Config
from data import get_dataloaders
from losses import build_regularizer
from models import LeJEPAEncoder, LinearProbe
from scheduler import make_scheduler
from train_loops import _train_step


def bench(fn, n=15, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.time()
        fn()
        torch.cuda.synchronize()
        times.append((time.time() - t) * 1000)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return statistics.median(times), peak_gb


def make_step_fn(cfg, device):
    """Build a callable that runs one _train_step and returns nothing."""
    train_ds, _, gpu_aug = get_dataloaders(cfg, device)
    encoder = LeJEPAEncoder(cfg).to(device)
    probe_dim = encoder.hidden_dim if cfg.probe_on_emb else cfg.proj_dim
    probe = LinearProbe(probe_dim, cfg.num_classes).to(device)
    reg_fn = build_regularizer(cfg, device)

    total_steps = 1000
    enc_opt = torch.optim.AdamW(
        encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    probe_opt = torch.optim.AdamW(
        probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_wd)
    enc_sched = make_scheduler(enc_opt, total_steps, cfg.lr)
    probe_sched = make_scheduler(probe_opt, total_steps, cfg.probe_lr)

    gen = torch.Generator(device=device).manual_seed(42)

    def step():
        images, labels = train_ds.sample_batch(cfg.batch_size, gen)
        _train_step(images, labels, encoder, probe, reg_fn, gpu_aug, cfg,
                    enc_opt, probe_opt, enc_sched, probe_sched)

    # Return references so they don't get GC'd
    return step, (train_ds, gpu_aug, encoder, probe, reg_fn,
                  enc_opt, probe_opt, enc_sched, probe_sched)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="convnextv2_pico")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--grad-views", type=int, nargs="+", default=[1, 2],
                        help="N: number of augmented grad views")
    parser.add_argument("--nograd-views", type=int, nargs="+",
                        default=[0, 1, 2, 4, 8, 16],
                        help="M: number of no-grad invariance views")
    parser.add_argument("--n-iters", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Backbone: {args.backbone}, resolution: {args.resolution}, "
          f"BS: {args.batch_size}")
    print(f"Iters per cell: {args.n_iters} (median)")
    print()

    # Header
    print(f"{'N_grad':>6} {'M_nograd':>8} {'total_views':>11} "
          f"{'step_ms':>8} {'peak_GB':>8} {'overhead':>8} {'views_per_ms':>11}")
    print("-" * 72)

    # Collect baseline (M=0) times per N for computing overhead
    baselines = {}

    for n_grad in args.grad_views:
        for m_nograd in args.nograd_views:
            torch.cuda.empty_cache()
            total_views = 1 + n_grad + m_nograd  # 1 orig + N aug + M nograd

            cfg = Config(
                dataset="cifar100",
                data_dir=args.data_dir,
                encoder_scale=args.backbone,
                crop_size=args.resolution,
                batch_size=args.batch_size,
                num_aug_views=n_grad,
                num_inv_nograd_views=m_nograd,
            )

            try:
                step_fn, refs = make_step_fn(cfg, device)
                t_ms, peak = bench(step_fn, n=args.n_iters)

                if m_nograd == 0:
                    baselines[n_grad] = t_ms

                base = baselines.get(n_grad, t_ms)
                overhead = (t_ms - base) / base if base > 0 else 0.0
                vpm = total_views / t_ms if t_ms > 0 else float("nan")

                print(f"{n_grad:6d} {m_nograd:8d} {total_views:11d} "
                      f"{t_ms:8.1f} {peak:8.2f} {overhead:>7.0%} {vpm:11.3f}")

                del step_fn, refs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{n_grad:6d} {m_nograd:8d} {total_views:11d} "
                      f"{'OOM':>8} {'':>8} {'':>8} {'':>11}")

    # Also show the "all grad" oracle cost for comparison
    print()
    print("--- Oracle: all views with gradients (no nograd trick) ---")
    print(f"{'N_grad':>6} {'step_ms':>8} {'peak_GB':>8}")
    print("-" * 30)
    for total_grad in sorted(set(
        1 + n + m for n in args.grad_views for m in args.nograd_views if m > 0
    )):
        torch.cuda.empty_cache()
        cfg = Config(
            dataset="cifar100",
            data_dir=args.data_dir,
            encoder_scale=args.backbone,
            crop_size=args.resolution,
            batch_size=args.batch_size,
            num_aug_views=total_grad - 1,  # -1 because orig is always there
            num_inv_nograd_views=0,
        )
        try:
            step_fn, refs = make_step_fn(cfg, device)
            t_ms, peak = bench(step_fn, n=args.n_iters)
            print(f"{total_grad:6d} {t_ms:8.1f} {peak:8.2f}")
            del step_fn, refs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{total_grad:6d} {'OOM':>8}")


if __name__ == "__main__":
    main()
