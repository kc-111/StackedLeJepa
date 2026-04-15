"""Measure pooled vs standard step cost across batch sizes.

Compares four step variants:
  1. std                 — 1F + 1B on BS samples (the LeJEPA baseline)
  2. pool_nograd         — 1F no-grad on (T-1)*BS in ONE call + 1F + 1B on BS
                           (our production implementation)
  3. pool_nograd_chunked — 1F no-grad in (T-1) sub-calls of BS each + 1F + 1B on BS
                           (supplementary: lower memory but only competitive at
                            specific BS — kept for the supplementary discussion)
  4. fullgrad            — 1F + 1B on T*BS samples — the "oracle" cost of
                           getting the same regularizer information as pool
                           if you ALSO computed gradients on every sample.
                           The efficiency claim: pool_nograd should be roughly
                           (T+2)/(3T) ≈ 0.42x the wall clock of fullgrad at
                           T=8, while delivering the same regularizer-accuracy
                           (since both see T*BS samples through the regularizer).
                           OOMs at large BS — gracefully reported as such.

Each cell INCLUDES the GPU augmentation cost (sample_batch + gpu_aug).
Each variant is timed N times → mean ± std reported.

Output: experiments/compute_cost/results/pooled_overhead_<gpu>.csv

Usage:
    python pooled_overhead.py --backbone convnextv2_nano --resolution 128
    python pooled_overhead.py --backbone tiny --resolution 128 --num-aug 2

Note: BN-based backbones (resnet18/34/50) are NOT pooled-safe — see
"Note on BatchNorm" in experiments/EXPERIMENT_PLAN.md. Defaults to the
LayerNorm-only convnextv2_nano.
"""

import argparse
import csv
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
from models import LeJEPAEncoder


def gpu_name():
    return torch.cuda.get_device_name().replace(" ", "_")


def time_step_with_std(step_fn, n=15, warmup=8):
    """Run step_fn many times, return (mean_ms, std_ms, peak_gb).

    Resets peak memory stats AFTER warmup so the peak reflects only the
    timed steps.
    """
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.time()
        step_fn()
        torch.cuda.synchronize()
        times.append((time.time() - t) * 1000)
    peak = torch.cuda.max_memory_allocated() / 1e9
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, peak


def fmt(mean, std):
    if mean != mean:  # NaN
        return "OOM"
    return f"{mean:.2f} ± {std:.2f} ms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="convnextv2_nano",
                        help="encoder_scale: convnextv2_atto/nano, convnext_tiny, "
                             "tiny (ViT), ... BN backbones (resnet18/34/50) "
                             "are NOT pooled-safe — see EXPERIMENT_PLAN.md.")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--T", type=int, default=8, help="Pool depth")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[8, 16, 32, 48, 64])
    parser.add_argument("--num-aug", type=int, default=1,
                        help="Number of augmented views per sample (default 1).")
    parser.add_argument("--n-iters", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda")
    T = args.T
    V = args.num_aug
    per_sample_views = 1 + V  # 1 orig + V aug

    cfg_template = dict(
        dataset="cifar100",
        data_dir=args.data_dir,
        encoder_scale=args.backbone,
        crop_size=args.resolution,
        num_aug_views=V,
    )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Backbone: {args.backbone}, resolution: {args.resolution}², T={T}")
    print(f"Views: 1 orig + {V} aug, per-batch image-forwards = BS × {per_sample_views}")
    print(f"Each step INCLUDES sample_batch + gpu_aug")
    print(f"Iters per cell: {args.n_iters} (mean ± std across runs)")
    print()

    rows = []
    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        cfg = Config(batch_size=bs, **cfg_template)
        train_ds, _, gpu_aug = get_dataloaders(cfg, device)
        encoder = LeJEPAEncoder(cfg).to(device)
        opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
        gen = torch.Generator(device=device).manual_seed(42)

        def get_view_tensors(n_samples):
            """Return (1+V)*n_samples views as a single flat tensor."""
            imgs, _ = train_ds.sample_batch(n_samples, gen)
            x = imgs.float() / 255.0
            orig, aug = gpu_aug(x)
            return torch.cat([orig, aug.reshape(-1, *aug.shape[2:])], dim=0)

        # 1) Standard step
        def std_step():
            opt.zero_grad(set_to_none=True)
            v = get_view_tensors(bs)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb, proj = encoder(v)
                loss = (proj ** 2).mean() + (emb ** 2).mean() * 0.001
            loss.backward()
            opt.step()

        # 2) pool_nograd: pooled, ONE no-grad forward on (T-1)*BS samples
        def pool_nograd_step():
            opt.zero_grad(set_to_none=True)
            v = get_view_tensors(bs)
            ng_v = get_view_tensors((T - 1) * bs)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, ng_proj = encoder(ng_v)
            ng_proj = ng_proj.detach()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb, proj = encoder(v)
                loss = (proj ** 2).mean() + (ng_proj ** 2).mean() * 0.0
                loss = loss + (emb ** 2).mean() * 0.001
            loss.backward()
            opt.step()

        # 3) pool_nograd_chunked: SUPPLEMENTARY. Split the no-grad pass into
        #    (T-1) sub-batches of BS each. Lower peak memory (cuDNN intermediate
        #    buffers sized for BS, not (T-1)*BS). NOT used in production
        #    because we don't need the memory savings — peak is ~2GB on 24GB
        #    available — and chunking is only competitive at one specific BS
        #    (see EXPERIMENT_PLAN.md "Supplementary: chunked Pass 1").
        #    (Earlier drafts also cited BN-stat aggregation across chunks as
        #    a complication; that's now moot since we use LN-only backbones.)

        # 4) fullgrad: 1F + 1B on T*BS samples. The "oracle" baseline for the
        #    efficiency claim — what you'd pay to get the same regularizer
        #    information as pool_nograd IF you also wanted gradients on every
        #    sample (which you don't need). Pool should be ~0.42x its wall
        #    clock at T=8 while giving the same regularizer accuracy.
        def fullgrad_step():
            opt.zero_grad(set_to_none=True)
            v = get_view_tensors(T * bs)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb, proj = encoder(v)
                loss = (proj ** 2).mean() + (emb ** 2).mean() * 0.001
            loss.backward()
            opt.step()

        def pool_nograd_chunked_step():
            opt.zero_grad(set_to_none=True)
            v = get_view_tensors(bs)
            ng_proj_chunks = []
            for _ in range(T - 1):
                ng_v_chunk = get_view_tensors(bs)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, chunk_proj = encoder(ng_v_chunk)
                ng_proj_chunks.append(chunk_proj.detach())
            ng_proj = torch.cat(ng_proj_chunks, dim=0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb, proj = encoder(v)
                loss = (proj ** 2).mean() + (ng_proj ** 2).mean() * 0.0
                loss = loss + (emb ** 2).mean() * 0.001
            loss.backward()
            opt.step()

        def safe(fn, n=args.n_iters):
            try:
                return time_step_with_std(fn, n=n)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return float("nan"), float("nan"), float("nan")

        mean_std, std_std, peak_std = safe(std_step)
        if mean_std != mean_std:
            print(f"BS={bs}: standard OOM — skipping")
            del encoder, opt, train_ds, gpu_aug
            continue
        mean_pn, std_pn, peak_pn = safe(pool_nograd_step)
        mean_pc, std_pc, peak_pc = safe(pool_nograd_chunked_step)
        mean_fg, std_fg, peak_fg = safe(fullgrad_step)

        ratio_pn = mean_pn / mean_std if mean_std else float("nan")
        ratio_pc = mean_pc / mean_std if mean_std else float("nan")
        # Pool / fullgrad efficiency: how much cheaper is pool than the
        # equivalent-quality oracle? Lower is better; theory predicts ~0.42 at T=8.
        ratio_pn_fg = (mean_pn / mean_fg) if (mean_fg == mean_fg and mean_fg) else float("nan")
        ratio_fg_std = (mean_fg / mean_std) if (mean_fg == mean_fg and mean_std) else float("nan")

        print(f"BS={bs:4d}  "
              f"std={fmt(mean_std, std_std)} ({peak_std:.2f}GB)")
        print(f"        pool_nograd={fmt(mean_pn, std_pn)} ({peak_pn:.2f}GB)  "
              f"ratio={ratio_pn:.2f}x")
        print(f"        pool_nograd_chunked={fmt(mean_pc, std_pc)} ({peak_pc:.2f}GB)  "
              f"ratio={ratio_pc:.2f}x")
        if mean_fg == mean_fg:
            print(f"        fullgrad (T*BS={T*bs})={fmt(mean_fg, std_fg)} ({peak_fg:.2f}GB)  "
                  f"vs std={ratio_fg_std:.2f}x  pool/fullgrad={ratio_pn_fg:.2f}x")
        else:
            print(f"        fullgrad (T*BS={T*bs}) = OOM")

        rows.append(dict(
            bs=bs,
            mean_std=mean_std, std_std=std_std, peak_std=peak_std,
            mean_pool=mean_pn, std_pool=std_pn, peak_pool=peak_pn,
            mean_pool_chunked=mean_pc, std_pool_chunked=std_pc,
            peak_pool_chunked=peak_pc,
            mean_fullgrad=mean_fg, std_fullgrad=std_fg, peak_fullgrad=peak_fg,
            ratio=ratio_pn, ratio_chunked=ratio_pc,
            ratio_pool_vs_fullgrad=ratio_pn_fg,
            ratio_fullgrad_vs_std=ratio_fg_std,
        ))
        del encoder, opt, train_ds, gpu_aug

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_V{V}_T{T}"
    out_path = out_dir / f"pooled_overhead_{gpu_name()}_{args.backbone}_{args.resolution}{suffix}.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bs",
            "std_mean_ms", "std_std_ms", "std_peak_gb",
            "pool_nograd_mean_ms", "pool_nograd_std_ms", "pool_nograd_peak_gb",
            "pool_chunked_mean_ms", "pool_chunked_std_ms", "pool_chunked_peak_gb",
            "fullgrad_mean_ms", "fullgrad_std_ms", "fullgrad_peak_gb",
            "ratio_pool_vs_std", "ratio_chunked_vs_std",
            "ratio_pool_vs_fullgrad", "ratio_fullgrad_vs_std",
        ])
        for r in rows:
            writer.writerow([
                r["bs"],
                r["mean_std"], r["std_std"], r["peak_std"],
                r["mean_pool"], r["std_pool"], r["peak_pool"],
                r["mean_pool_chunked"], r["std_pool_chunked"], r["peak_pool_chunked"],
                r["mean_fullgrad"], r["std_fullgrad"], r["peak_fullgrad"],
                r["ratio"], r["ratio_chunked"],
                r["ratio_pool_vs_fullgrad"], r["ratio_fullgrad_vs_std"],
            ])
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
