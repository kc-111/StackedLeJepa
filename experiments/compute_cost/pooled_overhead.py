"""Measure pooled vs standard step cost across batch sizes.

Compares three step variants:
  1. std                 — 1F + 1B on BS samples (the LeJEPA baseline)
  2. pool_nograd         — 1F no-grad on (T-1)*BS in ONE call + 1F + 1B on BS
                           (our production implementation)
  3. pool_nograd_chunked — 1F no-grad in (T-1) sub-calls of BS each + 1F + 1B on BS
                           (supplementary: lower memory but only competitive at
                            specific BS — kept for the supplementary discussion)

Each cell INCLUDES the GPU augmentation cost (sample_batch + gpu_aug).
Each variant is timed N times → mean ± std reported.

Output: experiments/compute_cost/results/pooled_overhead_<gpu>.md

Usage:
    python pooled_overhead.py --backbone resnet18 --resolution 128
    python pooled_overhead.py --backbone tiny --resolution 128 --multicrop
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
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--T", type=int, default=8, help="Pool depth")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[8, 16, 32, 64])
    parser.add_argument("--multicrop", action="store_true",
                        help="Enable multi-crop locals (V_g=2 + V_l=6). "
                             "Requires a ViT backbone.")
    parser.add_argument("--n-iters", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda")
    T = args.T
    use_locals = args.multicrop

    cfg_template = dict(
        dataset="cifar100",
        data_dir=args.data_dir,
        encoder_scale=args.backbone,
        global_crop_size=args.resolution,
        local_crop_size=max(args.resolution // 2, 56),
        num_global_views=2,
        num_local_views=6 if use_locals else 0,
    )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Backbone: {args.backbone}, resolution: {args.resolution}², T={T}")
    print(f"Views: V_g=2{' + V_l=6' if use_locals else ''}, "
          f"per-batch image-forwards = BS × {2 + (6 if use_locals else 0)}")
    print(f"Each step INCLUDES sample_batch + gpu_aug (multi-crop view generation)")
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
            imgs, _ = train_ds.sample_batch(n_samples, gen)
            x = imgs.float() / 255.0
            g, l = gpu_aug(x)
            g_flat = g.reshape(-1, *g.shape[2:])
            l_flat = l.reshape(-1, *l.shape[2:]) if l is not None else None
            return g_flat, l_flat

        # 1) Standard step
        def std_step():
            opt.zero_grad(set_to_none=True)
            g, l = get_view_tensors(bs)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb_g, proj_g = encoder(g)
                loss = (proj_g ** 2).mean() + (emb_g ** 2).mean() * 0.001
                if l is not None:
                    emb_l, proj_l = encoder(l)
                    loss = loss + (proj_l ** 2).mean() + (emb_l ** 2).mean() * 0.001
            loss.backward()
            opt.step()

        # 2) pool_nograd: pooled, ONE no-grad forward on (T-1)*BS samples
        def pool_nograd_step():
            opt.zero_grad(set_to_none=True)
            g, l = get_view_tensors(bs)
            ng_g, ng_l = get_view_tensors((T - 1) * bs)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, ng_proj_g = encoder(ng_g)
                if ng_l is not None:
                    _, _ = encoder(ng_l)
            ng_proj_g = ng_proj_g.detach()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb_g, proj_g = encoder(g)
                loss = (proj_g ** 2).mean() + (ng_proj_g ** 2).mean() * 0.0
                loss = loss + (emb_g ** 2).mean() * 0.001
                if l is not None:
                    emb_l, proj_l = encoder(l)
                    loss = loss + (proj_l ** 2).mean() + (emb_l ** 2).mean() * 0.001
            loss.backward()
            opt.step()

        # 3) pool_nograd_chunked: SUPPLEMENTARY. Split the no-grad pass into
        #    (T-1) sub-batches of BS each. Lower peak memory (cuDNN intermediate
        #    buffers sized for BS, not (T-1)*BS). NOT used in production:
        #    BN consistency would require parallel-variance aggregation, and
        #    we don't actually need the memory savings.
        def pool_nograd_chunked_step():
            opt.zero_grad(set_to_none=True)
            g, l = get_view_tensors(bs)
            ng_proj_chunks = []
            for _ in range(T - 1):
                ng_g_chunk, ng_l_chunk = get_view_tensors(bs)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, chunk_proj = encoder(ng_g_chunk)
                    if ng_l_chunk is not None:
                        _, _ = encoder(ng_l_chunk)
                ng_proj_chunks.append(chunk_proj.detach())
            ng_proj_g = torch.cat(ng_proj_chunks, dim=0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb_g, proj_g = encoder(g)
                loss = (proj_g ** 2).mean() + (ng_proj_g ** 2).mean() * 0.0
                loss = loss + (emb_g ** 2).mean() * 0.001
                if l is not None:
                    emb_l, proj_l = encoder(l)
                    loss = loss + (proj_l ** 2).mean() + (emb_l ** 2).mean() * 0.001
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

        ratio_pn = mean_pn / mean_std if mean_std else float("nan")
        ratio_pc = mean_pc / mean_std if mean_std else float("nan")

        print(f"BS={bs:4d}  "
              f"std={fmt(mean_std, std_std)} ({peak_std:.2f}GB)")
        print(f"        pool_nograd={fmt(mean_pn, std_pn)} ({peak_pn:.2f}GB)  "
              f"ratio={ratio_pn:.2f}x")
        print(f"        pool_nograd_chunked={fmt(mean_pc, std_pc)} ({peak_pc:.2f}GB)  "
              f"ratio={ratio_pc:.2f}x")

        rows.append(dict(
            bs=bs,
            mean_std=mean_std, std_std=std_std, peak_std=peak_std,
            mean_pool=mean_pn, std_pool=std_pn, peak_pool=peak_pn,
            mean_pool_chunked=mean_pc, std_pool_chunked=std_pc,
            peak_pool_chunked=peak_pc,
            ratio=ratio_pn, ratio_chunked=ratio_pc,
        ))
        del encoder, opt, train_ds, gpu_aug

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{'_multicrop' if use_locals else ''}_T{T}"
    out_path = out_dir / f"pooled_overhead_{gpu_name()}_{args.backbone}_{args.resolution}{suffix}.md"

    if use_locals:
        view_line = (f"- **Resolution**: {args.resolution}² (global), "
                     f"{cfg_template['local_crop_size']}² (local)")
        views_str = "V_g=2 + V_l=6 (multi-crop)"
    else:
        view_line = f"- **Resolution**: {args.resolution}²"
        views_str = "V_g=2 (no locals)"

    lines = [
        f"# Pooled overhead — {torch.cuda.get_device_name()}",
        f"",
        f"- **Backbone**: `{args.backbone}` ({cfg_template['encoder_scale']})",
        view_line,
        f"- **Views**: {views_str}",
        f"- **T (pool depth)**: {T}",
        f"- **Precision**: bf16",
        f"- **Iters per cell**: {args.n_iters} (mean ± std)",
        f"- **Each cell INCLUDES** sample_batch + gpu_aug (the full per-step cost as it appears in training)",
        "",
        "## Variants",
        "- `std` — standard step (1F + 1B on BS samples). Baseline.",
        "- `pool_nograd` — **production**. ONE big no-grad forward of (T-1)·BS + 1F + 1B on BS.",
        "- `pool_nograd_chunked` — **supplementary only**. (T-1) sequential no-grad sub-forwards of BS each. Lower peak memory but extra kernel launch overhead. Not used in production because we don't need the memory savings.",
        "",
        "## Results — timing (mean ± std)",
        "",
        "| BS | std | pool_nograd | pool_chunked | pool/std | chunked/std |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['bs']} | {fmt(r['mean_std'], r['std_std'])} | "
            f"{fmt(r['mean_pool'], r['std_pool'])} | "
            f"{fmt(r['mean_pool_chunked'], r['std_pool_chunked'])} | "
            f"{r['ratio']:.2f}× | {r['ratio_chunked']:.2f}× |"
        )
    lines.extend([
        "",
        "## Results — peak GPU memory (per method)",
        "",
        "| BS | std | pool_nograd | pool_chunked | pool / std | chunked / std |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for r in rows:
        m_pn = r['peak_pool'] / r['peak_std'] if r['peak_std'] else float("nan")
        m_pc = r['peak_pool_chunked'] / r['peak_std'] if r['peak_std'] else float("nan")
        lines.append(
            f"| {r['bs']} | {r['peak_std']:.2f} GB | "
            f"{r['peak_pool']:.2f} GB | {r['peak_pool_chunked']:.2f} GB | "
            f"{m_pn:.2f}× | {m_pc:.2f}× |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
