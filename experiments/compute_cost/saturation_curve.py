"""Measure GPU saturation curve for forward pass.

Sweeps batch size from 1 to 2048 on a single forward pass and reports
ms/batch and ms/img. The curve shows three regimes:
  - Flat zone (ms constant): GPU under-utilized; adding samples is free
  - Knee: saturation point where throughput peaks
  - Linear zone (ms scales with batch): compute-bound; cost = constant * batch

Output: markdown table at experiments/compute_cost/results/saturation_<gpu>.md

Usage:
    python saturation_curve.py --backbone resnet18 --resolution 128
    python saturation_curve.py --backbone vit_tiny --resolution 128 --multicrop
"""

import argparse
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


def time_forward(encoder, x, n=20, warmup=8):
    for _ in range(warmup):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoder(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoder(x)
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18",
                        help="encoder_scale: resnet18, resnet34, tiny, ...")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--max-imgs", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda")

    cfg = Config(
        dataset="cifar100",
        data_dir=args.data_dir,
        encoder_scale=args.backbone,
        batch_size=64,
        global_crop_size=args.resolution,
    )
    train_ds, _, _ = get_dataloaders(cfg, device)
    encoder = LeJEPAEncoder(cfg).to(device).eval()

    images_big, _ = train_ds.sample_batch(args.max_imgs, torch.Generator(device=device).manual_seed(42))
    x = images_big.float() / 255.0
    x = torch.nn.functional.interpolate(
        x, size=args.resolution, mode="bilinear", align_corners=False)

    sweep = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    sweep = [n for n in sweep if n <= args.max_imgs]

    rows = []
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Backbone: {cfg.backbone_name}, resolution: {args.resolution}²")
    print()
    print(f'{"n_imgs":>8s} {"time":>10s} {"ms/img":>10s} {"throughput":>14s}  regime')
    print("-" * 60)

    peak_thru = 0.0
    flat_floor = None
    for n in sweep:
        try:
            ms = time_forward(encoder, x[:n])
        except torch.cuda.OutOfMemoryError:
            print(f"{n:>8d}: OOM, stopping")
            break
        per_img = ms / n
        thru = n / ms * 1000
        if flat_floor is None or ms < flat_floor:
            flat_floor = ms
        if thru > peak_thru:
            peak_thru = thru

        # Classify regime
        if ms < flat_floor * 1.15:
            regime = "flat (under-utilized)"
        elif thru > peak_thru * 0.9:
            regime = "near-peak"
        else:
            regime = "linear (compute-bound)"

        line = (f"{n:>8d} {ms:>8.2f}ms {per_img:>8.4f}ms {thru:>11.0f}/s  {regime}")
        print(line)
        rows.append(dict(n_imgs=n, ms=ms, ms_per_img=per_img, throughput=thru, regime=regime))

    # Write markdown
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"saturation_{gpu_name()}_{args.backbone}_{args.resolution}.md"
    lines = [
        f"# Saturation curve — {torch.cuda.get_device_name()}",
        f"Backbone: `{cfg.backbone_name}`, resolution: {args.resolution}², bf16, eval mode",
        "",
        "| n_imgs | time | ms/img | throughput | regime |",
        "|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_imgs']} | {r['ms']:.2f} ms | {r['ms_per_img']:.4f} ms "
            f"| {r['throughput']:.0f}/s | {r['regime']} |")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
