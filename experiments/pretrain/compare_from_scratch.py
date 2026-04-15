"""Compare 4 LeJEPA methods (sigreg / sigreg_pooled / w1 / w1_pooled)
from scratch on a small dataset.

Usage:
    python experiments/pretrain/compare_from_scratch.py \
        --dataset cifar10 --epochs 10 --batch-size 64 \
        --out-dir runs/compare_from_scratch
"""

import argparse
import json
import sys
import time
from pathlib import Path

# sys.path setup so this works from any cwd (mirrors trainer.py:21-29)
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for p in (HERE, REPO_ROOT, REPO_ROOT / "src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch

from configs import Config
from data import get_dataloaders, InMemoryGPUDataset
from models import LeJEPAEncoder, LinearProbe
from losses import build_regularizer, build_sigreg
from scheduler import make_scheduler
from trainer import setup_seed
from train_loops import (
    train_epoch_standard_inmem,
    train_epoch_standard_loader,
    train_epoch_pooled_inmem,
    train_epoch_pooled_loader,
    make_nograd_loader,
    evaluate,
)

METHODS = [
    ("sigreg", False),
    ("sigreg", True),
    ("w1",     False),
    ("w1",     True),
]


def run_training(cfg, device):
    """One in-process training run, returns per-epoch history dict.

    Mirrors the 4-branch dispatch from trainer.py:189-209.
    """
    setup_seed(cfg.seed)
    train_source, val_source, gpu_aug = get_dataloaders(cfg, device)
    in_memory = isinstance(train_source, InMemoryGPUDataset)

    encoder = LeJEPAEncoder(cfg).to(device)
    probe = LinearProbe(encoder.hidden_dim, cfg.num_classes).to(device)

    enc_opt = torch.optim.AdamW(
        encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    probe_opt = torch.optim.AdamW(
        probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_wd)

    if in_memory:
        steps_per_epoch = len(train_source) // cfg.batch_size
    else:
        steps_per_epoch = len(train_source)
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    enc_sched = make_scheduler(enc_opt, warmup_steps, cfg.lr)
    probe_sched = make_scheduler(probe_opt, warmup_steps, cfg.probe_lr)

    reg_fn = None
    sigreg_mod = None
    nograd_loader = None
    nograd_iter_state = [None]

    if not cfg.accumulate:
        reg_fn = build_regularizer(cfg, device)
    else:
        sigreg_mod = build_sigreg(cfg, device)
        if not in_memory and cfg.nograd_pool_size > 0:
            nograd_loader = make_nograd_loader(
                train_source.dataset, cfg.nograd_pool_size, cfg.num_workers)
            nograd_iter_state = [iter(nograd_loader)]

    sample_gen = (torch.Generator(device=device).manual_seed(cfg.seed)
                  if in_memory else None)

    history = {"epoch": [], "train_loss": [], "val_acc": []}
    global_step = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        if cfg.accumulate and in_memory:
            avg_loss, global_step = train_epoch_pooled_inmem(
                epoch, encoder, probe, train_source, gpu_aug,
                enc_opt, probe_opt, enc_sched, probe_sched, cfg,
                global_step, sample_gen, sigreg_mod)
        elif cfg.accumulate:
            avg_loss, global_step = train_epoch_pooled_loader(
                epoch, encoder, probe, train_source, gpu_aug,
                nograd_loader, nograd_iter_state,
                enc_opt, probe_opt, enc_sched, probe_sched, cfg,
                global_step, sigreg_mod)
        elif in_memory:
            avg_loss, global_step = train_epoch_standard_inmem(
                epoch, encoder, probe, reg_fn, train_source, gpu_aug,
                enc_opt, probe_opt, enc_sched, probe_sched, cfg,
                global_step, sample_gen)
        else:
            avg_loss, global_step = train_epoch_standard_loader(
                epoch, encoder, probe, reg_fn, train_source, gpu_aug,
                enc_opt, probe_opt, enc_sched, probe_sched, cfg, global_step)

        val_acc = evaluate(encoder, probe, val_source, cfg)
        elapsed = time.time() - t0
        history["epoch"].append(epoch)
        history["train_loss"].append(float(avg_loss))
        history["val_acc"].append(float(val_acc))
        print(f"  epoch {epoch}: loss={avg_loss:.4f} val_acc={val_acc:.4f} "
              f"({elapsed:.1f}s)", flush=True)
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="cifar10")
    p.add_argument("--data-dir",      default="./data")
    p.add_argument("--encoder-scale", default="convnextv2_nano")
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--epochs",        type=int, default=10)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--out-dir",       default="runs/compare_from_scratch")
    p.add_argument("--no-plot",       action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for reg, pooled in METHODS:
        label = f"{reg}{'_pooled' if pooled else ''}"
        print(f"\n=== {label} ===", flush=True)
        cfg = Config(
            dataset=args.dataset,
            data_dir=args.data_dir,
            encoder_scale=args.encoder_scale,
            regularizer=reg,
            accumulate=pooled,
            batch_size=args.batch_size,
            epochs=args.epochs,
            log_interval=9999,
            eval_interval=1,
            seed=args.seed,
        )
        results[label] = run_training(cfg, device)

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {out_json}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, h in results.items():
            ls    = "-" if "pooled" in label else "--"
            color = "tab:blue" if "w1" in label else "tab:orange"
            ax.plot(h["epoch"], h["val_acc"], label=label,
                    linestyle=ls, color=color, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Accuracy")
        ax.set_title(f"{args.dataset} from-scratch | bs={args.batch_size}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_png = out_dir / "compare.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
