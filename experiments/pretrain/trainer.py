"""LeJEPA pretraining entry point.

Cluster-friendly: all config via CLI, deterministic output dirs,
resume support, skip-if-final-exists.

Run:
    python trainer.py --dataset cifar100 --encoder-scale resnet18
    python trainer.py --dataset cifar100 --encoder-scale tiny --patch-size 8
    python trainer.py --dataset cifar100 --regularizer w1 --accumulate
"""

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add paths (works from any working directory)
REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import Config
from data import get_dataloaders, InMemoryGPUDataset
from models import LeJEPAEncoder, LinearProbe
from losses import build_regularizer, build_sigreg
from scheduler import make_scheduler
from checkpoint import save_checkpoint, load_checkpoint
from train_loops import (
    train_epoch_standard_inmem, train_epoch_standard_loader,
    train_epoch_pooled_inmem, train_epoch_pooled_loader,
    make_nograd_loader, evaluate,
)


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_run_dir(cfg):
    method = cfg.regularizer
    if cfg.accumulate:
        method += "_pooled"
    base = (f"{cfg.dataset}_{cfg.encoder_scale}_{method}"
            f"_bs{cfg.batch_size}_seed{cfg.seed}")
    if cfg.continue_from:
        # Tag the dir as a continuation; the base checkpoint path's parent name
        # is folded in so different bases produce distinct cont dirs.
        base_tag = Path(cfg.continue_from).parent.name
        base = f"cont_{base_tag}__{base}"
    return Path(cfg.save_dir) / base


def load_for_continuation(checkpoint_path: str, encoder, probe, device):
    """Load encoder and probe state from a base checkpoint.

    Optimizer / scheduler / RNG / epoch counters are NOT loaded — continuation
    starts a fresh training phase with its own LR schedule.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    probe.load_state_dict(ckpt["probe"])
    return ckpt


def main():
    cfg = Config.from_cli()
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = build_run_dir(cfg)
    final_path = save_dir / "final.pt"
    if final_path.exists():
        print(f"SKIP: {final_path} already exists")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {cfg.dataset} ({cfg.num_classes} classes)")
    print(f"Backbone: {cfg.backbone_name}")
    print(f"Multi-crop: V_g={cfg.num_global_views}@{cfg.global_crop_size} "
          f"+ V_l={cfg.num_local_views}@{cfg.local_crop_size}")
    print(f"Regularizer: {cfg.regularizer}, accumulate={cfg.accumulate}")
    print(f"Projector: {cfg.proj_hidden}→{cfg.proj_dim}")
    print(f"Training: bs={cfg.batch_size}, epochs={cfg.epochs}, "
          f"lr={cfg.lr}, wd={cfg.weight_decay}, λ={cfg.lambd}")
    if cfg.continue_from:
        print(f"Continuation from: {cfg.continue_from}")
    print(f"Output: {save_dir}")

    # Data
    train_source, val_source, gpu_aug = get_dataloaders(cfg, device)
    in_memory = isinstance(train_source, InMemoryGPUDataset)

    if in_memory:
        steps_per_epoch = len(train_source) // cfg.batch_size
    else:
        steps_per_epoch = len(train_source)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    print(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}, "
          f"in_memory={in_memory}")

    # Model
    encoder = LeJEPAEncoder(cfg).to(device)
    probe = LinearProbe(encoder.hidden_dim, cfg.num_classes).to(device)
    if cfg.use_compile:
        encoder = torch.compile(encoder)
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # Optimizers + schedulers
    enc_opt = torch.optim.AdamW(
        encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    probe_opt = torch.optim.AdamW(
        probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_wd)
    enc_sched = make_scheduler(enc_opt, warmup_steps, total_steps,
                               cfg.lr, cfg.eta_min)
    probe_sched = make_scheduler(probe_opt, warmup_steps, total_steps,
                                 cfg.probe_lr, 1e-5)

    # Regularizer
    reg_fn = None
    sigreg_mod = None
    if not cfg.accumulate:
        reg_fn = build_regularizer(cfg, device)
    else:
        if cfg.regularizer == "sigreg":
            sigreg_mod = build_sigreg(cfg, device)

    # Resume / continuation
    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0

    # Continuation: load encoder+probe from a base checkpoint, then train
    # K=cfg.epochs more from epoch 0 with this run's optimizer/scheduler.
    if cfg.continue_from:
        if not os.path.exists(cfg.continue_from):
            raise FileNotFoundError(
                f"continue_from checkpoint not found: {cfg.continue_from}")
        load_for_continuation(cfg.continue_from, encoder, probe, device)
        print(f"Loaded encoder+probe from {cfg.continue_from} for continuation")
    else:
        # Standard resume from a partial run in the same save_dir
        if cfg.resume_from:
            resume_path = cfg.resume_from
        else:
            ckpts = sorted(save_dir.glob("epoch*.pt"))
            resume_path = str(ckpts[-1]) if ckpts else ""

        if resume_path and os.path.exists(resume_path):
            start_epoch, global_step, best_val_acc = load_checkpoint(
                resume_path, encoder, probe, enc_opt, probe_opt,
                enc_sched, probe_sched, cfg)
            print(f"Resumed from {resume_path} (epoch {start_epoch}, "
                  f"step {global_step}, best_acc {best_val_acc:.4f})")

    # Pooled mode: 2-step procedure. Each step samples (T-1)*BS extra
    # images and runs them through the encoder with no_grad to get FRESH
    # detached projections (current weights), then pools with the BS grad
    # samples for the regularizer. Cost per step ≈ 2× standard.
    nograd_loader = None
    nograd_iter_state = [None]
    if cfg.accumulate and not in_memory:
        nograd_bs = (cfg.accum_steps - 1) * cfg.batch_size
        nograd_loader = make_nograd_loader(
            train_source.dataset, nograd_bs, cfg.num_workers)
        nograd_iter_state = [iter(nograd_loader)]

    # GPU generator for in-memory sampling
    sample_gen = torch.Generator(device=device).manual_seed(cfg.seed) \
        if in_memory else None

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        if cfg.accumulate and in_memory:
            avg_loss, global_step = train_epoch_pooled_inmem(
                epoch, encoder, probe, train_source,
                gpu_aug, enc_opt, probe_opt, enc_sched, probe_sched, cfg,
                global_step, sample_gen, sigreg_mod)
        elif cfg.accumulate:
            avg_loss, global_step = train_epoch_pooled_loader(
                epoch, encoder, probe, train_source,
                gpu_aug, nograd_loader, nograd_iter_state,
                enc_opt, probe_opt, enc_sched, probe_sched, cfg,
                global_step, sigreg_mod)
        elif in_memory:
            avg_loss, global_step = train_epoch_standard_inmem(
                epoch, encoder, probe, reg_fn, train_source,
                gpu_aug, enc_opt, probe_opt, enc_sched, probe_sched,
                cfg, global_step, sample_gen)
        else:
            avg_loss, global_step = train_epoch_standard_loader(
                epoch, encoder, probe, reg_fn, train_source,
                gpu_aug, enc_opt, probe_opt, enc_sched, probe_sched,
                cfg, global_step)

        elapsed = time.time() - t0
        print(f"Epoch {epoch} | avg_loss {avg_loss:.4f} | {elapsed:.1f}s",
              flush=True)

        # Eval
        if (epoch + 1) % cfg.eval_interval == 0:
            val_acc = evaluate(encoder, probe, val_source, cfg)
            print(f"  val_acc: {val_acc:.4f} (best: {best_val_acc:.4f})",
                  flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    str(save_dir / "best.pt"), encoder, probe,
                    enc_opt, probe_opt, enc_sched, probe_sched,
                    epoch, global_step, cfg, best_val_acc)

        # Periodic save
        if (epoch + 1) % cfg.save_interval == 0:
            save_checkpoint(
                str(save_dir / f"epoch{epoch+1}.pt"), encoder, probe,
                enc_opt, probe_opt, enc_sched, probe_sched,
                epoch, global_step, cfg, best_val_acc)

    # Final save
    save_checkpoint(
        str(save_dir / "final.pt"), encoder, probe,
        enc_opt, probe_opt, enc_sched, probe_sched,
        cfg.epochs - 1, global_step, cfg, best_val_acc)
    print(f"Done. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
