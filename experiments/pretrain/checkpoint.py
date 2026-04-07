"""Checkpoint save/load utilities."""

import os
import random

import numpy as np
import torch


def save_checkpoint(path, encoder, probe, enc_opt, probe_opt,
                    enc_sched, probe_sched, epoch, global_step, cfg,
                    best_val_acc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "probe": probe.state_dict(),
        "enc_optimizer": enc_opt.state_dict(),
        "probe_optimizer": probe_opt.state_dict(),
        "enc_scheduler": enc_sched.state_dict(),
        "probe_scheduler": probe_sched.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": vars(cfg),
        "best_val_acc": best_val_acc,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, path)


def load_checkpoint(path, encoder, probe, enc_opt, probe_opt,
                    enc_sched, probe_sched, cfg):
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    probe.load_state_dict(ckpt["probe"])

    if not cfg.swap_regularizer:
        enc_opt.load_state_dict(ckpt["enc_optimizer"])
        probe_opt.load_state_dict(ckpt["probe_optimizer"])
        enc_sched.load_state_dict(ckpt["enc_scheduler"])
        probe_sched.load_state_dict(ckpt["probe_scheduler"])

    if "rng" in ckpt:
        torch.set_rng_state(ckpt["rng"]["torch"])
        if ckpt["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(ckpt["rng"]["cuda"])
        np.random.set_state(ckpt["rng"]["numpy"])
        random.setstate(ckpt["rng"]["python"])

    return ckpt["epoch"] + 1, ckpt.get("global_step", 0), ckpt.get("best_val_acc", 0.0)
