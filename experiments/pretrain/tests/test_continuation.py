"""Test the continuation training mechanism."""

from pathlib import Path

import torch

from configs import Config
from data import get_dataloaders
from models import LeJEPAEncoder, LinearProbe
from checkpoint import save_checkpoint
from trainer import load_for_continuation


def _make_fake_checkpoint(tmp_path: Path, device, cfg) -> str:
    """Train one tiny step and save a checkpoint to disk."""
    encoder = LeJEPAEncoder(cfg).to(device)
    probe = LinearProbe(encoder.hidden_dim, cfg.num_classes).to(device)
    enc_opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    from scheduler import make_scheduler
    enc_sched = make_scheduler(enc_opt, 1, 100, 1e-3, 1e-5)
    probe_sched = make_scheduler(probe_opt, 1, 100, 1e-3, 1e-5)

    ckpt_path = tmp_path / "fake_base.pt"
    save_checkpoint(str(ckpt_path), encoder, probe, enc_opt, probe_opt,
                    enc_sched, probe_sched, epoch=10, global_step=100,
                    cfg=cfg, best_val_acc=0.42)
    return str(ckpt_path)


def test_continuation_loads_encoder_and_probe(device, small_cfg, tmp_path):
    ckpt_path = _make_fake_checkpoint(tmp_path, device, small_cfg)

    # Build a fresh encoder and probe (random init)
    encoder = LeJEPAEncoder(small_cfg).to(device)
    probe = LinearProbe(encoder.hidden_dim, small_cfg.num_classes).to(device)

    # Snapshot a parameter to verify it changes after loading
    first_conv = next(m for m in encoder.modules() if hasattr(m, "weight")
                      and m.weight is not None and m.weight.dim() == 4)
    fresh_w = first_conv.weight.detach().clone()

    ckpt = load_for_continuation(ckpt_path, encoder, probe, device)
    loaded_w = first_conv.weight.detach().clone()

    # The loaded weight should differ from the fresh init (load actually happened)
    assert not torch.allclose(loaded_w, fresh_w), \
        "Encoder weights didn't change after load_for_continuation"

    # The returned ckpt should contain the saved metadata
    assert ckpt.get("epoch") == 10
    assert ckpt.get("best_val_acc") == 0.42


def test_continuation_does_not_load_optimizer(device, small_cfg, tmp_path):
    """Continuation should leave optimizer/scheduler state untouched so the
    new training phase starts with a fresh LR schedule."""
    ckpt_path = _make_fake_checkpoint(tmp_path, device, small_cfg)

    encoder = LeJEPAEncoder(small_cfg).to(device)
    probe = LinearProbe(encoder.hidden_dim, small_cfg.num_classes).to(device)
    enc_opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3)

    # Optimizer state should be empty (no steps taken) before AND after loading
    assert len(enc_opt.state) == 0
    load_for_continuation(ckpt_path, encoder, probe, device)
    assert len(enc_opt.state) == 0, \
        "load_for_continuation should NOT load optimizer state"


def test_continuation_run_dir_namespacing(device, small_cfg, tmp_path):
    """Continuation runs from the same base + same target config should
    not collide with each other when seed differs."""
    from trainer import build_run_dir
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    fake_base = base_dir / "final.pt"
    fake_base.touch()

    cfg1 = Config(**{**vars(small_cfg), "continue_from": str(fake_base), "seed": 0})
    cfg2 = Config(**{**vars(small_cfg), "continue_from": str(fake_base), "seed": 1})
    cfg1.__post_init__()
    cfg2.__post_init__()

    dir1 = build_run_dir(cfg1)
    dir2 = build_run_dir(cfg2)
    assert dir1 != dir2, "Different seeds should produce different cont dirs"
    assert "cont_" in str(dir1), "Continuation dir should be tagged with 'cont_'"
