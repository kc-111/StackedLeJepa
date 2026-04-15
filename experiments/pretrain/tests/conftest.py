"""Shared pytest fixtures for the pretrain test suite."""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PRETRAIN_DIR = Path(__file__).resolve().parents[1]
for p in (str(PRETRAIN_DIR), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — pooled training tests need GPU")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def small_cfg():
    """Tiny config for fast tests."""
    from configs import Config
    cfg = Config(
        dataset="cifar100",
        data_dir=str(REPO_ROOT / "data"),
        encoder_scale="resnet18",
        regularizer="w1",
        accumulate=False,
        batch_size=16,
        nograd_pool_size=0,
        num_aug_views=1,
        epochs=1,
    )
    return cfg


@pytest.fixture(scope="session")
def pooled_cfg():
    from configs import Config
    cfg = Config(
        dataset="cifar100",
        data_dir=str(REPO_ROOT / "data"),
        encoder_scale="resnet18",
        regularizer="w1",
        accumulate=True,
        batch_size=16,
        nograd_pool_size=48,  # equivalent to old accum_steps=4: (4-1)*16=48
        num_aug_views=1,
        epochs=1,
    )
    return cfg


@pytest.fixture(scope="session")
def fifo_cfg():
    """Config for FIFO mode (single_view_reg=True)."""
    from configs import Config
    cfg = Config(
        dataset="cifar100",
        data_dir=str(REPO_ROOT / "data"),
        encoder_scale="resnet18",
        regularizer="w1",
        accumulate=True,
        batch_size=16,
        nograd_pool_size=32,
        fifo_size=64,
        single_view_reg=True,
        num_aug_views=2,
        epochs=1,
    )
    return cfg
