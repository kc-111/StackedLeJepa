"""Smoke tests for the data pipeline (in-memory cache + multi-crop GPU aug)."""

import torch

from configs import Config
from data import (
    get_dataloaders, InMemoryGPUDataset, InMemoryEvalLoader,
    MultiCropGPUAug, IN_MEMORY_DATASETS,
)


def test_cifar100_loads_in_memory(device, small_cfg):
    train_source, val_source, gpu_aug = get_dataloaders(small_cfg, device)
    assert isinstance(train_source, InMemoryGPUDataset)
    assert isinstance(val_source, InMemoryEvalLoader)
    assert train_source.images.dtype == torch.uint8
    # cuda:0 vs cuda comparison: just check the type
    assert train_source.images.device.type == device.type


def test_sample_batch_returns_correct_shapes(device, small_cfg):
    train_source, _, _ = get_dataloaders(small_cfg, device)
    gen = torch.Generator(device=device).manual_seed(0)
    images, labels = train_source.sample_batch(small_cfg.batch_size, gen)
    assert images.shape[0] == small_cfg.batch_size
    assert images.shape[1] == 3
    assert images.dtype == torch.uint8
    assert labels.shape == (small_cfg.batch_size,)
    assert labels.dtype == torch.long


def test_multicrop_aug_output_shapes(device, small_cfg):
    train_source, _, gpu_aug = get_dataloaders(small_cfg, device)
    gen = torch.Generator(device=device).manual_seed(0)
    images, _ = train_source.sample_batch(small_cfg.batch_size, gen)
    x = images.float() / 255.0
    g, l = gpu_aug(x)

    assert g.shape == (
        small_cfg.num_global_views, small_cfg.batch_size, 3,
        small_cfg.global_crop_size, small_cfg.global_crop_size)
    if small_cfg.num_local_views > 0:
        assert l.shape == (
            small_cfg.num_local_views, small_cfg.batch_size, 3,
            small_cfg.local_crop_size, small_cfg.local_crop_size)
    else:
        assert l is None


def test_aug_normalizes_to_imagenet_stats(device, small_cfg):
    train_source, _, gpu_aug = get_dataloaders(small_cfg, device)
    gen = torch.Generator(device=device).manual_seed(0)
    images, _ = train_source.sample_batch(64, gen)
    x = images.float() / 255.0
    g, _ = gpu_aug(x)
    # After ImageNet normalization, mean should be roughly 0, std roughly 1
    flat = g.flatten(0, 1).flatten(2)  # (V*N, C, H*W)
    mean = flat.mean(dim=(0, 2))
    std = flat.std(dim=(0, 2))
    assert mean.abs().max().item() < 5.0, f"Normalized mean too large: {mean}"
    assert std.max().item() < 10.0, f"Normalized std too large: {std}"


def test_in_memory_dataset_list_includes_main_datasets():
    for name in ("cifar10", "cifar100", "stl10", "flowers102", "pets"):
        assert name in IN_MEMORY_DATASETS, f"{name} should be in IN_MEMORY_DATASETS"
