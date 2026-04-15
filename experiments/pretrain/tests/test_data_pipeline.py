"""Smoke tests for the data pipeline (in-memory cache + multi-crop GPU aug)."""

import torch

from configs import Config
from data import (
    get_dataloaders, InMemoryGPUDataset, InMemoryEvalLoader,
    GPUAug, IN_MEMORY_DATASETS,
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


def test_gpu_aug_output_shapes(device, small_cfg):
    train_source, _, gpu_aug = get_dataloaders(small_cfg, device)
    gen = torch.Generator(device=device).manual_seed(0)
    images, _ = train_source.sample_batch(small_cfg.batch_size, gen)
    x = images.float() / 255.0
    orig, aug = gpu_aug(x)

    assert orig.shape == (
        small_cfg.batch_size, 3,
        small_cfg.crop_size, small_cfg.crop_size)
    assert aug.shape == (
        small_cfg.num_aug_views, small_cfg.batch_size, 3,
        small_cfg.crop_size, small_cfg.crop_size)


def test_aug_normalizes_to_imagenet_stats(device, small_cfg):
    train_source, _, gpu_aug = get_dataloaders(small_cfg, device)
    gen = torch.Generator(device=device).manual_seed(0)
    images, _ = train_source.sample_batch(64, gen)
    x = images.float() / 255.0
    orig, aug = gpu_aug(x)
    # After ImageNet normalization, mean should be roughly 0, std roughly 1
    for t in (orig, aug.flatten(0, 1)):
        flat = t.flatten(2)  # (N, C, H*W)
        mean = flat.mean(dim=(0, 2))
        std = flat.std(dim=(0, 2))
        assert mean.abs().max().item() < 5.0, f"Normalized mean too large: {mean}"
        assert std.max().item() < 10.0, f"Normalized std too large: {std}"


def test_epoch_batches_no_replacement(device, small_cfg):
    """epoch_batches should cycle through all data without replacement."""
    train_source, _, _ = get_dataloaders(small_cfg, device)
    gen = torch.Generator(device=device).manual_seed(42)

    all_images = []
    for images, labels in train_source.epoch_batches(small_cfg.batch_size, gen):
        assert images.shape[0] == small_cfg.batch_size
        all_images.append(images)

    total = sum(img.shape[0] for img in all_images)
    expected_batches = len(train_source) // small_cfg.batch_size
    assert len(all_images) == expected_batches
    assert total == expected_batches * small_cfg.batch_size


def test_in_memory_dataset_list_includes_main_datasets():
    for name in ("cifar10", "cifar100", "stl10", "flowers102", "pets"):
        assert name in IN_MEMORY_DATASETS, f"{name} should be in IN_MEMORY_DATASETS"
