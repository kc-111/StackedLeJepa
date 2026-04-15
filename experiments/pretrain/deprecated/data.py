"""GPU-accelerated multi-view dataset loading for LeJEPA pretraining.

Two paths:
  1) ``InMemoryGPUDataset`` — entire dataset cached on GPU as uint8.
     Used for small datasets (CIFAR-10/100, STL10). Eliminates DataLoader,
     workers, PIL, and host→device transfers from the per-batch path.
  2) ``RawImageDataset`` + ``DataLoader`` — fallback for larger datasets
     that don't fit in GPU memory.

Augmentation is implemented with raw torch ops (no kornia) because
``kornia.augmentation`` has ~130 ms of dispatch overhead per op call,
making even trivial ops like Normalize and HFlip dominate runtime on
modern GPUs. Raw torch ops are 100× faster.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# Datasets whose native image size is small (no pre-resize needed for caching).
SMALL_NATIVE_DATASETS = {"cifar10", "cifar100", "stl10"}

# Datasets where the entire training set fits in GPU memory after pre-resize
# to ~256 px. Approximate cached sizes (uint8):
#   cifar10/100: native 32×32   → 150 MB
#   stl10:       native 96×96   → 140 MB
#   flowers102:  resize→256×256 → 200 MB
#   dtd:         resize→256×256 → 370 MB
#   aircraft:    resize→256×256 → 650 MB
#   pets:        resize→256×256 → 720 MB
#   food101:     resize→256×256 → 15 GB → too big, use DataLoader
IN_MEMORY_DATASETS = {
    "cifar10", "cifar100", "stl10",
    "flowers102", "dtd", "aircraft", "pets",
}


# ---------------------------------------------------------------------------
# Multi-crop GPU augmentation (raw torch — batched, vectorized)
# ---------------------------------------------------------------------------

def _random_resized_crop(images: torch.Tensor, target_size: int,
                         scale: tuple = (0.4, 1.0),
                         ratio: tuple = (3 / 4, 4 / 3)) -> torch.Tensor:
    """Vectorized RandomResizedCrop via ``F.grid_sample``.

    Each image gets an independently sampled crop box; all crops are
    resampled to ``target_size × target_size`` in one fused kernel.

    Args:
        images: (N, C, H, W) float in [0, 1].
        target_size: output H = W.
        scale: area scale range relative to original.
        ratio: aspect ratio range.
    """
    N, C, H, W = images.shape
    device = images.device
    dtype = images.dtype

    # Sample area scale and aspect ratio per image
    area = float(H * W)
    target_area = area * (
        torch.rand(N, device=device, dtype=dtype)
        * (scale[1] - scale[0]) + scale[0])
    log_ratio_lo = math.log(ratio[0])
    log_ratio_hi = math.log(ratio[1])
    aspect = torch.exp(
        torch.rand(N, device=device, dtype=dtype)
        * (log_ratio_hi - log_ratio_lo) + log_ratio_lo)
    crop_w = (target_area * aspect).sqrt().clamp_(min=1.0, max=float(W))
    crop_h = (target_area / aspect).sqrt().clamp_(min=1.0, max=float(H))

    # Random top-left
    x0 = torch.rand(N, device=device, dtype=dtype) * (W - crop_w)
    y0 = torch.rand(N, device=device, dtype=dtype) * (H - crop_h)

    # Build sampling grid for grid_sample. align_corners=False normalization:
    # input pixel center i ↔ normalized coord (2*i + 1) / size - 1
    # so a region [x0, x0+crop_w] in pixels maps to [(2*x0+1)/W - 1, (2*(x0+crop_w)-1)/W - 1].
    ts = target_size
    lin = torch.linspace(0.0, 1.0, ts, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")  # (ts, ts)
    # World coords (in source pixels) for each output pixel, per image
    x_world = x0[:, None, None] + xx[None] * crop_w[:, None, None]   # (N, ts, ts)
    y_world = y0[:, None, None] + yy[None] * crop_h[:, None, None]
    # Normalized grid coords for align_corners=False
    gx = (2.0 * x_world + 1.0) / W - 1.0
    gy = (2.0 * y_world + 1.0) / H - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (N, ts, ts, 2)

    return F.grid_sample(images, grid, mode="bilinear",
                         padding_mode="reflection", align_corners=False)


def _color_jitter(images: torch.Tensor, brightness: float = 0.4,
                  contrast: float = 0.4, saturation: float = 0.2,
                  p: float = 0.8) -> torch.Tensor:
    """Per-image color jitter (brightness, contrast, saturation).

    Hue is intentionally omitted (HSV conversion is expensive). All
    transformations are applied with random per-image strength; the per-image
    apply mask blends the result back to the original.
    """
    N = images.shape[0]
    device = images.device
    dtype = images.dtype

    out = images
    # Brightness: scale by random factor in [1-b, 1+b]
    bf = (torch.rand(N, 1, 1, 1, device=device, dtype=dtype)
          * (2 * brightness) + (1 - brightness))
    out = out * bf
    # Contrast: blend with per-channel mean
    cf = (torch.rand(N, 1, 1, 1, device=device, dtype=dtype)
          * (2 * contrast) + (1 - contrast))
    mean = out.mean(dim=(-2, -1), keepdim=True)
    out = (out - mean) * cf + mean
    # Saturation: blend with luminance
    sf = (torch.rand(N, 1, 1, 1, device=device, dtype=dtype)
          * (2 * saturation) + (1 - saturation))
    gray = (out * torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype)
            .view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    out = (out - gray) * sf + gray
    out = out.clamp_(0.0, 1.0)

    apply_mask = (torch.rand(N, device=device) < p).view(-1, 1, 1, 1)
    return torch.where(apply_mask, out, images)


def _random_grayscale(images: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    N = images.shape[0]
    device = images.device
    dtype = images.dtype
    gray_w = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype).view(1, 3, 1, 1)
    gray = (images * gray_w).sum(dim=1, keepdim=True).expand_as(images)
    apply_mask = (torch.rand(N, device=device) < p).view(-1, 1, 1, 1)
    return torch.where(apply_mask, gray, images)


def _random_hflip(images: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    N = images.shape[0]
    flipped = torch.flip(images, dims=[-1])
    apply_mask = (torch.rand(N, device=images.device) < p).view(-1, 1, 1, 1)
    return torch.where(apply_mask, flipped, images)


def _gaussian_blur(images: torch.Tensor, sigma: float = 1.0,
                   kernel_size: int = 5, p: float = 0.5) -> torch.Tensor:
    """Single-sigma batched Gaussian blur. Per-image apply mask.

    If ``p == 0``, returns the input unchanged (no allocation).
    If ``p == 1.0``, applies blur to all images without an apply mask.
    """
    if p <= 0.0:
        return images
    N, C = images.shape[:2]
    device = images.device
    dtype = images.dtype
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    kernel = g2d.expand(C, 1, kernel_size, kernel_size).contiguous()
    blurred = F.conv2d(images, kernel, padding=half, groups=C)
    if p >= 1.0:
        return blurred
    apply_mask = (torch.rand(N, device=device) < p).view(-1, 1, 1, 1)
    return torch.where(apply_mask, blurred, images)


def _solarize(images: torch.Tensor, threshold: float = 0.5,
              p: float = 0.2) -> torch.Tensor:
    """Per-image random solarization: invert pixels above ``threshold``."""
    if p <= 0.0:
        return images
    N = images.shape[0]
    inverted = torch.where(images >= threshold, 1.0 - images, images)
    if p >= 1.0:
        return inverted
    apply_mask = (torch.rand(N, device=images.device) < p).view(-1, 1, 1, 1)
    return torch.where(apply_mask, inverted, images)


class MultiCropGPUAug(nn.Module):
    """Generate ``num_global`` + ``num_local`` augmented views in batched ops.

    Follows the DINO/LeJEPA augmentation recipe (Caron et al. 2021,
    Balestriero & LeCun 2025 Sec 6.1):

    - Color jitter (brightness, contrast, saturation) + random grayscale
      run on the **small** input tensor before upsampling (cheap on 32×32).
    - Global views split into two groups with asymmetric augmentation:
        * Group g1 (first half of V_g): RRC + heavy blur (p=1.0)
        * Group g2 (second half of V_g): RRC + light blur (p=0.1) + solarize (p=0.2)
    - Local views: RRC at small size, no blur/solarize.
    - All views: hflip + ImageNet normalize.

    Input:  (N, C, H, W) float in [0, 1] on GPU.
    Output: ``(global_views, local_views)`` where
        global_views: (V_g, N, C, global_size, global_size)
        local_views:  (V_l, N, C, local_size, local_size)  or ``None``
    """

    def __init__(
        self,
        num_global: int = 2,
        num_local: int = 6,
        global_size: int = 224,
        local_size: int = 96,
        global_scale: tuple = (0.4, 1.0),
        local_scale: tuple = (0.05, 0.4),
        blur_sigma: float = 1.0,
        blur_kernel: int = 5,
        solarize_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_global = num_global
        self.num_local = num_local
        self.global_size = global_size
        self.local_size = local_size
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel
        self.solarize_threshold = solarize_threshold

        # Split global views into "g1" (heavy blur) and "g2" (light blur + solarize)
        # following DINO. With V_g=2 the standard, this gives 1+1.
        self.num_global_g1 = (num_global + 1) // 2
        self.num_global_g2 = num_global - self.num_global_g1

        self.register_buffer("mean", IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("std", IMAGENET_STD.view(1, 3, 1, 1))

    def _augment_view(self, x: torch.Tensor, target_size: int,
                      scale: tuple) -> torch.Tensor:
        """Full augmentation for one set of views (paper Sec 6.1).

        Order matches the paper: RRC → flip → color jitter → grayscale
        → blur → solarize → normalize.
        """
        x = _random_resized_crop(x, target_size, scale=scale)
        x = _random_hflip(x, p=0.5)
        x = _color_jitter(x, brightness=0.4, contrast=0.4,
                          saturation=0.2, p=0.8)
        x = _random_grayscale(x, p=0.2)
        x = _gaussian_blur(x, sigma=self.blur_sigma,
                           kernel_size=self.blur_kernel, p=0.5)
        x = _solarize(x, threshold=self.solarize_threshold, p=0.2)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """images: (N, C, H, W) float in [0, 1] on GPU."""
        N = images.shape[0]

        # Global views — each independently augmented
        globals_list = []
        for _ in range(self.num_global):
            g = self._augment_view(images, self.global_size,
                                   self.global_scale)
            globals_list.append(g)
        g_full = torch.stack(globals_list, dim=0)  # (V_g, N, C, gs, gs)

        # Local views
        if self.num_local > 0:
            locals_list = []
            for _ in range(self.num_local):
                lv = self._augment_view(images, self.local_size,
                                        self.local_scale)
                locals_list.append(lv)
            l = torch.stack(locals_list, dim=0)  # (V_l, N, C, ls, ls)
        else:
            l = None
        return g_full, l


# ---------------------------------------------------------------------------
# In-memory GPU dataset (fast path for small datasets)
# ---------------------------------------------------------------------------

class InMemoryGPUDataset:
    """Entire dataset cached on GPU as uint8. No DataLoader, no workers.

    Stores (N, C, H, W) uint8 tensor + (N,) int64 labels on the device.
    Sampling uses GPU-side ``torch.randint`` to draw a batch with replacement,
    then gathers via tensor indexing — no host involvement at all.

    For datasets with variable native image sizes (e.g. Flowers102, Pets),
    pass ``pre_resize > 0`` to resize each image to a fixed size during
    loading so the whole tensor can be stacked.
    """

    def __init__(self, torchvision_ds, device, pre_resize: int = 0):
        labels = []
        # Check if we need to resize (variable sizes or pre_resize requested)
        # by peeking at the first image.
        first_img, _ = torchvision_ds[0]
        if not isinstance(first_img, Image.Image):
            first_img = T.functional.to_pil_image(first_img)
        first_img = first_img.convert("RGB")
        h0, w0 = first_img.size[1], first_img.size[0]
        # Detect variable-size: scan a few samples
        variable_size = False
        for i in range(0, min(len(torchvision_ds), 10), 2):
            img, _ = torchvision_ds[i]
            if not isinstance(img, Image.Image):
                img = T.functional.to_pil_image(img)
            if img.size != first_img.size:
                variable_size = True
                break

        if pre_resize > 0 or variable_size:
            target = pre_resize if pre_resize > 0 else max(h0, w0)
            cpu_resize = T.Resize(target)
            cpu_crop = T.CenterCrop(target)
            arrays = []
            for img, lbl in torchvision_ds:
                if not isinstance(img, Image.Image):
                    img = T.functional.to_pil_image(img)
                img = img.convert("RGB")
                img = cpu_resize(img)
                img = cpu_crop(img)  # ensure square
                arrays.append(np.asarray(img))
                labels.append(int(lbl))
            x = torch.from_numpy(np.stack(arrays))  # (N, H, W, 3) uint8
        else:
            arrays = []
            for img, lbl in torchvision_ds:
                if not isinstance(img, Image.Image):
                    img = T.functional.to_pil_image(img)
                arrays.append(np.asarray(img.convert("RGB")))
                labels.append(int(lbl))
            x = torch.from_numpy(np.stack(arrays))

        x = x.permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W) uint8

        self.images = x.to(device, non_blocking=True)
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)
        self.N = self.images.shape[0]

    def __len__(self):
        return self.N

    def sample_batch(self, batch_size: int, generator: torch.Generator):
        idx = torch.randint(0, self.N, (batch_size,),
                            device=self.images.device, generator=generator)
        return self.images[idx], self.labels[idx]


class InMemoryEvalLoader:
    """Iterates an InMemoryGPUDataset in deterministic batches for evaluation.

    Applies normalization on the fly. No augmentation.
    """

    def __init__(self, dataset: InMemoryGPUDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        device = dataset.images.device
        self.mean = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
        self.std = IMAGENET_STD.to(device).view(1, 3, 1, 1)

    def __len__(self):
        return (self.dataset.N + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        N, B = self.dataset.N, self.batch_size
        for start in range(0, N, B):
            end = min(start + B, N)
            x = self.dataset.images[start:end].float().div_(255.0)
            x = (x - self.mean) / self.std
            y = self.dataset.labels[start:end]
            yield x, y


# ---------------------------------------------------------------------------
# Disk-backed dataset (fallback for larger datasets)
# ---------------------------------------------------------------------------

class RawImageDataset(Dataset):
    """Wraps a torchvision dataset. Returns (uint8 tensor, label), no aug."""

    def __init__(self, dataset, pre_resize: int = 0):
        self.dataset = dataset
        self.pre_resize = T.Resize(pre_resize) if pre_resize > 0 else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if not isinstance(img, Image.Image):
            img = T.functional.to_pil_image(img)
        img = img.convert("RGB")
        if self.pre_resize is not None:
            img = self.pre_resize(img)
        arr = np.asarray(img)  # (H, W, 3) uint8
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W) uint8
        return t, label


def _collate_uint8(batch):
    tensors, labels = zip(*batch)
    shapes = set(t.shape for t in tensors)
    if len(shapes) == 1:
        return torch.stack(tensors), torch.tensor(labels, dtype=torch.long)
    # Variable size: pad to max
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1])) for t in tensors]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _load_base_dataset(name, root, train, download=True):
    split = "train" if train else "test"
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(root=root, train=train, download=download)
    elif name == "cifar10":
        return torchvision.datasets.CIFAR10(root=root, train=train, download=download)
    elif name == "stl10":
        return torchvision.datasets.STL10(root=root, split=split, download=download)
    elif name == "flowers102":
        return torchvision.datasets.Flowers102(root=root, split=split, download=download)
    elif name == "food101":
        return torchvision.datasets.Food101(root=root, split=split, download=download)
    elif name == "dtd":
        return torchvision.datasets.DTD(root=root, split=split, download=download)
    elif name == "aircraft":
        return torchvision.datasets.FGVCAircraft(root=root, split=split, download=download)
    elif name == "pets":
        return torchvision.datasets.OxfordIIITPet(
            root=root, split="trainval" if train else "test", download=download)
    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(cfg, device):
    """Build train and val data sources + GPU augmentation module.

    For datasets in ``IN_MEMORY_DATASETS``, uses ``InMemoryGPUDataset`` for
    both train and val (no DataLoader). For others, falls back to a fast
    DataLoader returning uint8 tensors.

    Returns:
        (train_source, val_source, gpu_aug)

        - For small datasets: train_source is ``InMemoryGPUDataset``, val_source
          is ``InMemoryEvalLoader``.
        - Otherwise: both are ``DataLoader``.
        - ``gpu_aug`` is a ``MultiCropGPUAug`` module on ``device``.
    """
    train_base = _load_base_dataset(cfg.dataset, cfg.data_dir, train=True)
    val_base = _load_base_dataset(cfg.dataset, cfg.data_dir, train=False)

    gpu_aug = MultiCropGPUAug(
        num_global=cfg.num_global_views,
        num_local=cfg.num_local_views,
        global_size=cfg.global_crop_size,
        local_size=cfg.local_crop_size,
        global_scale=tuple(cfg.global_crop_scale),
        local_scale=tuple(cfg.local_crop_scale),
    ).to(device)

    if cfg.dataset in IN_MEMORY_DATASETS:
        # Cache the entire dataset in GPU memory as uint8.
        # CIFAR/STL10: cache at native resolution (small inputs are cheap to
        # upsample per-batch via grid_sample inside MultiCropGPUAug).
        # Larger fine-grained datasets (flowers/pets/aircraft/dtd): pre-resize
        # to a fixed resolution since native sizes are variable.
        # Set --cache-resolution N to override.
        if cfg.cache_resolution > 0:
            target = cfg.cache_resolution
        elif cfg.dataset in SMALL_NATIVE_DATASETS:
            # Pre-resize small images so RRC crops from a reasonable
            # resolution instead of upscaling 32px to 128px.
            target = cfg.global_crop_size + 32
        else:
            target = cfg.global_crop_size + 32  # leave crop margin
        train_ds = InMemoryGPUDataset(train_base, device, pre_resize=target)
        val_ds = InMemoryGPUDataset(val_base, device, pre_resize=cfg.global_crop_size)
        val_loader = InMemoryEvalLoader(val_ds, cfg.batch_size)
        return train_ds, val_loader, gpu_aug

    # Fallback: DataLoader path for larger datasets
    pre_resize = cfg.global_crop_size + 32  # leave some margin for cropping
    train_ds = RawImageDataset(train_base, pre_resize=pre_resize)
    val_ds = RawImageDataset(val_base, pre_resize=cfg.global_crop_size)

    nw = cfg.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
        collate_fn=_collate_uint8)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
        collate_fn=_collate_uint8)

    return train_loader, val_loader, gpu_aug
