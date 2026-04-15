"""GPU-accelerated dataset loading for LeJEPA pretraining.

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

``GPUAug`` produces one unaugmented original view + N augmented views.
The older multi-crop (global/local) module lives in ``deprecated/data.py``.
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import torchvision
import torchvision.transforms as T

# All datasets normalized to [-1, 1] via x * 2 - 1 (input in [0, 1]).
# No per-dataset mean/std needed when training from scratch.

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
# GPU augmentation (raw torch — batched, vectorized)
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


def _hue_jitter(images: torch.Tensor, hue: float = 0.1) -> torch.Tensor:
    """Per-image hue rotation via Rodrigues' rotation around (1,1,1) in RGB.

    Rotates each image's RGB values by a random angle in [-hue*pi, +hue*pi]
    around the gray axis (1,1,1)/sqrt(3). Equivalent to shifting H in HSV
    but avoids the RGB↔HSV round-trip.
    """
    N = images.shape[0]
    device, dtype = images.device, images.dtype
    angles = (torch.rand(N, device=device, dtype=dtype) * 2 - 1) * (hue * math.pi)
    cos_a = angles.cos()
    sin_a = angles.sin()
    # Rodrigues: R = cos(a)*I + (1-cos(a))*k*k^T + sin(a)*K
    # k = (1,1,1)/sqrt(3), so k*k^T = 1/3 * ones(3,3)
    # K (skew-symmetric of k) has off-diag entries ±1/sqrt(3)
    one_third = 1.0 / 3.0
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    # Build (N, 3, 3) rotation matrices
    c = cos_a
    s = sin_a
    oc = 1.0 - c
    R = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = c + oc * one_third
    R[:, 1, 1] = c + oc * one_third
    R[:, 2, 2] = c + oc * one_third
    R[:, 0, 1] = oc * one_third - s * inv_sqrt3
    R[:, 0, 2] = oc * one_third + s * inv_sqrt3
    R[:, 1, 0] = oc * one_third + s * inv_sqrt3
    R[:, 1, 2] = oc * one_third - s * inv_sqrt3
    R[:, 2, 0] = oc * one_third - s * inv_sqrt3
    R[:, 2, 1] = oc * one_third + s * inv_sqrt3
    # (N, 3, 3) @ (N, 3, H*W) → (N, 3, H*W)
    H, W = images.shape[2], images.shape[3]
    flat = images.reshape(N, 3, H * W)
    rotated = torch.bmm(R, flat).reshape(N, 3, H, W)
    return rotated


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


class GPUAug(nn.Module):
    """Produce one unaugmented original + ``num_aug_views`` augmented views.

    Simpler than the older multi-crop module: no global/local split, no
    g1/g2 asymmetric blur, no local crops. Per-view augmentation order
    follows the LeJEPA paper exactly: RRC → jitter → gray → blur(p=0.1) →
    solarize(p=0.2) → flip → normalize.

    The "original" is a deterministic center-crop of the input to
    ``crop_size`` followed by ImageNet normalization. The input is
    expected to already be at least ``crop_size`` in each spatial dim —
    ``InMemoryGPUDataset`` caches at ``crop_size + 32`` by default.

    Input:  (N, C, H, W) float in [0, 1] on GPU, H, W >= crop_size.
    Output: ``(orig, aug_views)`` where
        orig:      (N, C, crop_size, crop_size)
        aug_views: (V, N, C, crop_size, crop_size)
    """

    def __init__(
        self,
        num_aug_views: int = 1,
        crop_size: int = 128,
        crop_scale: tuple = (0.3, 1.0),
        blur_p: float = 0.1,
        solarize_p: float = 0.2,
        blur_sigma: float = 1.0,
        blur_kernel: int = 5,
        solarize_threshold: float = 0.5,
        motion_length: int = 9,
        n_motion_angles: int = 8,
        local_blur_sigmas: tuple = (4.0, 8.0, 14.0, 22.0),
        local_blur_min_kernel: int = 15,
        local_blur_radius_frac: tuple = (0.1, 0.4),
        num_local_blur_blobs: int = 4,
        local_blur_core_frac: float = 0.85,
    ):
        super().__init__()
        self.num_aug_views = num_aug_views
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.blur_p = blur_p
        self.solarize_p = solarize_p
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel
        self.solarize_threshold = solarize_threshold
        self.n_motion_angles = n_motion_angles
        self.local_blur_radius_frac = local_blur_radius_frac
        self.num_local_blur_blobs = num_local_blur_blobs
        self.local_blur_core_frac = local_blur_core_frac
        self.n_local_sigmas = len(local_blur_sigmas)

        # Cached gaussian blur kernel — built once, reused every call.
        # Depthwise: (3, 1, k, k).
        half = blur_kernel // 2
        coords = torch.arange(-half, half + 1, dtype=torch.float32)
        g1d = torch.exp(-(coords ** 2) / (2 * blur_sigma * blur_sigma))
        g1d = g1d / g1d.sum()
        g2d = g1d[:, None] * g1d[None, :]
        bk = g2d.expand(3, 1, blur_kernel, blur_kernel).contiguous()
        self.register_buffer("blur_w", bk)
        self.blur_pad = half

        # Cached luminance weights and RRC base sampling lattice. The lattice
        # is the only part of the grid that doesn't depend on per-image random
        # crop params, so it can live as a buffer.
        self.register_buffer(
            "gray_w",
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1),
        )
        lin = torch.linspace(0.0, 1.0, crop_size)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("rrc_xx", xx.contiguous())
        self.register_buffer("rrc_yy", yy.contiguous())

        # Cached motion-blur kernels at evenly spaced angles in [0, π).
        # Picked one-per-call (cheap) instead of per-image (would need
        # the grouped-conv-with-N-kernels trick). Different views in the
        # same step still pick independently.
        motion_K = motion_length if motion_length % 2 == 1 else motion_length + 1
        half_K = motion_K // 2
        half_L = (motion_length - 1) / 2
        mk_list = []
        for i in range(n_motion_angles):
            angle = i * math.pi / n_motion_angles
            kk = torch.zeros(motion_K, motion_K, dtype=torch.float32)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for t in range(motion_length):
                off = t - half_L
                xi = int(round(half_K + off * cos_a))
                yi = int(round(half_K + off * sin_a))
                if 0 <= xi < motion_K and 0 <= yi < motion_K:
                    kk[yi, xi] = 1.0
            kk = kk / kk.sum()
            mk_list.append(kk)
        mk = torch.stack(mk_list)                                   # (A, K, K)
        mk = mk[:, None, None, :, :].expand(
            n_motion_angles, 3, 1, motion_K, motion_K).contiguous()
        self.register_buffer("motion_w", mk)
        self.motion_pad = motion_K // 2

        # Multi-sigma cached separable gaussians for local blur. One sigma is
        # picked per call so blur strength varies. Stored as 1-D kernels and
        # applied via two conv2d passes (cheap for the larger sigmas, where a
        # full 2-D kernel would be ~50x more compute).
        pads = []
        for i, sigma in enumerate(local_blur_sigmas):
            K = max(local_blur_min_kernel, 2 * int(math.ceil(3 * sigma)) + 1)
            if K % 2 == 0:
                K += 1
            half = K // 2
            pads.append(half)
            coords = torch.arange(-half, half + 1, dtype=torch.float32)
            g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
            g1d = g1d / g1d.sum()
            h = g1d.view(1, 1, 1, K).expand(3, 1, 1, K).contiguous()
            v = g1d.view(1, 1, K, 1).expand(3, 1, K, 1).contiguous()
            self.register_buffer(f"local_blur_h_{i}", h)
            self.register_buffer(f"local_blur_v_{i}", v)
        self._local_blur_pads = pads

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map [0, 1] → [-1, 1]."""
        return x * 2 - 1

    def _rrc(self, images: torch.Tensor) -> torch.Tensor:
        """RandomResizedCrop using the cached base lattice."""
        N, _, H, W = images.shape
        device, dtype = images.device, images.dtype
        scale_lo, scale_hi = self.crop_scale
        target_area = (
            torch.rand(N, device=device, dtype=dtype)
            * (scale_hi - scale_lo) + scale_lo
        ) * float(H * W)
        log_lo, log_hi = math.log(3 / 4), math.log(4 / 3)
        aspect = torch.exp(
            torch.rand(N, device=device, dtype=dtype)
            * (log_hi - log_lo) + log_lo)
        crop_w = (target_area * aspect).sqrt().clamp_(1.0, float(W))
        crop_h = (target_area / aspect).sqrt().clamp_(1.0, float(H))
        x0 = torch.rand(N, device=device, dtype=dtype) * (W - crop_w)
        y0 = torch.rand(N, device=device, dtype=dtype) * (H - crop_h)

        xx = self.rrc_xx.to(dtype)
        yy = self.rrc_yy.to(dtype)
        x_world = x0[:, None, None] + xx[None] * crop_w[:, None, None]
        y_world = y0[:, None, None] + yy[None] * crop_h[:, None, None]
        gx = (2.0 * x_world + 1.0) / W - 1.0
        gy = (2.0 * y_world + 1.0) / H - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(images, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    def _color_jitter(self, images: torch.Tensor,
                      brightness: float = 0.4, contrast: float = 0.4,
                      saturation: float = 0.2, hue: float = 0.1,
                      p: float = 0.8) -> torch.Tensor:
        """Brightness/contrast/saturation/hue jitter using in-place ops on a clone."""
        N = images.shape[0]
        device, dtype = images.device, images.dtype
        out = images.clone()
        bf = (torch.rand(N, 1, 1, 1, device=device, dtype=dtype)
              * (2 * brightness) + (1 - brightness))
        out.mul_(bf)
        cf = (torch.rand(N, 1, 1, 1, device=device, dtype=dtype)
              * (2 * contrast) + (1 - contrast))
        mean = out.mean(dim=(-2, -1), keepdim=True)
        out.sub_(mean).mul_(cf).add_(mean)
        sf = (torch.rand(N, 1, 1, 1, device=device, dtype=dtype)
              * (2 * saturation) + (1 - saturation))
        gray = (out * self.gray_w).sum(dim=1, keepdim=True)
        out.sub_(gray).mul_(sf).add_(gray)
        if hue > 0:
            out = _hue_jitter(out, hue)
        out.clamp_(0.0, 1.0)
        apply_mask = (torch.rand(N, device=device) < p).view(-1, 1, 1, 1)
        return torch.where(apply_mask, out, images)

    def _grayscale(self, images: torch.Tensor, p: float = 0.2) -> torch.Tensor:
        N = images.shape[0]
        gray = (images * self.gray_w).sum(dim=1, keepdim=True).expand_as(images)
        apply_mask = (torch.rand(N, device=images.device) < p).view(-1, 1, 1, 1)
        return torch.where(apply_mask, gray, images)

    def _blur_gaussian(self, images: torch.Tensor) -> torch.Tensor:
        return F.conv2d(images, self.blur_w,
                        padding=self.blur_pad, groups=3)

    def _blur_gaussian_only(self, images: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """LeJEPA-style gaussian blur with per-image probability mask."""
        if p <= 0.0:
            return images
        N = images.shape[0]
        blurred = self._blur_gaussian(images)
        if p >= 1.0:
            return blurred
        apply_mask = (torch.rand(N, device=images.device) < p).view(-1, 1, 1, 1)
        return torch.where(apply_mask, blurred, images)

    def _blur_motion(self, images: torch.Tensor) -> torch.Tensor:
        idx = random.randrange(self.n_motion_angles)
        return F.conv2d(images, self.motion_w[idx],
                        padding=self.motion_pad, groups=3)

    def _blur_local(self, images: torch.Tensor) -> torch.Tensor:
        """Heavy blur in K random soft circular regions per image.

        Per call: picks one sigma from ``local_blur_sigmas`` (so blur strength
        varies between calls) and applies the corresponding cached kernel as
        a separable conv. Per image: draws ``num_local_blur_blobs`` independent
        random circular blobs whose gaussian-feathered masks are unioned
        (taken via ``max``) so blobs may overlap or be disjoint naturally.
        """
        N, _, H, W = images.shape
        device, dtype = images.device, images.dtype

        s_idx = random.randrange(self.n_local_sigmas)
        h_kernel = getattr(self, f"local_blur_h_{s_idx}")
        v_kernel = getattr(self, f"local_blur_v_{s_idx}")
        pad = self._local_blur_pads[s_idx]
        blurred = F.conv2d(images, h_kernel, padding=(0, pad), groups=3)
        blurred = F.conv2d(blurred, v_kernel, padding=(pad, 0), groups=3)

        K = self.num_local_blur_blobs
        R_lo, R_hi = self.local_blur_radius_frac
        radius = (
            torch.rand(K, N, device=device, dtype=dtype) * (R_hi - R_lo) + R_lo
        ) * float(min(H, W))
        cx = torch.rand(K, N, device=device, dtype=dtype) * float(W)
        cy = torch.rand(K, N, device=device, dtype=dtype) * float(H)

        yy = torch.arange(H, device=device, dtype=dtype).view(1, 1, 1, H, 1)
        xx = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, 1, W)
        cx_b = cx.view(K, N, 1, 1, 1)
        cy_b = cy.view(K, N, 1, 1, 1)
        r_b = radius.view(K, N, 1, 1, 1)

        # Flat-top per-blob mask: fully 1 inside ``core_frac * r``, linear
        # falloff from 1 to 0 between ``core_frac * r`` and ``r``, then 0.
        # This destroys all info inside the core (vs gaussian falloff which
        # only partially blends).
        dist2 = (xx - cx_b) ** 2 + (yy - cy_b) ** 2
        dist = dist2.sqrt()
        edge_width = 1.0 - self.local_blur_core_frac
        masks = ((1.0 - dist / r_b) / edge_width).clamp_(min=0.0, max=1.0)
        mask = masks.amax(dim=0)                     # soft union over K blobs
        return mask * blurred + (1.0 - mask) * images

    def _blur(self, images: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """Pick one of {gaussian, motion, local} per call, then per-image apply mask.

        Per-call (not per-image) choice keeps the conv work to a single kernel
        application; randomness across calls/views still gives the network all
        three blur types.
        """
        if p <= 0.0:
            return images
        N = images.shape[0]
        choice = random.random()
        if choice < 1.0 / 3.0:
            blurred = self._blur_gaussian(images)
        elif choice < 2.0 / 3.0:
            blurred = self._blur_motion(images)
        else:
            blurred = self._blur_local(images)
        if p >= 1.0:
            return blurred
        apply_mask = (torch.rand(N, device=images.device) < p).view(-1, 1, 1, 1)
        return torch.where(apply_mask, blurred, images)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Augmentation: RRC → Jitter → Gray → Blur → Solarize → HFlip → Norm.

        For CIFAR (32×32), set blur_p=0 and solarize_p=0 to match VICReg recipe.
        For ImageNet-scale, use blur_p=0.1 and solarize_p=0.2 (LeJEPA defaults).
        """
        x = self._rrc(x)
        x = self._color_jitter(x, brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1, p=0.8)
        x = self._grayscale(x, p=0.2)
        if self.blur_p > 0:
            x = self._blur_gaussian_only(x, p=self.blur_p)
        if self.solarize_p > 0:
            x = _solarize(x, threshold=self.solarize_threshold, p=self.solarize_p)
        x = _random_hflip(x, p=0.5)
        return self._normalize(x)

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """images: (N, C, H, W) float in [0, 1] on GPU."""
        cs = self.crop_size
        H, W = images.shape[-2], images.shape[-1]
        if H < cs or W < cs:
            images = F.interpolate(images, size=cs, mode="bilinear",
                                   align_corners=False)
            H = W = cs
        top = (H - cs) // 2
        left = (W - cs) // 2
        orig = self._normalize(images[:, :, top:top + cs, left:left + cs])

        V = self.num_aug_views
        N = images.shape[0]
        tiled = images.repeat(V, 1, 1, 1)          # (V*N, C, H, W)
        aug_flat = self._augment(tiled)              # (V*N, C, cs, cs)
        aug = aug_flat.view(V, N, *aug_flat.shape[1:])
        return orig, aug


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

    def epoch_batches(self, batch_size: int, generator: torch.Generator):
        """Yield (images, labels) batches without replacement for one epoch.

        Shuffles all N indices via randperm, yields consecutive slices.
        Drops the last incomplete batch.
        """
        perm = torch.randperm(self.N, device=self.images.device, generator=generator)
        for i in range(self.N // batch_size):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            yield self.images[idx], self.labels[idx]


class InMemoryEvalLoader:
    """Iterates an InMemoryGPUDataset in deterministic batches for evaluation.

    Applies normalization on the fly. No augmentation.
    """

    def __init__(self, dataset: InMemoryGPUDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return (self.dataset.N + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        N, B = self.dataset.N, self.batch_size
        for start in range(0, N, B):
            end = min(start + B, N)
            # uint8 [0, 255] → float [-1, 1]
            x = self.dataset.images[start:end].float().div_(127.5).sub_(1.0)
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
    # For SSL pretraining we want every non-test image. Torchvision's default
    # "train" split is tiny for several datasets (STL10 ignores 100k unlabeled
    # images; flowers102/dtd hide half the data in "val"; aircraft has a
    # separate "trainval" split). Pretraining doesn't need labels, so label=-1
    # on STL10's unlabeled portion is harmless.
    split = "train" if train else "test"
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(root=root, train=train, download=download)
    elif name == "cifar10":
        return torchvision.datasets.CIFAR10(root=root, train=train, download=download)
    elif name == "stl10":
        stl_split = "train+unlabeled" if train else "test"
        return torchvision.datasets.STL10(root=root, split=stl_split, download=download)
    elif name == "flowers102":
        if train:
            return ConcatDataset([
                torchvision.datasets.Flowers102(root=root, split="train", download=download),
                torchvision.datasets.Flowers102(root=root, split="val", download=download),
            ])
        return torchvision.datasets.Flowers102(root=root, split="test", download=download)
    elif name == "food101":
        return torchvision.datasets.Food101(root=root, split=split, download=download)
    elif name == "dtd":
        if train:
            return ConcatDataset([
                torchvision.datasets.DTD(root=root, split="train", download=download),
                torchvision.datasets.DTD(root=root, split="val", download=download),
            ])
        return torchvision.datasets.DTD(root=root, split="test", download=download)
    elif name == "aircraft":
        aircraft_split = "trainval" if train else "test"
        return torchvision.datasets.FGVCAircraft(root=root, split=aircraft_split, download=download)
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
        - ``gpu_aug`` is a ``GPUAug`` module on ``device``.
    """
    train_base = _load_base_dataset(cfg.dataset, cfg.data_dir, train=True)
    val_base = _load_base_dataset(cfg.dataset, cfg.data_dir, train=False)

    gpu_aug = GPUAug(
        num_aug_views=cfg.num_aug_views,
        crop_size=cfg.crop_size,
        crop_scale=tuple(cfg.crop_scale),
        blur_p=cfg.blur_p,
        solarize_p=cfg.solarize_p,
    ).to(device)

    if cfg.dataset in IN_MEMORY_DATASETS:
        # Cache at cfg.cache_resolution (auto-resolved in Config.__post_init__
        # to native_size — no wasteful upscaling for small datasets).
        train_ds = InMemoryGPUDataset(train_base, device,
                                      pre_resize=cfg.cache_resolution)
        val_ds = InMemoryGPUDataset(val_base, device,
                                    pre_resize=cfg.cache_resolution)
        val_loader = InMemoryEvalLoader(val_ds, cfg.batch_size)
        return train_ds, val_loader, gpu_aug

    # Fallback: DataLoader path for larger datasets
    train_ds = RawImageDataset(train_base, pre_resize=cfg.cache_resolution)
    val_ds = RawImageDataset(val_base, pre_resize=cfg.cache_resolution)

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
