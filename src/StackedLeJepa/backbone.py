"""Backbone creation with automatic low-resolution stem patching.

ImageNet-designed stems aggressively downsample, destroying spatial
information on small images like CIFAR (32x32):

    ResNet:     7x7 stride-2 + maxpool stride-2  ->  32 -> 8   (4x reduction)
    ConvNeXt:   4x4 stride-4 patchify            ->  32 -> 8   (4x reduction)
    ViT:        handled via patch_size arg (no patching needed)

For low-resolution inputs (crop_size <= 64) we automatically replace
these with stride-1 convolutions:

    ResNet:     3x3 stride-1, maxpool -> Identity ->  32 -> 32  (1x)
    ConvNeXt:   3x3 stride-1                      ->  32 -> 32  (1x)

This matches what stable_pretraining does for ResNet via
``from_torchvision(..., low_resolution=True)`` and extends it to
ConvNeXt/ConvNeXtV2.
"""

import torch.nn as nn
import timm

_LOW_RES_THRESHOLD = 64


def _is_vit(name: str) -> bool:
    return name.startswith("vit_") or name.startswith("deit_")


def _patch_resnet_stem(backbone: nn.Module) -> bool:
    """ResNet: 7x7 stride-2 + maxpool -> 3x3 stride-1 + Identity."""
    if not (hasattr(backbone, "conv1") and hasattr(backbone, "maxpool")):
        return False
    old = backbone.conv1
    backbone.conv1 = nn.Conv2d(
        old.in_channels, old.out_channels,
        kernel_size=3, stride=1, padding=1, bias=False,
    )
    backbone.maxpool = nn.Identity()
    return True


def _patch_convnext_stem(backbone: nn.Module) -> bool:
    """ConvNeXt/V2: 4x4 stride-4 patchify -> 3x3 stride-1."""
    if not (hasattr(backbone, "stem") and isinstance(backbone.stem, nn.Sequential)):
        return False
    old_conv = backbone.stem[0]
    if not (isinstance(old_conv, nn.Conv2d) and old_conv.stride[0] >= 4):
        return False
    backbone.stem[0] = nn.Conv2d(
        old_conv.in_channels, old_conv.out_channels,
        kernel_size=3, stride=1, padding=1,
    )
    return True


def patch_stem_low_res(backbone: nn.Module) -> bool:
    """Auto-detect backbone family and patch stem for low-resolution inputs.

    Returns True if a patch was applied.
    """
    return _patch_resnet_stem(backbone) or _patch_convnext_stem(backbone)


def create_backbone(
    model_name: str,
    crop_size: int,
    patch_size: int | None = None,
    pretrained: bool = False,
    grad_checkpoint: bool = False,
) -> nn.Module:
    """Create a timm backbone with automatic low-res stem patching.

    Args:
        model_name: timm model name (e.g. "resnet18", "convnextv2_pico",
            "vit_tiny_patch16_224").
        crop_size: spatial size of training crops. If <= 64, the stem is
            patched for low-resolution inputs.
        patch_size: ViT patch size override (ignored for CNNs).
        pretrained: load pretrained weights.
        grad_checkpoint: enable gradient checkpointing.

    Returns:
        Backbone module with ``num_features`` attribute indicating
        the embedding dimension.
    """
    kwargs = dict(pretrained=pretrained, num_classes=0)

    if _is_vit(model_name):
        kwargs["dynamic_img_size"] = True
        kwargs["img_size"] = crop_size
        if patch_size is not None:
            kwargs["patch_size"] = patch_size

    backbone = timm.create_model(model_name, **kwargs)

    # Auto-patch stem for low-resolution inputs (CIFAR, etc.)
    if crop_size <= _LOW_RES_THRESHOLD and not _is_vit(model_name):
        patched = patch_stem_low_res(backbone)
        if patched:
            print(f"  [backbone] Patched {model_name} stem for "
                  f"low-res input ({crop_size}x{crop_size})")
        else:
            print(f"  [backbone] WARNING: could not patch {model_name} stem "
                  f"for low-res input ({crop_size}x{crop_size})")

    if grad_checkpoint and hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(True)

    return backbone
