"""Configuration for LeJEPA pretraining.

Per-dataset defaults from DATASET_INFO. Hyperparameters follow the LeJEPA
paper (Section 6.1, Tables 4-5).
"""

import argparse
from dataclasses import dataclass, field, fields
from typing import Tuple


DATASET_INFO = {
    # Paper Table 5: per-dataset architectures and training budgets.
    # native_size: max(H, W) of the dataset's canonical image size.
    # All datasets normalized to [-1, 1] (no per-dataset mean/std needed).
    "cifar100":   {"num_classes": 100, "train_size": 50000,  "default_epochs": 400, "native_size": 32},
    "cifar10":    {"num_classes": 10,  "train_size": 50000,  "default_epochs": 400, "native_size": 32},
    "food101":    {"num_classes": 101, "train_size": 75750,  "default_epochs": 400, "native_size": 512},
    "stl10":      {"num_classes": 10,  "train_size": 5000,   "default_epochs": 400, "native_size": 96},
    "pets":       {"num_classes": 37,  "train_size": 3680,   "default_epochs": 800, "native_size": 256},
    "aircraft":   {"num_classes": 100, "train_size": 3334,   "default_epochs": 800, "native_size": 256},
    "dtd":        {"num_classes": 47,  "train_size": 1880,   "default_epochs": 1000, "native_size": 256},
    "flowers102": {"num_classes": 102, "train_size": 1020,   "default_epochs": 2000, "native_size": 256},
}


# encoder_scale → timm model name. ViT entries reference the patch16_224
# registry key only because timm doesn't ship patch8 variants for tiny/small;
# the effective patch size comes from cfg.patch_size (default 8), which is
# passed to timm.create_model and overrides the name's "_patch16_".
#
# The default for new experiments is convnextv2_nano (LayerNorm-only CNN,
# ~15M params). BatchNorm-based CNNs (resnet18/34/50) also work with pooled
# training — both passes run in train mode with independent batch stats.
BACKBONE_MAP = {
    # ViTs (patch_size overridden in models.py) — LayerNorm-based, pooled-safe
    "tiny":   "vit_tiny_patch16_224",
    "small":  "vit_small_patch16_224",
    "base":   "vit_base_patch16_224",
    # ConvNeXtV2 / ConvNeXt CNNs — LayerNorm-only, pooled-safe
    "convnextv2_atto": "convnextv2_atto",
    "convnextv2_femto": "convnextv2_femto",
    "convnextv2_pico": "convnextv2_pico",
    "convnextv2_nano": "convnextv2_nano.fcmae",
    "convnext_tiny":   "convnext_tiny",
    # ResNets — BatchNorm.
    "resnet18":        "resnet18",
    "resnet34":        "resnet34",
    "resnet50":        "resnet50",
}


def resolve_backbone(encoder_scale: str) -> str:
    """Resolve an encoder_scale to a timm model name."""
    if encoder_scale not in BACKBONE_MAP:
        # Treat as a literal timm name
        return encoder_scale
    return BACKBONE_MAP[encoder_scale]


def is_vit(backbone_name: str) -> bool:
    return backbone_name.startswith("vit_") or backbone_name.startswith("deit_")


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def auto_crop_size(dataset: str, cap: int = 128) -> int:
    """Determine crop size from native resolution.

    If native <= cap: use next_pow2(native) (GPU-friendly, no wasteful upscale).
    If native > cap: use cap (crop down).
    """
    info = DATASET_INFO.get(dataset, {})
    native = info.get("native_size", cap)
    if native <= cap:
        return _next_pow2(native)
    return cap


@dataclass
class Config:
    # Data
    dataset: str = "cifar100"
    data_dir: str = "./data"
    patch_size: int = 8                  # paper default for ViT-S; also needed so 56px local crops divide evenly
    num_workers: int = 8

    # Augmentation: one unaugmented original + N augmented views.
    # crop_size=-1 auto-resolves to next_pow2(native) capped at 128:
    #   CIFAR (32px) → 32, STL-10 (96px) → 128, larger → 128.
    num_aug_views: int = 1
    crop_size: int = -1
    crop_scale: Tuple[float, float] = (0.2, 1.0)
    # Blur and solarize probabilities. For CIFAR (32×32), set both to 0
    # to match VICReg recipe. For ImageNet-scale, use 0.1 and 0.2 (LeJEPA).
    blur_p: float = -1.0                  # -1 = auto (0 for CIFAR, 0.1 otherwise)
    solarize_p: float = -1.0              # -1 = auto (0 for CIFAR, 0.2 otherwise)

    # In-memory dataset cache (small datasets only).
    # -1 → use native_size (no upscaling); for variable-size datasets the
    # cache pre-resizes to crop_size so images can be stacked.
    cache_resolution: int = -1

    # Model
    encoder_scale: str = "convnextv2_pico"
    backbone_name: str = ""              # explicit override; empty → from encoder_scale
    proj_hidden: int = 4096               # authors use 2048 (>= hidden_dim)
    proj_dim: int = 64
    num_classes: int = -1                # auto from dataset

    # Regularizer
    regularizer: str = "sigreg"          # "sigreg" | "w1" | "w2" | "sigreg+w1" | "sigreg+w2"
    accumulate: bool = False             # turn on 2-step pooled training
    nograd_pool_size: int = 0            # number of no-grad samples for regularizer pool (0=grad batch only)
    fifo_size: int = 0                   # number of samples to retain in FIFO buffer (0=disabled)
    single_view_reg: bool = False        # use 1 random view per image for regularizer (required if fifo_size > 0)
    num_proj: int = 2048
    lambd: float = 0.05
    lambd_emb: float = 0.0               # weight for embedding regularization (0=disabled)
    # If True, the view-invariance MSE is computed on the pre-projection
    # encoder embeddings, not the projector output. The distributional
    # regularizer still acts on the projector output. This decouples "what
    # we make Gaussian" (proj) from "what we make view-invariant" (emb),
    # which matters when proj_dim is small enough that pushing it to N(0,I)
    # could collapse the backbone to that dim.
    inv_on_emb: bool = False
    # No-grad views for invariance centroid estimation. These views are
    # forwarded without gradients and used only to improve the centroid
    # estimate in inv_loss_fn. Cost: num_inv_nograd_views extra forward
    # passes (no backward). 0=disabled (original behavior).
    num_inv_nograd_views: int = 0
    # If True, stopgrad the invariance centroid so each grad view is pulled
    # toward a fixed target (one-directional). If False, gradients flow
    # through the grad views' contribution to the centroid (bidirectional).
    detach_inv_centroid: bool = False

    # Training
    batch_size: int = 32
    epochs: int = -1                     # auto from dataset
    lr: float = 1e-3
    weight_decay: float = -1             # auto: 5e-4 for CNNs, 5e-2 for ViTs
    warmup_epochs: int = 1
    use_compile: bool = False
    grad_checkpoint: bool = False        # enable for memory-tight ViT configs
    seed: int = 42

    # Probe
    probe_on_emb: bool = True            # True=probe on emb (default), False=probe on proj
    probe_lr: float = 1e-3
    probe_wd: float = 1e-6

    # Logging
    log_interval: int = 50
    eval_interval: int = 5
    save_interval: int = 25
    save_dir: str = "runs"
    resume_from: str = ""
    swap_regularizer: bool = False

    # Continuation training: load encoder+probe from a base checkpoint, then
    # train K=epochs more with the chosen (regularizer, accumulate, batch_size).
    # Optimizer + scheduler are reset (linear warmup + cosine over the K epochs).
    continue_from: str = ""              # path to base checkpoint .pt; empty = from scratch

    def __post_init__(self):
        info = DATASET_INFO.get(self.dataset, {})
        self.num_classes = info.get("num_classes", self.num_classes)
        if self.epochs == -1:
            self.epochs = info.get("default_epochs", 400)

        # Auto crop size from native resolution
        if self.crop_size < 0:
            self.crop_size = auto_crop_size(self.dataset)

        # Auto cache resolution: native size (no upscaling).
        # For variable-size datasets, InMemoryGPUDataset detects this and
        # pre-resizes to the native_size anyway.
        if self.cache_resolution < 0:
            native = info.get("native_size", self.crop_size)
            self.cache_resolution = max(native, self.crop_size)

        # Resolve backbone name from encoder_scale if not explicitly set
        if not self.backbone_name:
            self.backbone_name = resolve_backbone(self.encoder_scale)

        # Auto weight decay: 5e-2 for ViTs, 5e-4 for CNNs (paper Sec 6.1)
        if self.weight_decay < 0:
            self.weight_decay = 5e-2 if is_vit(self.backbone_name) else 5e-4

        # Auto blur/solarize: use LeJEPA defaults for all resolutions
        # (Removing these for CIFAR didn't help with sigreg)
        if self.blur_p < 0:
            self.blur_p = 0.1
        if self.solarize_p < 0:
            self.solarize_p = 0.2

        # FIFO requires single_view_reg for i.i.d. correctness
        if self.fifo_size > 0 and not self.single_view_reg:
            raise ValueError(
                "fifo_size > 0 requires single_view_reg=True for i.i.d. samples. "
                "FIFO mixes samples across batches, so each image must contribute "
                "only 1 randomly-selected view to maintain independence."
            )

    @classmethod
    def from_cli(cls) -> "Config":
        parser = argparse.ArgumentParser(description="LeJEPA Pretraining")
        # Field names that are 2-float tuples (handled specially in argparse)
        TUPLE_FIELDS = {"crop_scale"}
        # IMPORTANT: use the *field* defaults (not post-init values), so that
        # post-init resolution (e.g. backbone_name from encoder_scale) still
        # runs after CLI parsing.
        for f in fields(cls):
            name = f"--{f.name.replace('_', '-')}"
            default_val = f.default
            if f.type is bool:
                parser.add_argument(name, action="store_true", default=default_val)
                parser.add_argument(f"--no-{f.name.replace('_', '-')}",
                                    dest=f.name, action="store_false")
            elif f.name in TUPLE_FIELDS:
                parser.add_argument(name, type=float, nargs=2,
                                    default=list(default_val))
            else:
                parser.add_argument(name, type=f.type, default=default_val)
        args = parser.parse_args()
        kwargs = {}
        for f in fields(cls):
            v = getattr(args, f.name)
            if f.name in TUPLE_FIELDS:
                v = tuple(v)
            kwargs[f.name] = v
        return cls(**kwargs)
