"""Configuration for LeJEPA pretraining.

Per-dataset defaults from DATASET_INFO. Hyperparameters follow the LeJEPA
paper (Section 6.1, Tables 4-5).
"""

import argparse
from dataclasses import dataclass, field, fields
from typing import Tuple


DATASET_INFO = {
    # Paper Table 5: per-dataset architectures and training budgets.
    "cifar100":   {"num_classes": 100, "train_size": 50000,  "default_epochs": 400},
    "cifar10":    {"num_classes": 10,  "train_size": 50000,  "default_epochs": 400},
    "food101":    {"num_classes": 101, "train_size": 75750,  "default_epochs": 400},
    "stl10":      {"num_classes": 10,  "train_size": 5000,   "default_epochs": 400},
    "pets":       {"num_classes": 37,  "train_size": 3680,   "default_epochs": 800},
    "aircraft":   {"num_classes": 100, "train_size": 3334,   "default_epochs": 800},
    "dtd":        {"num_classes": 47,  "train_size": 1880,   "default_epochs": 1000},
    "flowers102": {"num_classes": 102, "train_size": 1020,   "default_epochs": 2000},
}


# encoder_scale → timm model name. For ViTs we always use the patch16 base
# model and override patch_size at create time (timm doesn't ship a
# vit_tiny_patch8_224 entry, but the architecture supports any patch size).
BACKBONE_MAP = {
    # ViTs (patch_size overridden in models.py)
    "tiny":   "vit_tiny_patch16_224",
    "small":  "vit_small_patch16_224",
    "base":   "vit_base_patch16_224",
    # CNNs (patch_size ignored)
    "resnet18":        "resnet18",
    "resnet34":        "resnet34",
    "resnet50":        "resnet50",
    "convnextv2_nano": "convnextv2_nano.fcmae",
    "convnext_tiny":   "convnext_tiny",
}


def resolve_backbone(encoder_scale: str) -> str:
    """Resolve an encoder_scale to a timm model name."""
    if encoder_scale not in BACKBONE_MAP:
        # Treat as a literal timm name
        return encoder_scale
    return BACKBONE_MAP[encoder_scale]


def is_vit(backbone_name: str) -> bool:
    return backbone_name.startswith("vit_") or backbone_name.startswith("deit_")


@dataclass
class Config:
    # Data
    dataset: str = "cifar100"
    data_dir: str = "./data"
    patch_size: int = 16                 # timm default; paper uses 8 only for ImageNet ViT-S
    num_workers: int = 8

    # Multi-crop (paper Sec 6.1: V_g=2, V_l=6)
    # Default global resolution is 128 (not the paper's 224) for ~3x speedup
    # in our compute-bound regime — see EXPERIMENT_PLAN.md "Compute-bound
    # vs memory-bound" note. Run a sanity check at 224 for the supplementary.
    num_global_views: int = 2
    num_local_views: int = 6             # auto-disabled for non-ViT backbones
    global_crop_size: int = 128
    local_crop_size: int = 56            # ~ 96 * (128/224)
    global_crop_scale: Tuple[float, float] = (0.4, 1.0)
    local_crop_scale: Tuple[float, float] = (0.05, 0.4)

    # In-memory dataset cache (small datasets only)
    cache_resolution: int = -1           # -1 → use global_crop_size

    # Model
    encoder_scale: str = "tiny"          # "tiny"/"small"/"base"/"resnet18"/...
    backbone_name: str = ""              # explicit override; empty → from encoder_scale
    proj_hidden: int = 512
    proj_dim: int = 64
    num_classes: int = -1                # auto from dataset

    # Regularizer
    regularizer: str = "sigreg"          # "sigreg" | "w1" | "w2"
    accumulate: bool = False             # turn on 2-step pooled training
    accum_steps: int = 8                 # T: pool = current BS + (T-1)*BS no-grad
    num_proj: int = 1024
    lambd: float = 0.05

    # Training
    batch_size: int = 64
    epochs: int = -1                     # auto from dataset
    lr: float = 1e-3
    weight_decay: float = 5e-2
    warmup_epochs: int = 1
    eta_min: float = 1e-5
    use_compile: bool = False
    grad_checkpoint: bool = False        # enable for memory-tight ViT configs
    seed: int = 42

    # Probe
    probe_lr: float = 1e-3
    probe_wd: float = 0.0

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

        # Resolve backbone name from encoder_scale if not explicitly set
        if not self.backbone_name:
            self.backbone_name = resolve_backbone(self.encoder_scale)

        # Auto-disable local views for non-ViT backbones
        if not is_vit(self.backbone_name) and self.num_local_views > 0:
            self.num_local_views = 0

    @classmethod
    def from_cli(cls) -> "Config":
        parser = argparse.ArgumentParser(description="LeJEPA Pretraining")
        # Field names that are 2-float tuples (handled specially in argparse)
        TUPLE_FIELDS = {"global_crop_scale", "local_crop_scale"}
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
