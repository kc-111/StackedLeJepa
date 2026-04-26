"""Predictor-loss variant of train.py.

Same skeleton as ``train.py`` but the invariance loss is replaced with
an *augmentation-conditioned predictor* head. For each sample we draw
two views v1, v2 and record the **full augmentation chain** that
produced each — every random decision lands in a per-view action
vector::

    t_i = (
        crop_cx, crop_cy, crop_w, crop_h,         # 4: random resized crop
        flip,                                     # 1: horizontal flip
        color_applied, brightness, contrast,
        saturation, hue,                          # 5: color jitter
        gray_applied,                             # 1: random grayscale
        blur_applied, blur_sigma,                 # 2: gaussian blur
        solarize_applied,                         # 1: solarize
    )

(14 numbers per view → 28 concatenated as ``t = [t_v1 ‖ t_v2]``.) The
predictor is a small MLP::

    g : (z_1, t)  ↦  ẑ_2

trained to predict v2's projection from v1's projection plus the
action. Loss::

    L = λ · reg((V=2, B, D))  +  (1 − λ) · (pred + inv)

with::

    pred = mean ‖ẑ_2 − z_2‖² / D                          (no margin)
    inv  = mean[ max(0, ‖z_v − z̄‖² − margin) ] / D       (margin = inv_tol · D/2)

The ``inv`` term is a small *anchor* on the projected embeddings: the
predictor alone could in principle satisfy ``pred`` while letting z_1
and z_2 drift apart in directions the action vector encodes — the
encoder would just route distribution shifts through z_1's null space
that the predictor's MLP can rotate out. Adding the centroid-MSE term
(same form as train.py's ``mse`` invariance with the same margin
recipe) keeps the projection space anchored so the predictor has
something stable to map between.

Why one-way, not symmetric — we want the encoder to produce a z_1 the
predictor can *forward-map* with only a small action vector to z_2.
Symmetrizing makes the loss easier to satisfy without forcing this.

Why concatenated action, not delta — t_v1 says what frame z_1 is in,
t_v2 says what frame to land in; the MLP figures out the relation.
A delta bakes in a group structure the photometric ops don't satisfy
(solarize isn't invertible).

Why every aug — color jitter / grayscale / blur / solarize aren't
invertible from t alone (they depend on image content), but knowing
they were applied lets the predictor *expect* the corresponding
distribution shift in z_2 instead of treating it as noise. This gives
the encoder an incentive to retain enough content info in z_1 for the
predictor to forecast that shift — i.e. *not* collapse to invariance.

Usage:
    python experiments/train/train_pred.py --dataset cifar10 \\
        --backbone resnet18 --regularizer w1 --epochs 200
"""

import argparse
import random
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import SlicedEppsPulley

sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses import make_regularizer  # noqa: E402
from train import (  # noqa: E402
    _DATASETS,
    allocate_run,
    make_backbone,
    make_projector,
)


# ---------------------------------------------------------------------------
# Per-view action vector layout
# ---------------------------------------------------------------------------
# Continuous params normalized so the predictor's first linear sees roughly
# matched scales. Identity values fill in when an op is gated off — that
# way the encoder can't tell "off" from "no-change with these factors" any
# more than the image can.
#
#   idx  field             range          identity (gate=0)
#   0    crop_cx           [0, 1]         (always applied)
#   1    crop_cy           [0, 1]
#   2    crop_w            (0, 1]
#   3    crop_h            (0, 1]
#   4    flip              {0, 1}         0
#   5    color_applied     {0, 1}         0
#   6    brightness        [-1, 1]        0   (factor 1 → 0 after norm)
#   7    contrast          [-1, 1]        0
#   8    saturation        [-1, 1]        0
#   9    hue               [-1, 1]        0   (factor 0 → 0)
#   10   gray_applied      {0, 1}         0
#   11   blur_applied      {0, 1}         0
#   12   blur_sigma        [0, 1]         0   (sigma 0 → 0)
#   13   solarize_applied  {0, 1}         0

T_DIM_PER_VIEW = 14
T_DIM = 2 * T_DIM_PER_VIEW

# Match train.py's ColorJitter recipe.
BRIGHTNESS_RANGE = 0.4
CONTRAST_RANGE = 0.4
SATURATION_RANGE = 0.2
HUE_RANGE = 0.1
COLOR_APPLY_P = 0.8
GRAY_APPLY_P = 0.2
BLUR_APPLY_P = 0.5
BLUR_SIGMA_RANGE = (0.1, 2.0)
SOLARIZE_APPLY_P = 0.2
SOLARIZE_THRESHOLD = 128  # PIL uint8 — applied before ToTensor


# ---------------------------------------------------------------------------
# Pair-augmentation dataset: emits (image_v1, image_v2, t_v1, t_v2, label)
# ---------------------------------------------------------------------------

class _PairAugDataset(Dataset):
    """Wrap a (PIL, label)-yielding source. Each sample is a dict::

        {image_v1, image_v2, t_v1, t_v2, label}

    where ``t_*`` records every random parameter from the augmentation
    chain that produced that view. We bypass spt's MultiViewTransform
    here because its ``Compose`` of v2 transforms hides the per-call
    sampled parameters — we need them.
    """

    def __init__(self, src, image_size, mean, std,
                 scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                 photo_aug=True):
        self.src = src
        self.image_size = image_size
        self._mean = mean
        self._std = std
        self.scale = scale
        self.ratio = ratio
        self.photo_aug = photo_aug

    def __len__(self):
        return len(self.src)

    def _view(self, img):
        """Sample one full augmentation chain on ``img`` and return
        (normalized tensor view, T_DIM_PER_VIEW action vector).

        Photometric ops use TF.* directly so we can record both the
        applied bit and the sampled continuous parameter. T.ColorJitter
        would shuffle the apply order each call AND hide its samples,
        so we don't use it.
        """
        W, H = img.size

        # 1. Geometry (always applied).
        i, j, h, w = _sample_resized_crop(img, self.scale, self.ratio)
        flip = random.random() < 0.5
        v = TF.resized_crop(img, i, j, h, w,
                            [self.image_size, self.image_size])
        if flip:
            v = TF.hflip(v)

        # Identity defaults for gated photometric ops.
        color_applied = 0
        b_factor, c_factor, s_factor, h_factor = 1.0, 1.0, 1.0, 0.0
        gray_applied = 0
        blur_applied = 0
        sigma = 0.0
        solar_applied = 0

        if self.photo_aug:
            # 2. Color jitter — fixed b → c → s → h order so action ↔
            #    pixels map is consistent across calls.
            if random.random() < COLOR_APPLY_P:
                color_applied = 1
                b_factor = random.uniform(1.0 - BRIGHTNESS_RANGE,
                                          1.0 + BRIGHTNESS_RANGE)
                c_factor = random.uniform(1.0 - CONTRAST_RANGE,
                                          1.0 + CONTRAST_RANGE)
                s_factor = random.uniform(1.0 - SATURATION_RANGE,
                                          1.0 + SATURATION_RANGE)
                h_factor = random.uniform(-HUE_RANGE, HUE_RANGE)
                v = TF.adjust_brightness(v, b_factor)
                v = TF.adjust_contrast(v, c_factor)
                v = TF.adjust_saturation(v, s_factor)
                v = TF.adjust_hue(v, h_factor)

            # 3. Random grayscale (3-channel out preserves tensor shape).
            if random.random() < GRAY_APPLY_P:
                gray_applied = 1
                v = TF.rgb_to_grayscale(v, num_output_channels=3)

            # 4. Gaussian blur via PIL — sigma-only, no kernel sizing
            #    edge cases on small images (cifar 32×32).
            if random.random() < BLUR_APPLY_P:
                blur_applied = 1
                sigma = random.uniform(*BLUR_SIGMA_RANGE)
                v = v.filter(ImageFilter.GaussianBlur(radius=sigma))

            # 5. Solarize (uint8 PIL — must run before ToTensor below).
            if random.random() < SOLARIZE_APPLY_P:
                solar_applied = 1
                v = ImageOps.solarize(v, threshold=SOLARIZE_THRESHOLD)

        # 6. Tensor + per-dataset normalize.
        v = TF.to_tensor(v)
        v = TF.normalize(v, mean=self._mean, std=self._std)

        t = torch.tensor([
            (j + w / 2) / W,
            (i + h / 2) / H,
            w / W,
            h / H,
            float(flip),
            float(color_applied),
            (b_factor - 1.0) / BRIGHTNESS_RANGE,
            (c_factor - 1.0) / CONTRAST_RANGE,
            (s_factor - 1.0) / SATURATION_RANGE,
            h_factor / HUE_RANGE,
            float(gray_applied),
            float(blur_applied),
            sigma / BLUR_SIGMA_RANGE[1],
            float(solar_applied),
        ], dtype=torch.float32)
        return v, t

    def __getitem__(self, idx):
        img, label = self.src[idx]
        if not isinstance(img, Image.Image):
            img = TF.to_pil_image(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        v1, t1 = self._view(img)
        v2, t2 = self._view(img)
        return {
            "image_v1": v1,
            "image_v2": v2,
            "t_v1": t1,
            "t_v2": t2,
            "label": int(label),
        }


def _sample_resized_crop(img, scale, ratio):
    """Wraps torchvision's RandomResizedCrop sampling; pulled out so the
    import surface stays small and the call site reads cleanly."""
    from torchvision.transforms import RandomResizedCrop
    return RandomResizedCrop.get_params(img, scale=list(scale),
                                        ratio=list(ratio))


# ---------------------------------------------------------------------------
# Data: train pair-aug + standard val
# ---------------------------------------------------------------------------

def make_data(name, batch_size, num_workers=8, image_size=None,
              photo_aug=True, scale=(0.08, 1.0)):
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(_DATASETS)}")
    loaders_fn, num_classes, default_size, norm = _DATASETS[name]
    size = image_size or default_size

    train_raw, val_raw = loaders_fn()

    train_ds = _PairAugDataset(
        src=train_raw, image_size=size,
        mean=norm["mean"], std=norm["std"],
        scale=scale, photo_aug=photo_aug,
    )
    # Val: same single-image pipeline as train.py — probes consume it.
    val_tf = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((size, size)),
        transforms.ToImage(**norm),
    )
    val_ds = spt.data.FromTorchDataset(
        val_raw, names=["image", "label"], transform=val_tf)

    train_dl = DataLoader(
        dataset=train_ds, batch_size=batch_size,
        num_workers=num_workers, drop_last=True, shuffle=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        dataset=val_ds, batch_size=batch_size, num_workers=num_workers)

    return spt.data.DataModule(train=train_dl, val=val_dl), num_classes


# ---------------------------------------------------------------------------
# Predictor MLP
# ---------------------------------------------------------------------------

def make_predictor(z_dim, t_dim, hidden=512):
    """Two-hidden-layer MLP: input = [z_1 ‖ t_cat], output = ẑ_2 (z_dim)."""
    return nn.Sequential(
        nn.Linear(z_dim + t_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, z_dim),
    )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
#
# Train: V=2 paired views per sample. Both run through backbone+projector,
# producing z_1, z_2. Predictor maps (z_1, t_cat) → ẑ_2 and we hinge-MSE
# against z_2. Distribution loss runs on the (V=2, B, D) stack — same
# per-view averaging as train.py's FIFO-off branch.
#
# Eval: standard single-image forward (val pipeline), so OnlineProbe's
# input="embedding"/"projection" finds what it needs.

def make_pred_forward():
    def pred_forward(self, batch, stage):
        out = {}

        if "image_v1" in batch:
            v1 = batch["image_v1"]
            v2 = batch["image_v2"]
            t1 = batch["t_v1"]
            t2 = batch["t_v2"]
            t_cat = torch.cat([t1, t2], dim=-1)  # (B, T_DIM)

            h1 = self.backbone(v1)
            h2 = self.backbone(v2)
            z1 = self.projector(h1)
            z2 = self.projector(h2)

            # One-way: predict z_2 from z_1 + actions. No stop-grad on z_2 —
            # the encoder should learn from this signal too (dropping grad
            # through z_2 would let the encoder satisfy the loss by ignoring
            # actions in z_1, since the target moves with it).
            pred = self.predictor(torch.cat([z1, t_cat], dim=-1))

            # Regularizer: per-view (V=2) by default. SlicedEppsPulley
            # only takes [N, D] so it always gets the flat pool; --flatten-reg
            # opts our regs into the same one-big-bag behavior (biases the
            # estimator since across-view samples are correlated).
            z_stack = torch.stack([z1, z2], dim=0)  # (V=2, B, D)
            V, B, D = z_stack.shape
            do_flatten = self.flatten_reg or isinstance(
                self.regularizer, SlicedEppsPulley)
            if do_flatten:
                flat = z_stack.reshape(-1, D)
                reg = self.regularizer(flat)
                pool_rows = flat.shape[0]
            else:
                reg = self.regularizer(z_stack)  # averages over V
                pool_rows = B
            if self.regularizer.needs_compensation:
                reg = reg * (pool_rows / B)

            # Prediction loss: plain MSE between ẑ_2 and z_2. The predictor
            # has the action vector and can in principle fit exactly, so
            # we don't apply a margin here — leaving residual energy means
            # the encoder is hiding action-conditioned info in directions
            # the predictor can't reach.
            pred_loss = (pred - z2).square().sum(dim=-1).mean() / D

            # Invariance anchor: centroid MSE on z_1, z_2. Same margin recipe
            # as train.py's mse term — under N(0, I), E‖z_v − z̄‖² = D·(V−1)/V
            # = D/2 here, so margin = inv_tol · D/2 zeroes the penalty below
            # that floor (anything tighter fights the regularizer). Without
            # this term, pred + reg could be satisfied by z_1, z_2 drifting
            # apart along action-correlated axes that the MLP rotates out.
            mean_z = z_stack.mean(dim=0, keepdim=True)             # (1, B, D)
            per_sample_sq = (z_stack - mean_z).square().sum(dim=-1)  # (V, B)
            prior_floor = D * (V - 1) / V
            margin = self.inv_tol * prior_floor
            inv_loss = torch.clamp(per_sample_sq - margin, min=0.0).mean() / D

            # Diagnostic: fraction of (V, B) entries actively contributing
            # (above the margin). 1 at inv_tol=0 with non-collapsed projs;
            # falls toward 0 as the anchor pulls samples below the floor —
            # if it saturates near 0 the inv term has effectively turned off
            # (lower --inv-tol if you want it to keep working).
            inv_margin_active = (per_sample_sq > margin).float().mean()

            loss = self.lambd * reg + (1.0 - self.lambd) * (inv_loss + pred_loss)
            out["loss"] = loss

            # Probes: concat both views' features + duplicated labels.
            out["embedding"] = torch.cat([h1, h2], dim=0)
            out["projection"] = torch.cat([z1, z2], dim=0)
            if "label" in batch:
                out["label"] = torch.cat([batch["label"], batch["label"]], dim=0)

            self.log(f"{stage}/pred_loss", pred_loss,
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/inv_loss", inv_loss,
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/inv_margin_active", inv_margin_active,
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/reg_loss", reg,
                     on_step=False, on_epoch=True, sync_dist=True)
        else:
            # Eval / inference: single-image pipeline.
            emb = self.backbone(batch["image"])
            out["embedding"] = emb
            out["projection"] = self.projector(emb)
            if "label" in batch:
                out["label"] = batch["label"]

        return out

    return pred_forward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="imagenette",
                   choices=["cifar10", "cifar100", "imagenette", "imagenet-100"])
    p.add_argument("--image-size", type=int, default=None,
                   help="Override dataset default image size")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--regularizer", default="w1",
                   choices=["sigreg", "sigreg_raw", "w1", "w2"])
    p.add_argument("--lambd", type=float, default=0.95,
                   help="Weight on reg vs prediction loss.")
    p.add_argument("--inv-tol", type=float, default=0.1,
                   help="Margin on the *invariance anchor* (centroid MSE on "
                        "z_1, z_2), as a fraction of the N(0, I) prior floor "
                        "D·(V−1)/V (V=2 → D/2). 0 = strict invariance; higher "
                        "= more slack so reg has room to spread the projections. "
                        "Prediction loss is unmargined.") # callback near 1 means margin is active.
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--proj-hidden", type=int, default=2048)
    p.add_argument("--pred-hidden", type=int, default=2048,
                   help="Predictor MLP hidden width.")
    p.add_argument("--num-proj", type=int, default=2048)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--flatten-reg", action="store_true",
                   help="Flatten (V=2, B, D) → (V*B, D) before reg. "
                        "Matches the reference LeJEPA impl but biases the "
                        "estimator (across-view samples are correlated).")
    p.add_argument("--photo-aug", choices=["none", "full"], default="full",
                   help="full: keep color jitter / grayscale / blur / "
                        "solarize on top of crop+flip. none: only crop+flip "
                        "(then the action vector is mostly the geometric bits).")
    p.add_argument("--scale-min", type=float, default=0.3)
    p.add_argument("--scale-max", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed. If omitted, auto-picks the smallest "
                        "nonneg int unused by prior runs with identical hparams.")
    return p


def main():
    args = build_parser().parse_args()

    # SlicedEppsPulley always sees flat pool — record that in args so
    # post-hoc analysis isn't misled.
    if args.regularizer == "sigreg_raw":
        args.flatten_reg = True

    log_dir = Path(__file__).resolve().parent / "logs"
    run_dir, run_name, seed = allocate_run(
        log_dir, args.dataset, f"{args.backbone}_pred", vars(args), args.seed)
    args.seed = seed
    pl.seed_everything(seed, workers=True)
    print(f"[run] {run_name}  seed={seed}  dir={run_dir}")

    data, num_classes = make_data(
        args.dataset, args.batch_size, args.num_workers,
        image_size=args.image_size,
        photo_aug=args.photo_aug == "full",
        scale=(args.scale_min, args.scale_max),
    )
    low_res = args.dataset.startswith("cifar")
    backbone, emb_dim = make_backbone(args.backbone, low_resolution=low_res)
    projector = make_projector(emb_dim, args.proj_dim, args.proj_hidden)
    predictor = make_predictor(args.proj_dim, T_DIM, args.pred_hidden)
    regularizer = make_regularizer(
        args.regularizer, num_proj=args.num_proj, knots=args.knots)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        predictor=predictor,
        forward=make_pred_forward(),
        regularizer=regularizer,
        lambd=args.lambd,
        inv_tol=args.inv_tol,
        flatten_reg=args.flatten_reg,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )

    def _cls_metrics():
        return {
            "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
            "top5": torchmetrics.classification.MulticlassAccuracy(
                num_classes, top_k=5),
        }

    linear_probe_emb = spt.callbacks.OnlineProbe(
        module, name="linear_probe_emb",
        input="embedding", target="label",
        probe=nn.Linear(emb_dim, num_classes),
        loss=nn.CrossEntropyLoss(),
        metrics=_cls_metrics(),
    )
    linear_probe_proj = spt.callbacks.OnlineProbe(
        module, name="linear_probe_proj",
        input="projection", target="label",
        probe=nn.Linear(args.proj_dim, num_classes),
        loss=nn.CrossEntropyLoss(),
        metrics=_cls_metrics(),
    )
    knn_probe_emb = spt.callbacks.OnlineKNN(
        name="knn_probe_emb", input="embedding", target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes)},
        input_dim=emb_dim, k=20,
    )
    knn_probe_proj = spt.callbacks.OnlineKNN(
        name="knn_probe_proj", input="projection", target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes)},
        input_dim=args.proj_dim, k=20,
    )

    logger = CSVLogger(save_dir=str(log_dir), name=run_name, version="")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        save_last=True, save_top_k=0,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[
            linear_probe_emb, linear_probe_proj,
            knn_probe_emb, knn_probe_proj,
            ckpt_cb,
        ],
        precision="16-mixed",
        logger=logger,
    )

    # Same callback surgery as train.py: drop spt's env dump, swap the
    # default LoggingCallback for an eval-only one (printing inside
    # on_train_epoch_end so val + train aggregates land in the same table).
    from stable_pretraining.callbacks import (
        EnvironmentDumpCallback,
        LoggingCallback,
    )
    from lightning.pytorch.utilities import rank_zero_only
    from prettytable import PrettyTable
    import logging as _logging

    class EvalOnlyLoggingCallback(LoggingCallback):
        @rank_zero_only
        def on_validation_end(self, trainer, pl_module):
            pass  # printing happens in on_train_epoch_end below

        @rank_zero_only
        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            table = PrettyTable()
            table.field_names = ["Metric", "Value"]
            for key in sorted(metrics):
                if not (key.startswith("eval/") or key.startswith("fit/")):
                    continue
                if key.endswith("_epoch") and key[:-len("_epoch")] in metrics:
                    continue
                table.add_row([
                    "\033[0;34;40m" + key + "\033[0m",
                    "\033[0;32;40m" + str(metrics[key].item()) + "\033[0m",
                ])
            _logging.info(f"\n{table}")

    trainer.callbacks = [
        EvalOnlyLoggingCallback() if isinstance(cb, LoggingCallback) else cb
        for cb in trainer.callbacks
        if not isinstance(cb, EnvironmentDumpCallback)
    ]

    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=seed)
    manager()


if __name__ == "__main__":
    main()
