"""Multi-head LeJEPA-style pretrainer.

Stripped-down sibling of ``train.py``: no FIFO, no extra-forward factor,
no extra no-grad views. Adds N independent projection heads at different
output dimensions, each with its own invariance + regularizer objective.
The total loss averages the per-head terms so its scale is independent
of the number of heads.

Usage:
    python experiments/train/train_multihead.py --dataset cifar10 \\
        --backbone resnet18 --regularizer w1 \\
        --proj-dims 64 128 256 --num-views 2 --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.forward import _get_views_list
from stable_pretraining.methods.lejepa import SlicedEppsPulley

sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses import make_regularizer  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _build_cifar(num_classes: int, cls):
    def loaders():
        return (
            cls(root=str(DATA_DIR), train=True, download=True),
            cls(root=str(DATA_DIR), train=False, download=True),
        )
    return loaders, num_classes


def _build_imagefolder(subdir: str, num_classes: int):
    def loaders():
        root = DATA_DIR / subdir
        return (
            torchvision.datasets.ImageFolder(root=str(root / "train")),
            torchvision.datasets.ImageFolder(root=str(root / "val")),
        )
    return loaders, num_classes


class _HFTupleAdapter(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        return row["image"], row["label"]


def _build_imagenet100_parquet(num_classes: int = 100):
    def loaders():
        import datasets as hf_datasets
        root = DATA_DIR / "imagenet-100" / "data"
        ds = hf_datasets.load_dataset("parquet", data_files={
            "train": sorted(str(p) for p in root.glob("train-*.parquet")),
            "validation": sorted(str(p) for p in root.glob("validation-*.parquet")),
        })
        return _HFTupleAdapter(ds["train"]), _HFTupleAdapter(ds["validation"])
    return loaders, num_classes


_DATASETS = {
    "cifar10": (*_build_cifar(10, torchvision.datasets.CIFAR10),
                32, spt.data.static.CIFAR10),
    "cifar100": (*_build_cifar(100, torchvision.datasets.CIFAR100),
                 32, spt.data.static.CIFAR100),
    "imagenette": (*_build_imagefolder("imagenette2-160", 10),
                   128, spt.data.static.ImageNet),
    "imagenet-100": (*_build_imagenet100_parquet(100),
                     224, spt.data.static.ImageNet),
}


def make_data(name: str, batch_size: int, num_workers: int = 8,
              num_views: int = 2, image_size: int = None):
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(_DATASETS)}")
    loaders_fn, num_classes, default_size, norm = _DATASETS[name]
    size = image_size or default_size

    aug = transforms.Compose(
        transforms.RGB(),
        # transforms.RandomResizedCrop((size, size), scale=(0.08, 1.0)),
        transforms.RandomResizedCrop((size, size), scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILGaussianBlur(p=0.5),
        transforms.RandomSolarize(p=0.2, threshold=0.5),
        transforms.ToImage(**norm),
    )
    train_tf = transforms.MultiViewTransform([aug] * num_views)
    val_tf = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((size, size)),
        transforms.ToImage(**norm),
    )

    train_raw, val_raw = loaders_fn()
    train_ds = spt.data.FromTorchDataset(
        train_raw, names=["image", "label"], transform=train_tf)
    val_ds = spt.data.FromTorchDataset(
        val_raw, names=["image", "label"], transform=val_tf)

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=batch_size,
        num_workers=num_workers, drop_last=True, shuffle=True)
    val_dl = torch.utils.data.DataLoader(
        dataset=val_ds, batch_size=batch_size, num_workers=num_workers)

    return spt.data.DataModule(train=train_dl, val=val_dl), num_classes


# ---------------------------------------------------------------------------
# Backbone + multi-head projector
# ---------------------------------------------------------------------------

_EMB_DIM = {"resnet18": 512, "resnet50": 2048}


def make_backbone(name: str, low_resolution: bool = True):
    if name in _EMB_DIM:
        b = spt.backbone.from_torchvision(name, low_resolution=low_resolution)
        return b, _EMB_DIM[name]
    raise ValueError(f"unknown backbone {name!r}")


def make_projector(in_dim: int, out_dim: int, hidden: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class MultiHeadProjector(nn.Module):
    """Bank of independent projectors mapping a shared embedding to N output dims."""

    def __init__(self, in_dim: int, out_dims, hidden: int = 2048):
        super().__init__()
        self.heads = nn.ModuleList([
            make_projector(in_dim, d, hidden) for d in out_dims
        ])
        self.out_dims = list(out_dims)

    def forward(self, emb):
        return [head(emb) for head in self.heads]


# ---------------------------------------------------------------------------
# Forward: per-head invariance + regularizer
# ---------------------------------------------------------------------------
#
# Each backbone forward processes one view of B images. Multi-head projector
# fans the embedding out into H tensors of shape (B, D_h). For each head we
# compute regularizer (per-view across V, averaged) and invariance (mean-target
# variance across V views). Total loss averages over heads — keeps the scale
# stable as H varies.

def make_lejepa_forward(num_views: int):
    def lejepa_forward(self, batch, stage):
        out = {}
        views = _get_views_list(batch)

        if views is not None:
            V = num_views
            embs = [self.backbone(v["image"]) for v in views[:V]]
            # projector returns a list of H tensors per view; transpose to
            # heads_z[h] = list of V tensors, each (B, D_h).
            per_view_heads = [self.projector(e) for e in embs]
            heads_z = [list(h) for h in zip(*per_view_heads)]
            H = len(heads_z)

            if self.training:
                total_inv = 0.0
                total_reg = 0.0

                for h, head_views in enumerate(heads_z):
                    D = head_views[0].size(-1)
                    B = head_views[0].size(0)

                    # --- Regularizer: per-view (iid-within-view), averaged over V ---
                    # SlicedEppsPulley only takes [N, D] flat input, so route it
                    # through the flat path; bias from cross-view non-iid is
                    # accepted as the cost of using the reference impl.
                    if isinstance(self.regularizer, SlicedEppsPulley):
                        flat = torch.cat(head_views, dim=0)  # (V*B, D)
                        reg_h = self.regularizer(flat)
                        pool_rows = flat.shape[0]
                    else:
                        cols = torch.stack(head_views, dim=0)  # (V, B, D)
                        reg_h = self.regularizer(cols)
                        pool_rows = cols.shape[1]
                    if getattr(self.regularizer, "needs_compensation", False):
                        reg_h = reg_h * (pool_rows / B)
                    total_reg = total_reg + reg_h
                    self.log(f"{stage}/reg_loss_h{h}", reg_h,
                             on_step=False, on_epoch=True, sync_dist=True)

                    # --- Invariance: mean-target variance across views ---
                    if V >= 2:
                        stacked = torch.stack(head_views, dim=0)  # (V, B, D)
                        mean_z = stacked.mean(dim=0, keepdim=True)
                        norms = stacked.norm(dim=-1)
                        mean_norm = norms.mean(dim=0, keepdim=True).clamp_min(1e-8)
                        mag_loss = ((norms - mean_norm) / mean_norm).square().mean()
                        if self.inv_loss == "cosine":
                            z_bar_norm = mean_z.norm(dim=-1).clamp_min(1e-8)
                            inner = (stacked * mean_z).sum(dim=-1)
                            cos = inner / (norms.clamp_min(1e-8) * z_bar_norm)
                            cos_loss = torch.clamp(
                                (1.0 - cos) - self.inv_tol, min=0.0).mean()
                            inv_h = cos_loss + mag_loss
                        else:  # "mse"
                            per_sample_sq = (stacked - mean_z).square().sum(dim=-1)
                            # N(0, I) prior floor on ‖z_v − z̄‖²; divide final
                            # inv by D so heads with different D sit on the
                            # same scale before averaging.
                            prior_floor = D * (V - 1) / V
                            margin = self.inv_tol * prior_floor
                            inv_h = torch.clamp(
                                per_sample_sq - margin, min=0.0).mean() / D
                    else:
                        inv_h = torch.zeros(
                            (), device=head_views[0].device,
                            dtype=head_views[0].dtype)
                    total_inv = total_inv + inv_h
                    self.log(f"{stage}/inv_loss_h{h}", inv_h,
                             on_step=False, on_epoch=True, sync_dist=True)

                inv = total_inv / H
                reg = total_reg / H
                loss = self.lambd * reg + (1.0 - self.lambd) * inv
                out["loss"] = loss

                self.log(f"{stage}/inv_loss", inv,
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{stage}/reg_loss", reg,
                         on_step=False, on_epoch=True, sync_dist=True)

            # Probe-facing outputs (train + eval): embedding + each head's projection
            out["embedding"] = torch.cat(embs, dim=0)
            for h, head_views in enumerate(heads_z):
                out[f"projection_h{h}"] = torch.cat(head_views, dim=0)
            if "label" in views[0]:
                out["label"] = torch.cat(
                    [v["label"] for v in views[:V]], dim=0)
        else:
            emb = self.backbone(batch["image"])
            zs = self.projector(emb)
            out["embedding"] = emb
            for h, z in enumerate(zs):
                out[f"projection_h{h}"] = z
            if "label" in batch:
                out["label"] = batch["label"]

        return out

    return lejepa_forward


# ---------------------------------------------------------------------------
# Run allocation
# ---------------------------------------------------------------------------

_HP_EXCLUDE = {"seed", "run_name"}


def allocate_run(log_dir: Path, dataset: str, backbone: str,
                 args_dict: dict, seed):
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset}_{backbone}_"

    def _hp(d):
        return {k: v for k, v in d.items() if k not in _HP_EXCLUDE}

    target_hp = _hp(args_dict)
    existing_ids = []
    used_seeds = set()

    for d in log_dir.iterdir():
        if not (d.is_dir() and d.name.startswith(prefix)):
            continue
        try:
            existing_ids.append(int(d.name[len(prefix):]))
        except ValueError:
            continue
        ap = d / "args.json"
        if not ap.exists():
            continue
        prev = json.loads(ap.read_text())
        if _hp(prev) == target_hp and isinstance(prev.get("seed"), int):
            used_seeds.add(prev["seed"])

    run_id = max(existing_ids, default=0) + 1
    if seed is None:
        seed = 0
        while seed in used_seeds:
            seed += 1

    run_name = f"{prefix}{run_id}"
    run_dir = log_dir / run_name
    run_dir.mkdir()

    saved = {**args_dict, "seed": seed, "run_name": run_name}
    (run_dir / "args.json").write_text(json.dumps(saved, indent=2, sort_keys=True))
    return run_dir, run_name, seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar100",
                   choices=["cifar10", "cifar100", "imagenette", "imagenet-100"])
    p.add_argument("--image-size", type=int, default=None,
                   help="Override dataset default image size")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--regularizer", default="w1",
                   choices=["sigreg", "sigreg_raw", "w1", "w2"])
    p.add_argument("--lambd", type=float, default=0.8)
    p.add_argument("--inv-loss", default="mse", choices=["mse", "cosine"])
    p.add_argument("--inv-tol", type=float, default=0.0,
                   help="Invariance margin (see train.py for semantics).")
    p.add_argument("--proj-dims", type=int, nargs="+", default=[16, 32, 128],
                   help="Output dim of each projection head. One head + "
                        "one (inv + reg) objective per dim. "
                        "e.g. --proj-dims 64 128 256")
    p.add_argument("--proj-hidden", type=int, default=512)
    p.add_argument("--num-proj", type=int, default=2048,
                   help="Number of random slices for the regularizer (shared "
                        "across heads — reg is dim-agnostic).")
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--num-views", type=int, default=2,
                   help="V: grad-carrying views per sample. All views feed "
                        "both invariance and regularizer.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()

    log_dir = Path(__file__).resolve().parent / "logs"
    run_dir, run_name, seed = allocate_run(
        log_dir, args.dataset, args.backbone, vars(args), args.seed)
    args.seed = seed
    pl.seed_everything(seed, workers=True)
    print(f"[run] {run_name}  seed={seed}  dir={run_dir}  heads={args.proj_dims}")

    data, num_classes = make_data(
        args.dataset, args.batch_size, args.num_workers,
        num_views=args.num_views, image_size=args.image_size,
    )
    low_res = args.dataset.startswith("cifar")
    backbone, emb_dim = make_backbone(args.backbone, low_resolution=low_res)
    projector = MultiHeadProjector(emb_dim, args.proj_dims, args.proj_hidden)
    regularizer = make_regularizer(
        args.regularizer, num_proj=args.num_proj, knots=args.knots)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=make_lejepa_forward(num_views=args.num_views),
        regularizer=regularizer,
        lambd=args.lambd,
        inv_loss=args.inv_loss,
        inv_tol=args.inv_tol,
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

    callbacks = [
        spt.callbacks.OnlineProbe(
            module, name="linear_probe_emb",
            input="embedding", target="label",
            probe=nn.Linear(emb_dim, num_classes),
            loss=nn.CrossEntropyLoss(),
            metrics=_cls_metrics(),
        ),
        spt.callbacks.OnlineKNN(
            name="knn_probe_emb", input="embedding", target="label",
            queue_length=20000,
            metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes)},
            input_dim=emb_dim, k=20,
        ),
    ]
    for h, d in enumerate(args.proj_dims):
        callbacks.append(spt.callbacks.OnlineProbe(
            module, name=f"linear_probe_proj_h{h}",
            input=f"projection_h{h}", target="label",
            probe=nn.Linear(d, num_classes),
            loss=nn.CrossEntropyLoss(),
            metrics=_cls_metrics(),
        ))
        callbacks.append(spt.callbacks.OnlineKNN(
            name=f"knn_probe_proj_h{h}",
            input=f"projection_h{h}", target="label",
            queue_length=20000,
            metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes)},
            input_dim=d, k=20,
        ))

    logger = CSVLogger(save_dir=str(log_dir), name=run_name, version="")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        save_last=True, save_top_k=0,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=callbacks + [ckpt_cb],
        precision="16-mixed",
        logger=logger,
    )

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
            pass

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
