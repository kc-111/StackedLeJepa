"""CIFAR-100 pretraining with sigreg regularizer.

Goal: achieve ~70% linear probe accuracy using sigreg (distributional
regularizer toward N(0, I)) instead of VICReg on CIFAR-100.

Adapted from a VICReg CIFAR-10 recipe:
  - CIFAR-10 -> CIFAR-100 (100 classes)
  - VICReg loss -> sigreg (Epps-Pulley test) + invariance MSE
  - Uses stable_pretraining (spt) + PyTorch Lightning

Run:
    python train.py
"""

import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from lightning.pytorch.loggers import CSVLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.forward import _get_views_list

# SIGReg from repo src/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from sliced_gauss_reg import SIGReg


# ---------------------------------------------------------------------------
# SIGReg forward (replaces vicreg_forward)
# ---------------------------------------------------------------------------

def sigreg_forward(self, batch, stage):
    """Forward with sigreg regularizer instead of VICReg.

    loss = lambd * sigreg(projections) + (1 - lambd) * MSE(z_i, z_j)

    sigreg pushes the projection distribution toward N(0, I) via the
    Epps-Pulley characteristic function test, replacing VICReg's
    variance + covariance terms.
    """
    out = {}
    views = _get_views_list(batch)

    if views is not None:
        if len(views) != 2:
            raise ValueError(f"SIGReg requires 2 views, got {len(views)}")

        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]

            # Invariance: MSE between two views (same as VICReg sim term)
            inv_loss = F.mse_loss(projections[0], projections[1])

            # Regularization: sigreg on stacked projections -> N(0, I)
            stacked = torch.stack(projections, dim=0)  # (2, B, D)
            reg_loss = self.sigreg(stacked)

            loss = self.lambd * reg_loss + (1.0 - self.lambd) * inv_loss
            out["loss"] = loss

            self.log(f"{stage}/loss", loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/inv_loss", inv_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/reg_loss", reg_loss,
                     on_step=True, on_epoch=True, sync_dist=True)
    else:
        # Single-view (validation)
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LAMBD = 0.05          # sigreg weight: lambd*reg + (1-lambd)*inv
BATCH_SIZE = 256
MAX_EPOCHS = 1000
NUM_CLASSES = 100
EMB_DIM = 512         # resnet18 embedding dim
PROJ_DIM = 64       # projector output dim (matching VICReg reference)
DATA_DIR = "./data"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

aug_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToImage(**spt.data.static.CIFAR100),
)

train_transform = transforms.MultiViewTransform([aug_transform, aug_transform])

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**spt.data.static.CIFAR100),
)

cifar_train = torchvision.datasets.CIFAR100(
    root=DATA_DIR, train=True, download=True)
cifar_val = torchvision.datasets.CIFAR100(
    root=DATA_DIR, train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train, names=["image", "label"], transform=train_transform)
val_dataset = spt.data.FromTorchDataset(
    cifar_val, names=["image", "label"], transform=val_transform)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE,
    num_workers=8, drop_last=True, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=10)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
backbone.fc = nn.Identity()

projector = nn.Sequential(
    nn.Linear(EMB_DIM, PROJ_DIM),
    nn.BatchNorm1d(PROJ_DIM),
    nn.ReLU(inplace=True),
    nn.Linear(PROJ_DIM, PROJ_DIM),
    nn.BatchNorm1d(PROJ_DIM),
    nn.ReLU(inplace=True),
    nn.Linear(PROJ_DIM, PROJ_DIM),
)

sigreg = SIGReg(knots=17, num_proj=2048)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=sigreg_forward,
    sigreg=sigreg,
    lambd=LAMBD,
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 5,
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(EMB_DIM, NUM_CLASSES),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
        "top5": torchmetrics.classification.MulticlassAccuracy(
            NUM_CLASSES, top_k=5),
    },
)

# knn_probe = spt.callbacks.OnlineKNN(
#     name="knn_probe",
#     input="embedding",
#     target="label",
#     queue_length=20000,
#     metrics={
#         "accuracy": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
#     },
#     input_dim=EMB_DIM,
#     k=10,
# )

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    num_sanity_val_steps=0,
    # callbacks=[knn_probe, linear_probe],
    callbacks=[linear_probe],
    precision="16-mixed",
    logger=CSVLogger("logs", name="cifar100_sigreg"),
    enable_checkpointing=False,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
