"""General LeJEPA-style pretrainer on stable_pretraining.

Supports any (backbone × dataset × regularizer) combination. All pool /
FIFO knobs are *factors* of ``--batch-size`` so each backbone forward
processes exactly ``batch_size`` images — BN running stats update
uniformly across live and no-grad calls.

Knobs (all default 0):
    --live-views V              grad-carrying views per sample
    --extra-view-factor K       total views = V * (1 + K); extras no-grad,
                                feed invariance only
    --extra-forward-factor M    dataloader batch = batch_size * (1 + M);
                                no-grad rows chunked into M * V forwards
                                of batch_size each
    --fifo-factor F             FIFO holds F * batch_size past detached
                                projections; switches reg into
                                1-random-view-per-image mode (iid-per-image)

Regularizer modes:
    F = 0: per-view averaged over V views (iid-within-view).
    F > 0: flat pool of 1 random view per image + FIFO contents.

W1/W2 get ``pool_size / batch_size`` compensation (sort-based loss has
O(1/n_total) per-sample gradient); SIGReg does not (×n normalization).

Usage:
    python experiments/train/train.py --dataset cifar10 --backbone resnet18 \\
        --regularizer w1 --fifo-factor 4 --epochs 2
"""

import argparse
import json
import sys
from collections import deque
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
    """CIFAR spec: torchvision dataset with train=True/False."""
    def loaders():
        return (
            cls(root=str(DATA_DIR), train=True, download=True),
            cls(root=str(DATA_DIR), train=False, download=True),
        )
    return loaders, num_classes


def _build_imagefolder(subdir: str, num_classes: int):
    """ImageFolder spec: {DATA_DIR}/{subdir}/{train,val}."""
    def loaders():
        root = DATA_DIR / subdir
        return (
            torchvision.datasets.ImageFolder(root=str(root / "train")),
            torchvision.datasets.ImageFolder(root=str(root / "val")),
        )
    return loaders, num_classes


# (loaders_fn, num_classes, default_image_size, normalization_constants)
_DATASETS = {
    "cifar10": (*_build_cifar(10, torchvision.datasets.CIFAR10),
                32, spt.data.static.CIFAR10),
    "cifar100": (*_build_cifar(100, torchvision.datasets.CIFAR100),
                 32, spt.data.static.CIFAR100),
    "imagenette": (*_build_imagefolder("imagenette2-160", 10),
                   128, spt.data.static.ImageNet),
    "imagenet-100": (*_build_imagefolder("imagenet-100", 100),
                     224, spt.data.static.ImageNet),
}


def make_data(name: str, batch_size: int, num_workers: int = 8,
              live_views: int = 2, extra_view_factor: int = 0,
              extra_forward_factor: int = 0, image_size: int = None):
    """Build train/val DataModule.

    Both ``extra_view_factor`` and ``extra_forward_factor`` are
    multipliers that preserve per-forward batch size (and therefore
    BN running-stats consistency) — each backbone forward always
    processes exactly ``batch_size`` images.

    - ``batch_size``: live (grad-carrying) samples per step.
    - ``extra_forward_factor M``: dataloader yields ``batch_size*(1+M)``
      rows. For each live view v, the first ``batch_size`` rows are live
      (grad) and the remaining ``M*batch_size`` rows are processed as M
      separate no-grad forwards of ``batch_size`` each.
    - ``extra_view_factor K``: MultiViewTransform produces
      ``live_views*(1+K)`` views per sample. The first ``live_views``
      are live; the remaining ``live_views*K`` are processed as
      ``live_views*K`` separate no-grad forwards (one per extra view,
      each on the live-portion ``batch_size`` images).
    """
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(_DATASETS)}")
    loaders_fn, num_classes, default_size, norm = _DATASETS[name]
    size = image_size or default_size

    aug = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((size, size), scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILGaussianBlur(p=0.5),
        transforms.RandomSolarize(p=0.2, threshold=0.5),
        transforms.ToImage(**norm), 
    )
    n_views = live_views * (1 + extra_view_factor)
    train_tf = transforms.MultiViewTransform([aug] * n_views)
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

    dl_bs = batch_size * (1 + extra_forward_factor)
    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=dl_bs,
        num_workers=num_workers, drop_last=True, shuffle=True)
    val_dl = torch.utils.data.DataLoader(
        dataset=val_ds, batch_size=batch_size, num_workers=num_workers)

    return spt.data.DataModule(train=train_dl, val=val_dl), num_classes


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

_EMB_DIM = {"resnet18": 512, "resnet50": 2048}


def make_backbone(name: str, low_resolution: bool = True):
    """Returns (backbone, embedding_dim). spt.backbone.from_torchvision
    already swaps fc→Identity for us."""
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


# ---------------------------------------------------------------------------
# Forward: distinct roles for views / extra views / extra forwards / FIFO
# ---------------------------------------------------------------------------
#
# Invariance loss (matches images across augmentations):
#   - mean-target variance: z̄ = mean over all views (live + extras) per
#     image; inv = mean MSE(z_v, z̄). Grad flows through live views only.
#
# Regularizer / distribution loss (empirical distribution → N(0, I)):
#   Two modes, chosen by FIFO state:
#
#   (a) FIFO off: PER-VIEW, AVERAGED.
#       Within a single view transform, samples across images are iid;
#       across views they are correlated (same image, different augs).
#       Stacking all V*B samples into one bag violates iid, which biases
#       W1/W2/SIGReg estimators. Instead, compute reg separately on each
#       view column v (live + same-view extra-forward rows, size dl_bs)
#       and average over V. Extras (K no-grad views) don't enter reg —
#       they only feed invariance.
#
#   (b) FIFO on: 1 RANDOM VIEW PER IMAGE, FLAT.
#       Pick 1 random view per image (iid-per-image) from live and from
#       extra-forward rows; concat with FIFO contents. One reg call on
#       the flat pool. FIFO stores the live picks, so stale content is
#       already iid-per-image. Per-view mode doesn't compose with FIFO
#       (past steps gave different view assignments → no view axis to
#       average over consistently).
#
# Gradient-dilution compensation (W1/W2 only): multiply reg by
# pool_size / n_live, where n_live counts grad-carrying rows.

def make_lejepa_forward(live_batch_size: int, live_views: int):
    def lejepa_forward(self, batch, stage):
        out = {}
        views = _get_views_list(batch)

        if views is not None:
            if self.training:
                V = live_views
                B = live_batch_size
                dev = views[0]["image"].device
                dl_bs = views[0]["image"].shape[0]
                # Factors are derived from tensor shapes so the forward stays
                # in sync with whatever make_data produced.
                M = dl_bs // B - 1                       # extra_forward_factor
                K = len(views) // V - 1                  # extra_view_factor

                # --- Live (grad) forwards: V calls, each on B images ---
                live_emb = [self.backbone(v["image"][:B]) for v in views[:V]]
                live_z = [self.projector(e) for e in live_emb]   # V × (B, D)
                live_emb_cat = torch.cat(live_emb, dim=0)         # for probes

                # --- NG extra forwards: V*M calls, each on B images ---
                # Each live view v has its no-grad rows sliced into M chunks
                # of size B, then forwarded separately. Keeps BN consistent
                # with the live forwards (all calls see exactly B images).
                ng_fwd_z = [[] for _ in range(V)]  # per-view list of M tensors
                if M > 0:
                    with torch.no_grad():
                        for v in range(V):
                            for r in range(M):
                                off = B * (1 + r)
                                x = views[v]["image"][off:off + B]
                                ng_fwd_z[v].append(self.projector(self.backbone(x)))

                # --- NG extra views: V*K calls, each on B images ---
                # Extras are K additional "rounds" of V views (independent
                # random augs), applied to the live-portion images. They
                # feed invariance only.
                ng_extra_z = []  # V*K tensors, each (B, D)
                if K > 0:
                    with torch.no_grad():
                        for ev in range(V, V * (1 + K)):
                            x = views[ev]["image"][:B]
                            ng_extra_z.append(self.projector(self.backbone(x)))

                # --- Regularizer: per-view (FIFO off) or flat-sampled (FIFO on) ---
                # Per-view ng_fwd: for each view v, concat its M chunks into
                # (M*B, D). Across V views → (V, M*B, D). Keep ng_fwd as a
                # per-view stack so both branches below can share it.
                if M > 0:
                    ng_fwd_per_view = torch.stack(
                        [torch.cat(ng_fwd_z[v], dim=0) for v in range(V)],
                        dim=0,
                    )  # (V, M*B, D)
                else:
                    ng_fwd_per_view = None

                if self.fifo_size > 0:
                    # Mode (b): 1 random view per image, flat pool with FIFO
                    def pick_one(stacked):
                        Vv, N, _ = stacked.shape
                        vi = torch.randint(0, Vv, (N,), device=stacked.device)
                        bi = torch.arange(N, device=stacked.device)
                        return stacked[vi, bi]

                    live_pick = pick_one(torch.stack(live_z, dim=0))  # (B, D), grad
                    pool = [live_pick]
                    if ng_fwd_per_view is not None:
                        pool.append(pick_one(ng_fwd_per_view))  # (M*B, D), no grad
                    if len(self.fifo_buffer) > 0:
                        for t in self.fifo_buffer:
                            pool.append(t.to(dev, non_blocking=True))
                    pooled = torch.cat(pool, dim=0)
                    reg = self.regularizer(pooled)
                    if self.regularizer.needs_compensation:
                        reg = reg * (pooled.shape[0] / B)
                    pool_size_log = float(pooled.shape[0])
                else:
                    # Mode (a): per-view regularizer, averaged over V views.
                    # Stack live + ng_fwd per view → (V, dl_bs, D) and call the
                    # regularizer once; it reduces over the leading V axis so
                    # one vectorized call == V per-view calls then mean.
                    live_stack = torch.stack(live_z, dim=0)  # (V, B, D)
                    if ng_fwd_per_view is not None:
                        cols = torch.cat([live_stack, ng_fwd_per_view], dim=1)  # (V, dl_bs, D)
                    else:
                        cols = live_stack
                    # SlicedEppsPulley only accepts [N, D], so it always gets
                    # a flat pool. For our SIGReg/W1/W2, default is per-view
                    # (iid-within-view, paper-faithful); --flatten-reg opts
                    # into the reference LeJEPA impl's one-big-bag behavior,
                    # which violates iid across views (same image → correlated
                    # projections) and biases the estimator.
                    do_flatten = self.flatten_reg or isinstance(
                        self.regularizer, SlicedEppsPulley)
                    if do_flatten:
                        flat = cols.reshape(-1, cols.size(-1))  # (V*dl_bs, D)
                        reg = self.regularizer(flat)
                        pool_rows = flat.shape[0]
                    else:
                        reg = self.regularizer(cols)  # averages over V axis
                        pool_rows = cols.shape[1]
                    if self.regularizer.needs_compensation:
                        reg = reg * (pool_rows / B)
                    live_pick = None
                    pool_size_log = float(V * cols.shape[1])

                # --- Invariance: mean-target variance across views ---
                # For each image, compute z̄ = mean over views (live + extras);
                # invariance = mean MSE(z_v, z̄) over all views v. Equivalent
                # to the view-axis variance of z, averaged over (image, dim).
                # Gradient w.r.t. each live view j reduces to (2/V_total)(z_j -
                # z̄) — pulls each live view toward the per-image centroid.
                # No-grad extras pull the centroid without receiving gradient.
                all_inv_views = live_z + ng_extra_z  # V*(1+K) tensors, each (B, D)
                if len(all_inv_views) >= 2:
                    stacked = torch.stack(all_inv_views, dim=0)   # (V*(1+K), B, D)
                    mean_z = stacked.mean(dim=0, keepdim=True)    # (1, B, D)
                    inv = (stacked - mean_z).square().mean()
                else:
                    inv = torch.zeros((), device=dev, dtype=live_z[0].dtype)

                loss = self.lambd * reg + (1.0 - self.lambd) * inv
                out["loss"] = loss

                # Expose embeddings for online probes (live rows only)
                out["embedding"] = live_emb_cat
                if "label" in views[0]:
                    out["label"] = torch.cat(
                        [v["label"][:B] for v in views[:V]], dim=0)

                self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"{stage}/inv_loss", inv, on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"{stage}/reg_loss", reg, on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"{stage}/pool_size", pool_size_log,
                         on_step=True, on_epoch=True, sync_dist=True)

                # FIFO update: only defined in flat-sampled mode (FIFO on).
                if self.fifo_size > 0 and live_pick is not None:
                    self.fifo_buffer.append(live_pick.detach().cpu())
                    total_n = sum(t.shape[0] for t in self.fifo_buffer)
                    while total_n > self.fifo_size and len(self.fifo_buffer) > 1:
                        total_n -= self.fifo_buffer[0].shape[0]
                        self.fifo_buffer.popleft()
            else:
                # Eval: one forward per live view (probe expects matched labels)
                V = live_views
                emb = [self.backbone(v["image"]) for v in views[:V]]
                out["embedding"] = torch.cat(emb, dim=0)
                if "label" in views[0]:
                    out["label"] = torch.cat([v["label"] for v in views[:V]], dim=0)
        else:
            out["embedding"] = self.backbone(batch["image"])
            if "label" in batch:
                out["label"] = batch["label"]

        return out

    return lejepa_forward


# ---------------------------------------------------------------------------
# Run allocation: unique dir per run, args.json for later analysis
# ---------------------------------------------------------------------------

_HP_EXCLUDE = {"seed", "run_name"}


def allocate_run(log_dir: Path, dataset: str, backbone: str,
                 args_dict: dict, seed):
    """Create a fresh {dataset}_{backbone}_{N} dir under log_dir.

    N auto-increments across existing runs so names are never duplicated.
    If seed is None, picks the smallest non-negative int not already used
    by a prior run with identical hyperparameters (everything in args_dict
    except seed/run_name). Writes args.json into the new dir.
    """
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
    p.add_argument("--dataset", default="imagenette",
                   choices=["cifar10", "cifar100" "imagenette", "imagenet-100"])
    p.add_argument("--image-size", type=int, default=None,
                   help="Override dataset default image size")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--regularizer", default="sigreg",
                   choices=["sigreg", "sigreg_raw", "w1", "w2"])
    p.add_argument("--lambd", type=float, default=0.05)
    p.add_argument("--proj-dim", type=int, default=512)
    p.add_argument("--proj-hidden", type=int, default=64)
    p.add_argument("--num-proj", type=int, default=2048)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--live-views", type=int, default=2,
                   help="V: grad-carrying views per sample. Invariance is "
                        "the view-axis variance (0 if V=1 and no extras)")
    p.add_argument("--fifo-factor", type=int, default=0,
                   help="F: FIFO holds F past batches' worth of detached "
                        "samples (F * batch_size). 0 = off. When on, reg "
                        "switches to 1-random-view-per-image (iid-per-image); "
                        "when off, reg is per-view averaged.")
    p.add_argument("--extra-view-factor", type=int, default=0,
                   help="K: multiplier of V. Total views = V*(1+K); the extra "
                        "V*K views are no-grad, feed invariance only. "
                        "Each no-grad forward processes exactly --batch-size "
                        "images so BN running stats stay consistent with live.")
    p.add_argument("--extra-forward-factor", type=int, default=0,
                   help="M: multiplier of --batch-size. Dataloader yields "
                        "batch_size*(1+M) rows; for each live view, the "
                        "first batch_size are live, remaining M*batch_size "
                        "are forwarded as M separate no-grad calls of "
                        "batch_size each (BN-consistent).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Live (grad-carrying) samples per step")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--flatten-reg", action="store_true",
                   help="Flatten (V, B, D) → (V*B, D) before calling the "
                        "regularizer (FIFO-off path only). Matches the "
                        "reference LeJEPA impl but violates iid across "
                        "views — biases SIGReg downward. Default: per-view.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed. If omitted, auto-picks the smallest "
                        "nonneg int unused by prior runs with identical hparams.")
    return p


def main():
    args = build_parser().parse_args()

    # sigreg_raw always flattens (SlicedEppsPulley only accepts [N, D]).
    # Reflect that in args.flatten_reg so args.json records the effective
    # behavior — otherwise analysis would show flatten_reg=False for a run
    # that actually pooled flat.
    if args.regularizer == "sigreg_raw":
        args.flatten_reg = True

    log_dir = Path(__file__).resolve().parent / "logs"
    run_dir, run_name, seed = allocate_run(
        log_dir, args.dataset, args.backbone, vars(args), args.seed)
    args.seed = seed
    pl.seed_everything(seed, workers=True)
    print(f"[run] {run_name}  seed={seed}  dir={run_dir}")

    data, num_classes = make_data(
        args.dataset, args.batch_size, args.num_workers,
        live_views=args.live_views,
        extra_view_factor=args.extra_view_factor,
        extra_forward_factor=args.extra_forward_factor,
        image_size=args.image_size,
    )
    # CIFAR uses the low-res stem (3×3 stride-1 conv, no maxpool);
    # ImageNet-scale datasets use the full ImageNet stem.
    low_res = args.dataset.startswith("cifar")
    backbone, emb_dim = make_backbone(args.backbone, low_resolution=low_res)
    projector = make_projector(emb_dim, args.proj_dim, args.proj_hidden)
    regularizer = make_regularizer(
        args.regularizer, num_proj=args.num_proj, knots=args.knots)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=make_lejepa_forward(
            live_batch_size=args.batch_size, live_views=args.live_views),
        regularizer=regularizer,
        lambd=args.lambd,
        flatten_reg=args.flatten_reg,
        # Internal fifo_size (sample cap) = factor × batch_size, so FIFO
        # stabilizes at exactly fifo_factor batches of past projections.
        fifo_size=args.fifo_factor * args.batch_size,
        fifo_buffer=deque(),
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

    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(emb_dim, num_classes),
        loss=nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
            "top5": torchmetrics.classification.MulticlassAccuracy(
                num_classes, top_k=5),
        },
    )
    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes)},
        input_dim=emb_dim,
        k=20,
    )

    logger = CSVLogger(save_dir=str(log_dir), name=run_name, version="")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        save_last=True,
        save_top_k=0,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[knn_probe, linear_probe, ckpt_cb],
        precision="16-mixed",
        logger=logger,
    )
    # Lightning auto-injects stable_pretraining's default callbacks (via the
    # "lightning.pytorch.callbacks_factory" entry point). Drop only the env
    # dump — it writes environment.json and requirements_frozen.txt to the
    # repo root on every fit. Keep LoggingCallback etc. so we still see the
    # per-epoch metrics table.
    from stable_pretraining.callbacks import EnvironmentDumpCallback
    trainer.callbacks = [
        cb for cb in trainer.callbacks
        if not isinstance(cb, EnvironmentDumpCallback)
    ]

    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=seed)
    manager()


if __name__ == "__main__":
    main()
