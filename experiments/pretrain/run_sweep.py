"""Launch a sweep of LeJEPA pretraining runs.

Builds the cartesian product of configs in a named plan and launches
trainer.py for each combination as a subprocess. Skip-if-final-exists is
handled inside trainer.py, so re-running this script resumes where it
stopped.

Available plans (see experiments/EXPERIMENT_PLAN.md):
    smoke           Tiny verification: 1 base + 1 continuation, 1 epoch (2 runs)
    phase0_base     Phase 0: base sigreg trainings on 3 datasets (3 runs)
    phase1_fig3     Phase 1: bs sweep continuation, Fig 3 headline (180 runs)
    phase2_tab1     Phase 2: 5 new dataset bases + headline-bs continuation (65 runs)
    phase3_tab2     Phase 3: architecture sweep with HEADLINE_METHODS (66 runs)

Phase 1/2/3 are continuation plans — they require Phase 0 (and Phase 2 for
the new-dataset bases) to be complete first so the base checkpoints exist.

Usage:
    python run_sweep.py --plan phase0_base --save-dir runs/lejepa_v1
    python run_sweep.py --plan phase1_fig3 --save-dir runs/lejepa_v1
    python run_sweep.py --plan smoke --dry-run

Override epochs / encoder etc. via the same flags trainer.py accepts:
    python run_sweep.py --plan phase1_fig3 --epochs 50
"""

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = Path(__file__).resolve().parent / "trainer.py"


@dataclass
class RunSpec:
    dataset: str
    encoder_scale: str
    regularizer: str
    accumulate: bool
    batch_size: int
    epochs: int
    nograd_pool_size: int = 0      # number of no-grad samples for regularizer pool
    continue_from: str = ""        # path to base ckpt; empty = from scratch
    extra_args: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Method tuples (regularizer, accumulate)
# ---------------------------------------------------------------------------

ALL_METHODS = [
    ("sigreg", False),
    ("sigreg", True),
    ("w1",     False),
    ("w1",     True),
]

POOLED_METHODS = [
    ("sigreg", True),
    ("w1",     True),
]

HEADLINE_METHODS = [
    ("sigreg", False),  # LeJEPA baseline
    ("w1",     True),   # our best
]


# ---------------------------------------------------------------------------
# Phase plan constants (see EXPERIMENT_PLAN.md)
# ---------------------------------------------------------------------------

HEADLINE_DATASETS = ["cifar100", "food101", "flowers102"]   # Phase 0 + Phase 1
TAB1_NEW_DATASETS = ["cifar10", "stl10", "dtd", "aircraft", "pets"]  # Phase 2
# Phase 3 architecture sweep — all LayerNorm-only. BN backbones (resnet*) have
# shown instability with pooled training; see "Note on BatchNorm" in EXPERIMENT_PLAN.md.
TAB2_ARCHS = ["convnextv2_atto", "convnextv2_pico", "convnext_tiny", "tiny"]
TAB2_DATASETS = ["cifar100", "flowers102", "pets"]

# 5-point continuation BS sweep. BS=128 was dropped because the per-pool-epoch
# cost is essentially constant past saturation, so 128 adds budget without
# adding new physics — and fullgrad OOMs at BS=64 already.
CONT_BS_VALUES = [8, 16, 32, 48, 64]
CONT_SEEDS = [0, 1, 2]
CONT_EPOCHS = 50
HEADLINE_CONT_BS = 32          # the bs used in Tab 1 / Tab 2

# Default backbone for sweep plans when --encoder isn't passed.
DEFAULT_ENCODER = "convnextv2_pico"

BASE_BS = 8
BASE_EPOCHS = 200
BASE_SEED = 42


def base_ckpt_path(save_dir: str, dataset: str, encoder_scale: str,
                   regularizer: str = "sigreg", accumulate: bool = False,
                   batch_size: int = BASE_BS, seed: int = BASE_SEED) -> str:
    """Build the path to a base checkpoint's final.pt.

    Must produce the same string as `trainer.build_run_dir(cfg) / 'final.pt'`
    when `cfg.continue_from == ""`.
    """
    method = regularizer + ("_pooled" if accumulate else "")
    return str(Path(save_dir) /
               f"{dataset}_{encoder_scale}_{method}_bs{batch_size}_seed{seed}" /
               "final.pt")


# ---------------------------------------------------------------------------
# Plan definitions
# ---------------------------------------------------------------------------

def plan_smoke(epochs: int, encoder: str, save_dir: str) -> List[RunSpec]:
    """Tiny verification — 1 base + 1 continuation, 1 epoch each.

    Exercises both the from-scratch and continuation code paths.
    """
    enc = encoder or DEFAULT_ENCODER
    ep = epochs or 1
    base_path = base_ckpt_path(save_dir, "cifar100", enc)
    return [
        RunSpec("cifar100", enc, "sigreg", False, BASE_BS, ep),
        RunSpec("cifar100", enc, "w1", True, HEADLINE_CONT_BS, ep,
                continue_from=base_path),
    ]


def plan_phase0_base(epochs: int, encoder: str, save_dir: str) -> List[RunSpec]:
    """Phase 0 — base sigreg trainings on 3 headline datasets, single seed."""
    enc = encoder or DEFAULT_ENCODER
    ep = epochs or BASE_EPOCHS
    return [
        RunSpec(
            dataset=ds, encoder_scale=enc,
            regularizer="sigreg", accumulate=False,
            batch_size=BASE_BS, epochs=ep,
        )
        for ds in HEADLINE_DATASETS
    ]


def plan_phase1_fig3(epochs: int, encoder: str, save_dir: str) -> List[RunSpec]:
    """Phase 1 — Fig 3: 3 datasets × 5 cont bs × 4 methods × 3 seeds = 180."""
    enc = encoder or DEFAULT_ENCODER
    ep = epochs or CONT_EPOCHS
    specs = []
    for ds in HEADLINE_DATASETS:
        base_path = base_ckpt_path(save_dir, ds, enc)
        for bs in CONT_BS_VALUES:
            for reg, acc in ALL_METHODS:
                for seed in CONT_SEEDS:
                    specs.append(RunSpec(
                        dataset=ds, encoder_scale=enc,
                        regularizer=reg, accumulate=acc,
                        batch_size=bs, epochs=ep,
                        continue_from=base_path,
                        extra_args=("--seed", str(seed)),
                    ))
    return specs


def plan_phase2_tab1(epochs: int, encoder: str, save_dir: str) -> List[RunSpec]:
    """Phase 2 — Tab 1 new datasets: 5 bases + 5 ds × 4 methods × 3 seeds = 65.

    HEADLINE_DATASETS (cifar100/food101/flowers102) are reused from Phase 0.
    """
    enc = encoder or DEFAULT_ENCODER
    ep_base = epochs or BASE_EPOCHS
    ep_cont = epochs or CONT_EPOCHS
    specs = []
    # 5 new bases (single seed each)
    for ds in TAB1_NEW_DATASETS:
        specs.append(RunSpec(
            dataset=ds, encoder_scale=enc,
            regularizer="sigreg", accumulate=False,
            batch_size=BASE_BS, epochs=ep_base,
        ))
    # 4 methods × 3 seeds × 5 datasets continuation, at HEADLINE_CONT_BS
    for ds in TAB1_NEW_DATASETS:
        base_path = base_ckpt_path(save_dir, ds, enc)
        for reg, acc in ALL_METHODS:
            for seed in CONT_SEEDS:
                specs.append(RunSpec(
                    dataset=ds, encoder_scale=enc,
                    regularizer=reg, accumulate=acc,
                    batch_size=HEADLINE_CONT_BS, epochs=ep_cont,
                    continue_from=base_path,
                    extra_args=("--seed", str(seed)),
                ))
    return specs


def plan_phase3_tab2(epochs: int, encoder: str, save_dir: str) -> List[RunSpec]:
    """Phase 3 — Tab 2 architecture sweep: 12 bases + 54 cont = 66.

    All 12 (arch, ds) bases are enumerated; the DEFAULT_ENCODER ones are
    no-ops at runtime via skip-if-final-exists (those checkpoints exist from
    Phase 0 for cifar100/flowers102 and from Phase 2 for pets).

    Continuations skip arch=DEFAULT_ENCODER entirely: every (DEFAULT_ENCODER,
    ds) at HEADLINE_CONT_BS with HEADLINE_METHODS is already covered by
    Phase 1 (cifar100/flowers102) or Phase 2 (pets) with 3 seeds each. That
    removes 18 cont runs (3 ds × 2 methods × 3 seeds), leaving 72 - 18 = 54.
    """
    ep_base = epochs or BASE_EPOCHS
    ep_cont = epochs or CONT_EPOCHS
    specs = []
    for arch in TAB2_ARCHS:
        for ds in TAB2_DATASETS:
            # Base for every (arch, ds) — runtime skip handles reused ones
            specs.append(RunSpec(
                dataset=ds, encoder_scale=arch,
                regularizer="sigreg", accumulate=False,
                batch_size=BASE_BS, epochs=ep_base,
            ))
            # Continuations: skip the default encoder — already covered
            # by Phase 1/2
            if arch == DEFAULT_ENCODER:
                continue
            base_path = base_ckpt_path(save_dir, ds, arch)
            for reg, acc in HEADLINE_METHODS:
                for seed in CONT_SEEDS:
                    specs.append(RunSpec(
                        dataset=ds, encoder_scale=arch,
                        regularizer=reg, accumulate=acc,
                        batch_size=HEADLINE_CONT_BS, epochs=ep_cont,
                        continue_from=base_path,
                        extra_args=("--seed", str(seed)),
                    ))
    return specs


PLANS = {
    "smoke":       plan_smoke,
    "phase0_base": plan_phase0_base,
    "phase1_fig3": plan_phase1_fig3,
    "phase2_tab1": plan_phase2_tab1,
    "phase3_tab2": plan_phase3_tab2,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def build_command(spec: RunSpec, common: List[str]) -> List[str]:
    cmd = [
        sys.executable, str(TRAINER),
        "--dataset", spec.dataset,
        "--encoder-scale", spec.encoder_scale,
        "--regularizer", spec.regularizer,
        "--batch-size", str(spec.batch_size),
        "--epochs", str(spec.epochs),
        "--nograd-pool-size", str(spec.nograd_pool_size),
    ]
    if spec.accumulate:
        cmd.append("--accumulate")
    if spec.continue_from:
        cmd.extend(["--continue-from", spec.continue_from])
    cmd.extend(common)
    # extra_args goes LAST so per-spec --seed (if any) wins over the
    # global --seed in `common` (argparse last-wins).
    cmd.extend(spec.extra_args)
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Launch a LeJEPA pretraining sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--plan", choices=list(PLANS.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=0,
                        help="Override epochs for every run (0 = use plan default)")
    parser.add_argument("--encoder-scale", type=str, default="",
                        help="Override encoder for every run (ignored by phase3_tab2)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="runs/sweep")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without launching subprocesses")
    args, passthrough = parser.parse_known_args()

    common = [
        "--data-dir", args.data_dir,
        "--save-dir", args.save_dir,
        "--seed", str(args.seed),
    ] + passthrough

    specs = PLANS[args.plan](args.epochs, args.encoder_scale, args.save_dir)

    print(f"Plan: {args.plan} — {len(specs)} runs")
    print(f"Save dir: {args.save_dir}")
    print()

    t_start = time.time()
    failed = []
    for i, spec in enumerate(specs, 1):
        cmd = build_command(spec, common)
        method = f"{spec.regularizer}{'_pooled' if spec.accumulate else ''}"
        cont_tag = " cont" if spec.continue_from else ""
        tag = (f"[{i}/{len(specs)}] {spec.dataset:11s} {spec.encoder_scale:16s} "
               f"{method:14s} bs={spec.batch_size:3d} pool={spec.nograd_pool_size:3d} "
               f"ep={spec.epochs:4d}{cont_tag}")
        print(tag)
        if args.dry_run:
            print("  $ " + shlex.join(cmd))
            continue
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"  → ok in {elapsed/60:.1f} min")
        else:
            failed.append((i, spec, result.returncode))
            print(f"  → FAIL ({result.returncode}) in {elapsed/60:.1f} min")
        print()

    total_min = (time.time() - t_start) / 60
    print(f"Sweep complete in {total_min:.1f} min ({total_min/60:.1f} h)")
    if failed:
        print(f"\n{len(failed)} runs failed:")
        for i, spec, rc in failed:
            print(f"  [{i}] {spec.dataset} {spec.regularizer}"
                  f"{'_pooled' if spec.accumulate else ''} bs={spec.batch_size}: rc={rc}")


if __name__ == "__main__":
    main()
