"""Smoke tests for the sweep launcher's plan enumeration.

These are pure-Python — no GPU, no model build, no data loading.
"""

from pathlib import Path

import pytest

from run_sweep import (
    PLANS,
    RunSpec,
    base_ckpt_path,
    build_command,
    plan_smoke,
    plan_phase0_base,
    plan_phase1_fig3,
    plan_phase2_tab1,
    plan_phase3_tab2,
    HEADLINE_DATASETS,
    TAB1_NEW_DATASETS,
    TAB2_ARCHS,
    TAB2_DATASETS,
    CONT_BS_VALUES,
    CONT_SEEDS,
    HEADLINE_CONT_BS,
    BASE_BS,
    BASE_SEED,
    DEFAULT_ENCODER,
)


SAVE_DIR = "runs/sweep"


# ---------------------------------------------------------------------------
# Plan enumeration counts
# ---------------------------------------------------------------------------

def test_smoke_plan_at_most_two_runs():
    specs = plan_smoke(epochs=1, encoder="convnextv2_nano", save_dir=SAVE_DIR)
    assert 1 <= len(specs) <= 2
    # Should exercise both code paths: at least one base, one continuation.
    assert any(s.continue_from == "" for s in specs)
    assert any(s.continue_from != "" for s in specs)


def test_phase0_base_has_three_runs_no_continuation():
    specs = plan_phase0_base(epochs=200, encoder="convnextv2_nano", save_dir=SAVE_DIR)
    assert len(specs) == 3
    assert all(s.continue_from == "" for s in specs)
    assert {s.dataset for s in specs} == set(HEADLINE_DATASETS)
    assert all(s.regularizer == "sigreg" and not s.accumulate for s in specs)
    assert all(s.batch_size == BASE_BS for s in specs)


def test_phase1_fig3_has_180_continuation_runs():
    specs = plan_phase1_fig3(epochs=50, encoder="convnextv2_nano", save_dir=SAVE_DIR)
    # 3 datasets × 5 bs × 4 methods × 3 seeds
    assert len(specs) == 180
    # Every run is a continuation
    assert all(s.continue_from for s in specs)
    # Every continue_from path ends in /final.pt
    assert all(s.continue_from.endswith("/final.pt") for s in specs)
    # Coverage of bs values, datasets, methods
    assert {s.dataset for s in specs} == set(HEADLINE_DATASETS)
    assert {s.batch_size for s in specs} == set(CONT_BS_VALUES)
    methods = {(s.regularizer, s.accumulate) for s in specs}
    assert methods == {
        ("sigreg", False), ("sigreg", True),
        ("w1", False), ("w1", True),
    }
    # Each spec carries a --seed override in extra_args
    for s in specs:
        assert "--seed" in s.extra_args


def test_phase2_tab1_has_65_runs():
    specs = plan_phase2_tab1(epochs=50, encoder="convnextv2_nano", save_dir=SAVE_DIR)
    # 5 new bases + 5 ds × 4 methods × 3 seeds = 5 + 60 = 65
    assert len(specs) == 65
    bases = [s for s in specs if not s.continue_from]
    conts = [s for s in specs if s.continue_from]
    assert len(bases) == 5
    assert len(conts) == 60
    assert {s.dataset for s in bases} == set(TAB1_NEW_DATASETS)
    assert {s.dataset for s in conts} == set(TAB1_NEW_DATASETS)
    # All continuations at HEADLINE_CONT_BS
    assert all(s.batch_size == HEADLINE_CONT_BS for s in conts)


def test_phase3_tab2_has_66_runs():
    specs = plan_phase3_tab2(epochs=50, encoder="convnextv2_nano", save_dir=SAVE_DIR)
    # 12 bases (4 archs × 3 ds) + 54 cont (3 non-default archs × 3 ds × 2 methods × 3 seeds)
    assert len(specs) == 66
    bases = [s for s in specs if not s.continue_from]
    conts = [s for s in specs if s.continue_from]
    assert len(bases) == 12
    assert len(conts) == 54
    # No DEFAULT_ENCODER continuations (those rows are reused from Phase 1/2)
    assert all(s.encoder_scale != DEFAULT_ENCODER for s in conts)
    # Continuation archs cover the other 3 backbones
    assert {s.encoder_scale for s in conts} == set(TAB2_ARCHS) - {DEFAULT_ENCODER}
    # Continuation datasets cover all of TAB2_DATASETS
    assert {s.dataset for s in conts} == set(TAB2_DATASETS)
    # Continuation methods are exactly HEADLINE_METHODS
    assert {(s.regularizer, s.accumulate) for s in conts} == {
        ("sigreg", False), ("w1", True),
    }
    # All cont specs have 3 seeds
    seeds_seen = {s.extra_args[s.extra_args.index("--seed") + 1] for s in conts}
    assert seeds_seen == {"0", "1", "2"}


def test_plans_dict_dispatches_with_save_dir():
    """Every plan must be callable as PLANS[name](epochs, encoder, save_dir)."""
    for name, fn in PLANS.items():
        specs = fn(0, "", SAVE_DIR)
        assert isinstance(specs, list) and len(specs) > 0, \
            f"Plan {name} produced empty list"
        for s in specs:
            assert isinstance(s, RunSpec)


# ---------------------------------------------------------------------------
# build_command CLI emission
# ---------------------------------------------------------------------------

def _common():
    return ["--data-dir", "./data", "--save-dir", SAVE_DIR, "--seed", "42"]


def test_build_command_omits_continue_from_for_base_runs():
    spec = RunSpec("cifar100", "resnet18", "sigreg", False, 8, 200)
    cmd = build_command(spec, _common())
    assert "--continue-from" not in cmd


def test_build_command_emits_continue_from_when_set():
    base = "runs/sweep/cifar100_resnet18_sigreg_bs8_seed42/final.pt"
    spec = RunSpec("cifar100", "resnet18", "w1", True, 32, 50,
                   continue_from=base, extra_args=("--seed", "1"))
    cmd = build_command(spec, _common())
    assert "--continue-from" in cmd
    assert cmd[cmd.index("--continue-from") + 1] == base
    # extra_args go LAST so per-spec --seed overrides the global one
    seed_positions = [i for i, a in enumerate(cmd) if a == "--seed"]
    assert len(seed_positions) == 2
    assert cmd[seed_positions[-1] + 1] == "1"
    # --accumulate emitted because spec.accumulate is True
    assert "--accumulate" in cmd


# ---------------------------------------------------------------------------
# base_ckpt_path matches trainer.build_run_dir
# ---------------------------------------------------------------------------

def test_base_ckpt_path_matches_trainer_build_run_dir():
    """The launcher's base_ckpt_path must match trainer.build_run_dir for the
    non-continuation case, otherwise Phase 1/2/3 will look for checkpoints
    that Phase 0/2 never wrote."""
    from configs import Config
    from trainer import build_run_dir

    cfg = Config(
        dataset="cifar100",
        encoder_scale="resnet18",
        regularizer="sigreg",
        accumulate=False,
        batch_size=BASE_BS,
        seed=BASE_SEED,
        save_dir=SAVE_DIR,
        epochs=200,
    )
    expected = str(build_run_dir(cfg) / "final.pt")
    actual = base_ckpt_path(SAVE_DIR, "cifar100", "resnet18")
    assert expected == actual, f"path mismatch:\n  expected: {expected}\n  actual:   {actual}"


def test_phase1_continue_from_paths_are_well_formed():
    """Phase 1 builds continue_from paths from base_ckpt_path; spot-check the
    structure for cifar100."""
    specs = plan_phase1_fig3(epochs=50, encoder="convnextv2_nano", save_dir=SAVE_DIR)
    cifar_specs = [s for s in specs if s.dataset == "cifar100"]
    expected = base_ckpt_path(SAVE_DIR, "cifar100", "convnextv2_nano")
    assert all(s.continue_from == expected for s in cifar_specs)
    assert "cifar100_convnextv2_nano_sigreg_bs8_seed42" in expected
    assert expected.endswith("/final.pt")
