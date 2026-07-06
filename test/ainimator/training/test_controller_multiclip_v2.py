"""Tests for multi-clip controller training (A2/A4 / A7 validation path).

``controller_multiclip_v2`` has been archived to ``legacy/`` (A7,
2026-06-25).  The equivalent functionality is now provided by
``controller_generalization_v2.runControllerGeneralization`` (the
``full`` profile of the unified ``train_controller_v2`` CLI).

These tests exercise the multi-clip code path via the new module.
The negative-import test asserts that the archived module cannot be
imported from ``ainimator.*`` (lint-imports contract
``no_legacy_imports``).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ainimator.core.constants.controller import PhaseMode
from ainimator.health.contract import Verdict
from ainimator.training.controller_generalization_v2 import (
    runControllerGeneralization,
)
from ainimator.training.controller_training_v2 import (
    ControllerTrainingConfig,
    loadControllerCheckpoint,
)


def _clip(seed: int, speed: float, frames: int = 56) -> tuple[
    torch.Tensor, torch.Tensor
]:
    """Synthetic clip with periodic motion."""
    phase = torch.linspace(0.0, 4.0 * math.pi, frames)
    rotation6d = torch.zeros(frames, 22, 6)
    for bone in range(22):
        for channel in range(6):
            rotation6d[:, bone, channel] = math.cos(
                bone + channel + seed
            ) + 0.3 * torch.sin(
                phase + 0.2 * bone + 0.5 * channel + seed
            )
    rootTranslation = torch.zeros(frames, 3)
    rootTranslation[:, 0] = speed * torch.arange(frames)
    rootTranslation[:, 2] = 0.05 * torch.cos(phase)
    return rotation6d, rootTranslation


def _clips() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Four synthetic clips with varied speeds."""
    return [
        _clip(0, 0.01),
        _clip(1, 0.03),
        _clip(2, 0.05),
        _clip(3, 0.02),
    ]


def _config(outputDir: Path) -> ControllerTrainingConfig:
    """Minimal fast training config for tests."""
    return ControllerTrainingConfig(
        outputDir=outputDir,
        epochs=250,
        device="cpu",
        embedDim=128,
        numHeads=4,
        numLayers=3,
        contextFrames=4,
        logEvery=999,
        seed=0,
    )


def test_multiclip_archived_module_not_importable() -> None:
    """Archived multiclip module must not be importable from ainimator.

    This is the negative-import test for lint-imports contract
    ``no_legacy_imports``.  Attempting to import
    ``ainimator.training.controller_multiclip_v2`` must fail with
    ``ModuleNotFoundError``.
    """
    with pytest.raises(ModuleNotFoundError):
        import ainimator.training.controller_multiclip_v2  # noqa: F401


def test_multiclip_requires_two_train_clips(tmp_path: Path) -> None:
    """generalization needs at least 2 train clips (mirrors old check)."""
    with pytest.raises(ValueError, match="at least 2"):
        runControllerGeneralization(
            [_clip(0, 0.01)], _clips(), _config(tmp_path)
        )


def test_multiclip_trains_and_reproduces(tmp_path: Path) -> None:
    """Multi-clip training converges; train loss is finite and low.

    Contract verdicts on the held-out set are NOT asserted here.  With
    2 train / 2 held-out clips and 250 epochs the model memorizes its
    2 train clips but is not expected to generalize to the held-out
    set — that requires many more clips (A6/A7 full profile).  We only
    verify that the run completes, the loss is low, and all held-out
    metrics are finite.
    """
    import math as _math

    trainClips = _clips()[:2]
    heldOut = _clips()[2:]
    result = runControllerGeneralization(
        trainClips, heldOut, _config(tmp_path)
    )
    assert result.finalLoss < 0.1
    for key, value in result.heldOutMetrics.items():
        assert _math.isfinite(value), f"{key} is not finite"


def test_multiclip_metrics_present(tmp_path: Path) -> None:
    """Held-out metrics dict carries all expected keys."""
    trainClips = _clips()[:2]
    heldOut = _clips()[2:]
    result = runControllerGeneralization(
        trainClips, heldOut, _config(tmp_path)
    )
    for key in (
        "control_sensitivity",
        "mean_collapse_rank",
        "mean_collapse_sim",
        "rollout_drift",
        "reconstruction_geodesic",
        "post_norm_stats",
    ):
        assert key in result.heldOutMetrics
        assert math.isfinite(result.heldOutMetrics[key])


def test_multiclip_checkpoint_round_trips(tmp_path: Path) -> None:
    """Saved checkpoint loads back with the correct architecture."""
    trainClips = _clips()[:2]
    heldOut = _clips()[2:]
    result = runControllerGeneralization(
        trainClips, heldOut, _config(tmp_path)
    )
    model, _s, _d, _m, _st = loadControllerCheckpoint(result.checkpointPath)
    assert model.config.contextFrames == 4
