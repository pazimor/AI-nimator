"""Tests for multi-clip controller training (C2/C4 validation path)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ainimator.core.constants.controller import PhaseMode
from ainimator.health.contract import Verdict
from ainimator.training.controller_multiclip_v2 import runControllerMultiClip
from ainimator.training.controller_training_v2 import (
    ControllerTrainingConfig,
    loadControllerCheckpoint,
)


def _clip(seed: int, speed: float, frames: int = 56) -> tuple[
    torch.Tensor, torch.Tensor
]:
    phase = torch.linspace(0.0, 4.0 * math.pi, frames)
    rotation6d = torch.zeros(frames, 22, 6)
    for bone in range(22):
        for channel in range(6):
            rotation6d[:, bone, channel] = math.cos(
                bone + channel + seed
            ) + 0.3 * torch.sin(phase + 0.2 * bone + 0.5 * channel + seed)
    rootTranslation = torch.zeros(frames, 3)
    rootTranslation[:, 0] = speed * torch.arange(frames)
    rootTranslation[:, 2] = 0.05 * torch.cos(phase)
    return rotation6d, rootTranslation


def _clips() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        _clip(0, 0.01),
        _clip(1, 0.03),
        _clip(2, 0.05),
        _clip(3, 0.02),
    ]


def _config(outputDir: Path) -> ControllerTrainingConfig:
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


def test_multiclip_requires_two_clips(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        runControllerMultiClip([_clip(0, 0.01)], _config(tmp_path))


def test_multiclip_trains_and_reproduces(tmp_path: Path) -> None:
    result = runControllerMultiClip(_clips(), _config(tmp_path))
    assert result.finalLoss < 0.1
    # Per-clip rollouts reproduce their clips.
    assert result.verdicts["rollout_drift"] is Verdict.OK
    assert result.verdicts["post_norm_stats"] is Verdict.OK


def test_multiclip_metrics_present(tmp_path: Path) -> None:
    result = runControllerMultiClip(_clips(), _config(tmp_path))
    for key in (
        "control_sensitivity",
        "mean_collapse_rank",
        "mean_collapse_sim",
        "rollout_drift",
        "post_norm_stats",
    ):
        assert key in result.metrics
        assert math.isfinite(result.metrics[key])


def test_multiclip_checkpoint_round_trips(tmp_path: Path) -> None:
    result = runControllerMultiClip(_clips(), _config(tmp_path))
    model, _s, _d, _m, _st = loadControllerCheckpoint(result.checkpointPath)
    assert model.config.contextFrames == 4


def test_multiclip_warm_start(tmp_path: Path) -> None:
    base = runControllerMultiClip(_clips(), _config(tmp_path / "base"))
    fineTune = ControllerTrainingConfig(
        outputDir=tmp_path / "ss",
        epochs=30,
        device="cpu",
        logEvery=999,
        scheduledSampling=0.25,
        resumeCheckpoint=base.checkpointPath,
    )
    result = runControllerMultiClip(_clips(), fineTune)
    assert math.isfinite(result.finalLoss)
