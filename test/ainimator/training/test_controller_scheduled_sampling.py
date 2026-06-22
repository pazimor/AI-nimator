"""Phase C4 tests — scheduled sampling + drift-vs-length curve."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ainimator.health.contract import Verdict
from ainimator.training.controller_scheduled_sampling import (
    scheduledSamplingProbability,
)
from ainimator.training.controller_training_v2 import (
    ControllerTrainingConfig,
    runControllerOverfit,
)


def _syntheticClip(frames: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    phase = torch.linspace(0.0, 4.0 * math.pi, frames)
    rotation6d = torch.zeros(frames, 22, 6)
    for bone in range(22):
        for channel in range(6):
            rotation6d[:, bone, channel] = math.cos(bone + channel) + (
                0.3 * torch.sin(phase + 0.2 * bone + 0.5 * channel)
            )
    rootTranslation = torch.zeros(frames, 3)
    rootTranslation[:, 0] = 0.02 * torch.arange(frames)
    rootTranslation[:, 2] = 0.05 * torch.cos(phase)
    return rotation6d, rootTranslation


# ---------------------------------------------------------------------
# Probability ramp
# ---------------------------------------------------------------------
def test_ramp_starts_at_zero() -> None:
    assert scheduledSamplingProbability(0, 100, 0.5) == pytest.approx(0.0)


def test_ramp_reaches_target_at_last_epoch() -> None:
    assert scheduledSamplingProbability(99, 100, 0.5) == pytest.approx(0.5)


def test_ramp_is_monotonic() -> None:
    values = [scheduledSamplingProbability(e, 50, 0.4) for e in range(50)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_ramp_single_epoch_returns_target() -> None:
    assert scheduledSamplingProbability(0, 1, 0.3) == pytest.approx(0.3)


# ---------------------------------------------------------------------
# End-to-end scheduled-sampling training
# ---------------------------------------------------------------------
def test_scheduled_sampling_trains_and_reports_drift_curve(
    tmp_path: Path,
) -> None:
    rotation6d, rootTranslation = _syntheticClip()
    config = ControllerTrainingConfig(
        outputDir=tmp_path,
        epochs=120,
        device="cpu",
        embedDim=64,
        numHeads=4,
        numLayers=2,
        logEvery=60,
        seed=0,
        scheduledSampling=0.25,
    )
    result = runControllerOverfit(rotation6d, rootTranslation, config)
    # The sequential scheduled-sampling loop produced a finite loss and
    # a bounded short-horizon rollout.
    assert math.isfinite(result.finalLoss)
    assert result.verdicts["rollout_drift"] is Verdict.OK
    # The drift-vs-length curve is recorded at increasing horizons.
    assert len(result.driftCurve) >= 3
    horizons = sorted(result.driftCurve)
    assert horizons[-1] >= horizons[0]
    for value in result.driftCurve.values():
        assert math.isfinite(value)
