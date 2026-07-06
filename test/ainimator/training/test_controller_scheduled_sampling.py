"""Phase A4 tests — scheduled sampling + drift-vs-length curve."""

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
    loadControllerCheckpoint,
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
    # Project onto valid 6D rotations: real (AMASS) states are always on
    # the rotation manifold, and the normative inference loop
    # (inference_contract.md §3.6) assumes it.
    from ainimator.geometry.components import orthonormalizeRot6d

    rotation6d = orthonormalizeRot6d(rotation6d)
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


def test_scheduled_sampling_warm_start_from_checkpoint(
    tmp_path: Path,
) -> None:
    """SS fine-tunes a teacher-forced checkpoint (the correct recipe)."""
    rotation6d, rootTranslation = _syntheticClip()
    teacherForced = ControllerTrainingConfig(
        outputDir=tmp_path / "tf",
        epochs=120,
        device="cpu",
        embedDim=64,
        numHeads=4,
        numLayers=2,
        contextFrames=4,
        logEvery=999,
        seed=0,
    )
    tfResult = runControllerOverfit(
        rotation6d, rootTranslation, teacherForced
    )
    fineTune = ControllerTrainingConfig(
        outputDir=tmp_path / "ss",
        epochs=40,
        device="cpu",
        logEvery=999,
        scheduledSampling=0.25,
        resumeCheckpoint=tfResult.checkpointPath,
    )
    ssResult = runControllerOverfit(rotation6d, rootTranslation, fineTune)
    assert math.isfinite(ssResult.finalLoss)
    # The resumed model kept the checkpoint architecture (ctx=4), not the
    # fine-tune config defaults (ctx=1).
    model, _s, _d, _m, _st = loadControllerCheckpoint(ssResult.checkpointPath)
    assert model.config.contextFrames == 4


def test_resume_preserves_architecture(tmp_path: Path) -> None:
    rotation6d, rootTranslation = _syntheticClip(frames=48)
    base = ControllerTrainingConfig(
        outputDir=tmp_path / "base",
        epochs=20,
        device="cpu",
        embedDim=64,
        numHeads=4,
        numLayers=2,
        contextFrames=3,
        logEvery=999,
    )
    baseResult = runControllerOverfit(rotation6d, rootTranslation, base)
    resumed = ControllerTrainingConfig(
        outputDir=tmp_path / "resumed",
        epochs=10,
        device="cpu",
        logEvery=999,
        resumeCheckpoint=baseResult.checkpointPath,
    )
    result = runControllerOverfit(rotation6d, rootTranslation, resumed)
    model, _s, _d, _m, _st = loadControllerCheckpoint(result.checkpointPath)
    assert model.config.contextFrames == 3
    assert model.config.embedDim == 64


# ---------------------------------------------------------------------
# Closed-loop rollout loss (2026-07-05 long-horizon divergence fix)
# ---------------------------------------------------------------------
def test_rollout_loss_trains_and_reports_components(tmp_path: Path) -> None:
    """rolloutLossHorizon > 0 adds finite rollout_* loss components."""
    rotation6d, rootTranslation = _syntheticClip()
    config = ControllerTrainingConfig(
        outputDir=tmp_path,
        epochs=30,
        device="cpu",
        embedDim=64,
        numHeads=4,
        numLayers=2,
        logEvery=30,
        seed=0,
        rolloutLossHorizon=6,
        rolloutLossWeight=0.5,
    )
    result = runControllerOverfit(rotation6d, rootTranslation, config)
    assert math.isfinite(result.finalLoss)


def test_rollout_loss_window_feeds_back_valid_rotations(
    tmp_path: Path,
) -> None:
    """The fed-back frames stay on the rotation manifold (contract §3.6)."""
    from ainimator.data.controller_sequences import (
        ControllerSequenceConfig,
        buildControllerSequences,
    )
    from ainimator.geometry.components import orthonormalizeRot6d
    from ainimator.model.controller_v2 import (
        ControllerV2Config,
        MotionController,
    )
    from ainimator.model.losses_controller_v2 import ControllerLossWeights
    from ainimator.training.controller_scheduled_sampling import (
        rolloutLossWindow,
    )
    from ainimator.training.controller_training_v2 import (
        _fitNormalizers,
        _normalizeTensors,
    )

    torch.manual_seed(0)
    rotation6d, rootTranslation = _syntheticClip(32)
    batch = buildControllerSequences(
        rotation6d, rootTranslation,
        ControllerSequenceConfig(contextFrames=4),
    )
    stateNormalizer, deltaNormalizer = _fitNormalizers(batch, 22)
    from ainimator.core.constants.controller import PhaseMode

    model = MotionController(
        ControllerV2Config(
            embedDim=32, numHeads=2, numLayers=1, contextFrames=4,
            phaseMode=PhaseMode.NONE,
        )
    )
    tensors = _normalizeTensors(batch, stateNormalizer, deltaNormalizer)
    controlNorm = torch.zeros(batch.numTransitions, 2)

    result = rolloutLossWindow(
        model, batch, stateNormalizer, deltaNormalizer, controlNorm,
        tensors, ControllerLossWeights(), horizon=8,
    )
    assert math.isfinite(float(result.total))
    result.total.backward()
    grads = [
        p.grad for p in model.parameters() if p.grad is not None
    ]
    assert grads, "rollout loss must produce gradients"
