"""Tests for the autoregressive controller rollout (incl. phase wiring)."""

from __future__ import annotations

import pytest
import torch

from ainimator.core.constants.controller import PhaseMode
from ainimator.core.types.controller import ControllerV2Config
from ainimator.model.controller_rollout import (
    rolloutController,
    rolloutControllerClosedLoop,
)
from ainimator.model.controller_v2 import MotionController
from ainimator.model.motion_normalizer import MotionNormalizer


def _normalizers() -> tuple[MotionNormalizer, MotionNormalizer]:
    state = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)
    delta = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)
    return state, delta


def _model(phaseMode: PhaseMode) -> MotionController:
    config = ControllerV2Config(
        embedDim=32, numHeads=4, numLayers=2, contextFrames=1,
        phaseMode=phaseMode,
    )
    return MotionController(config).eval()


def test_rollout_shapes_no_phase() -> None:
    model = _model(PhaseMode.NONE)
    stateNorm, deltaNorm = _normalizers()
    steps = 7
    result = rolloutController(
        model,
        stateNorm,
        deltaNorm,
        torch.randn(1, 1, 22, 6),
        torch.randn(1, 1, 3),
        torch.randn(1, steps, model.config.controlChannels),
    )
    assert result.rotation6d.shape == (1, 1 + steps, 22, 6)
    assert result.rootTranslation.shape == (1, 1 + steps, 3)


def test_rollout_with_explicit_phase_runs() -> None:
    model = _model(PhaseMode.EXPLICIT)
    stateNorm, deltaNorm = _normalizers()
    steps = 5
    result = rolloutController(
        model,
        stateNorm,
        deltaNorm,
        torch.randn(1, 1, 22, 6),
        torch.randn(1, 1, 3),
        torch.randn(1, steps, model.config.controlChannels),
        phaseSequence=torch.randn(1, steps, model.config.phaseChannels),
    )
    assert result.rotation6d.shape == (1, 1 + steps, 22, 6)


def test_closed_loop_reinjection_reduces_drift() -> None:
    """More frequent GT re-injection must not increase drift."""
    model = _model(PhaseMode.NONE)
    stateNorm, deltaNorm = _normalizers()
    frames, steps = 41, 40
    gtBone = torch.randn(1, frames, 22, 6)
    gtRoot = torch.randn(1, frames, 3)
    control = torch.randn(1, steps, model.config.controlChannels)

    def drift(period: int) -> float:
        rollout = rolloutControllerClosedLoop(
            model, stateNorm, deltaNorm, gtBone, gtRoot, control, period
        )
        return float(((rollout.rotation6d - gtBone) ** 2).mean())

    openLoop = drift(0)
    frequent = drift(4)
    veryFrequent = drift(1)
    assert frequent <= openLoop
    assert veryFrequent <= frequent
    # re-injecting every frame ⇒ trajectory is ground truth ⇒ ~0 drift.
    assert veryFrequent < 1e-6


def test_rollout_explicit_phase_missing_sequence_raises() -> None:
    """A phase-conditioned model must be given a phase sequence."""
    model = _model(PhaseMode.EXPLICIT)
    stateNorm, deltaNorm = _normalizers()
    with pytest.raises(ValueError, match="phase is required"):
        rolloutController(
            model,
            stateNorm,
            deltaNorm,
            torch.randn(1, 1, 22, 6),
            torch.randn(1, 1, 3),
            torch.randn(1, 4, model.config.controlChannels),
        )
