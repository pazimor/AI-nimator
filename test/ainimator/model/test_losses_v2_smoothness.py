"""Regression tests for the temporal-smoothness losses (2026-06-18).

Guards two things:
1. The ``temporalDifference`` dim bug — velocity/acceleration losses must
   difference over the FRAME axis (dim=1), not the batch (dim=0).  A
   batch of identical-but-time-varying samples must yield a NON-zero
   loss (the old dim=0 default returned ~0).
2. ``rotationJerkLossV2`` penalises rot6d trembling: a jittery prediction
   against a smooth target must score higher than a smooth one.
"""

from __future__ import annotations

import torch

from ainimator.model.losses_v2 import (
    accelerationXyzLossV2,
    rotationJerkLossV2,
    velocityXyzLossV2,
)

_BONES = 22
_FRAMES = 24
_ALPHAS = torch.linspace(1.0, 1e-3, 1000)
_TS = torch.zeros(4, dtype=torch.long)


def _batchIdenticalTimeVarying() -> tuple[torch.Tensor, torch.Tensor]:
    """4 identical samples, each varying over time (batch diff ≡ 0)."""
    one = torch.randn(1, _FRAMES, _BONES, 6)
    other = torch.randn(1, _FRAMES, _BONES, 6)
    return one.repeat(4, 1, 1, 1), other.repeat(4, 1, 1, 1)


def test_velocity_loss_differences_over_time_not_batch() -> None:
    """Velocity loss is non-zero for batch-identical, time-varying input."""
    pred, target = _batchIdenticalTimeVarying()
    value = velocityXyzLossV2(pred, target, _TS, _ALPHAS, schedule="none")
    assert value > 0.0


def test_acceleration_loss_differences_over_time_not_batch() -> None:
    """Acceleration loss is non-zero for batch-identical input."""
    pred, target = _batchIdenticalTimeVarying()
    value = accelerationXyzLossV2(pred, target, _TS, _ALPHAS, schedule="none")
    assert value > 0.0


def test_rotation_jerk_penalises_trembling() -> None:
    """A jittery rot6d prediction scores higher than a smooth one."""
    frames = 40
    smooth = torch.zeros(1, frames, _BONES, 6)
    for frame in range(frames):
        smooth[0, frame] = torch.sin(torch.tensor(frame * 0.2))
    jittery = smooth + 0.3 * torch.randn(1, frames, _BONES, 6)
    timestep = torch.zeros(1, dtype=torch.long)
    smoothScore = rotationJerkLossV2(
        smooth, smooth, timestep, _ALPHAS, schedule="none"
    )
    jitterScore = rotationJerkLossV2(
        jittery, smooth, timestep, _ALPHAS, schedule="none"
    )
    assert float(smoothScore) == 0.0
    assert float(jitterScore) > float(smoothScore)
