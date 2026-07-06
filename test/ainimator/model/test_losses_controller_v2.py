"""Phase A1 tests — controller loss functions."""

from __future__ import annotations

import math

import pytest
import torch

from ainimator.model.losses_controller_v2 import (
    ControllerLossWeights,
    combinedControllerLoss,
    footContactStepLoss,
    geodesicRotationLoss,
    velocityDeltaLoss,
)


def test_velocity_loss_zero_on_identical() -> None:
    delta = torch.randn(4, 22, 6)
    assert float(velocityDeltaLoss(delta, delta.clone())) == pytest.approx(0.0)


def test_velocity_loss_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        velocityDeltaLoss(torch.randn(4, 22, 6), torch.randn(4, 3))


def test_geodesic_zero_on_identical_rotation() -> None:
    rot6d = torch.randn(2, 22, 6)
    loss = geodesicRotationLoss(rot6d, rot6d.clone())
    # Inherent floor ~sqrt(2 * acos clamp eps) from the acos singularity.
    assert float(loss) == pytest.approx(0.0, abs=2e-3)


def test_geodesic_positive_on_rotated() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    predicted = identity.view(1, 1, 6).repeat(1, 22, 1)
    # 90° about Z: first axis -> (0,1,0), second -> (-1,0,0)
    rotated = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    target = rotated.view(1, 1, 6).repeat(1, 22, 1)
    loss = float(geodesicRotationLoss(predicted, target))
    assert loss == pytest.approx(math.pi / 2, abs=1e-2)


def test_geodesic_gradient_is_finite() -> None:
    rot6d = torch.randn(2, 22, 6, requires_grad=True)
    target = torch.randn(2, 22, 6)
    loss = geodesicRotationLoss(rot6d, target)
    loss.backward()
    assert rot6d.grad is not None
    assert torch.isfinite(rot6d.grad).all()


def test_combined_loss_weights_applied() -> None:
    velocity = torch.tensor(2.0)
    geodesic = torch.tensor(3.0)
    weights = ControllerLossWeights(velocity=0.5, geodesic=2.0)
    result = combinedControllerLoss(velocity, geodesic, weights)
    assert float(result.total) == pytest.approx(0.5 * 2.0 + 2.0 * 3.0)
    assert set(result.components) == {"loss_velocity", "loss_geodesic"}


def test_foot_contact_step_zero_without_contact() -> None:
    predNext = torch.randn(4, 22, 6)
    last = torch.randn(4, 22, 6)
    globalDelta = torch.randn(4, 3)
    contact = torch.zeros(4, 2)  # no foot in contact
    loss = footContactStepLoss(predNext, last, globalDelta, contact)
    assert float(loss) == pytest.approx(0.0)


def test_foot_contact_step_positive_under_contact_and_motion() -> None:
    last = torch.randn(4, 22, 6)
    predNext = last + 0.5  # moved rotations -> foot moves
    globalDelta = torch.ones(4, 3) * 0.2
    contact = torch.ones(4, 2)  # both feet in contact
    loss = footContactStepLoss(predNext, last, globalDelta, contact)
    assert float(loss) > 0.0


def test_combined_loss_includes_foot_contact_when_weighted() -> None:
    result = combinedControllerLoss(
        torch.tensor(1.0),
        torch.tensor(1.0),
        ControllerLossWeights(velocity=1.0, geodesic=1.0, footContact=1.0),
        footContact=torch.tensor(4.0),
    )
    assert "loss_foot_contact" in result.components
    assert float(result.total) == pytest.approx(6.0)
