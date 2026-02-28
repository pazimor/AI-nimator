"""Tests for generation loss functions."""

from __future__ import annotations

import torch

from src.shared.model.generation.losses import (
    _rot6dToJointXYZ,
    combinedGenerationLoss,
    velocityXyzLoss,
    xyzLoss,
)


def _randomMotion(
    batchSize: int = 2,
    frameCount: int = 6,
    boneCount: int = 22,
) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batchSize, frameCount, boneCount, 6)


def test_xyzLoss_is_zero_for_identical_inputs() -> None:
    motion = _randomMotion()
    mask = torch.ones(motion.shape[:2], dtype=torch.bool)
    lossValue = xyzLoss(motion, motion, motionMask=mask)
    assert torch.isclose(lossValue, torch.tensor(0.0), atol=1e-6)


def test_xyzLoss_ignores_masked_frames() -> None:
    target = _randomMotion(batchSize=1, frameCount=4)
    predicted = target.clone()
    predicted[:, 2:] = predicted[:, 2:] + 4.0
    mask = torch.tensor([[True, True, False, False]])
    lossValue = xyzLoss(predicted, target, motionMask=mask)
    assert float(lossValue.item()) < 1e-6


def test_xyzLoss_has_finite_gradients() -> None:
    predicted = _randomMotion(batchSize=1, frameCount=4).requires_grad_(True)
    target = _randomMotion(batchSize=1, frameCount=4) + 0.25
    lossValue = xyzLoss(predicted, target)
    lossValue.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum().item()) > 0.0


def test_combinedLoss_reports_xyz_component() -> None:
    targetMotion = _randomMotion(batchSize=1, frameCount=4)
    predictedMotion = targetMotion.clone()
    predictedMotion[:, :, 0, 0] = predictedMotion[:, :, 0, 0] + 0.2
    mask = torch.ones(targetMotion.shape[:2], dtype=torch.bool)

    totalLoss, components = combinedGenerationLoss(
        predictedMotion=predictedMotion,
        targetMotion=targetMotion,
        diffusionWeight=0.0,
        xyzWeight=2.0,
        velocityWeight=0.0,
        accelerationWeight=0.0,
        motionMask=mask,
    )

    expected = 2.0 * components["loss_xyz"]
    assert torch.isclose(totalLoss.detach(), expected, atol=1e-6)


def test_combinedLoss_uses_x0_for_diffusion_term() -> None:
    targetMotion = _randomMotion(batchSize=1, frameCount=4)
    predictedMotion = targetMotion.clone()
    predictedMotion[:, :, 0, 0] = predictedMotion[:, :, 0, 0] + 0.25

    totalLoss, components = combinedGenerationLoss(
        predictedMotion=predictedMotion,
        targetMotion=targetMotion,
        diffusionWeight=1.0,
        xyzWeight=0.0,
        velocityWeight=0.0,
        accelerationWeight=0.0,
    )

    expected = ((predictedMotion - targetMotion) ** 2).mean()
    assert torch.isclose(components["loss_diffusion"], expected, atol=1e-6)
    assert torch.isclose(totalLoss.detach(), expected, atol=1e-6)


def test_combinedLoss_matches_weighted_component_sum() -> None:
    targetMotion = _randomMotion(batchSize=1, frameCount=5)
    predictedMotion = targetMotion.clone()
    predictedMotion[:, :, 0, 0] = predictedMotion[:, :, 0, 0] + 0.1

    totalLoss, components = combinedGenerationLoss(
        predictedMotion=predictedMotion,
        targetMotion=targetMotion,
        diffusionWeight=0.7,
        xyzWeight=1.3,
        velocityXyzWeight=0.2,
        accelerationWeight=0.4,
    )

    expected = (
        0.7 * components["loss_diffusion"]
        + 1.3 * components["loss_xyz"]
        + components["loss_vel_xyz"]
        + components["loss_acceleration"]
    )
    assert torch.isclose(totalLoss.detach(), expected, atol=1e-6)


def test_velocityXyzLoss_is_zero_for_identical_motion() -> None:
    motion = _randomMotion(batchSize=1, frameCount=5)
    mask = torch.ones(motion.shape[:2], dtype=torch.bool)
    lossValue = velocityXyzLoss(
        predictedMotion=motion,
        targetMotion=motion,
        weight=1.0,
        motionMask=mask,
    )
    assert torch.isclose(lossValue, torch.tensor(0.0), atol=1e-6)


def test_velocityXyzLoss_detects_temporal_mismatch() -> None:
    target = _randomMotion(batchSize=1, frameCount=6)
    predicted = target.clone()
    predicted[:, 1:] = target[:, :-1]
    lossValue = velocityXyzLoss(
        predictedMotion=predicted,
        targetMotion=target,
        weight=1.0,
    )
    assert float(lossValue.item()) > 0.0


def test_combinedLoss_can_disable_vel_xyz_and_acc() -> None:
    targetMotion = _randomMotion(batchSize=1, frameCount=5)
    predictedMotion = targetMotion.clone()
    predictedMotion[:, 2:, :, :] = predictedMotion[:, 2:, :, :] + 0.3

    _, components = combinedGenerationLoss(
        predictedMotion=predictedMotion,
        targetMotion=targetMotion,
        diffusionWeight=0.0,
        xyzWeight=0.0,
        velocityXyzWeight=0.0,
        accelerationWeight=0.0,
    )

    assert torch.isclose(components["loss_vel_xyz"], torch.tensor(0.0))
    assert torch.isclose(
        components["loss_acceleration"],
        torch.tensor(0.0),
    )


def test_root_joint_stays_in_place_without_translation() -> None:
    motion = _randomMotion(batchSize=1, frameCount=5)
    jointXyz = _rot6dToJointXYZ(motion)
    rootTrajectory = jointXyz[:, :, 0, :]
    drift = (rootTrajectory - rootTrajectory[:, :1, :]).abs().max()
    assert float(drift.item()) < 1e-5
