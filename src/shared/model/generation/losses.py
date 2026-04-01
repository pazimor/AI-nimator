"""Loss functions for motion generation training."""

from __future__ import annotations

import torch

from src.shared.model.components.ops import (
    maskedMean,
    rot6dToJointXYZ,
    temporalDifference,
)

DEFAULT_DIFFUSION_WEIGHT = 1.0
DEFAULT_XYZ_WEIGHT = 0.1
DEFAULT_VELOCITY_WEIGHT = 0.01
DEFAULT_VELOCITY_XYZ_WEIGHT = 0.01
DEFAULT_ACCELERATION_WEIGHT = 0.001
XYZ_SCHEDULE_NONE = "none"
XYZ_SCHEDULE_TIMESTEP = "timestep"
MIN_DIFFUSION_STEPS = 1


def diffusionLoss(
    predictedNoise: torch.Tensor,
    targetNoise: torch.Tensor,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Standard MSE loss for noise prediction in diffusion models.

    Parameters
    ----------
    predictedNoise : torch.Tensor
        Predicted noise from the denoiser.
    targetNoise : torch.Tensor
        Ground truth noise that was added.
    motionMask : torch.Tensor | None, optional
        Boolean mask indicating valid (non-padded) frames.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss.
    """
    squaredError = (predictedNoise - targetNoise) ** 2
    return maskedMean(squaredError, motionMask)


def startMotionLoss(
    predictedMotion: torch.Tensor,
    targetMotion: torch.Tensor,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Direct MSE on predicted clean motion x0 (MDM-style target).

    Parameters
    ----------
    predictedMotion : torch.Tensor
        Predicted clean motion shaped (batch, frames, bones, 6).
    targetMotion : torch.Tensor
        Ground truth clean motion shaped (batch, frames, bones, 6).
    motionMask : torch.Tensor | None, optional
        Boolean mask indicating valid (non-padded) frames.

    Returns
    -------
    torch.Tensor
        Scalar masked MSE in raw motion space.
    """
    squaredError = (predictedMotion - targetMotion) ** 2
    return maskedMean(squaredError, motionMask)


def xyzLoss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    XYZ reconstruction loss for 6D rotation representations.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted 6D rotations shaped (batch, frames, bones, 6).
    target : torch.Tensor
        Target 6D rotations shaped (batch, frames, bones, 6).
    motionMask : torch.Tensor | None, optional
        Boolean mask indicating valid (non-padded) frames.

    Returns
    -------
    torch.Tensor
        Scalar masked XYZ MSE.
    """
    predictedXYZ = rot6dToJointXYZ(predicted)
    targetXYZ = rot6dToJointXYZ(target)
    squaredError = (predictedXYZ - targetXYZ) ** 2
    return maskedMean(squaredError, motionMask)


def velocityLoss(
    motion: torch.Tensor,
    weight: float = 1.0,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Temporal velocity regularization loss.

    Encourages smooth motion by penalizing large frame-to-frame differences.

    Parameters
    ----------
    motion : torch.Tensor
        Motion sequence shaped (batch, frames, bones, channels).
    weight : float, optional
        Loss weight, by default 1.0.
    motionMask : torch.Tensor | None, optional
        Boolean mask indicating valid (non-padded) frames.

    Returns
    -------
    torch.Tensor
        Scalar velocity loss.
    """
    if motion.shape[1] < 2:
        return torch.tensor(0.0, device=motion.device)

    velocity = temporalDifference(motion, dim=1)[:, 1:]
    if motionMask is not None:
        motionMask = motionMask[:, 1:] & motionMask[:, :-1]
    return weight * maskedMean(velocity ** 2, motionMask)


def velocityXyzLoss(
    predictedMotion: torch.Tensor,
    targetMotion: torch.Tensor,
    weight: float = 1.0,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Velocity matching loss in joint XYZ space (MDM-like vel_xyz term).

    Parameters
    ----------
    predictedMotion : torch.Tensor
        Predicted 6D rotations shaped (batch, frames, bones, 6).
    targetMotion : torch.Tensor
        Target 6D rotations shaped (batch, frames, bones, 6).
    weight : float, optional
        Loss weight, by default 1.0.
    motionMask : torch.Tensor | None, optional
        Boolean mask indicating valid (non-padded) frames.

    Returns
    -------
    torch.Tensor
        Scalar XYZ velocity matching loss.
    """
    if predictedMotion.shape[1] < 2:
        return torch.tensor(0.0, device=predictedMotion.device)

    predictedXyz = rot6dToJointXYZ(predictedMotion)
    targetXyz = rot6dToJointXYZ(targetMotion)
    predictedVelocity = temporalDifference(predictedXyz, dim=1)[:, 1:]
    targetVelocity = temporalDifference(targetXyz, dim=1)[:, 1:]
    velocityError = (predictedVelocity - targetVelocity) ** 2

    if motionMask is not None:
        motionMask = motionMask[:, 1:] & motionMask[:, :-1]
    return weight * maskedMean(velocityError, motionMask)


def accelerationLoss(
    motion: torch.Tensor,
    weight: float = 1.0,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Temporal acceleration regularization loss.

    Encourages smooth velocity changes.

    Parameters
    ----------
    motion : torch.Tensor
        Motion sequence shaped (batch, frames, bones, channels).
    weight : float, optional
        Loss weight, by default 1.0.
    motionMask : torch.Tensor | None, optional
        Boolean mask indicating valid (non-padded) frames.

    Returns
    -------
    torch.Tensor
        Scalar acceleration loss.
    """
    if motion.shape[1] < 3:
        return torch.tensor(0.0, device=motion.device)

    velocity = temporalDifference(motion, dim=1)
    acceleration = temporalDifference(velocity, dim=1)[:, 2:]
    if motionMask is not None:
        motionMask = (
            motionMask[:, 2:]
            & motionMask[:, 1:-1]
            & motionMask[:, :-2]
        )
    return weight * maskedMean(acceleration ** 2, motionMask)


DEFAULT_FOOT_SKATING_WEIGHT = 5.0
FOOT_JOINT_INDICES = (7, 10, 8, 11)  # leftAnkle, leftFoot, rightAnkle, rightFoot


def footSkatingLoss(
    predictedMotion: torch.Tensor,
    footContact: torch.Tensor | None,
    weight: float = DEFAULT_FOOT_SKATING_WEIGHT,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Penalize foot-joint velocity when the foot is in contact with the ground.

    This is the standard foot-skating loss used in MDM/MLD to enforce
    physically plausible foot plants during locomotion.

    Parameters
    ----------
    predictedMotion : torch.Tensor
        Predicted 6D rotations shaped (batch, frames, bones, 6).
    footContact : torch.Tensor | None
        Ground-truth contact labels shaped (batch, frames, 4).
        Channels: leftAnkle, leftFoot, rightAnkle, rightFoot.
        When None, returns zero.
    weight : float
        Loss weight.
    motionMask : torch.Tensor | None
        Valid-frame mask.

    Returns
    -------
    torch.Tensor
        Scalar skating penalty.
    """
    if footContact is None or predictedMotion.shape[1] < 2:
        return torch.tensor(0.0, device=predictedMotion.device)

    jointXyz = rot6dToJointXYZ(predictedMotion)
    footVel = temporalDifference(jointXyz[:, :, FOOT_JOINT_INDICES, :], dim=1)

    # Expand contact to match XYZ channels: (batch, frames, 4) → (batch, frames, 4, 3)
    contactMask = footContact.unsqueeze(-1).expand_as(footVel)
    skating = (footVel ** 2) * contactMask

    if motionMask is not None:
        frameMask = motionMask.unsqueeze(-1).unsqueeze(-1).expand_as(skating)
        skating = skating * frameMask.float()

    return weight * skating.mean()


def combinedGenerationLoss(
    predictedMotion: torch.Tensor,
    targetMotion: torch.Tensor,
    diffusionWeight: float = DEFAULT_DIFFUSION_WEIGHT,
    xyzWeight: float = DEFAULT_XYZ_WEIGHT,
    velocityWeight: float = DEFAULT_VELOCITY_WEIGHT,
    velocityXyzWeight: float | None = None,
    accelerationWeight: float = DEFAULT_ACCELERATION_WEIGHT,
    xyzWeightSchedule: str = XYZ_SCHEDULE_NONE,
    timesteps: torch.Tensor | None = None,
    numTimesteps: int | None = None,
    motionMask: torch.Tensor | None = None,
    footContact: torch.Tensor | None = None,
    footSkatingWeight: float = DEFAULT_FOOT_SKATING_WEIGHT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combined loss for motion generation training.

    Parameters
    ----------
    predictedMotion : torch.Tensor
        Predicted clean motion (x0) from the denoiser.
    targetMotion : torch.Tensor
        Ground truth motion.
    diffusionWeight : float, optional
        Weight for diffusion loss, by default 1.0.
    xyzWeight : float, optional
        Weight for XYZ reconstruction loss, by default 0.1.
    velocityWeight : float, optional
        Weight for velocity loss, by default 0.01.
    velocityXyzWeight : float | None, optional
        Weight for velocity matching loss in XYZ space.
        When None, velocityWeight is used for backward compatibility.
    accelerationWeight : float, optional
        Weight for acceleration loss, by default 0.001.
    xyzWeightSchedule : str, optional
        Schedule mode for XYZ weight, by default "none".
    timesteps : torch.Tensor | None, optional
        Diffusion timesteps for schedule-aware weighting.
    numTimesteps : int | None, optional
        Total diffusion steps for schedule-aware weighting.
    motionMask : torch.Tensor | None, optional
        Boolean mask indicating valid (non-padded) frames.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        Total loss and dictionary of individual loss components.
    """
    lossDiff = startMotionLoss(
        predictedMotion,
        targetMotion,
        motionMask=motionMask,
    )
    lossXyz = xyzLoss(
        predictedMotion,
        targetMotion,
        motionMask=motionMask,
    )
    weightedXyz = _resolveXyzWeight(
        xyzWeight,
        xyzWeightSchedule,
        timesteps,
        numTimesteps,
        predictedMotion.device,
    )
    resolvedVelocityXyzWeight = (
        velocityWeight
        if velocityXyzWeight is None
        else velocityXyzWeight
    )
    lossVel = velocityXyzLoss(
        predictedMotion,
        targetMotion,
        resolvedVelocityXyzWeight,
        motionMask=motionMask,
    )
    lossAcc = accelerationLoss(
        predictedMotion,
        accelerationWeight,
        motionMask=motionMask,
    )

    total = (
        diffusionWeight * lossDiff
        + weightedXyz * lossXyz
        + lossVel
        + lossAcc
    )

    lossSkating = footSkatingLoss(
        predictedMotion,
        footContact,
        weight=footSkatingWeight,
        motionMask=motionMask,
    )
    total = total + lossSkating

    components = {
        "loss_diffusion": lossDiff.detach(),
        "loss_xyz": lossXyz.detach(),
        "loss_vel_xyz": lossVel.detach(),
        # Backward-compatibility alias used in older logs/consumers.
        "loss_velocity": lossVel.detach(),
        "loss_acceleration": lossAcc.detach(),
        "loss_foot_skating": lossSkating.detach(),
    }

    return total, components


def _resolveXyzWeight(
    baseWeight: float,
    schedule: str,
    timesteps: torch.Tensor | None,
    numTimesteps: int | None,
    device: torch.device,
) -> torch.Tensor:
    """
    Resolve XYZ weight based on schedule and timesteps.
    """
    if schedule == XYZ_SCHEDULE_NONE:
        return torch.tensor(baseWeight, device=device)
    if schedule == XYZ_SCHEDULE_TIMESTEP:
        if timesteps is None or numTimesteps is None:
            return torch.tensor(baseWeight, device=device)
        denom = max(numTimesteps - 1, MIN_DIFFUSION_STEPS)
        weights = 1.0 - (timesteps.float() / float(denom))
        return weights.mean() * baseWeight
    raise ValueError(f"Unknown XYZ schedule: {schedule}")
