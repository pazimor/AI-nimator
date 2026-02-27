"""Loss functions for motion generation training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.shared.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL22_DEFAULT_OFFSETS,
    SMPL22_HIERARCHY,
)
from src.shared.types.generation import (
    PREDICTION_TARGET_EPSILON,
    PREDICTION_TARGET_X0,
)

DEFAULT_DIFFUSION_WEIGHT = 1.0
DEFAULT_XYZ_WEIGHT = 0.1
DEFAULT_VELOCITY_WEIGHT = 0.01
DEFAULT_VELOCITY_XYZ_WEIGHT = 0.01
DEFAULT_ACCELERATION_WEIGHT = 0.001
XYZ_SCHEDULE_NONE = "none"
XYZ_SCHEDULE_TIMESTEP = "timestep"
MIN_DIFFUSION_STEPS = 1
MIN_SIXD_CHANNELS = 6

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
    return _maskedMean(squaredError, motionMask)


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
    return _maskedMean(squaredError, motionMask)


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
    predictedXYZ = _rot6dToJointXYZ(predicted)
    targetXYZ = _rot6dToJointXYZ(target)
    squaredError = (predictedXYZ - targetXYZ) ** 2
    return _maskedMean(squaredError, motionMask)


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

    velocity = motion[:, 1:] - motion[:, :-1]
    if motionMask is not None:
        motionMask = motionMask[:, 1:] & motionMask[:, :-1]
    return weight * _maskedMean(velocity ** 2, motionMask)


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

    predictedXyz = _rot6dToJointXYZ(predictedMotion)
    targetXyz = _rot6dToJointXYZ(targetMotion)
    predictedVelocity = predictedXyz[:, 1:] - predictedXyz[:, :-1]
    targetVelocity = targetXyz[:, 1:] - targetXyz[:, :-1]
    velocityError = (predictedVelocity - targetVelocity) ** 2

    if motionMask is not None:
        motionMask = motionMask[:, 1:] & motionMask[:, :-1]
    return weight * _maskedMean(velocityError, motionMask)


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

    velocity = motion[:, 1:] - motion[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    if motionMask is not None:
        motionMask = (
            motionMask[:, 2:]
            & motionMask[:, 1:-1]
            & motionMask[:, :-2]
        )
    return weight * _maskedMean(acceleration ** 2, motionMask)


def combinedGenerationLoss(
    predictedNoise: torch.Tensor,
    targetNoise: torch.Tensor,
    predictedMotion: torch.Tensor,
    targetMotion: torch.Tensor,
    diffusionWeight: float = DEFAULT_DIFFUSION_WEIGHT,
    xyzWeight: float = DEFAULT_XYZ_WEIGHT,
    velocityWeight: float = DEFAULT_VELOCITY_WEIGHT,
    velocityXyzWeight: float | None = None,
    accelerationWeight: float = DEFAULT_ACCELERATION_WEIGHT,
    xyzWeightSchedule: str = XYZ_SCHEDULE_NONE,
    predictionTarget: str = PREDICTION_TARGET_EPSILON,
    timesteps: torch.Tensor | None = None,
    numTimesteps: int | None = None,
    motionMask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combined loss for motion generation training.

    Parameters
    ----------
    predictedNoise : torch.Tensor
        Predicted noise from denoiser.
    targetNoise : torch.Tensor
        Ground truth noise.
    predictedMotion : torch.Tensor
        Reconstructed motion (for regularization).
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
    predictionTarget : str, optional
        Main denoiser target ("epsilon" or "x0").
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
    if predictionTarget == PREDICTION_TARGET_X0:
        lossDiff = startMotionLoss(
            predictedMotion,
            targetMotion,
            motionMask=motionMask,
        )
    elif predictionTarget == PREDICTION_TARGET_EPSILON:
        lossDiff = diffusionLoss(predictedNoise, targetNoise, motionMask)
    else:
        raise ValueError(
            f"Unknown predictionTarget: {predictionTarget!r}"
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
        predictedNoise.device,
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

    contribDiffusion = (diffusionWeight * lossDiff).detach()
    contribXyz = (weightedXyz * lossXyz).detach()
    contribVelXyz = lossVel.detach()
    contribAcceleration = lossAcc.detach()

    components = {
        "loss_diffusion": lossDiff.detach(),
        "loss_xyz": lossXyz.detach(),
        "loss_vel_xyz": lossVel.detach(),
        # Backward-compatibility alias used in older logs/consumers.
        "loss_velocity": lossVel.detach(),
        "loss_acceleration": lossAcc.detach(),
        "contrib_diffusion": contribDiffusion,
        "contrib_xyz": contribXyz,
        "contrib_vel_xyz": contribVelXyz,
        "contrib_acceleration": contribAcceleration,
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

def _maskedMean(
    values: torch.Tensor,
    motionMask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute mean over valid frames when a motion mask is provided.
    """
    values = torch.nan_to_num(values)
    if motionMask is None:
        return values.mean()
    mask = motionMask.to(values.device).float()
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    masked = values * mask
    valid = mask.sum()
    if float(valid.item()) == 0.0:
        return torch.tensor(0.0, device=values.device)
    scale = values.numel() / mask.numel()
    return masked.sum() / (valid * scale)


def _sixdToRotationMatrix(sixd: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix.

    Uses Gram-Schmidt orthogonalization.

    Parameters
    ----------
    sixd : torch.Tensor
        6D rotation shaped (..., 6).

    Returns
    -------
    torch.Tensor
        Rotation matrix shaped (..., 3, 3).
    """
    a1 = sixd[..., :3]
    a2 = sixd[..., 3:6]

    # Normalize first vector
    b1 = F.normalize(a1, dim=-1)

    # Make second vector orthogonal to first
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = F.normalize(b2, dim=-1)

    # Third vector is cross product
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack into rotation matrix
    return torch.stack([b1, b2, b3], dim=-1)


def _rot6dToJointXYZ(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert local 6D rotations to global joint XYZ via FK.

    Parameters
    ----------
    rot6d : torch.Tensor
        Tensor shaped (batch, frames, bones, 6).

    Returns
    -------
    torch.Tensor
        Global joint positions shaped (batch, frames, bones, 3).
    """
    if rot6d.dim() != 4 or rot6d.shape[-1] != MIN_SIXD_CHANNELS:
        raise ValueError(
            "Expected rot6d shape (batch, frames, bones, 6), got "
            f"{tuple(rot6d.shape)}"
        )

    batchSize, frameCount, boneCount, _ = rot6d.shape
    parentIndices, offsets = _smpl22KinematicParams(
        boneCount,
        rot6d.device,
        rot6d.dtype,
    )
    localRotations = _sixdToRotationMatrix(rot6d)

    globalRotations: list[torch.Tensor] = []
    globalPositions: list[torch.Tensor] = []

    for boneIndex in range(boneCount):
        localRotation = localRotations[:, :, boneIndex]
        if parentIndices[boneIndex] < 0:
            globalRotations.append(localRotation)
            rootOffset = offsets[boneIndex].view(1, 1, 3)
            rootOffset = rootOffset.expand(batchSize, frameCount, 3)
            globalPositions.append(rootOffset)
            continue

        parentIndex = parentIndices[boneIndex]
        parentRotation = globalRotations[parentIndex]
        parentPosition = globalPositions[parentIndex]
        globalRotation = torch.matmul(parentRotation, localRotation)
        childOffset = offsets[boneIndex].view(1, 1, 3, 1)
        childOffset = torch.matmul(parentRotation, childOffset).squeeze(-1)
        globalRotations.append(globalRotation)
        globalPositions.append(parentPosition + childOffset)

    return torch.stack(globalPositions, dim=2)


def _smpl22KinematicParams(
    boneCount: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[int], torch.Tensor]:
    """
    Return parent indices and offsets for the first SMPL22 joints.
    """
    maxBones = len(SMPL22_BONE_ORDER)
    if boneCount > maxBones:
        raise ValueError(
            f"Unsupported boneCount={boneCount}, max supported is {maxBones}"
        )

    boneNames = SMPL22_BONE_ORDER[:boneCount]
    indexByName = {name: idx for idx, name in enumerate(boneNames)}
    parentIndices: list[int] = []
    offsetValues: list[list[float]] = []

    for boneName in boneNames:
        parentName = SMPL22_HIERARCHY[boneName]
        if parentName is None:
            parentIndices.append(-1)
        else:
            parentIndex = indexByName.get(parentName)
            if parentIndex is None:
                raise ValueError(
                    "Invalid skeleton order: parent "
                    f"{parentName} missing for {boneName}"
                )
            parentIndices.append(parentIndex)
        offsetValues.append(SMPL22_DEFAULT_OFFSETS[boneName])

    offsets = torch.tensor(
        offsetValues,
        device=device,
        dtype=dtype,
    )
    return parentIndices, offsets
