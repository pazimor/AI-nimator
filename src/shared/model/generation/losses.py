"""Loss functions for motion generation training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.constants.rotation import (
    DEFAULT_ROTATION_REPR,
    ROTATION_KIND_AXIS_ANGLE,
    ROTATION_REPR_AXIS_ANGLE,
    ROTATION_REPR_ROT6D,
)
from src.shared.quaternion import Rotation

DEFAULT_DIFFUSION_WEIGHT = 1.0
DEFAULT_GEODESIC_WEIGHT = 0.1
DEFAULT_VELOCITY_WEIGHT = 0.01
DEFAULT_ACCELERATION_WEIGHT = 0.001
GEODESIC_SCHEDULE_NONE = "none"
GEODESIC_SCHEDULE_TIMESTEP = "timestep"
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
    return _maskedMean(squaredError, motionMask)


def geodesicLoss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    rotationRepr: str = DEFAULT_ROTATION_REPR,
    eps: float = 1e-7,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Geodesic loss for rotation representations.

    Computes loss based on the rotation matrix difference.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted rotations shaped (..., C).
    target : torch.Tensor
        Target rotations shaped (..., C).
    rotationRepr : str, optional
        Rotation representation, by default "rot6d".
    eps : float, optional
        Epsilon for numerical stability, by default 1e-7.
    motionMask : torch.Tensor | None, optional
        Boolean mask indicating valid (non-padded) frames.

    Returns
    -------
    torch.Tensor
        Scalar geodesic loss.
    """
    # Convert rotations to rotation matrices
    predMat = _rotationToMatrix(predicted, rotationRepr)
    targetMat = _rotationToMatrix(target, rotationRepr)

    # Compute R_pred^T @ R_target
    diff = torch.matmul(predMat.transpose(-2, -1), targetMat)

    # Trace of rotation matrix
    trace = diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2]

    # Geodesic distance: arccos((trace - 1) / 2)
    cos = (trace - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos)

    return _maskedMean(angle, motionMask)


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
    geodesicWeight: float = DEFAULT_GEODESIC_WEIGHT,
    velocityWeight: float = DEFAULT_VELOCITY_WEIGHT,
    accelerationWeight: float = DEFAULT_ACCELERATION_WEIGHT,
    geodesicWeightSchedule: str = GEODESIC_SCHEDULE_NONE,
    rotationRepr: str = DEFAULT_ROTATION_REPR,
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
    geodesicWeight : float, optional
        Weight for geodesic loss, by default 0.1.
    velocityWeight : float, optional
        Weight for velocity loss, by default 0.01.
    accelerationWeight : float, optional
        Weight for acceleration loss, by default 0.001.
    geodesicWeightSchedule : str, optional
        Schedule mode for geodesic weight, by default "none".
    rotationRepr : str, optional
        Rotation representation, by default "rot6d".
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
    lossDiff = diffusionLoss(predictedNoise, targetNoise, motionMask)
    lossGeo = geodesicLoss(
        predictedMotion,
        targetMotion,
        rotationRepr=rotationRepr,
        motionMask=motionMask,
    )
    geoWeight = _resolveGeodesicWeight(
        geodesicWeight,
        geodesicWeightSchedule,
        timesteps,
        numTimesteps,
        predictedNoise.device,
    )
    lossVel = velocityLoss(
        predictedMotion,
        velocityWeight,
        motionMask=motionMask,
    )
    lossAcc = accelerationLoss(
        predictedMotion,
        accelerationWeight,
        motionMask=motionMask,
    )

    total = (
        diffusionWeight * lossDiff
        + geoWeight * lossGeo
        + lossVel
        + lossAcc
    )

    components = {
        "loss_diffusion": lossDiff.detach(),
        "loss_geodesic": lossGeo.detach(),
        "loss_velocity": lossVel.detach(),
        "loss_acceleration": lossAcc.detach(),
    }

    return total, components


def _resolveGeodesicWeight(
    baseWeight: float,
    schedule: str,
    timesteps: torch.Tensor | None,
    numTimesteps: int | None,
    device: torch.device,
) -> torch.Tensor:
    """
    Resolve geodesic weight based on schedule and timesteps.
    """
    if schedule == GEODESIC_SCHEDULE_NONE:
        return torch.tensor(baseWeight, device=device)
    if schedule == GEODESIC_SCHEDULE_TIMESTEP:
        if timesteps is None or numTimesteps is None:
            return torch.tensor(baseWeight, device=device)
        denom = max(numTimesteps - 1, MIN_DIFFUSION_STEPS)
        weights = 1.0 - (timesteps.float() / float(denom))
        return weights.mean() * baseWeight
    raise ValueError(f"Unknown geodesic schedule: {schedule}")


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


def _rotationToMatrix(
    rotation: torch.Tensor,
    rotationRepr: str,
) -> torch.Tensor:
    """
    Convert rotation representation to rotation matrix.

    Parameters
    ----------
    rotation : torch.Tensor
        Rotations shaped (..., C).
    rotationRepr : str
        Rotation representation.

    Returns
    -------
    torch.Tensor
        Rotation matrix shaped (..., 3, 3).
    """
    if rotationRepr == ROTATION_REPR_ROT6D:
        return _sixdToRotationMatrix(rotation)
    if rotationRepr == ROTATION_REPR_AXIS_ANGLE:
        return Rotation(
            rotation,
            kind=ROTATION_KIND_AXIS_ANGLE,
        ).matrix
    raise ValueError(f"Unknown rotation repr: {rotationRepr}")
