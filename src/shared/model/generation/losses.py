"""Loss functions for motion generation training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def diffusionLoss(
    predictedNoise: torch.Tensor,
    targetNoise: torch.Tensor,
) -> torch.Tensor:
    """
    Standard MSE loss for noise prediction in diffusion models.

    Parameters
    ----------
    predictedNoise : torch.Tensor
        Predicted noise from the denoiser.
    targetNoise : torch.Tensor
        Ground truth noise that was added.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss.
    """
    return F.mse_loss(predictedNoise, targetNoise)


def geodesicLoss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Geodesic loss for 6D rotation representations.

    Computes loss based on the rotation matrix difference.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted 6D rotations shaped (..., 6).
    target : torch.Tensor
        Target 6D rotations shaped (..., 6).
    eps : float, optional
        Epsilon for numerical stability, by default 1e-7.

    Returns
    -------
    torch.Tensor
        Scalar geodesic loss.
    """
    # Convert 6D to rotation matrices
    predMat = _sixdToRotationMatrix(predicted)
    targetMat = _sixdToRotationMatrix(target)

    # Compute R_pred^T @ R_target
    diff = torch.matmul(predMat.transpose(-2, -1), targetMat)

    # Trace of rotation matrix
    trace = diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2]

    # Clamp to valid range for arccos
    trace = torch.clamp(trace, -1.0 + eps, 3.0 - eps)

    # Geodesic distance: arccos((trace - 1) / 2)
    angle = torch.acos((trace - 1.0) / 2.0)

    return angle.mean()


def velocityLoss(
    motion: torch.Tensor,
    weight: float = 1.0,
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

    Returns
    -------
    torch.Tensor
        Scalar velocity loss.
    """
    if motion.shape[1] < 2:
        return torch.tensor(0.0, device=motion.device)

    velocity = motion[:, 1:] - motion[:, :-1]
    return weight * (velocity ** 2).mean()


def accelerationLoss(
    motion: torch.Tensor,
    weight: float = 1.0,
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

    Returns
    -------
    torch.Tensor
        Scalar acceleration loss.
    """
    if motion.shape[1] < 3:
        return torch.tensor(0.0, device=motion.device)

    velocity = motion[:, 1:] - motion[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    return weight * (acceleration ** 2).mean()


def combinedGenerationLoss(
    predictedNoise: torch.Tensor,
    targetNoise: torch.Tensor,
    predictedMotion: torch.Tensor,
    targetMotion: torch.Tensor,
    diffusionWeight: float = 1.0,
    geodesicWeight: float = 0.1,
    velocityWeight: float = 0.01,
    accelerationWeight: float = 0.001,
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

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        Total loss and dictionary of individual loss components.
    """
    lossDiff = diffusionLoss(predictedNoise, targetNoise)
    lossGeo = geodesicLoss(predictedMotion, targetMotion)
    lossVel = velocityLoss(predictedMotion, velocityWeight)
    lossAcc = accelerationLoss(predictedMotion, accelerationWeight)

    total = (
        diffusionWeight * lossDiff
        + geodesicWeight * lossGeo
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
