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
DEFAULT_VEL_XYZ_SCHEDULE = XYZ_SCHEDULE_NONE
MIN_DIFFUSION_STEPS = 1
# Default Min-SNR gamma from Hang et al. 2023 ("Efficient Diffusion Training
# via Min-SNR Weighting Strategy").  gamma=5 gives high-t samples a weight
# of ~5*SNR(t) instead of 1.0, so they no longer dominate the gradient and
# pull the x0 predictor toward a mean-pose output.  Set to 0 to disable.
DEFAULT_MIN_SNR_GAMMA = 5.0


def minSnrLossWeights(
    timesteps: torch.Tensor,
    alphasCumprod: torch.Tensor,
    gamma: float = DEFAULT_MIN_SNR_GAMMA,
) -> torch.Tensor:
    """Compute per-sample Min-SNR loss weights for x0-prediction training.

    For x0 parametrisation the appropriate weight is ``min(SNR(t), gamma) /
    SNR(t)``.  At low t (high SNR) the weight is ~gamma/SNR (down-weighted)
    and at high t (low SNR) the weight saturates at 1.0 -- the opposite of
    vanilla MSE, which over-weights the easy-to-reconstruct low-t samples
    and at the same time lets the impossible-to-recover high-t samples
    collapse the model toward the dataset mean.

    Parameters
    ----------
    timesteps : torch.Tensor
        Diffusion timesteps shaped (batch,).
    alphasCumprod : torch.Tensor
        Cumulative product of alphas from the DDIM scheduler.
    gamma : float, optional
        SNR clipping threshold, by default :data:`DEFAULT_MIN_SNR_GAMMA`.

    Returns
    -------
    torch.Tensor
        Per-sample loss weights shaped (batch,).
    """
    if gamma <= 0.0:
        return torch.ones_like(timesteps, dtype=torch.float32)

    alphasCumprod = alphasCumprod.to(device=timesteps.device)
    alphaT = alphasCumprod.gather(0, timesteps.long())
    snr = alphaT / torch.clamp(1.0 - alphaT, min=1e-8)
    gammaTensor = torch.full_like(snr, float(gamma))
    return torch.minimum(snr, gammaTensor) / torch.clamp(snr, min=1e-8)


def _perSampleMse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-squared error reduced to one scalar per batch item.

    Leaves a shape-(batch,) tensor that can be multiplied by Min-SNR
    weights before the final batch average.
    """
    squaredError = (predicted - target) ** 2
    if motionMask is not None:
        # motionMask shape: (batch, frames) -> broadcast over trailing dims.
        expandMask = motionMask
        while expandMask.dim() < squaredError.dim():
            expandMask = expandMask.unsqueeze(-1)
        expandMask = expandMask.float()
        numerator = (squaredError * expandMask).flatten(1).sum(dim=1)
        denom = expandMask.expand_as(squaredError).flatten(1).sum(dim=1)
        return numerator / torch.clamp(denom, min=1.0)
    return squaredError.flatten(1).mean(dim=1)


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
    perSampleWeights: torch.Tensor | None = None,
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
    perSampleWeights : torch.Tensor | None, optional
        Shape-(batch,) per-sample multiplier applied before averaging --
        used by Min-SNR weighting to de-emphasise high-t samples that
        otherwise collapse x0 predictions to a mean pose.

    Returns
    -------
    torch.Tensor
        Scalar masked MSE in raw motion space.
    """
    if perSampleWeights is None:
        squaredError = (predictedMotion - targetMotion) ** 2
        return maskedMean(squaredError, motionMask)

    perSampleLoss = _perSampleMse(predictedMotion, targetMotion, motionMask)
    weights = perSampleWeights.to(perSampleLoss.dtype).to(perSampleLoss.device)
    return (perSampleLoss * weights).mean()


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
    velXyzWeightSchedule: str = DEFAULT_VEL_XYZ_SCHEDULE,
    timesteps: torch.Tensor | None = None,
    numTimesteps: int | None = None,
    motionMask: torch.Tensor | None = None,
    footContact: torch.Tensor | None = None,
    footSkatingWeight: float = DEFAULT_FOOT_SKATING_WEIGHT,
    perSampleWeights: torch.Tensor | None = None,
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
    velXyzWeightSchedule : str, optional
        Schedule mode for velocity XYZ weight, by default "none".
        Using "timestep" gates the velocity loss to low-t where x0
        predictions are reliable — critical to avoid noisy FK-of-noise
        gradients at high diffusion timesteps pulling the model toward a
        temporally flat mean pose.
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
        perSampleWeights=perSampleWeights,
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
    # Apply timestep schedule to velocity XYZ weight.  When velXyzWeightSchedule
    # is "timestep" the weight scales as (1 - t/T), decaying to ~0 at high
    # diffusion timesteps.  This prevents FK-of-noise from generating chaotic
    # velocity gradients at high-t that would otherwise compete with the
    # diffusion MSE and push the denoiser toward a temporally flat mean pose.
    resolvedVelXyzWeightTensor = _resolveXyzWeight(
        resolvedVelocityXyzWeight,
        velXyzWeightSchedule,
        timesteps,
        numTimesteps,
        predictedMotion.device,
    )
    # Compute base velocity loss (unit-weighted) then scale by resolved tensor.
    lossVelBase = velocityXyzLoss(
        predictedMotion,
        targetMotion,
        weight=1.0,
        motionMask=motionMask,
    )
    lossVel = resolvedVelXyzWeightTensor * lossVelBase
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
