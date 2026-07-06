"""Post-processing helpers for v2 sampling output.

Phase 4 of the v2 quality plan — small filters applied **after** DDIM
sampling and **before** Collada export.  They never feed back into the
denoiser; their only job is to clean up visible artefacts the model
keeps producing despite the FK losses:

* :func:`smoothRootYaw` — Gaussian filter on the per-frame yaw of the
  pelvis (extracted from ``rotation6d``) to remove the spinning /
  jittering heading that small rotation errors compound into.
* :func:`dampFootMotionDuringContact` — given the FK-derived foot
  positions, identify frames where the foot ought to be still (low
  velocity) and pull subsequent foot positions back toward the contact
  frame, killing foot-skating without a full IK solve.

Both functions operate on a single sample (no batch dim) — the call
sites (CLI generation, test fixtures) hand us ``rotation6d`` of shape
``(F, 22, 6)`` and ``root_translation`` of shape ``(F, 3)``.

The helpers are deliberately conservative: by default they do nothing
(``sigma=0`` / ``threshold=0`` short-circuits).  The CLI flips the
flag on demand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ainimator.geometry.components.ops import (
    rot6dToJointXYZ,
    sixdToRotationMatrix,
)


# Foot bone indices (matches losses_v2.FOOT_CONTACT_BONE_INDICES).
_FOOT_BONE_INDICES = (7, 10, 8, 11)


# ---------------------------------------------------------------------------
# Phase 4.2 — root yaw temporal smoothing
# ---------------------------------------------------------------------------
def _rotationMatrixToYaw(rotationMatrix: torch.Tensor) -> torch.Tensor:
    """Extract the Y-axis (yaw) rotation from a (F, 3, 3) rotation tensor."""
    # atan2(R[0,2], R[2,2]) gives the rotation around Y for the
    # standard right-handed convention used elsewhere in the codebase.
    return torch.atan2(
        rotationMatrix[..., 0, 2], rotationMatrix[..., 2, 2]
    )


def _yawToRotationMatrix(yaw: torch.Tensor) -> torch.Tensor:
    """Build a (F, 3, 3) Y-axis rotation tensor from a (F,) yaw tensor."""
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    zero = torch.zeros_like(yaw)
    one = torch.ones_like(yaw)
    # Row-major build.
    row0 = torch.stack([cos, zero, sin], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-sin, zero, cos], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _gaussianKernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    """Build a normalised 1-D Gaussian kernel of radius ``ceil(3*sigma)``."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(
        -radius, radius + 1, dtype=torch.float32, device=device
    )
    kernel = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _circularGaussianSmooth(
    angles: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Smooth a (F,) angle signal on the circle.

    Filters cos / sin separately then re-extracts the angle so the
    [-pi, pi] discontinuity does not pollute the output.
    """
    if sigma <= 0.0 or angles.numel() < 3:
        return angles
    kernel = _gaussianKernel1d(sigma, angles.device)
    pad = kernel.numel() // 2
    cos = torch.cos(angles).view(1, 1, -1)
    sin = torch.sin(angles).view(1, 1, -1)
    cosPadded = torch.nn.functional.pad(cos, (pad, pad), mode="reflect")
    sinPadded = torch.nn.functional.pad(sin, (pad, pad), mode="reflect")
    kernel = kernel.view(1, 1, -1)
    cosSmoothed = torch.nn.functional.conv1d(cosPadded, kernel).squeeze()
    sinSmoothed = torch.nn.functional.conv1d(sinPadded, kernel).squeeze()
    return torch.atan2(sinSmoothed, cosSmoothed)


def smoothRootYaw(
    rotation6d: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Gaussian-smooth the pelvis yaw across time.

    Parameters
    ----------
    rotation6d : torch.Tensor
        ``(F, 22, 6)`` rotation6D tensor for a single sample.
    sigma : float
        Standard deviation of the Gaussian kernel, in frames.  ``0.0``
        disables the filter (returns the input unchanged).

    Returns
    -------
    torch.Tensor
        New ``(F, 22, 6)`` tensor with the pelvis yaw replaced by its
        smoothed version.  Pitch and roll of the pelvis are preserved
        by re-composing ``R_smooth = R_y(yaw_smooth) @ R_y(-yaw_raw) @
        R_pelvis``.  Other bones are returned untouched.
    """
    if sigma <= 0.0:
        return rotation6d
    if rotation6d.ndim != 3 or rotation6d.shape[-1] != 6:
        raise ValueError(
            "rotation6d must be (F, 22, 6); got "
            f"{tuple(rotation6d.shape)}."
        )

    pelvis6d = rotation6d[:, 0]  # (F, 6)
    pelvisRotation = sixdToRotationMatrix(pelvis6d)  # (F, 3, 3)
    yawRaw = _rotationMatrixToYaw(pelvisRotation)
    yawSmooth = _circularGaussianSmooth(yawRaw, sigma)
    yawDelta = yawSmooth - yawRaw
    deltaMatrix = _yawToRotationMatrix(yawDelta)
    pelvisSmoothed = deltaMatrix @ pelvisRotation  # (F, 3, 3)

    # Repack to 6D: take the first two rows (Zhou et al. continuity).
    pelvis6dSmooth = pelvisSmoothed[..., :2, :].reshape(
        pelvisSmoothed.shape[0], 6
    )
    output = rotation6d.clone()
    output[:, 0] = pelvis6dSmooth
    return output


def smoothRotation6dTemporal(
    rotation6d: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Gaussian-smooth every rotation6d channel across time.

    Removes the high-frequency frame-to-frame jitter ("trembling")
    diagnosed on 2026-06-05: generated motion has correct pose amplitude
    (global std matches real ~0.50) but ~30× the real frame-to-frame
    acceleration.  A temporal Gaussian filter (``sigma`` in frames)
    brings the jitter back to real-motion levels — ``sigma≈2`` matches
    the AMASS reference (Δ²/frame ~0.001) while preserving the motion
    (cosine to the raw sequence ~0.998).

    Parameters
    ----------
    rotation6d : torch.Tensor
        ``(F, 22, 6)`` rotation6D tensor for a single sample.
    sigma : float
        Gaussian std in frames.  ``0.0`` disables (returns the input).

    Returns
    -------
    torch.Tensor
        New ``(F, 22, 6)`` tensor with each channel filtered along time.
    """
    if sigma <= 0.0:
        return rotation6d
    if rotation6d.ndim != 3 or rotation6d.shape[-1] != 6:
        raise ValueError(
            "rotation6d must be (F, 22, 6); got "
            f"{tuple(rotation6d.shape)}."
        )
    frames = rotation6d.shape[0]
    if frames < 3:
        return rotation6d
    kernel = _gaussianKernel1d(sigma, rotation6d.device)
    pad = kernel.numel() // 2
    channels = rotation6d.permute(1, 2, 0).reshape(-1, 1, frames)
    padded = torch.nn.functional.pad(channels, (pad, pad), mode="replicate")
    smoothed = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1))
    return smoothed.reshape(
        rotation6d.shape[1], 6, frames
    ).permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# Phase 4.1 — foot stationarity (lightweight IK substitute)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FootDampReport:
    """Diagnostics returned by :func:`dampFootMotionDuringContact`."""

    contactFramesPerFoot: tuple[int, int, int, int]
    rootCorrectionNorm: float


def dampFootMotionDuringContact(
    rotation6d: torch.Tensor,
    rootTranslation: torch.Tensor,
    velocityThreshold: float = 0.04,
    blendAlpha: float = 0.6,
) -> tuple[torch.Tensor, FootDampReport]:
    """Pull the root translation back to suppress visible foot skating.

    We can't run a real per-bone IK here without a full SMPL skeleton
    in the export pipeline, so the heuristic is much simpler:

    1. FK the bones to recover world-space joint positions.
    2. For each foot frame, compute its per-frame velocity (Euclidean).
    3. When a foot is "in contact" (velocity < threshold), measure the
       residual displacement between that foot's current XYZ and the
       previous frame's XYZ — and **subtract** that residual from the
       root translation, blended by ``blendAlpha``.

    This counter-translates the whole body to cancel the unwanted slip
    of the contact foot, which is exactly the visual artefact users
    notice as "foot skating".  No bone rotations are modified.

    Parameters
    ----------
    rotation6d : torch.Tensor
        ``(F, 22, 6)`` for a single sample.
    rootTranslation : torch.Tensor
        ``(F, 3)`` for the same sample.  Modified in a copy and the
        new tensor is returned (input is left untouched).
    velocityThreshold : float
        Foot speed (m/frame) below which the foot is considered to be
        in contact with the ground.  ``0.04`` ≈ 24 cm/s @ 30 fps.
    blendAlpha : float
        ``0.0`` disables the correction; ``1.0`` cancels the slip
        completely on every contact frame.  ``0.6`` is a safe default.

    Returns
    -------
    (torch.Tensor, FootDampReport)
        Corrected root translation and a short diagnostic record.
    """
    if blendAlpha <= 0.0 or velocityThreshold <= 0.0:
        zeroReport = FootDampReport(
            contactFramesPerFoot=(0, 0, 0, 0),
            rootCorrectionNorm=0.0,
        )
        return rootTranslation.clone(), zeroReport
    if rotation6d.ndim != 3 or rotation6d.shape[-1] != 6:
        raise ValueError(
            "rotation6d must be (F, 22, 6); got "
            f"{tuple(rotation6d.shape)}."
        )
    if rootTranslation.ndim != 2 or rootTranslation.shape[-1] != 3:
        raise ValueError(
            "rootTranslation must be (F, 3); got "
            f"{tuple(rootTranslation.shape)}."
        )
    if rotation6d.shape[0] != rootTranslation.shape[0]:
        raise ValueError(
            "rotation6d and rootTranslation must share frame count."
        )

    # FK in the SMPL T-pose — gives world XYZ for every joint.  Root
    # translation is added afterwards in this helper because we are
    # about to modify it.
    joints = rot6dToJointXYZ(
        rotation6d.unsqueeze(0)
    ).squeeze(0)  # (F, 22, 3)
    joints = joints + rootTranslation.unsqueeze(1)  # add root XYZ

    feet = joints[:, list(_FOOT_BONE_INDICES), :]  # (F, 4, 3)
    velocity = feet[1:] - feet[:-1]  # (F-1, 4, 3)
    speed = velocity.norm(dim=-1)  # (F-1, 4)
    contactMask = speed < velocityThreshold  # (F-1, 4)

    correctedRoot = rootTranslation.clone()
    contactCounts = contactMask.sum(dim=0).tolist()
    cumulativeCorrection = torch.zeros_like(rootTranslation[0])
    totalCorrectionNorm = 0.0
    for frameIndex in range(1, rotation6d.shape[0]):
        framePair = frameIndex - 1
        activeFeet = torch.nonzero(
            contactMask[framePair], as_tuple=False
        ).squeeze(-1)
        if activeFeet.numel() == 0:
            correctedRoot[frameIndex] = (
                correctedRoot[frameIndex] + cumulativeCorrection
            )
            continue
        # Average residual slip across all contact feet in this frame.
        slip = velocity[framePair, activeFeet].mean(dim=0)  # (3,)
        # Counter-translate the root by ``-slip`` (blend in).
        increment = -slip * blendAlpha
        cumulativeCorrection = cumulativeCorrection + increment
        totalCorrectionNorm = (
            totalCorrectionNorm + float(increment.norm().item())
        )
        correctedRoot[frameIndex] = (
            correctedRoot[frameIndex] + cumulativeCorrection
        )

    report = FootDampReport(
        contactFramesPerFoot=tuple(int(c) for c in contactCounts),
        rootCorrectionNorm=totalCorrectionNorm,
    )
    return correctedRoot, report
