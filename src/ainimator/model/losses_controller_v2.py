"""Loss functions for the Goal A deterministic controller (phase A1).

The controller regresses ``Δstate`` per frame.  Its training objective
(ROADMAP_DETERMINIST A1) combines:

1. :func:`velocityDeltaLoss` — L2 on the regressed deltas ("velocities")
   in normalized space.  This is the primary signal: it directly
   supervises the per-frame motion the controller emits.
2. :func:`geodesicRotationLoss` — geodesic distance on SO(3) between the
   reconstructed next-frame rotation and the ground truth.  Unlike a raw
   6D MSE, the geodesic respects the rotation manifold, so small 6D
   errors near the root cannot masquerade as large/small depending on
   parameterisation.
3. :func:`footContactLossController` — FK foot-contact supervision
   (anti-skating).  Wired in phase A2; exposed here so the combiner is
   complete.

The geodesic operates on **absolute** reconstructed rotations (the
training loop denormalizes ``state + Δ`` before calling it); the velocity
loss operates on the **normalized deltas** the model emits directly.
This keeps every function pure and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ainimator.geometry.components.ops import (
    maskedMean,
    rot6dToJointXYZ,
    sixdToRotationMatrix,
)

# Clamp bound for ``acos`` to keep the geodesic gradient finite at the
# ±1 singularities of ``cos θ``.
_ACOS_EPS = 1e-6

# Default vertical axis index for foot-contact velocity (SMPL is Y-up).
_GROUND_PLANE_AXES = (0, 2)


# ---------------------------------------------------------------------
# Core losses
# ---------------------------------------------------------------------
def velocityDeltaLoss(
    predictedDelta: torch.Tensor,
    targetDelta: torch.Tensor,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L2 loss on regressed deltas ("velocities"), normalized space.

    Parameters
    ----------
    predictedDelta, targetDelta : torch.Tensor
        Matching-shape delta tensors (bone or global), e.g.
        ``(B, numBones, 6)`` or ``(B, 3)`` (single-frame) or with an
        extra window axis.
    motionMask : torch.Tensor or None
        Optional validity mask broadcast over the leading axes.

    Returns
    -------
    torch.Tensor
        Scalar masked mean of the squared error.
    """
    if predictedDelta.shape != targetDelta.shape:
        raise ValueError(
            "predicted and target delta shapes must match; got "
            f"{tuple(predictedDelta.shape)} vs {tuple(targetDelta.shape)}."
        )
    squaredError = (predictedDelta - targetDelta) ** 2
    return maskedMean(squaredError, motionMask)


def geodesicRotationLoss(
    predictedRotation6d: torch.Tensor,
    targetRotation6d: torch.Tensor,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Geodesic SO(3) distance between reconstructed absolute rotations.

    Parameters
    ----------
    predictedRotation6d, targetRotation6d : torch.Tensor
        Absolute rotations, shape ``(..., numBones, 6)``.
    motionMask : torch.Tensor or None
        Optional validity mask broadcast over the leading axes.

    Returns
    -------
    torch.Tensor
        Scalar masked mean of the per-bone geodesic angle (radians).
    """
    if predictedRotation6d.shape != targetRotation6d.shape:
        raise ValueError(
            "predicted and target rotation6d shapes must match; got "
            f"{tuple(predictedRotation6d.shape)} vs "
            f"{tuple(targetRotation6d.shape)}."
        )
    predictedMatrix = sixdToRotationMatrix(predictedRotation6d)
    targetMatrix = sixdToRotationMatrix(targetRotation6d)
    relative = torch.matmul(
        predictedMatrix.transpose(-1, -2), targetMatrix
    )
    trace = (
        relative[..., 0, 0]
        + relative[..., 1, 1]
        + relative[..., 2, 2]
    )
    cosAngle = ((trace - 1.0) / 2.0).clamp(
        min=-1.0 + _ACOS_EPS, max=1.0 - _ACOS_EPS
    )
    angle = torch.acos(cosAngle)
    return maskedMean(angle, motionMask)


def footContactStepLoss(
    predictedNextRotation6d: torch.Tensor,
    lastRotation6d: torch.Tensor,
    predictedGlobalDelta: torch.Tensor,
    contactTarget: torch.Tensor,
) -> torch.Tensor:
    """Per-step anti-skating loss (phase A2).

    For each predicted transition, penalises the planar velocity of a
    foot that the ground truth marks as in contact at the target frame.
    The FK joint velocity (world-frame, from rotations) is combined with
    the predicted root planar motion to capture foot sliding even as the
    pelvis advances — the failure mode a rotation-only loss misses.

    Parameters
    ----------
    predictedNextRotation6d : torch.Tensor
        Reconstructed absolute next-frame rotations, ``(N, numBones, 6)``.
    lastRotation6d : torch.Tensor
        Last-window absolute rotations, ``(N, numBones, 6)``.
    predictedGlobalDelta : torch.Tensor
        Predicted root-local motion delta, ``(N, 4)`` —
        ``(Δforward, Δlateral, Δheight, Δyaw)``.  Only the first two
        channels (ground-plane displacement magnitude) are used, expressed
        in the root-local frame as a proxy for the world-frame root
        velocity.  The magnitude is frame-invariant; the direction
        approximation is acceptable because LOT-1 targets straight-ahead
        motion and the loss is scale-dominated.
    contactTarget : torch.Tensor
        Foot-contact mask at the target frame, ``(N, numFeet)`` ordered
        ``(leftFoot, rightFoot)``.

    Returns
    -------
    torch.Tensor
        Scalar mean of the contact-weighted squared foot velocity.
    """
    footIndices = _footJointIndices()
    nextXyz = rot6dToJointXYZ(predictedNextRotation6d.unsqueeze(1))
    lastXyz = rot6dToJointXYZ(lastRotation6d.unsqueeze(1))
    footNext = nextXyz.squeeze(1)[:, footIndices, :]
    footLast = lastXyz.squeeze(1)[:, footIndices, :]
    localVelocity = (footNext - footLast)[..., _GROUND_PLANE_AXES]
    # Use (Δforward, Δlateral) channels — ground-plane displacement in the
    # root-local frame.  The magnitude equals the world-frame magnitude;
    # only the direction differs (acceptable proxy for the anti-skating
    # penalty).
    rootPlanar = predictedGlobalDelta[:, :2].unsqueeze(1)
    worldVelocity = localVelocity + rootPlanar
    weighted = (worldVelocity ** 2) * contactTarget.unsqueeze(-1)
    return weighted.mean()


def _footJointIndices() -> list[int]:
    """SMPL-22 indices of the foot/toe joints used for contact."""
    # leftFoot=10, rightFoot=11 in the canonical SMPL22_BONE_ORDER.
    return [10, 11]


# ---------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ControllerLossResult:
    """Total controller loss and its decomposed components.

    Attributes
    ----------
    total : torch.Tensor
        Weighted sum used for backprop.
    components : dict[str, torch.Tensor]
        Per-term scalar losses (unweighted), for ``loss_share`` health.
    """

    total: torch.Tensor
    components: dict[str, torch.Tensor]


@dataclass(frozen=True)
class ControllerLossWeights:
    """Weights for the controller loss terms.

    Attributes
    ----------
    velocity : float
        Weight of the delta (velocity) L2 term.
    geodesic : float
        Weight of the geodesic rotation term.
    footContact : float
        Weight of the foot-contact term (0 until A2).
    """

    velocity: float = 1.0
    geodesic: float = 1.0
    footContact: float = 0.0


def combinedControllerLoss(
    velocity: torch.Tensor,
    geodesic: torch.Tensor,
    weights: ControllerLossWeights,
    footContact: torch.Tensor | None = None,
) -> ControllerLossResult:
    """Combine the controller loss terms into a single objective.

    Parameters
    ----------
    velocity, geodesic : torch.Tensor
        Scalar losses from :func:`velocityDeltaLoss` /
        :func:`geodesicRotationLoss`.
    weights : ControllerLossWeights
        Term weights.
    footContact : torch.Tensor or None
        Optional foot-contact scalar (phase A2).

    Returns
    -------
    ControllerLossResult
    """
    total = weights.velocity * velocity + weights.geodesic * geodesic
    components: dict[str, torch.Tensor] = {
        "loss_velocity": velocity,
        "loss_geodesic": geodesic,
    }
    if footContact is not None and weights.footContact > 0.0:
        total = total + weights.footContact * footContact
        components["loss_foot_contact"] = footContact
    return ControllerLossResult(total=total, components=components)
