"""Autoregressive sequence builder for the Goal C controller (C1).

Turns a ground-truth motion clip (lean representation) into the windows,
targets and control signal the :class:`MotionController` trains on
(ROADMAP_DETERMINIST C1: "builder de séquences autorégressives — fenêtres,
cibles Δstate dérivées de la GT, signal de contrôle dérivé de la GT").

This module lives in the ``data`` layer: ``training`` (the only junction
module) imports it; ``model`` never does.  It produces tensors in **raw**
(un-normalized) space — z-normalization is applied by the training loop
so the same builder serves train, rollout and export.

Conventions
-----------
* Frames are the leading axis.  A window of length ``K = contextFrames``
  ending at frame ``t`` predicts the transition ``Δ = state[t+1] -
  state[t]`` and the absolute next rotation ``state[t+1]`` (geodesic
  target).
* The control signal is *derived from the ground truth* so a rollout
  driven by it must reproduce the clip (the C1 smoke test).  For C1 the
  control is the desired **planar root velocity** (world-frame x/z
  displacement per frame); the aim direction is added in C2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ainimator.core.constants.controller import (
    CONTROL_PLANAR_VELOCITY_CHANNELS,
)

# Ground-plane axes of the SMPL Y-up convention (x, z).
_GROUND_PLANE_AXES = (0, 2)


@dataclass(frozen=True)
class ControllerSequenceConfig:
    """Knobs for :func:`buildControllerSequences`.

    Attributes
    ----------
    contextFrames : int
        Window length ``K`` (frames seen per forward).
    useAimDirection : bool
        Append the aim-direction control channels (C2 rich control).
    """

    contextFrames: int = 1
    useAimDirection: bool = False

    def __post_init__(self) -> None:
        if self.contextFrames < 1:
            raise ValueError("contextFrames must be >= 1.")


@dataclass(frozen=True)
class ControllerSequenceBatch:
    """Flat batch of autoregressive transitions from one clip.

    All tensors share a leading axis ``N = F - K`` (the number of
    predictable transitions).

    Attributes
    ----------
    boneWindow : torch.Tensor
        ``(N, K, numBones, 6)`` raw rotation6d windows.
    globalWindow : torch.Tensor
        ``(N, K, 3)`` raw root_translation windows.
    targetBoneNext : torch.Tensor
        ``(N, numBones, 6)`` absolute next-frame rotation (geodesic
        target).
    targetBoneDelta : torch.Tensor
        ``(N, numBones, 6)`` next-frame rotation delta.
    targetGlobalDelta : torch.Tensor
        ``(N, 3)`` next-frame root_translation delta.
    control : torch.Tensor
        ``(N, controlChannels)`` GT-derived control signal.
    """

    boneWindow: torch.Tensor
    globalWindow: torch.Tensor
    targetBoneNext: torch.Tensor
    targetBoneDelta: torch.Tensor
    targetGlobalDelta: torch.Tensor
    control: torch.Tensor

    @property
    def numTransitions(self) -> int:
        """Number of predictable transitions in this batch."""
        return int(self.boneWindow.shape[0])


def buildControllerSequences(
    rotation6d: torch.Tensor,
    rootTranslation: torch.Tensor,
    config: ControllerSequenceConfig,
) -> ControllerSequenceBatch:
    """Build autoregressive windows + targets + control from one clip.

    Parameters
    ----------
    rotation6d : torch.Tensor
        Ground-truth rotations, shape ``(F, numBones, 6)``.
    rootTranslation : torch.Tensor
        Ground-truth root translation, shape ``(F, 3)``.
    config : ControllerSequenceConfig
        Window / control configuration.

    Returns
    -------
    ControllerSequenceBatch

    Raises
    ------
    ValueError
        If shapes are inconsistent or the clip is too short for one
        transition (``F <= contextFrames``).
    """
    _validateClip(rotation6d, rootTranslation, config)
    frames = rotation6d.shape[0]
    window = config.contextFrames
    numTransitions = frames - window

    starts = torch.arange(numTransitions, device=rotation6d.device)
    windowIndex = starts[:, None] + torch.arange(
        window, device=rotation6d.device
    )[None, :]
    lastFrame = starts + window - 1
    nextFrame = starts + window

    boneWindow = rotation6d[windowIndex]
    globalWindow = rootTranslation[windowIndex]
    targetBoneNext = rotation6d[nextFrame]
    targetBoneDelta = targetBoneNext - rotation6d[lastFrame]
    targetGlobalDelta = rootTranslation[nextFrame] - rootTranslation[lastFrame]

    control = _deriveControl(targetGlobalDelta, config)
    return ControllerSequenceBatch(
        boneWindow=boneWindow,
        globalWindow=globalWindow,
        targetBoneNext=targetBoneNext,
        targetBoneDelta=targetBoneDelta,
        targetGlobalDelta=targetGlobalDelta,
        control=control,
    )


def _deriveControl(
    globalDelta: torch.Tensor,
    config: ControllerSequenceConfig,
) -> torch.Tensor:
    """Derive the GT control signal from the root translation delta."""
    planarVelocity = globalDelta[:, _GROUND_PLANE_AXES]
    if not config.useAimDirection:
        return planarVelocity
    aim = _aimDirection(planarVelocity)
    return torch.cat([planarVelocity, aim], dim=-1)


def _aimDirection(planarVelocity: torch.Tensor) -> torch.Tensor:
    """Unit heading from planar velocity (C2 rich control).

    Falls back to ``(1, 0)`` for near-stationary frames so the unit
    vector is always well defined (no NaN from normalizing a zero).
    """
    norm = planarVelocity.norm(dim=-1, keepdim=True)
    safe = norm.clamp(min=1e-6)
    unit = planarVelocity / safe
    forward = torch.zeros_like(unit)
    forward[..., 0] = 1.0
    isMoving = (norm > 1e-6).to(unit.dtype)
    return isMoving * unit + (1.0 - isMoving) * forward


def _validateClip(
    rotation6d: torch.Tensor,
    rootTranslation: torch.Tensor,
    config: ControllerSequenceConfig,
) -> None:
    """Validate clip ranks, lengths and minimum duration."""
    if rotation6d.ndim != 3 or rotation6d.shape[-1] != 6:
        raise ValueError(
            "rotation6d must be (F, numBones, 6); got "
            f"{tuple(rotation6d.shape)}."
        )
    if rootTranslation.ndim != 2 or rootTranslation.shape[-1] != 3:
        raise ValueError(
            "rootTranslation must be (F, 3); got "
            f"{tuple(rootTranslation.shape)}."
        )
    if rotation6d.shape[0] != rootTranslation.shape[0]:
        raise ValueError(
            "rotation6d and rootTranslation must share frame count; got "
            f"{rotation6d.shape[0]} vs {rootTranslation.shape[0]}."
        )
    if rotation6d.shape[0] <= config.contextFrames:
        raise ValueError(
            f"clip too short: {rotation6d.shape[0]} frames for "
            f"contextFrames={config.contextFrames} (need > K)."
        )
    expectedPlanar = CONTROL_PLANAR_VELOCITY_CHANNELS
    if expectedPlanar != len(_GROUND_PLANE_AXES):
        raise ValueError(
            "planar control channels do not match ground-plane axes."
        )
