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

import math
from dataclasses import dataclass

import torch

from ainimator.core.constants.controller import (
    CONTROL_PLANAR_VELOCITY_CHANNELS,
)
from ainimator.geometry.components.ops import rot6dToJointXYZ

# Ground-plane axes of the SMPL Y-up convention (x, z).
_GROUND_PLANE_AXES = (0, 2)
# Vertical axis of the SMPL Y-up convention.
_VERTICAL_AXIS = 1
# SMPL-22 foot joint indices (leftFoot, rightFoot).
_FOOT_JOINT_INDICES = (10, 11)
# Default contact thresholds: a foot is "in contact" when it sits below
# ``heightThreshold`` (meters) AND moves slower than ``speedThreshold``
# (meters/frame) on the ground plane.  Conservative defaults; tuned per
# dataset during C2 validation.
_DEFAULT_CONTACT_HEIGHT = 0.05
_DEFAULT_CONTACT_SPEED = 0.01
# Half a gait cycle advances the phase by π (one foot strike).
_HALF_CYCLE = math.pi


@dataclass(frozen=True)
class ControllerSequenceConfig:
    """Knobs for :func:`buildControllerSequences`.

    Attributes
    ----------
    contextFrames : int
        Window length ``K`` (frames seen per forward).
    useAimDirection : bool
        Append the aim-direction control channels (C2 rich control).
    emitPhase : bool
        Derive a foot-contact gait phase and emit it per transition (C2).
    emitContacts : bool
        Derive foot-contact labels and emit them per transition (C2,
        consumed by the anti-skating loss).
    contactHeight : float
        Vertical threshold (m) below which a foot may be in contact.
    contactSpeed : float
        Planar speed threshold (m/frame) below which a foot may be in
        contact.
    """

    contextFrames: int = 1
    useAimDirection: bool = False
    emitPhase: bool = False
    emitContacts: bool = False
    contactHeight: float = _DEFAULT_CONTACT_HEIGHT
    contactSpeed: float = _DEFAULT_CONTACT_SPEED

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
    phase : torch.Tensor or None
        ``(N, 2)`` gait phase ``(cos φ, sin φ)`` at the target frame
        (C2), or ``None`` when phase emission is off.
    contactTarget : torch.Tensor or None
        ``(N, 2)`` foot-contact labels ``(left, right)`` at the target
        frame (C2), or ``None`` when contact emission is off.
    """

    boneWindow: torch.Tensor
    globalWindow: torch.Tensor
    targetBoneNext: torch.Tensor
    targetBoneDelta: torch.Tensor
    targetGlobalDelta: torch.Tensor
    control: torch.Tensor
    phase: torch.Tensor | None = None
    contactTarget: torch.Tensor | None = None

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
    phase, contactTarget = _derivePhaseAndContacts(
        rotation6d, rootTranslation, nextFrame, config
    )
    return ControllerSequenceBatch(
        boneWindow=boneWindow,
        globalWindow=globalWindow,
        targetBoneNext=targetBoneNext,
        targetBoneDelta=targetBoneDelta,
        targetGlobalDelta=targetGlobalDelta,
        control=control,
        phase=phase,
        contactTarget=contactTarget,
    )


def _derivePhaseAndContacts(
    rotation6d: torch.Tensor,
    rootTranslation: torch.Tensor,
    nextFrame: torch.Tensor,
    config: ControllerSequenceConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Derive per-transition gait phase and contact labels (C2)."""
    if not (config.emitPhase or config.emitContacts):
        return None, None
    contacts = deriveFootContacts(
        rotation6d, rootTranslation, config.contactHeight, config.contactSpeed
    )
    phase: torch.Tensor | None = None
    if config.emitPhase:
        phase = deriveGaitPhase(contacts)[nextFrame]
    contactTarget = contacts[nextFrame] if config.emitContacts else None
    return phase, contactTarget


def deriveFootContacts(
    rotation6d: torch.Tensor,
    rootTranslation: torch.Tensor,
    heightThreshold: float = _DEFAULT_CONTACT_HEIGHT,
    speedThreshold: float = _DEFAULT_CONTACT_SPEED,
) -> torch.Tensor:
    """Derive per-frame foot-contact labels from the ground truth.

    A foot is "in contact" when its world height sits below
    ``heightThreshold`` and its planar speed is below ``speedThreshold``.

    Parameters
    ----------
    rotation6d : torch.Tensor
        ``(F, numBones, 6)`` ground-truth rotations.
    rootTranslation : torch.Tensor
        ``(F, 3)`` ground-truth root translation.
    heightThreshold, speedThreshold : float
        Contact thresholds (meters / meters-per-frame).

    Returns
    -------
    torch.Tensor
        ``(F, 2)`` float contact labels ``(leftFoot, rightFoot)``.
    """
    jointXyz = rot6dToJointXYZ(rotation6d.unsqueeze(0)).squeeze(0)
    footXyz = jointXyz[:, _FOOT_JOINT_INDICES, :]
    worldFoot = footXyz + rootTranslation.unsqueeze(1)
    height = worldFoot[..., _VERTICAL_AXIS]
    planar = worldFoot[..., _GROUND_PLANE_AXES]
    velocity = torch.zeros_like(planar)
    velocity[1:] = planar[1:] - planar[:-1]
    speed = velocity.norm(dim=-1)
    contact = (height < heightThreshold) & (speed < speedThreshold)
    return contact.to(rotation6d.dtype)


def deriveGaitPhase(contacts: torch.Tensor) -> torch.Tensor:
    """Derive a continuous gait phase from foot-contact onsets (PFNN).

    Each new foot strike (a 0→1 contact transition on either foot)
    advances the phase by π; the phase is linearly interpolated between
    consecutive strikes.  Frames before the first strike (or clips with
    fewer than two strikes) hold phase 0.

    Parameters
    ----------
    contacts : torch.Tensor
        ``(F, 2)`` foot-contact labels.

    Returns
    -------
    torch.Tensor
        ``(F, 2)`` phase encoded as ``(cos φ, sin φ)``.
    """
    onsets = _contactOnsets(contacts)
    device = contacts.device
    phase = torch.zeros(
        contacts.shape[0], dtype=contacts.dtype, device=device
    )
    for index in range(len(onsets) - 1):
        start, end = onsets[index], onsets[index + 1]
        ramp = torch.linspace(
            index * _HALF_CYCLE,
            (index + 1) * _HALF_CYCLE,
            end - start + 1,
            device=device,
            dtype=contacts.dtype,
        )
        phase[start:end] = ramp[:-1]
    if onsets:
        phase[onsets[-1]:] = float((len(onsets) - 1) * _HALF_CYCLE)
    return torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)


def _contactOnsets(contacts: torch.Tensor) -> list[int]:
    """Return frame indices where any foot transitions to contact."""
    rising = (contacts[1:] > 0.5) & (contacts[:-1] <= 0.5)
    onsetFrames = torch.nonzero(rising.any(dim=-1), as_tuple=False) + 1
    return [int(frame) for frame in onsetFrames.flatten().tolist()]


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
