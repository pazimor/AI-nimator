"""Autoregressive sequence builder for the Goal A controller (A1).

Turns a ground-truth motion clip (lean representation) into the windows,
targets and control signal the :class:`MotionController` trains on
(ROADMAP_DETERMINIST A1: "builder de séquences autorégressives — fenêtres,
cibles Δstate dérivées de la GT, signal de contrôle dérivé de la GT").

This module lives in the ``data`` layer: ``training`` (the only junction
module) imports it; ``model`` never does.  It produces tensors in **raw**
(un-normalized) space — z-normalization is applied by the training loop
so the same builder serves train, rollout and export.

State representation (ROADMAP_DETERMINIST §2.2.a)
-------------------------------------------------
The state is **136 channels** = rotation6d (132) + root-local motion (4).
The root-local motion ``(Δforward, Δlateral, Δheight, Δyaw)`` replaces the
absolute ``root_translation`` (3) of the diffusion lean representation.
Conversion from GT absolute trajectories is handled by
:mod:`ainimator.geometry.root_local`.

Control signal (ROADMAP_DETERMINIST §2.2.b)
-------------------------------------------
``(vx, vz)`` is the desired planar velocity in the **root-local frame**
at each step (not world frame).  ``(aim_x, aim_z)`` is the facing
direction unit vector derived from the **pelvis yaw** — decoupled from
locomotion direction (a character can face east while walking north).

Conventions
-----------
* Frames are the leading axis.  A window of length ``K = contextFrames``
  ending at frame ``t`` predicts the transition ``Δ = state[t+1] -
  state[t]`` and the absolute next rotation ``state[t+1]`` (geodesic
  target).
* The control signal is *derived from the ground truth* so a rollout
  driven by it must reproduce the clip (the A1 smoke test).
* The ``globalWindow`` tensor holds the **root-local motion deltas**
  (not absolute translations) so the model's context window is
  stationary-frame-invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ainimator.core.constants.controller import (
    CONTROL_PLANAR_VELOCITY_CHANNELS,
    ROOT_LOCAL_MOTION_CHANNELS,
)
from ainimator.geometry.components.ops import rot6dToJointXYZ
from ainimator.geometry.root_local import (
    absoluteToRootLocalDeltas,
    aimDirectionFromPelvisYaw,
)

# Ground-plane axes of the SMPL Y-up convention (x, z).
_GROUND_PLANE_AXES = (0, 2)
# Vertical axis of the SMPL Y-up convention.
_VERTICAL_AXIS = 1
# SMPL-22 foot joint indices (leftFoot, rightFoot).
_FOOT_JOINT_INDICES = (10, 11)
# Default contact thresholds: a foot is "in contact" when it sits below
# ``heightThreshold`` (meters) AND moves slower than ``speedThreshold``
# (meters/frame) on the ground plane.  Conservative defaults; tuned per
# dataset during A2 validation.
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
        Append the aim-direction control channels (A2 rich control).
    emitPhase : bool
        Derive a foot-contact gait phase and emit it per transition (A2).
    emitContacts : bool
        Derive foot-contact labels and emit them per transition (A2,
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
        ``(N, K, 4)`` root-local motion delta windows
        ``(Δforward, Δlateral, Δheight, Δyaw)``.  Each frame in the
        window is the local-frame displacement at that frame (computed
        relative to the previous frame's pelvis yaw).
    targetBoneNext : torch.Tensor
        ``(N, numBones, 6)`` absolute next-frame rotation (geodesic
        target).
    targetBoneDelta : torch.Tensor
        ``(N, numBones, 6)`` next-frame rotation delta.
    targetGlobalDelta : torch.Tensor
        ``(N, 4)`` next-frame root-local motion delta
        ``(Δforward, Δlateral, Δheight, Δyaw)``.
    control : torch.Tensor
        ``(N, controlChannels)`` GT-derived control signal.
        Channel layout: ``(vx, vz [, aim_x, aim_z])``.
        ``(vx, vz)`` is in the root-local frame (z-normalized by the
        training loop); ``(aim_x, aim_z)`` is a unit facing vector
        derived from pelvis yaw (excluded from z-norm).
    phase : torch.Tensor or None
        ``(N, 2)`` gait phase ``(cos φ, sin φ)`` at the target frame
        (A2), or ``None`` when phase emission is off.
    contactTarget : torch.Tensor or None
        ``(N, 2)`` foot-contact labels ``(left, right)`` at the target
        frame (A2), or ``None`` when contact emission is off.
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

    Converts the absolute ``rootTranslation`` GT to root-local motion
    deltas (ROADMAP_DETERMINIST §2.2.a) before building windows.

    Parameters
    ----------
    rotation6d : torch.Tensor
        Ground-truth rotations, shape ``(F, numBones, 6)``.
    rootTranslation : torch.Tensor
        Ground-truth root translation, shape ``(F, 3)`` — absolute,
        world-space XYZ (as stored in the AMASS dataset).
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

    # Convert absolute GT trajectory → root-local motion deltas (F, 4).
    rootLocalMotion = absoluteToRootLocalDeltas(rootTranslation, rotation6d)

    starts = torch.arange(numTransitions, device=rotation6d.device)
    windowIndex = starts[:, None] + torch.arange(
        window, device=rotation6d.device
    )[None, :]
    lastFrame = starts + window - 1
    nextFrame = starts + window

    boneWindow = rotation6d[windowIndex]
    # globalWindow holds root-local motion delta history (N, K, 4).
    globalWindow = rootLocalMotion[windowIndex]
    targetBoneNext = rotation6d[nextFrame]
    targetBoneDelta = targetBoneNext - rotation6d[lastFrame]
    # Target global delta: the root-local motion at the *next* frame.
    targetGlobalDelta = rootLocalMotion[nextFrame]

    control = _deriveControl(rotation6d, targetGlobalDelta, nextFrame, config)
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
    """Derive per-transition gait phase and contact labels (A2)."""
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
    rotation6d: torch.Tensor,
    targetGlobalDelta: torch.Tensor,
    nextFrame: torch.Tensor,
    config: ControllerSequenceConfig,
) -> torch.Tensor:
    """Derive the GT control signal.

    ``(vx, vz)`` = the root-local planar velocity (Δforward, Δlateral)
    taken directly from ``targetGlobalDelta[:, :2]`` — already in the
    local frame.

    ``(aim_x, aim_z)`` = unit facing vector derived from the pelvis yaw
    at the target frame, **decoupled** from locomotion direction
    (ROADMAP_DETERMINIST §2.2.b).
    """
    # Root-local planar velocity: first two channels of the 4-channel delta.
    planarVelocity = targetGlobalDelta[:, :2]  # (N, 2) — already local frame
    if not config.useAimDirection:
        return planarVelocity
    # Aim from pelvis facing at the *next* (target) frame.
    aim = aimDirectionFromPelvisYaw(rotation6d)[nextFrame]  # (N, 2)
    return torch.cat([planarVelocity, aim], dim=-1)


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
    if expectedPlanar != 2:
        raise ValueError(
            "planar control channels do not match expected 2."
        )
    if ROOT_LOCAL_MOTION_CHANNELS != 4:
        raise ValueError(
            "ROOT_LOCAL_MOTION_CHANNELS must be 4."
        )
