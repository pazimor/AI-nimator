"""Autoregressive rollout for the Goal A controller.

The rollout loop lives **outside** the model graph (ROADMAP_DETERMINIST
§2.1 truth #10): ONNX exports only the single :meth:`MotionController.
forward`, and the engine (or this helper) drives the loop frame by frame.

State representation (ROADMAP_DETERMINIST §2.2.a)
-------------------------------------------------
The controller predicts ``Δstate`` where the global branch is the
**root-local motion delta** ``(Δforward, Δlateral, Δheight, Δyaw)``
(4 channels), not an absolute world-space translation.

The rollout accumulates these local deltas into a world-space trajectory
via :func:`~ainimator.geometry.root_local.rootLocalDeltasToAbsolute`.
This integration lives *here* (outside the ONNX graph) as required.

I/O convention
--------------
* Inputs: raw (un-normalized) tensors.
* Normalization happens internally through the two
  :class:`MotionNormalizer` instances (state and delta).
* ``RolloutResult.rootTranslation`` is the **world-space** XYZ
  trajectory, reconstructed from the local deltas by integration.
  Callers — generation CLI and the ``rollout_drift`` health metric —
  receive the same world-space result they expect.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ainimator.geometry.root_local import (
    pelvisYawFromRot6d,
    rootLocalDeltasToAbsolute,
)
from ainimator.model.controller_v2 import MotionController
from ainimator.model.motion_normalizer import (
    MotionNormalizer,
    denormalizeStepDelta,
)

# Pelvis is bone 0 in SMPL-22.
_PELVIS_BONE_INDEX: int = 0


@dataclass(frozen=True)
class RolloutResult:
    """A rolled-out trajectory in raw space.

    Attributes
    ----------
    rotation6d : torch.Tensor
        ``(B, K + N, numBones, 6)`` — the seed window followed by the
        ``N`` rolled-out frames.
    rootTranslation : torch.Tensor
        ``(B, K + N, 3)`` world-space XYZ trajectory, reconstructed from
        root-local motion deltas by integration.
    rootLocalMotion : torch.Tensor
        ``(B, K + N, 4)`` root-local motion delta history
        ``(Δforward, Δlateral, Δheight, Δyaw)``.  Useful for computing
        drift in the local-motion domain.
    """

    rotation6d: torch.Tensor
    rootTranslation: torch.Tensor
    rootLocalMotion: torch.Tensor


@torch.no_grad()
def rolloutController(
    model: MotionController,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    seedRotation6d: torch.Tensor,
    seedRootTranslation: torch.Tensor,
    controlSequence: torch.Tensor,
    phaseSequence: torch.Tensor | None = None,
    seedRootLocalMotion: torch.Tensor | None = None,
) -> RolloutResult:
    """Roll the controller autoregressively under a control sequence.

    Parameters
    ----------
    model : MotionController
        The trained controller (set to eval by the caller if needed).
    stateNormalizer, deltaNormalizer : MotionNormalizer
        Z-norm statistics for the state window and the predicted delta.
    seedRotation6d : torch.Tensor
        ``(B, K, numBones, 6)`` raw seed window (``K = contextFrames``).
    seedRootTranslation : torch.Tensor
        ``(B, K, 3)`` raw seed window in **world-space** XYZ.  The
        final seed frame is used as the integration origin.
    controlSequence : torch.Tensor
        ``(B, N, controlChannels)`` per-frame control.
    phaseSequence : torch.Tensor or None
        ``(B, N, phaseChannels)`` per-frame phase, or ``None``.
    seedRootLocalMotion : torch.Tensor or None
        ``(B, K, 4)`` seed window for the global (root-local) branch.
        When ``None``, a zero tensor is used (no motion history before
        the seed).

    Returns
    -------
    RolloutResult
    """
    window = model.config.contextFrames
    _validateSeed(model, seedRotation6d, seedRootTranslation, controlSequence)

    batchSize = seedRotation6d.shape[0]
    device = seedRotation6d.device
    dtype = seedRotation6d.dtype

    boneHistory = list(seedRotation6d.unbind(dim=1))
    globalChannels = model.config.globalChannels  # 4

    if seedRootLocalMotion is not None:
        globalHistory = list(seedRootLocalMotion.unbind(dim=1))
    else:
        # No prior local motion context: use zeros.
        zeroFrame = torch.zeros(
            batchSize, globalChannels, device=device, dtype=dtype
        )
        globalHistory = [zeroFrame.clone() for _ in range(window)]

    # Track cumulative world-space position and yaw for integration.
    # Start from the last seed frame.
    currentWorldPos = seedRootTranslation[:, -1, :]          # (B, 3)
    lastSeedPelvis = seedRotation6d[:, -1, _PELVIS_BONE_INDEX, :]  # (B, 6)
    currentYaw = pelvisYawFromRot6d(lastSeedPelvis)           # (B,)

    # Accumulate world-space positions (seed window + rolled frames).
    worldPositions = list(seedRootTranslation.unbind(dim=1))   # K frames
    localMotionHistory = list(
        (seedRootLocalMotion if seedRootLocalMotion is not None
         else torch.zeros(batchSize, window, globalChannels,
                          device=device, dtype=dtype)
         ).unbind(dim=1)
    )

    steps = controlSequence.shape[1]

    for step in range(steps):
        boneWindow = torch.stack(boneHistory[-window:], dim=1)
        globalWindow = torch.stack(globalHistory[-window:], dim=1)
        control = controlSequence[:, step, :]
        phase = None if phaseSequence is None else phaseSequence[:, step, :]
        nextBone, nextLocalDelta = _stepOnce(
            model,
            stateNormalizer,
            deltaNormalizer,
            boneWindow,
            globalWindow,
            control,
            phase,
        )
        # Integrate local delta → world position.
        nextWorldPos, nextYaw = _integrateOneStep(
            currentWorldPos, currentYaw, nextLocalDelta
        )
        currentWorldPos = nextWorldPos
        currentYaw = nextYaw

        boneHistory.append(nextBone)
        globalHistory.append(nextLocalDelta)
        worldPositions.append(nextWorldPos)
        localMotionHistory.append(nextLocalDelta)

    return RolloutResult(
        rotation6d=torch.stack(boneHistory, dim=1),
        rootTranslation=torch.stack(worldPositions, dim=1),
        rootLocalMotion=torch.stack(localMotionHistory, dim=1),
    )


@torch.no_grad()
def rolloutControllerClosedLoop(
    model: MotionController,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    groundTruthRotation6d: torch.Tensor,
    groundTruthRootTranslation: torch.Tensor,
    controlSequence: torch.Tensor,
    reinjectEvery: int,
    phaseSequence: torch.Tensor | None = None,
    groundTruthRootLocalMotion: torch.Tensor | None = None,
) -> RolloutResult:
    """Roll out with periodic ground-truth state re-injection.

    Models the deployment regime: a game engine re-grounds the character
    every few frames (foot-lock IK, physics), so the controller does not
    accumulate error indefinitely.  Every ``reinjectEvery`` steps the
    working frame is replaced by ground truth; ``reinjectEvery <= 0`` is
    pure open-loop.

    Parameters
    ----------
    model, stateNormalizer, deltaNormalizer : see :func:`rolloutController`.
    groundTruthRotation6d : torch.Tensor
        ``(B, K + N, numBones, 6)`` reference frames (seed + targets).
    groundTruthRootTranslation : torch.Tensor
        ``(B, K + N, 3)`` reference root translation in world space.
    controlSequence : torch.Tensor
        ``(B, N, controlChannels)`` per-frame control.
    reinjectEvery : int
        Re-injection period in frames (``<= 0`` → open-loop).
    phaseSequence : torch.Tensor or None
        ``(B, N, phaseChannels)`` per-frame phase.
    groundTruthRootLocalMotion : torch.Tensor or None
        ``(B, K + N, 4)`` reference root-local motion deltas.  Used for
        re-injection of the global branch.  When ``None``, the global
        branch is re-injected with zeros (conservative).

    Returns
    -------
    RolloutResult
    """
    window = model.config.contextFrames
    batchSize = groundTruthRotation6d.shape[0]
    device = groundTruthRotation6d.device
    dtype = groundTruthRotation6d.dtype
    globalChannels = model.config.globalChannels  # 4

    boneHistory = list(groundTruthRotation6d[:, :window].unbind(dim=1))
    worldPositions = list(groundTruthRootTranslation[:, :window].unbind(dim=1))

    if groundTruthRootLocalMotion is not None:
        globalHistory = list(
            groundTruthRootLocalMotion[:, :window].unbind(dim=1)
        )
    else:
        zeroFrame = torch.zeros(
            batchSize, globalChannels, device=device, dtype=dtype
        )
        globalHistory = [zeroFrame.clone() for _ in range(window)]

    localMotionHistory = list(globalHistory)

    currentWorldPos = groundTruthRootTranslation[:, window - 1, :]
    lastPelvis = groundTruthRotation6d[:, window - 1, _PELVIS_BONE_INDEX, :]
    currentYaw = pelvisYawFromRot6d(lastPelvis)

    steps = controlSequence.shape[1]

    for step in range(steps):
        boneWindow = torch.stack(boneHistory[-window:], dim=1)
        globalWindow = torch.stack(globalHistory[-window:], dim=1)
        phase = None if phaseSequence is None else phaseSequence[:, step, :]
        nextBone, nextLocalDelta = _stepOnce(
            model, stateNormalizer, deltaNormalizer, boneWindow,
            globalWindow, controlSequence[:, step, :], phase,
        )
        nextWorldPos, nextYaw = _integrateOneStep(
            currentWorldPos, currentYaw, nextLocalDelta
        )
        targetIndex = window + step
        if reinjectEvery > 0 and (step + 1) % reinjectEvery == 0:
            nextBone = groundTruthRotation6d[:, targetIndex]
            nextWorldPos = groundTruthRootTranslation[:, targetIndex]
            if groundTruthRootLocalMotion is not None:
                nextLocalDelta = groundTruthRootLocalMotion[:, targetIndex]
            # Re-sync yaw from the GT pelvis.
            nextYaw = pelvisYawFromRot6d(
                groundTruthRotation6d[:, targetIndex, _PELVIS_BONE_INDEX, :]
            )

        currentWorldPos = nextWorldPos
        currentYaw = nextYaw

        boneHistory.append(nextBone)
        globalHistory.append(nextLocalDelta)
        worldPositions.append(nextWorldPos)
        localMotionHistory.append(nextLocalDelta)

    return RolloutResult(
        rotation6d=torch.stack(boneHistory, dim=1),
        rootTranslation=torch.stack(worldPositions, dim=1),
        rootLocalMotion=torch.stack(localMotionHistory, dim=1),
    )


def _integrateOneStep(
    currentPos: torch.Tensor,
    currentYaw: torch.Tensor,
    localDelta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate one root-local motion delta to world space.

    Parameters
    ----------
    currentPos : torch.Tensor
        Current world-space XYZ, shape ``(B, 3)``.
    currentYaw : torch.Tensor
        Current world-space yaw in radians, shape ``(B,)``.
    localDelta : torch.Tensor or None
        Root-local motion delta ``(Δfwd, Δlat, Δheight, Δyaw)``, shape
        ``(B, 4)``.  When ``None`` the position is unchanged.

    Returns
    -------
    nextPos : torch.Tensor
        New world-space XYZ, shape ``(B, 3)``.
    nextYaw : torch.Tensor
        New world-space yaw, shape ``(B,)``.
    """
    if localDelta is None:
        return currentPos, currentYaw

    dFwd = localDelta[:, 0]
    dLat = localDelta[:, 1]
    dHeight = localDelta[:, 2]
    dYaw = localDelta[:, 3]

    cos = torch.cos(currentYaw)
    sin = torch.sin(currentYaw)
    worldDx = cos * dFwd - sin * dLat
    worldDz = sin * dFwd + cos * dLat

    nextPos = currentPos + torch.stack([worldDx, dHeight, worldDz], dim=-1)
    nextYaw = currentYaw + dYaw
    return nextPos, nextYaw


def _stepOnce(
    model: MotionController,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    boneWindow: torch.Tensor,
    globalWindow: torch.Tensor,
    control: torch.Tensor,
    phase: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Advance one frame; return the next raw bone delta and global delta."""
    normBone = stateNormalizer.normalizeBone(boneWindow)
    normGlobal = stateNormalizer.normalizeGlobal(globalWindow)
    output = model(normBone, control, globalWindow=normGlobal, phase=phase)

    rawBoneDelta, rawGlobalDelta = denormalizeStepDelta(
        deltaNormalizer, output.boneDelta, output.globalDelta
    )
    nextBone = boneWindow[:, -1, :, :] + rawBoneDelta
    return nextBone, rawGlobalDelta


def _validateSeed(
    model: MotionController,
    seedRotation6d: torch.Tensor,
    seedRootTranslation: torch.Tensor,
    controlSequence: torch.Tensor,
) -> None:
    """Validate seed-window shapes against the model config."""
    window = model.config.contextFrames
    if seedRotation6d.ndim != 4 or seedRotation6d.shape[1] != window:
        raise ValueError(
            "seedRotation6d must be (B, contextFrames, bones, 6); got "
            f"{tuple(seedRotation6d.shape)} for K={window}."
        )
    if seedRootTranslation.shape[1] != window:
        raise ValueError(
            "seedRootTranslation window length must equal contextFrames."
        )
    if controlSequence.ndim != 3:
        raise ValueError(
            "controlSequence must be (B, N, controlChannels); got "
            f"{tuple(controlSequence.shape)}."
        )
    if model.config.phaseMode.value == "explicit" and False:
        # Phase validation happens at call sites when phaseSequence is None.
        pass
