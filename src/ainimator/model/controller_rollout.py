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
        ``(B, K, 3)`` raw seed window in **world-space** XYZ.
    controlSequence : torch.Tensor
        ``(B, N, controlChannels)`` per-frame control.
    phaseSequence : torch.Tensor or None
        ``(B, N, phaseChannels)`` per-frame phase, or ``None``.
    seedRootLocalMotion : torch.Tensor or None
        ``(B, K, 4)`` seed window for the global (root-local) branch.

    Returns
    -------
    RolloutResult
    """
    _validateSeed(model, seedRotation6d, seedRootTranslation, controlSequence)
    _validatePhaseSequence(model, phaseSequence)
    state = _initRolloutState(model, seedRotation6d, seedRootTranslation,
                              seedRootLocalMotion)
    _runRolloutLoop(model, stateNormalizer, deltaNormalizer,
                    controlSequence, phaseSequence, state)
    return _buildResult(state)


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
    every few frames.  Every ``reinjectEvery`` steps the working frame
    is replaced by ground truth; ``reinjectEvery <= 0`` is open-loop.

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
        ``(B, K + N, 4)`` reference root-local motion deltas.

    Returns
    -------
    RolloutResult
    """
    state = _initClosedLoopState(
        model, groundTruthRotation6d,
        groundTruthRootTranslation, groundTruthRootLocalMotion,
    )
    _runClosedLoopBody(
        model, stateNormalizer, deltaNormalizer, state,
        groundTruthRotation6d, groundTruthRootTranslation,
        groundTruthRootLocalMotion, controlSequence, phaseSequence,
        reinjectEvery,
    )
    return _buildResult(state)


@dataclass
class _RolloutState:
    """Mutable accumulator for one rollout pass (open-loop or closed-loop)."""

    boneHistory: list[torch.Tensor]
    globalHistory: list[torch.Tensor]
    worldPositions: list[torch.Tensor]
    localMotionHistory: list[torch.Tensor]
    currentWorldPos: torch.Tensor
    currentYaw: torch.Tensor
    window: int


def _zeroGlobalHistory(
    window: int,
    batchSize: int,
    globalChannels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Return a list of ``window`` zero tensors for the global branch."""
    zeroFrame = torch.zeros(batchSize, globalChannels, device=device, dtype=dtype)
    return [zeroFrame.clone() for _ in range(window)]


def _initRolloutState(
    model: MotionController,
    seedRotation6d: torch.Tensor,
    seedRootTranslation: torch.Tensor,
    seedRootLocalMotion: torch.Tensor | None,
) -> _RolloutState:
    """Build the initial mutable state for an open-loop rollout."""
    window = model.config.contextFrames
    batchSize = seedRotation6d.shape[0]
    device = seedRotation6d.device
    dtype = seedRotation6d.dtype
    globalChannels = model.config.globalChannels
    boneHistory = list(seedRotation6d.unbind(dim=1))
    globalHistory = (
        list(seedRootLocalMotion.unbind(dim=1))
        if seedRootLocalMotion is not None
        else _zeroGlobalHistory(window, batchSize, globalChannels, device, dtype)
    )
    worldPositions = list(seedRootTranslation.unbind(dim=1))
    localMotionSeed = (
        seedRootLocalMotion
        if seedRootLocalMotion is not None
        else torch.zeros(batchSize, window, globalChannels, device=device, dtype=dtype)
    )
    lastPelvis = seedRotation6d[:, -1, _PELVIS_BONE_INDEX, :]
    return _RolloutState(
        boneHistory=boneHistory,
        globalHistory=globalHistory,
        worldPositions=worldPositions,
        localMotionHistory=list(localMotionSeed.unbind(dim=1)),
        currentWorldPos=seedRootTranslation[:, -1, :],
        currentYaw=pelvisYawFromRot6d(lastPelvis),
        window=window,
    )


def _runRolloutLoop(
    model: MotionController,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    controlSequence: torch.Tensor,
    phaseSequence: torch.Tensor | None,
    state: _RolloutState,
) -> None:
    """Advance the rollout state in-place for all steps."""
    for step in range(controlSequence.shape[1]):
        boneWin = torch.stack(state.boneHistory[-state.window:], dim=1)
        globalWin = torch.stack(state.globalHistory[-state.window:], dim=1)
        phase = None if phaseSequence is None else phaseSequence[:, step, :]
        nextBone, nextDelta = _stepOnce(
            model, stateNormalizer, deltaNormalizer,
            boneWin, globalWin, controlSequence[:, step, :], phase,
        )
        nextPos, nextYaw = _integrateOneStep(
            state.currentWorldPos, state.currentYaw, nextDelta
        )
        state.currentWorldPos = nextPos
        state.currentYaw = nextYaw
        state.boneHistory.append(nextBone)
        state.globalHistory.append(nextDelta)
        state.worldPositions.append(nextPos)
        state.localMotionHistory.append(nextDelta)


def _initClosedLoopState(
    model: MotionController,
    groundTruthRotation6d: torch.Tensor,
    groundTruthRootTranslation: torch.Tensor,
    groundTruthRootLocalMotion: torch.Tensor | None,
) -> _RolloutState:
    """Build the initial mutable state for a closed-loop rollout."""
    window = model.config.contextFrames
    batchSize = groundTruthRotation6d.shape[0]
    device = groundTruthRotation6d.device
    dtype = groundTruthRotation6d.dtype
    globalChannels = model.config.globalChannels
    boneHistory = list(groundTruthRotation6d[:, :window].unbind(dim=1))
    worldPositions = list(groundTruthRootTranslation[:, :window].unbind(dim=1))
    globalHistory = (
        list(groundTruthRootLocalMotion[:, :window].unbind(dim=1))
        if groundTruthRootLocalMotion is not None
        else _zeroGlobalHistory(window, batchSize, globalChannels, device, dtype)
    )
    lastPelvis = groundTruthRotation6d[:, window - 1, _PELVIS_BONE_INDEX, :]
    return _RolloutState(
        boneHistory=boneHistory,
        globalHistory=globalHistory,
        worldPositions=worldPositions,
        localMotionHistory=list(globalHistory),
        currentWorldPos=groundTruthRootTranslation[:, window - 1, :],
        currentYaw=pelvisYawFromRot6d(lastPelvis),
        window=window,
    )


def _runClosedLoopBody(
    model: MotionController,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    state: _RolloutState,
    gtRotation6d: torch.Tensor,
    gtRootTranslation: torch.Tensor,
    gtRootLocalMotion: torch.Tensor | None,
    controlSequence: torch.Tensor,
    phaseSequence: torch.Tensor | None,
    reinjectEvery: int,
) -> None:
    """Closed-loop rollout with optional GT re-injection."""
    for step in range(controlSequence.shape[1]):
        boneWin = torch.stack(state.boneHistory[-state.window:], dim=1)
        globalWin = torch.stack(state.globalHistory[-state.window:], dim=1)
        phase = None if phaseSequence is None else phaseSequence[:, step, :]
        nextBone, nextDelta = _stepOnce(
            model, stateNormalizer, deltaNormalizer,
            boneWin, globalWin, controlSequence[:, step, :], phase,
        )
        nextPos, nextYaw = _integrateOneStep(
            state.currentWorldPos, state.currentYaw, nextDelta
        )
        targetIndex = state.window + step
        if reinjectEvery > 0 and (step + 1) % reinjectEvery == 0:
            nextBone, nextPos, nextDelta, nextYaw = _reinjectGT(
                gtRotation6d, gtRootTranslation,
                gtRootLocalMotion, targetIndex, nextDelta,
            )
        state.currentWorldPos = nextPos
        state.currentYaw = nextYaw
        state.boneHistory.append(nextBone)
        state.globalHistory.append(nextDelta)
        state.worldPositions.append(nextPos)
        state.localMotionHistory.append(nextDelta)


def _reinjectGT(
    gtRotation6d: torch.Tensor,
    gtRootTranslation: torch.Tensor,
    gtRootLocalMotion: torch.Tensor | None,
    targetIndex: int,
    predictedDelta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Return GT bone/pos/delta/yaw for one re-injection step.

    When ``gtRootLocalMotion`` is ``None`` the predicted delta is kept
    so the history lists never contain ``None`` entries.
    """
    nextBone = gtRotation6d[:, targetIndex]
    nextPos = gtRootTranslation[:, targetIndex]
    nextDelta = (
        gtRootLocalMotion[:, targetIndex]
        if gtRootLocalMotion is not None
        else predictedDelta
    )
    nextYaw = pelvisYawFromRot6d(
        gtRotation6d[:, targetIndex, _PELVIS_BONE_INDEX, :]
    )
    return nextBone, nextPos, nextDelta, nextYaw


def _buildResult(state: _RolloutState) -> RolloutResult:
    """Assemble a :class:`RolloutResult` from the accumulated state."""
    return RolloutResult(
        rotation6d=torch.stack(state.boneHistory, dim=1),
        rootTranslation=torch.stack(state.worldPositions, dim=1),
        rootLocalMotion=torch.stack(state.localMotionHistory, dim=1),
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


def _validatePhaseSequence(
    model: MotionController,
    phaseSequence: torch.Tensor | None,
) -> None:
    """Raise ValueError if phase is required but not supplied.

    This guard lives here (outside ``forward()``) so the ONNX graph
    stays free of data-dependent control flow (G-ONNX).
    """
    if model.config.phaseChannels > 0 and phaseSequence is None:
        raise ValueError(
            "phase is required when phaseMode is not 'none', "
            "but phaseSequence was not provided."
        )
