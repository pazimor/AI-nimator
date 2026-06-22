"""Autoregressive rollout for the Goal C controller.

The rollout loop lives **outside** the model graph (ROADMAP_DETERMINIST
§2.1 truth #10): ONNX exports only the single :meth:`MotionController.
forward`, and the engine (or this helper) drives the loop frame by frame.

I/O is in **raw** (un-normalized) space; normalization happens internally
through the two :class:`MotionNormalizer` instances (state and delta) so
callers — generation CLI and the ``rollout_drift`` health metric — share
one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ainimator.model.controller_v2 import MotionController
from ainimator.model.motion_normalizer import (
    MotionNormalizer,
    denormalizeStepDelta,
)


@dataclass(frozen=True)
class RolloutResult:
    """A rolled-out trajectory in raw space.

    Attributes
    ----------
    rotation6d : torch.Tensor
        ``(B, K + N, numBones, 6)`` — the seed window followed by the
        ``N`` rolled-out frames.
    rootTranslation : torch.Tensor
        ``(B, K + N, 3)``.
    """

    rotation6d: torch.Tensor
    rootTranslation: torch.Tensor


@torch.no_grad()
def rolloutController(
    model: MotionController,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    seedRotation6d: torch.Tensor,
    seedRootTranslation: torch.Tensor,
    controlSequence: torch.Tensor,
    phaseSequence: torch.Tensor | None = None,
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
        ``(B, K, 3)`` raw seed window.
    controlSequence : torch.Tensor
        ``(B, N, controlChannels)`` per-frame control.
    phaseSequence : torch.Tensor or None
        ``(B, N, phaseChannels)`` per-frame phase, or ``None``.

    Returns
    -------
    RolloutResult
    """
    window = model.config.contextFrames
    _validateSeed(model, seedRotation6d, seedRootTranslation, controlSequence)

    boneHistory = list(seedRotation6d.unbind(dim=1))
    globalHistory = list(seedRootTranslation.unbind(dim=1))
    steps = controlSequence.shape[1]

    for step in range(steps):
        boneWindow = torch.stack(boneHistory[-window:], dim=1)
        globalWindow = torch.stack(globalHistory[-window:], dim=1)
        control = controlSequence[:, step, :]
        phase = None if phaseSequence is None else phaseSequence[:, step, :]
        nextBone, nextGlobal = _stepOnce(
            model,
            stateNormalizer,
            deltaNormalizer,
            boneWindow,
            globalWindow,
            control,
            phase,
        )
        boneHistory.append(nextBone)
        globalHistory.append(nextGlobal)

    return RolloutResult(
        rotation6d=torch.stack(boneHistory, dim=1),
        rootTranslation=torch.stack(globalHistory, dim=1),
    )


def _stepOnce(
    model: MotionController,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    boneWindow: torch.Tensor,
    globalWindow: torch.Tensor,
    control: torch.Tensor,
    phase: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one frame; return the next raw bone / global frame."""
    normBone = stateNormalizer.normalizeBone(boneWindow)
    normGlobal = stateNormalizer.normalizeGlobal(globalWindow)
    output = model(normBone, control, globalWindow=normGlobal, phase=phase)

    rawBoneDelta, rawGlobalDelta = denormalizeStepDelta(
        deltaNormalizer, output.boneDelta, output.globalDelta
    )
    nextBone = boneWindow[:, -1, :, :] + rawBoneDelta
    if rawGlobalDelta is not None:
        nextGlobal = globalWindow[:, -1, :] + rawGlobalDelta
    else:
        nextGlobal = globalWindow[:, -1, :]
    return nextBone, nextGlobal


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
