"""Scheduled sampling for the Goal C controller (phase C4).

Corrects the **exposure bias** of the C1/C2 teacher-forced loop: at
training time the controller only ever sees ground-truth windows, but at
rollout it must consume its own (slightly wrong) predictions, and the
errors compound (ROADMAP_DETERMINIST C4).  Scheduled sampling feeds the
model its own predictions during training with a probability ramped from
0 to a target, so it learns to recover from its own drift.

The fed-back predictions are **detached** (no backprop-through-time):
each step's loss is still computed against the ground truth, but the
*input* window may be a past prediction.  This is the standard, stable
formulation (Bengio et al. 2015).

This sequential per-step loop is opt-in (``scheduledSampling > 0``); the
default training path stays the fast parallel teacher-forced step.  It
operates on a single clip (batch size 1) — the overfit / sanity regime;
multi-clip batching is a later optimisation.
"""

from __future__ import annotations

import torch

from ainimator.data.controller_sequences import ControllerSequenceBatch
from ainimator.model.controller_v2 import MotionController
from ainimator.model.losses_controller_v2 import (
    ControllerLossResult,
    ControllerLossWeights,
    combinedControllerLoss,
    footContactStepLoss,
    geodesicRotationLoss,
    velocityDeltaLoss,
)
from ainimator.model.motion_normalizer import (
    MotionNormalizer,
    denormalizeStepDelta,
)


def scheduledSamplingProbability(
    epoch: int,
    totalEpochs: int,
    target: float,
) -> float:
    """Linear ramp of the scheduled-sampling probability ``0 → target``.

    Parameters
    ----------
    epoch : int
        Current 0-based epoch.
    totalEpochs : int
        Total epochs (the ramp reaches ``target`` at the last epoch).
    target : float
        Final scheduled-sampling probability.

    Returns
    -------
    float
        Probability in ``[0, target]``.
    """
    if totalEpochs <= 1:
        return target
    fraction = min(1.0, epoch / (totalEpochs - 1))
    return target * fraction


def scheduledSamplingStep(
    model: MotionController,
    batch: ControllerSequenceBatch,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    controlNorm: torch.Tensor,
    deltaTensors: dict[str, torch.Tensor],
    probability: float,
    weights: ControllerLossWeights,
) -> ControllerLossResult:
    """One sequential training step with scheduled sampling (batch 1).

    Walks the clip in order; at each transition the input window is the
    last ``K`` frames of a working buffer that mixes ground truth and
    (detached) past predictions according to ``probability``.

    Returns
    -------
    ControllerLossResult
        Mean loss over the clip's transitions.
    """
    window = model.config.contextFrames
    workingBone = list(batch.boneWindow[0].unbind(dim=0))
    workingGlobal = list(batch.globalWindow[0].unbind(dim=0))
    accumulator = _LossAccumulator()

    for step in range(batch.numTransitions):
        boneWindow = torch.stack(workingBone[-window:], dim=0).unsqueeze(0)
        globalWindow = torch.stack(
            workingGlobal[-window:], dim=0
        ).unsqueeze(0)
        output = model(
            stateNormalizer.normalizeBone(boneWindow),
            controlNorm[step : step + 1],
            globalWindow=stateNormalizer.normalizeGlobal(globalWindow),
            phase=_phaseAt(batch, step),
        )
        _accumulateStepLoss(
            accumulator, output, batch, deltaTensors, step, weights,
            deltaNormalizer, boneWindow,
        )
        nextBone, nextGlobal = _nextWorkingFrame(
            output, batch, deltaNormalizer, boneWindow, globalWindow,
            step, probability,
        )
        workingBone.append(nextBone.detach())
        workingGlobal.append(nextGlobal.detach())

    return accumulator.result(weights)


class _LossAccumulator:
    """Sum per-step loss components for a scheduled-sampling pass."""

    def __init__(self) -> None:
        self.velocity = torch.zeros(())
        self.geodesic = torch.zeros(())
        self.footContact = torch.zeros(())
        self.hasFoot = False
        self.count = 0

    def add(
        self,
        velocity: torch.Tensor,
        geodesic: torch.Tensor,
        footContact: torch.Tensor | None,
    ) -> None:
        """Accumulate one step's components."""
        self.velocity = self.velocity + velocity
        self.geodesic = self.geodesic + geodesic
        if footContact is not None:
            self.footContact = self.footContact + footContact
            self.hasFoot = True
        self.count += 1

    def result(self, weights: ControllerLossWeights) -> ControllerLossResult:
        """Mean the components and combine into the final objective."""
        steps = max(self.count, 1)
        foot = self.footContact / steps if self.hasFoot else None
        return combinedControllerLoss(
            self.velocity / steps,
            self.geodesic / steps,
            weights,
            footContact=foot,
        )


def _phaseAt(
    batch: ControllerSequenceBatch, step: int
) -> torch.Tensor | None:
    """Return the phase row for ``step`` (or ``None`` when phase is off)."""
    if batch.phase is None:
        return None
    return batch.phase[step : step + 1]


def _accumulateStepLoss(
    accumulator: _LossAccumulator,
    output: object,
    batch: ControllerSequenceBatch,
    deltaTensors: dict[str, torch.Tensor],
    step: int,
    weights: ControllerLossWeights,
    deltaNormalizer: MotionNormalizer,
    boneWindow: torch.Tensor,
) -> None:
    """Compute and accumulate one step's loss components."""
    boneDelta = output.boneDelta  # type: ignore[attr-defined]
    globalDelta = output.globalDelta  # type: ignore[attr-defined]
    velocity = velocityDeltaLoss(
        boneDelta, deltaTensors["normBoneDelta"][step : step + 1]
    ) + velocityDeltaLoss(
        globalDelta, deltaTensors["normGlobalDelta"][step : step + 1]
    )
    rawBoneDelta, rawGlobalDelta = denormalizeStepDelta(
        deltaNormalizer, boneDelta, globalDelta
    )
    predictedNextBone = boneWindow[:, -1, :, :] + rawBoneDelta
    geodesic = geodesicRotationLoss(
        predictedNextBone, batch.targetBoneNext[step : step + 1]
    )
    foot = _stepFootLoss(
        batch, predictedNextBone, rawGlobalDelta, step, weights
    )
    accumulator.add(velocity, geodesic, foot)


def _stepFootLoss(
    batch: ControllerSequenceBatch,
    predictedNextBone: torch.Tensor,
    rawGlobalDelta: torch.Tensor | None,
    step: int,
    weights: ControllerLossWeights,
) -> torch.Tensor | None:
    """Per-step anti-skating loss, or ``None`` when disabled."""
    if weights.footContact <= 0.0 or batch.contactTarget is None:
        return None
    if rawGlobalDelta is None:
        return None
    return footContactStepLoss(
        predictedNextBone,
        batch.boneWindow[step : step + 1, -1, :, :],
        rawGlobalDelta,
        batch.contactTarget[step : step + 1],
    )


def _nextWorkingFrame(
    output: object,
    batch: ControllerSequenceBatch,
    deltaNormalizer: MotionNormalizer,
    boneWindow: torch.Tensor,
    globalWindow: torch.Tensor,
    step: int,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the next working frame: prediction (prob) vs ground truth."""
    usePrediction = bool(torch.rand(()) < probability)
    if not usePrediction:
        return batch.targetBoneNext[step], _gtNextGlobal(batch, step)
    rawBoneDelta, rawGlobalDelta = denormalizeStepDelta(
        deltaNormalizer,
        output.boneDelta,  # type: ignore[attr-defined]
        output.globalDelta,  # type: ignore[attr-defined]
    )
    nextBone = (boneWindow[:, -1, :, :] + rawBoneDelta)[0]
    if rawGlobalDelta is None:
        return nextBone, globalWindow[0, -1, :]
    nextGlobal = (globalWindow[:, -1, :] + rawGlobalDelta)[0]
    return nextBone, nextGlobal


def _gtNextGlobal(
    batch: ControllerSequenceBatch, step: int
) -> torch.Tensor:
    """Ground-truth next root translation frame for ``step``."""
    return batch.globalWindow[step, -1, :] + batch.targetGlobalDelta[step]
