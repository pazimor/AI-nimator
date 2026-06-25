"""ARCHIVED (A7, 2026-06-25) — replaced by the unified full profile.

The multi-clip training logic has been superseded by
``ainimator.training.controller_generalization_v2`` (the ``full``
profile of the unified ``train_controller_v2`` CLI).  This file is kept
in ``legacy/`` for historical reference only.  Nothing in the live
``ainimator.*`` package may import from ``legacy.*``
(lint-imports contract ``no_legacy_imports``).

Original module docstring follows.
----------------------------------------------------------------------
Small-N multi-clip controller training (Goal A — A2/A4 validation).

The single-clip overfit (``controller_training_v2``) validates A1 and the
A4 mechanism, but two acceptance criteria are *only* meaningful with
several clips and **varied control** (ROADMAP_DETERMINIST §4/§5):

* ``mean_collapse`` SAIN — ``effective_rank`` only rises when the batch
  carries genuinely different controls (different walks/turns); on one
  clip it is structurally low.
* long-horizon ``rollout_drift`` — measured per clip and averaged, on a
  model that learned a generalisable control→motion map rather than one
  memorised trajectory.

This module reuses the whole single-clip machinery (model, losses,
rollout, metrics, scheduled sampling, warm-start) and only adds the
multi-clip plumbing: per-clip sequence batches, global normalizers /
control stats fitted across all clips, a merged batch for the parallel
teacher-forced step and the collapse metrics, and per-clip rollouts for
the drift metric.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ainimator.core.constants.controller import PhaseMode, ROOT_LOCAL_MOTION_CHANNELS
from ainimator.core.resolved_config import writeResolvedConfig
from ainimator.data.controller_sequences import (
    ControllerSequenceBatch,
    ControllerSequenceConfig,
    buildControllerSequences,
)
from ainimator.health.contract import Verdict
from ainimator.health.controller_metrics import (
    controlSensitivity,
    meanCollapse,
    postNormStats,
    rolloutDrift,
    rolloutDriftCurve,
)
from ainimator.model.controller_v2 import MotionController
from ainimator.model.motion_normalizer import MotionNormalizer
from ainimator.training.controller_scheduled_sampling import (
    scheduledSamplingProbability,
    scheduledSamplingStep,
)
from ainimator.training.controller_training_v2 import (
    ControllerOverfitResult,
    ControllerTrainingConfig,
    _driftHorizons,
    _evaluateContracts,
    _forwardLosses,
    _groundTruthTrajectory,
    _normalizeTensors,
    _rolloutFromClip,
    _saveRollout,
    buildControllerModelConfig,
    loadControllerCheckpoint,
    resolveControllerDevice,
    saveControllerCheckpoint,
    _DEFAULT_HEALTH_PATH,
)

LOGGER = logging.getLogger("ainimator.training.controller_multiclip")

Clip = tuple[torch.Tensor, torch.Tensor]
_CONTROL_STD_FLOOR = 1e-5


# ---------------------------------------------------------------------
# Per-clip preparation
# ---------------------------------------------------------------------
def _sequenceConfig(
    config: ControllerTrainingConfig,
    phaseMode: PhaseMode,
    useAim: bool,
    contextFrames: int,
) -> ControllerSequenceConfig:
    """Build the sequence config shared by every clip."""
    return ControllerSequenceConfig(
        contextFrames=contextFrames,
        useAimDirection=useAim,
        emitPhase=phaseMode is not PhaseMode.NONE,
        emitContacts=config.lossWeights.footContact > 0.0,
    )


def _buildClipBatches(
    clips: list[Clip],
    sequenceConfig: ControllerSequenceConfig,
    device: torch.device,
) -> list[ControllerSequenceBatch]:
    """Build one autoregressive batch per clip (on ``device``)."""
    batches: list[ControllerSequenceBatch] = []
    for rotation6d, rootTranslation in clips:
        batches.append(
            buildControllerSequences(
                rotation6d.to(device),
                rootTranslation.to(device),
                sequenceConfig,
            )
        )
    return batches


def _concatClipBatches(
    batches: list[ControllerSequenceBatch],
) -> ControllerSequenceBatch:
    """Concatenate per-clip batches into one (for the parallel step)."""
    phaseAll = all(b.phase is not None for b in batches)
    contactAll = all(b.contactTarget is not None for b in batches)
    return ControllerSequenceBatch(
        boneWindow=torch.cat([b.boneWindow for b in batches], dim=0),
        globalWindow=torch.cat([b.globalWindow for b in batches], dim=0),
        targetBoneNext=torch.cat(
            [b.targetBoneNext for b in batches], dim=0
        ),
        targetBoneDelta=torch.cat(
            [b.targetBoneDelta for b in batches], dim=0
        ),
        targetGlobalDelta=torch.cat(
            [b.targetGlobalDelta for b in batches], dim=0
        ),
        control=torch.cat([b.control for b in batches], dim=0),
        phase=(
            torch.cat([b.phase for b in batches], dim=0)  # type: ignore[arg-type]
            if phaseAll
            else None
        ),
        contactTarget=(
            torch.cat(
                [b.contactTarget for b in batches], dim=0  # type: ignore[arg-type]
            )
            if contactAll
            else None
        ),
    )


def _fitNormalizersMulti(
    batches: list[ControllerSequenceBatch],
    numBones: int,
) -> tuple[MotionNormalizer, MotionNormalizer]:
    """Fit state + delta z-normalizers across ALL clips (truth #3).

    Uses ``ROOT_LOCAL_MOTION_CHANNELS`` (4) for the global branch to
    match the root-local motion representation (§2.2.a).
    """
    stateNormalizer = MotionNormalizer(
        numBones=numBones,
        motionChannels=6,
        globalChannels=ROOT_LOCAL_MOTION_CHANNELS,
    )
    stateNormalizer.fitFromTensors(
        boneSamples=[b.boneWindow for b in batches],
        globalSamples=[b.globalWindow for b in batches],
    )
    deltaNormalizer = MotionNormalizer(
        numBones=numBones,
        motionChannels=6,
        globalChannels=ROOT_LOCAL_MOTION_CHANNELS,
    )
    deltaNormalizer.fitFromTensors(
        boneSamples=[b.targetBoneDelta for b in batches],
        globalSamples=[b.targetGlobalDelta for b in batches],
    )
    return stateNormalizer, deltaNormalizer


def _globalControlStats(
    merged: ControllerSequenceBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean / std of the control across all clips (global standardisation)."""
    mean = merged.control.mean(dim=0, keepdim=True)
    std = merged.control.std(dim=0, unbiased=False, keepdim=True)
    return mean, std.clamp(min=_CONTROL_STD_FLOOR)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def _trainMultiClip(
    model: MotionController,
    batches: list[ControllerSequenceBatch],
    merged: ControllerSequenceBatch,
    perClip: list[dict[str, torch.Tensor]],
    mergedTensors: dict[str, torch.Tensor],
    controlMean: torch.Tensor,
    controlStd: torch.Tensor,
    deltaNormalizer: MotionNormalizer,
    stateNormalizer: MotionNormalizer,
    config: ControllerTrainingConfig,
) -> float:
    """Train across clips; return the final total loss."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learningRate,
        weight_decay=config.weightDecay,
    )
    mergedControlNorm = (merged.control - controlMean) / controlStd
    lastLoss = float("nan")
    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        loss = _multiClipLoss(
            model, batches, merged, perClip, mergedTensors,
            mergedControlNorm, controlMean, controlStd, stateNormalizer,
            deltaNormalizer, config, epoch,
        )
        loss.backward()
        optimizer.step()
        lastLoss = float(loss.item())
        if epoch % config.logEvery == 0 or epoch == config.epochs - 1:
            LOGGER.info("epoch %d — loss %.6f", epoch, lastLoss)
    return lastLoss


def _multiClipLoss(
    model: MotionController,
    batches: list[ControllerSequenceBatch],
    merged: ControllerSequenceBatch,
    perClip: list[dict[str, torch.Tensor]],
    mergedTensors: dict[str, torch.Tensor],
    mergedControlNorm: torch.Tensor,
    controlMean: torch.Tensor,
    controlStd: torch.Tensor,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    config: ControllerTrainingConfig,
    epoch: int,
) -> torch.Tensor:
    """Teacher-forced (parallel, merged) or scheduled-sampling (per clip)."""
    if config.scheduledSampling <= 0.0:
        result = _forwardLosses(
            model, merged, mergedTensors, mergedControlNorm,
            deltaNormalizer, config.lossWeights,
        )
        return result.total
    probability = scheduledSamplingProbability(
        epoch, config.epochs, config.scheduledSampling
    )
    total = torch.zeros((), device=mergedControlNorm.device)
    for batch, tensors in zip(batches, perClip):
        controlNorm = (batch.control - controlMean) / controlStd
        stepResult = scheduledSamplingStep(
            model, batch, stateNormalizer, deltaNormalizer, controlNorm,
            tensors, probability, config.lossWeights,
        )
        total = total + stepResult.total
    return total / max(len(batches), 1)


# ---------------------------------------------------------------------
# Evaluation (varied-control collapse + per-clip drift)
# ---------------------------------------------------------------------
def _evaluateMultiClip(
    model: MotionController,
    batches: list[ControllerSequenceBatch],
    clipRootTranslations: list[torch.Tensor],
    merged: ControllerSequenceBatch,
    mergedTensors: dict[str, torch.Tensor],
    controlMean: torch.Tensor,
    controlStd: torch.Tensor,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    healthPath: Path,
) -> tuple[dict[str, float], dict[str, Verdict], dict[int, float]]:
    """Collapse/sensitivity on the varied-control batch; drift per clip."""
    model.eval()
    mergedControlNorm = (merged.control - controlMean) / controlStd
    output = model(
        mergedTensors["normBoneWindow"],
        mergedControlNorm,
        globalWindow=mergedTensors["normGlobalWindow"],
        phase=merged.phase,
    )
    rank, sim = meanCollapse(output)
    sensitivity = controlSensitivity(
        model,
        mergedTensors["normBoneWindow"],
        mergedControlNorm,
        globalWindow=mergedTensors["normGlobalWindow"],
        phase=merged.phase,
    )
    drift, driftCurve = _perClipDrift(
        model, batches, clipRootTranslations, controlMean, controlStd,
        stateNormalizer, deltaNormalizer,
    )
    postNorm = max(
        postNormStats(mergedTensors["normBoneWindow"]),
        postNormStats(mergedTensors["normBoneDelta"]),
    )
    metrics = {
        "control_sensitivity": sensitivity,
        "mean_collapse_rank": rank,
        "mean_collapse_sim": sim,
        "rollout_drift": drift,
        "post_norm_stats": postNorm,
    }
    return metrics, _evaluateContracts(metrics, healthPath), driftCurve


def _perClipDrift(
    model: MotionController,
    batches: list[ControllerSequenceBatch],
    clipRootTranslations: list[torch.Tensor],
    controlMean: torch.Tensor,
    controlStd: torch.Tensor,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
) -> tuple[float, dict[int, float]]:
    """Mean rollout drift over clips; drift curve on the longest clip."""
    drifts: list[float] = []
    curve: dict[int, float] = {}
    longest = max(range(len(batches)), key=lambda i: batches[i].numTransitions)
    for index, batch in enumerate(batches):
        controlNorm = (batch.control - controlMean) / controlStd
        rollout = _rolloutFromClip(
            model, batch, stateNormalizer, deltaNormalizer, controlNorm,
            clipRootTranslations[index],
        )
        gtBone, gtRoot = _groundTruthTrajectory(
            batch, clipRootTranslations[index]
        )
        drifts.append(rolloutDrift(rollout, gtBone, gtRoot))
        if index == longest:
            curve = rolloutDriftCurve(
                rollout, gtBone, gtRoot,
                _driftHorizons(rollout.rotation6d.shape[1]),
            )
    return sum(drifts) / max(len(drifts), 1), curve


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def runControllerMultiClip(
    clips: list[Clip],
    config: ControllerTrainingConfig,
    healthPath: Path = _DEFAULT_HEALTH_PATH,
) -> ControllerOverfitResult:
    """Train the controller on several clips; gate on A2/A4 contracts."""
    if len(clips) < 2:
        raise ValueError("multi-clip training needs at least 2 clips.")
    torch.manual_seed(config.seed)
    device = resolveControllerDevice(config.device)
    numBones = int(clips[0][0].shape[-2])

    model, stateNormalizer, deltaNormalizer, sequenceConfig = (
        _buildOrResumeModel(config, device, numBones)
    )
    batches = _buildClipBatches(clips, sequenceConfig, device)
    if config.resumeCheckpoint is None:
        stateNormalizer, deltaNormalizer = _fitNormalizersMulti(
            batches, numBones
        )
        stateNormalizer = stateNormalizer.to(device)
        deltaNormalizer = deltaNormalizer.to(device)
    merged = _concatClipBatches(batches)
    controlMean, controlStd = _globalControlStats(merged)

    perClip = [
        _normalizeTensors(batch, stateNormalizer, deltaNormalizer)
        for batch in batches
    ]
    mergedTensors = _normalizeTensors(
        merged, stateNormalizer, deltaNormalizer
    )

    finalLoss = _trainMultiClip(
        model, batches, merged, perClip, mergedTensors, controlMean,
        controlStd, deltaNormalizer, stateNormalizer, config,
    )

    clipRootTranslations = [
        rot.to(device) for _, rot in [
            (clips[i][0], clips[i][1]) for i in range(len(clips))
        ]
    ]

    config.outputDir.mkdir(parents=True, exist_ok=True)
    metrics, verdicts, driftCurve = _evaluateMultiClip(
        model, batches, clipRootTranslations, merged, mergedTensors,
        controlMean, controlStd, stateNormalizer, deltaNormalizer, healthPath,
    )
    rollout = _rolloutFromClip(
        model, batches[0], stateNormalizer, deltaNormalizer,
        (batches[0].control - controlMean) / controlStd,
        clipRootTranslations[0],
    )
    checkpointPath = saveControllerCheckpoint(
        model, stateNormalizer, deltaNormalizer, controlMean, controlStd,
        config.outputDir,
    )
    rolloutPath = _saveRollout(rollout, config.outputDir)
    writeResolvedConfig(config, config.outputDir)
    return ControllerOverfitResult(
        finalLoss=finalLoss,
        metrics=metrics,
        verdicts=verdicts,
        driftCurve=driftCurve,
        checkpointPath=checkpointPath,
        rolloutPath=rolloutPath,
    )


def _buildOrResumeModel(
    config: ControllerTrainingConfig,
    device: torch.device,
    numBones: int,
) -> tuple[
    MotionController, MotionNormalizer, MotionNormalizer,
    ControllerSequenceConfig,
]:
    """Build a fresh model or warm-start; return it + the sequence config."""
    if config.resumeCheckpoint is not None:
        model, stateNormalizer, deltaNormalizer, _mean, _std = (
            loadControllerCheckpoint(config.resumeCheckpoint, device)
        )
        model = model.to(device)
        sequenceConfig = _sequenceConfig(
            config, model.config.phaseMode, model.config.useAimDirection,
            model.config.contextFrames,
        )
        return (
            model,
            stateNormalizer.to(device),
            deltaNormalizer.to(device),
            sequenceConfig,
        )
    model = MotionController(
        buildControllerModelConfig(config, numBones)
    ).to(device)
    sequenceConfig = _sequenceConfig(
        config, config.phaseMode, config.useAimDirection,
        config.contextFrames,
    )
    placeholderState = MotionNormalizer(
        numBones, 6, ROOT_LOCAL_MOTION_CHANNELS
    ).to(device)
    placeholderDelta = MotionNormalizer(
        numBones, 6, ROOT_LOCAL_MOTION_CHANNELS
    ).to(device)
    return model, placeholderState, placeholderDelta, sequenceConfig
