"""Generalization training for the Goal A controller (phase A6/A7).

A0–A5 were validated in **overfit** (1, then 16 clips): "the rollout
reproduces the clip it was trained on".  A6 is the first regime that tests
**generalization** — train on many clips, then judge on clips the model
**never saw** (ROADMAP_DETERMINIST A6).

This module is also the backing implementation of the ``full`` profile
of the unified ``train_controller_v2`` CLI (phase A7).

Two differences with the archived multi-clip path:

* **Held-out split.** Training consumes only the train clips; the honest
  metrics (reconstruction, drift, control sensitivity) are measured on a
  disjoint held-out set whose control is derived from *its own* ground
  truth.
* **Clip-minibatched loop.** The multi-clip path concatenated every clip
  into one full-batch step — fine at 16 clips, an OOM at ~1000.  Here
  each optimizer step samples ``clipBatchSize`` clips, builds their
  sequences on device and discards them, so memory is bounded by the
  minibatch, not by the dataset size.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from ainimator.core.constants.controller import (
    PhaseMode,
    ROOT_LOCAL_MOTION_CHANNELS,
)
from ainimator.core.resolved_config import writeResolvedConfig
from ainimator.data.controller_sequences import (
    ControllerSequenceBatch,
    ControllerSequenceConfig,
    buildControllerSequences,
)
from ainimator.geometry.root_local import absoluteToRootLocalDeltas
from ainimator.health.contract import Verdict
from ainimator.health.controller_metrics import (
    controlSensitivity,
    meanCollapse,
    postNormStats,
    rolloutDrift,
    rolloutDriftCurve,
)
from ainimator.model.controller_rollout import rolloutController
from ainimator.model.controller_v2 import MotionController
from ainimator.model.losses_controller_v2 import geodesicRotationLoss
from ainimator.model.motion_normalizer import (
    MotionNormalizer,
    denormalizeStepDelta,
)
from ainimator.training.controller_training_v2 import (
    ControllerTrainingConfig,
    _DEFAULT_HEALTH_PATH,
    _driftHorizons,
    _evaluateContracts,
    _forwardLosses,
    _groundTruthTrajectory,
    _normalizeTensors,
    _rolloutFromClip,
    buildControllerModelConfig,
    resolveControllerDevice,
    saveControllerCheckpoint,
)

LOGGER = logging.getLogger("ainimator.training.controller_generalization")

Clip = tuple[torch.Tensor, torch.Tensor]
_CONTROL_STD_FLOOR = 1e-5


# ---------------------------------------------------------------------
# Clip-batch utilities (inlined from archived controller_multiclip_v2)
# ---------------------------------------------------------------------
def _sequenceConfig(
    config: ControllerTrainingConfig,
    phaseMode: PhaseMode,
    useAim: bool,
    contextFrames: int,
) -> ControllerSequenceConfig:
    """Build the sequence config shared by every clip.

    Parameters
    ----------
    config : ControllerTrainingConfig
        Training config (used for loss-weight flags).
    phaseMode : PhaseMode
        Locomotor phase regime.
    useAim : bool
        Whether to emit aim-direction control channels.
    contextFrames : int
        Autoregressive context window length.

    Returns
    -------
    ControllerSequenceConfig
        Shared config for all clip sequence builds.
    """
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
    """Build one autoregressive batch per clip (on ``device``).

    Parameters
    ----------
    clips : list[Clip]
        List of (rotation6d, rootTranslation) tensor pairs.
    sequenceConfig : ControllerSequenceConfig
        Shared sequence config.
    device : torch.device
        Target device for tensors.

    Returns
    -------
    list[ControllerSequenceBatch]
        One batch per clip.
    """
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
    """Concatenate per-clip batches into one merged batch.

    Parameters
    ----------
    batches : list[ControllerSequenceBatch]
        Per-clip batches to merge along the first (sample) axis.

    Returns
    -------
    ControllerSequenceBatch
        Merged batch containing all clips.
    """
    phaseAll = all(b.phase is not None for b in batches)
    contactAll = all(b.contactTarget is not None for b in batches)
    return ControllerSequenceBatch(
        boneWindow=torch.cat([b.boneWindow for b in batches], dim=0),
        globalWindow=torch.cat(
            [b.globalWindow for b in batches], dim=0
        ),
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
            torch.cat(
                [b.phase for b in batches], dim=0  # type: ignore[arg-type]
            )
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


def _perClipDrift(
    model: MotionController,
    batches: list[ControllerSequenceBatch],
    clipRootTranslations: list[torch.Tensor],
    controlMean: torch.Tensor,
    controlStd: torch.Tensor,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
) -> tuple[float, dict[int, float]]:
    """Mean rollout drift over clips; drift curve on the longest clip.

    Parameters
    ----------
    model : MotionController
        Trained controller (eval mode expected).
    batches : list[ControllerSequenceBatch]
        Per-clip sequence batches.
    clipRootTranslations : list[torch.Tensor]
        GT root translation per clip (for absolute-world reconstruction).
    controlMean, controlStd : torch.Tensor
        Global control statistics for normalization.
    stateNormalizer, deltaNormalizer : MotionNormalizer
        State / delta z-normalizers.

    Returns
    -------
    tuple[float, dict[int, float]]
        Mean drift across clips, and drift-vs-horizon curve of the
        longest clip.
    """
    drifts: list[float] = []
    curve: dict[int, float] = {}
    longest = max(
        range(len(batches)),
        key=lambda index: batches[index].numTransitions,
    )
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


@dataclass(frozen=True)
class GeneralizationResult:
    """Outcome of :func:`runControllerGeneralization`.

    Attributes
    ----------
    finalLoss : float
        Mean train loss at the last epoch.
    trainMetrics : dict[str, float]
        Metrics on a sample of *train* clips (the reference to compare the
        held-out numbers against).
    heldOutMetrics : dict[str, float]
        Metrics on the disjoint **held-out** clips — the honest judge.
    verdicts : dict[str, Verdict]
        Controller contracts evaluated on the held-out metrics.
    driftCurve : dict[int, float]
        Held-out drift vs horizon (longest held-out clip).
    checkpointPath : Path
        Saved checkpoint location.
    numTrainClips, numHeldOutClips : int
        Split sizes actually used.
    """

    finalLoss: float
    trainMetrics: dict[str, float]
    heldOutMetrics: dict[str, float]
    verdicts: dict[str, Verdict]
    driftCurve: dict[int, float]
    checkpointPath: Path
    numTrainClips: int
    numHeldOutClips: int


# ---------------------------------------------------------------------
# Streaming normalizer / control statistics (memory-bounded for large N)
# ---------------------------------------------------------------------
def _fitStateDeltaNormalizers(
    trainClips: list[Clip],
    numBones: int,
) -> tuple[MotionNormalizer, MotionNormalizer]:
    """Fit state + delta z-normalizers from the raw clip frames.

    Raw clips are small (~0.2 MB each); fitting on the clip frames and
    their consecutive differences matches the windowed-sequence statistics
    up to negligible clip-boundary effects, and avoids materialising every
    clip's sequence batch at once.
    """
    # Root-local motion deltas per clip (F, 4) — converted from absolute GT.
    rootLocalByClip = [
        absoluteToRootLocalDeltas(clip[1], clip[0]) for clip in trainClips
    ]
    stateNormalizer = MotionNormalizer(
        numBones, 6, ROOT_LOCAL_MOTION_CHANNELS
    )
    stateNormalizer.fitFromTensors(
        boneSamples=[clip[0] for clip in trainClips],
        globalSamples=rootLocalByClip,
    )
    deltaNormalizer = MotionNormalizer(
        numBones, 6, ROOT_LOCAL_MOTION_CHANNELS
    )
    deltaNormalizer.fitFromTensors(
        boneSamples=[clip[0][1:] - clip[0][:-1] for clip in trainClips],
        globalSamples=[rm[1:] - rm[:-1] for rm in rootLocalByClip],
    )
    return stateNormalizer, deltaNormalizer


def _fitControlStats(
    trainClips: list[Clip],
    sequenceConfig: ControllerSequenceConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean / std of the GT-derived control, streamed over train clips.

    The control signal lives in the sequence batch, so we build each
    clip's sequences once, keep only its (tiny) control tensor, and
    discard the rest before moving on — memory stays O(one clip).
    """
    controls: list[torch.Tensor] = []
    for rotation6d, rootTranslation in trainClips:
        batch = buildControllerSequences(
            rotation6d.to(device), rootTranslation.to(device), sequenceConfig
        )
        controls.append(batch.control.detach().cpu())
    merged = torch.cat(controls, dim=0)
    mean = merged.mean(dim=0, keepdim=True)
    std = merged.std(dim=0, unbiased=False, keepdim=True)
    return mean.to(device), std.clamp(min=_CONTROL_STD_FLOOR).to(device)


# ---------------------------------------------------------------------
# Clip-minibatched training
# ---------------------------------------------------------------------
def _mergedMinibatch(
    trainClips: list[Clip],
    indices: list[int],
    sequenceConfig: ControllerSequenceConfig,
    device: torch.device,
) -> ControllerSequenceBatch:
    """Build + concat the sequences of the selected clips (on device)."""
    chunk = [trainClips[index] for index in indices]
    batches = _buildClipBatches(chunk, sequenceConfig, device)
    return _concatClipBatches(batches)


def _trainEpoch(
    model: MotionController,
    trainClips: list[Clip],
    order: list[int],
    sequenceConfig: ControllerSequenceConfig,
    optimizer: torch.optim.Optimizer,
    norms: tuple[MotionNormalizer, MotionNormalizer],
    controlStats: tuple[torch.Tensor, torch.Tensor],
    config: ControllerTrainingConfig,
    clipBatchSize: int,
    device: torch.device,
) -> float:
    """Run one clip-minibatched epoch; return the mean step loss."""
    stateNormalizer, deltaNormalizer = norms
    controlMean, controlStd = controlStats
    total = 0.0
    steps = 0
    for start in range(0, len(order), clipBatchSize):
        indices = order[start : start + clipBatchSize]
        merged = _mergedMinibatch(
            trainClips, indices, sequenceConfig, device
        )
        tensors = _normalizeTensors(merged, stateNormalizer, deltaNormalizer)
        controlNorm = (merged.control - controlMean) / controlStd
        optimizer.zero_grad()
        result = _forwardLosses(
            model, merged, tensors, controlNorm, deltaNormalizer,
            config.lossWeights,
        )
        result.total.backward()
        optimizer.step()
        total += float(result.total.item())
        steps += 1
    return total / max(steps, 1)


def _train(
    model: MotionController,
    trainClips: list[Clip],
    sequenceConfig: ControllerSequenceConfig,
    norms: tuple[MotionNormalizer, MotionNormalizer],
    controlStats: tuple[torch.Tensor, torch.Tensor],
    config: ControllerTrainingConfig,
    clipBatchSize: int,
    device: torch.device,
) -> float:
    """Clip-minibatched training loop; return the final mean epoch loss."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learningRate,
        weight_decay=config.weightDecay,
    )
    rng = random.Random(config.seed)
    order = list(range(len(trainClips)))
    lastLoss = float("nan")
    model.train()
    for epoch in range(config.epochs):
        rng.shuffle(order)
        lastLoss = _trainEpoch(
            model, trainClips, order, sequenceConfig, optimizer, norms,
            controlStats, config, clipBatchSize, device,
        )
        if epoch % config.logEvery == 0 or epoch == config.epochs - 1:
            LOGGER.info("epoch %d — mean loss %.6f", epoch, lastLoss)
    return lastLoss


# ---------------------------------------------------------------------
# Evaluation on a clip set (held-out, or a train sample for reference)
# ---------------------------------------------------------------------
@torch.no_grad()
def _reconstructionGeodesic(
    model: MotionController,
    merged: ControllerSequenceBatch,
    tensors: dict[str, torch.Tensor],
    controlNorm: torch.Tensor,
    deltaNormalizer: MotionNormalizer,
) -> float:
    """Teacher-forced per-step geodesic error (radians) on a batch."""
    output = model(
        tensors["normBoneWindow"], controlNorm,
        globalWindow=tensors["normGlobalWindow"], phase=merged.phase,
    )
    rawBoneDelta, _ = denormalizeStepDelta(
        deltaNormalizer, output.boneDelta, output.globalDelta
    )
    predictedNextBone = merged.boneWindow[:, -1, :, :] + rawBoneDelta
    return float(
        geodesicRotationLoss(predictedNextBone, merged.targetBoneNext).item()
    )


@torch.no_grad()
def _evaluateClips(
    model: MotionController,
    clips: list[Clip],
    sequenceConfig: ControllerSequenceConfig,
    norms: tuple[MotionNormalizer, MotionNormalizer],
    controlStats: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, float], dict[int, float]]:
    """Metrics on a (small) clip set; reuses the controller probes.

    Runs under ``torch.no_grad()``: evaluation concatenates *every* clip
    of the set into one forward, so retaining the autograd graph over all
    of them (one per train-sample / held-out clip) blows the MPS budget at
    large N.  No metric here needs gradients (the probes are forward-only),
    so disabling grad keeps peak memory bounded by the activations alone.
    """
    stateNormalizer, deltaNormalizer = norms
    controlMean, controlStd = controlStats
    batches = _buildClipBatches(clips, sequenceConfig, device)
    merged = _concatClipBatches(batches)
    tensors = _normalizeTensors(merged, stateNormalizer, deltaNormalizer)
    controlNorm = (merged.control - controlMean) / controlStd
    model.eval()
    output = model(
        tensors["normBoneWindow"], controlNorm,
        globalWindow=tensors["normGlobalWindow"], phase=merged.phase,
    )
    rank, sim = meanCollapse(output)
    sensitivity = controlSensitivity(
        model, tensors["normBoneWindow"], controlNorm,
        globalWindow=tensors["normGlobalWindow"], phase=merged.phase,
    )
    clipRootTranslations = [clip[1].to(device) for clip in clips]
    drift, curve = _perClipDrift(
        model, batches, clipRootTranslations, controlMean, controlStd,
        stateNormalizer, deltaNormalizer,
    )
    metrics = {
        "control_sensitivity": sensitivity,
        "mean_collapse_rank": rank,
        "mean_collapse_sim": sim,
        "rollout_drift": drift,
        "reconstruction_geodesic": _reconstructionGeodesic(
            model, merged, tensors, controlNorm, deltaNormalizer
        ),
        "post_norm_stats": max(
            postNormStats(tensors["normBoneWindow"]),
            postNormStats(tensors["normBoneDelta"]),
        ),
    }
    return metrics, curve


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def runControllerGeneralization(
    trainClips: list[Clip],
    heldOutClips: list[Clip],
    config: ControllerTrainingConfig,
    clipBatchSize: int = 8,
    evalSampleClips: int = 32,
    healthPath: Path = _DEFAULT_HEALTH_PATH,
) -> GeneralizationResult:
    """Train on ``trainClips``; judge generalization on ``heldOutClips``."""
    if len(trainClips) < 2:
        raise ValueError("generalization needs at least 2 train clips.")
    if len(heldOutClips) < 2:
        raise ValueError("generalization needs at least 2 held-out clips.")
    torch.manual_seed(config.seed)
    device = resolveControllerDevice(config.device)
    numBones = int(trainClips[0][0].shape[-2])

    sequenceConfig = _sequenceConfig(
        config, config.phaseMode, config.useAimDirection,
        config.contextFrames,
    )
    stateNormalizer, deltaNormalizer = _fitStateDeltaNormalizers(
        trainClips, numBones
    )
    norms = (stateNormalizer.to(device), deltaNormalizer.to(device))
    controlStats = _fitControlStats(trainClips, sequenceConfig, device)

    model = MotionController(
        buildControllerModelConfig(config, numBones)
    ).to(device)
    finalLoss = _train(
        model, trainClips, sequenceConfig, norms, controlStats, config,
        clipBatchSize, device,
    )

    trainSample = trainClips[: max(2, min(evalSampleClips, len(trainClips)))]
    trainMetrics, _ = _evaluateClips(
        model, trainSample, sequenceConfig, norms, controlStats, device
    )
    if device.type == "mps":
        torch.mps.empty_cache()
    heldOutMetrics, driftCurve = _evaluateClips(
        model, heldOutClips, sequenceConfig, norms, controlStats, device
    )

    config.outputDir.mkdir(parents=True, exist_ok=True)
    checkpointPath = saveControllerCheckpoint(
        model, norms[0], norms[1], controlStats[0], controlStats[1],
        config.outputDir,
    )
    writeResolvedConfig(config, config.outputDir)
    return GeneralizationResult(
        finalLoss=finalLoss,
        trainMetrics=trainMetrics,
        heldOutMetrics=heldOutMetrics,
        verdicts=_evaluateContracts(heldOutMetrics, healthPath),
        driftCurve=driftCurve,
        checkpointPath=checkpointPath,
        numTrainClips=len(trainClips),
        numHeldOutClips=len(heldOutClips),
    )
