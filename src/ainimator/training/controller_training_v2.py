"""Overfit / rollout training loop for the Goal A controller (A1).

This is the deterministic counterpart of ``training_v2.runOverfit``: the
**feasibility gate** of ROADMAP_DETERMINIST A1.  Its single objective is
to prove that the autoregressive loop *trains and rolls out without
exploding* — overfit one sequence, then roll the controller forward under
the ground-truth-derived control and check that it reproduces the clip.

It is intentionally self-contained: it consumes raw clip tensors so the
smoke test can run on synthetic data with no dataset on disk (mirrors the
``make smoke-test`` contract).  Z-normalization (truth #3) is applied to
the state *and* the deltas; the ``post_norm_stats`` health metric asserts
both are ≈ N(0, 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from ainimator.core.checkpoint_io import checkpointDir
from ainimator.core.constants.controller import PhaseMode, ROOT_LOCAL_MOTION_CHANNELS
from ainimator.core.resolved_config import writeResolvedConfig
from ainimator.core.types.controller import ControllerV2Config
from ainimator.data.controller_sequences import (
    ControllerSequenceBatch,
    ControllerSequenceConfig,
    buildControllerSequences,
)
from ainimator.health.contract import Verdict
from ainimator.health.controller_metrics import (
    controlSensitivity,
    loadControllerContracts,
    meanCollapse,
    postNormStats,
    rolloutDrift,
    rolloutDriftCurve,
)
from ainimator.model.controller_rollout import (
    RolloutResult,
    rolloutController,
)
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
    normalizeStepDelta,
)
from ainimator.training.controller_scheduled_sampling import (
    scheduledSamplingProbability,
    scheduledSamplingStep,
)

LOGGER = logging.getLogger("ainimator.training.controller")

_DEFAULT_HEALTH_PATH = Path("src/configs/health.yaml")
_CONTROL_STD_FLOOR = 1e-5
_CHECKPOINT_NAME = "controller_overfit_checkpoint.pt"
_ROLLOUT_NAME = "controller_overfit_rollout.pt"


@dataclass(frozen=True)
class ControllerTrainingConfig:
    """Knobs for the controller overfit / sanity loop.

    Attributes
    ----------
    outputDir : Path
        Where to write the checkpoint, rollout and resolved config.
    epochs : int
        Optimisation epochs (one full-batch step each).
    learningRate : float
        AdamW learning rate.
    weightDecay : float
        AdamW weight decay (0 for overfit).
    embedDim, numHeads, numLayers : int
        Controller transformer dims.
    contextFrames : int
        Autoregressive window length.
    phaseMode : PhaseMode
        Locomotor phase regime (A1 default: ``none``).
    useAimDirection : bool
        Append aim-direction control channels (A2; off for A1).
    seed : int
        RNG seed.
    device : str
        Torch device (``auto`` | ``cpu`` | ``mps`` | ``cuda``).
    logEvery : int
        Log a loss line every N epochs.
    lossWeights : ControllerLossWeights
        Loss term weights.
    scheduledSampling : float
        Target scheduled-sampling probability (A4).  ``0`` keeps the fast
        parallel teacher-forced loop; ``> 0`` enables the sequential
        scheduled-sampling loop with a linear ramp ``0 → target``.
    resumeCheckpoint : Optional[Path]
        Warm-start: load model + normalizers + control stats from this
        checkpoint instead of building fresh.  The architecture and
        normalization come from the checkpoint (arch flags are ignored).
        This is the correct way to apply scheduled sampling — fine-tune a
        teacher-forced-converged model, not train SS from scratch.
    """

    outputDir: Path
    epochs: int = 300
    learningRate: float = 1e-3
    weightDecay: float = 0.0
    embedDim: int = 128
    numHeads: int = 4
    numLayers: int = 3
    contextFrames: int = 1
    phaseMode: PhaseMode = PhaseMode.NONE
    useAimDirection: bool = False
    seed: int = 0
    device: str = "auto"
    logEvery: int = 50
    lossWeights: ControllerLossWeights = field(
        default_factory=ControllerLossWeights
    )
    scheduledSampling: float = 0.0
    resumeCheckpoint: Optional[Path] = None


@dataclass(frozen=True)
class ControllerOverfitResult:
    """Outcome of :func:`runControllerOverfit`.

    Attributes
    ----------
    finalLoss : float
        Total loss at the last epoch.
    metrics : dict[str, float]
        Controller health metrics.
    verdicts : dict[str, Verdict]
        Contract verdicts keyed by contract name.
    driftCurve : dict[int, float]
        Rollout drift vs horizon (A4 "drift vs length" curve).
    checkpointPath : Path
        Saved checkpoint location.
    rolloutPath : Path
        Saved rollout trajectory location.
    """

    finalLoss: float
    metrics: dict[str, float]
    verdicts: dict[str, Verdict]
    driftCurve: dict[int, float]
    checkpointPath: Path
    rolloutPath: Path


# ---------------------------------------------------------------------
# Helpers — device, configs, normalizers
# ---------------------------------------------------------------------
def resolveControllerDevice(requested: str) -> torch.device:
    """Resolve a torch device, honouring ``auto`` (prefers MPS/CUDA)."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def buildControllerModelConfig(
    config: ControllerTrainingConfig,
    numBones: int,
) -> ControllerV2Config:
    """Map a training config onto the model architecture config."""
    return ControllerV2Config(
        embedDim=config.embedDim,
        numHeads=config.numHeads,
        numLayers=config.numLayers,
        numBones=numBones,
        contextFrames=config.contextFrames,
        phaseMode=config.phaseMode,
        useAimDirection=config.useAimDirection,
    )


def _standardiseControl(
    control: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-standardise the control signal; return ``(norm, mean, std)``."""
    mean = control.mean(dim=0, keepdim=True)
    std = control.std(dim=0, unbiased=False, keepdim=True)
    std = std.clamp(min=_CONTROL_STD_FLOOR)
    return (control - mean) / std, mean, std


def _fitNormalizers(
    batch: ControllerSequenceBatch,
    numBones: int,
) -> tuple[MotionNormalizer, MotionNormalizer]:
    """Fit the state and delta z-normalizers (truth #3).

    Uses ``ROOT_LOCAL_MOTION_CHANNELS`` (4) for the global branch so the
    normalizer matches the root-local motion representation
    (ROADMAP_DETERMINIST §2.2.a).
    """
    stateNormalizer = MotionNormalizer(
        numBones=numBones,
        motionChannels=6,
        globalChannels=ROOT_LOCAL_MOTION_CHANNELS,
    )
    stateNormalizer.fitFromTensors(
        boneSamples=[batch.boneWindow], globalSamples=[batch.globalWindow]
    )
    deltaNormalizer = MotionNormalizer(
        numBones=numBones,
        motionChannels=6,
        globalChannels=ROOT_LOCAL_MOTION_CHANNELS,
    )
    deltaNormalizer.fitFromTensors(
        boneSamples=[batch.targetBoneDelta],
        globalSamples=[batch.targetGlobalDelta],
    )
    return stateNormalizer, deltaNormalizer


# ---------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------
def _forwardLosses(
    model: MotionController,
    batch: ControllerSequenceBatch,
    tensors: dict[str, torch.Tensor],
    controlNorm: torch.Tensor,
    deltaNormalizer: MotionNormalizer,
    weights: ControllerLossWeights,
) -> ControllerLossResult:
    """Forward + combined controller loss over the full batch."""
    output = model(
        tensors["normBoneWindow"],
        controlNorm,
        globalWindow=tensors["normGlobalWindow"],
        phase=batch.phase,
    )
    velocity = velocityDeltaLoss(
        output.boneDelta, tensors["normBoneDelta"]
    ) + velocityDeltaLoss(output.globalDelta, tensors["normGlobalDelta"])
    rawBoneDelta, rawGlobalDelta = denormalizeStepDelta(
        deltaNormalizer, output.boneDelta, output.globalDelta
    )
    predictedNextBone = batch.boneWindow[:, -1, :, :] + rawBoneDelta
    geodesic = geodesicRotationLoss(predictedNextBone, batch.targetBoneNext)
    footContact = _footContactTerm(
        batch, predictedNextBone, rawGlobalDelta, weights
    )
    return combinedControllerLoss(
        velocity, geodesic, weights, footContact=footContact
    )


def _footContactTerm(
    batch: ControllerSequenceBatch,
    predictedNextBone: torch.Tensor,
    rawGlobalDelta: torch.Tensor | None,
    weights: ControllerLossWeights,
) -> torch.Tensor | None:
    """Compute the anti-skating loss when enabled and data is present."""
    if weights.footContact <= 0.0 or batch.contactTarget is None:
        return None
    if rawGlobalDelta is None:
        return None
    return footContactStepLoss(
        predictedNextBone,
        batch.boneWindow[:, -1, :, :],
        rawGlobalDelta,
        batch.contactTarget,
    )


def _trainLoop(
    model: MotionController,
    batch: ControllerSequenceBatch,
    tensors: dict[str, torch.Tensor],
    controlNorm: torch.Tensor,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    config: ControllerTrainingConfig,
) -> float:
    """Overfit loop; return the final total loss.

    Uses the fast parallel teacher-forced step by default; when
    ``config.scheduledSampling > 0`` it runs the sequential
    scheduled-sampling step with a per-epoch ramped probability (A4).
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learningRate,
        weight_decay=config.weightDecay,
    )
    lastLoss = float("nan")
    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        result = _trainStep(
            model, batch, tensors, controlNorm, stateNormalizer,
            deltaNormalizer, config, epoch,
        )
        result.total.backward()
        optimizer.step()
        lastLoss = float(result.total.item())
        if epoch % config.logEvery == 0 or epoch == config.epochs - 1:
            LOGGER.info("epoch %d — loss %.6f", epoch, lastLoss)
    return lastLoss


def _trainStep(
    model: MotionController,
    batch: ControllerSequenceBatch,
    tensors: dict[str, torch.Tensor],
    controlNorm: torch.Tensor,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    config: ControllerTrainingConfig,
    epoch: int,
) -> ControllerLossResult:
    """Dispatch to the teacher-forced or scheduled-sampling step."""
    if config.scheduledSampling <= 0.0:
        return _forwardLosses(
            model, batch, tensors, controlNorm, deltaNormalizer,
            config.lossWeights,
        )
    probability = scheduledSamplingProbability(
        epoch, config.epochs, config.scheduledSampling
    )
    return scheduledSamplingStep(
        model, batch, stateNormalizer, deltaNormalizer, controlNorm,
        tensors, probability, config.lossWeights,
    )


def _normalizeTensors(
    batch: ControllerSequenceBatch,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
) -> dict[str, torch.Tensor]:
    """Pre-compute the normalized training tensors."""
    normBoneDelta, normGlobalDelta = normalizeStepDelta(
        deltaNormalizer, batch.targetBoneDelta, batch.targetGlobalDelta
    )
    return {
        "normBoneWindow": stateNormalizer.normalizeBone(batch.boneWindow),
        "normGlobalWindow": stateNormalizer.normalizeGlobal(
            batch.globalWindow
        ),
        "normBoneDelta": normBoneDelta,
        "normGlobalDelta": normGlobalDelta,
    }


# ---------------------------------------------------------------------
# Evaluation (health metrics + contracts + rollout)
# ---------------------------------------------------------------------
def _rolloutFromClip(
    model: MotionController,
    batch: ControllerSequenceBatch,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    controlNorm: torch.Tensor,
    clipRootTranslation: torch.Tensor,
) -> RolloutResult:
    """Roll out from the first window under the GT-derived control.

    ``clipRootTranslation`` provides the world-space XYZ seed needed by
    the integration step in :func:`rolloutController`.
    """
    window = model.config.contextFrames
    seedBone = batch.boneWindow[:1]
    # globalWindow is now root-local motion deltas (N, K, 4).
    seedRootLocalMotion = batch.globalWindow[:1]
    # Seed world-space positions: first K frames of the clip.
    seedRootTranslation = clipRootTranslation[:window].unsqueeze(0)
    controlSequence = controlNorm.unsqueeze(0)
    phaseSequence = (
        None if batch.phase is None else batch.phase.unsqueeze(0)
    )
    return rolloutController(
        model,
        stateNormalizer,
        deltaNormalizer,
        seedBone,
        seedRootTranslation,
        controlSequence,
        phaseSequence=phaseSequence,
        seedRootLocalMotion=seedRootLocalMotion,
    )


def _groundTruthTrajectory(
    batch: ControllerSequenceBatch,
    clipRootTranslation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the GT trajectory (world-space) aligned with the rollout.

    Returns bone and root tensors of shape ``(1, K + N, ...)``.
    The root is the absolute world-space XYZ trajectory rebuilt by
    integrating the root-local motion deltas from the seed origin.
    """
    window = batch.boneWindow.shape[1]
    seedBone = batch.boneWindow[0]
    bone = torch.cat([seedBone, batch.targetBoneNext], dim=0).unsqueeze(0)

    # Seed root positions (K frames of absolute world XYZ).
    seedRoot = clipRootTranslation[:window]  # (K, 3)

    # Integrate the per-transition local deltas from the seed origin.
    # ``targetGlobalDelta`` holds the local delta at each *next* frame.
    from ainimator.geometry.root_local import (
        pelvisYawFromRot6d,
        rootLocalDeltasToAbsolute,
    )

    seedPelvisRot6d = batch.boneWindow[0, -1, 0, :]  # pelvis of last seed
    seedYaw = pelvisYawFromRot6d(seedPelvisRot6d.unsqueeze(0)).squeeze(0)
    seedOrigin = seedRoot[-1]  # last seed frame's world position

    # localDeltas: (N, 4) from the sequence batch.
    localDeltas = batch.targetGlobalDelta  # (N, 4)
    rootAbsolute, _ = rootLocalDeltasToAbsolute(
        seedOrigin, seedYaw, localDeltas
    )  # (N, 3)

    root = torch.cat(
        [seedRoot, rootAbsolute], dim=0
    ).unsqueeze(0)  # (1, K+N, 3)
    return bone, root


def _evaluate(
    model: MotionController,
    batch: ControllerSequenceBatch,
    tensors: dict[str, torch.Tensor],
    controlNorm: torch.Tensor,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    healthPath: Path,
    clipRootTranslation: torch.Tensor,
) -> tuple[
    dict[str, float], dict[str, Verdict], dict[int, float], RolloutResult
]:
    """Compute controller metrics, evaluate contracts, run a rollout."""
    model.eval()
    output = model(
        tensors["normBoneWindow"],
        controlNorm,
        globalWindow=tensors["normGlobalWindow"],
        phase=batch.phase,
    )
    rank, sim = meanCollapse(output)
    sensitivity = controlSensitivity(
        model,
        tensors["normBoneWindow"],
        controlNorm,
        globalWindow=tensors["normGlobalWindow"],
        phase=batch.phase,
    )
    rollout = _rolloutFromClip(
        model, batch, stateNormalizer, deltaNormalizer, controlNorm,
        clipRootTranslation,
    )
    gtBone, gtRoot = _groundTruthTrajectory(batch, clipRootTranslation)
    drift = rolloutDrift(rollout, gtBone, gtRoot)
    driftCurve = rolloutDriftCurve(
        rollout, gtBone, gtRoot, _driftHorizons(rollout.rotation6d.shape[1])
    )
    postNorm = max(
        postNormStats(tensors["normBoneWindow"]),
        postNormStats(tensors["normBoneDelta"]),
    )
    metrics = {
        "control_sensitivity": sensitivity,
        "mean_collapse_rank": rank,
        "mean_collapse_sim": sim,
        "rollout_drift": drift,
        "post_norm_stats": postNorm,
    }
    verdicts = _evaluateContracts(metrics, healthPath)
    return metrics, verdicts, driftCurve, rollout


def _driftHorizons(totalFrames: int) -> list[int]:
    """Quartile horizons for the drift-vs-length curve (A4 health report)."""
    quarters = [totalFrames // 4, totalFrames // 2,
                (3 * totalFrames) // 4, totalFrames]
    return sorted({max(1, horizon) for horizon in quarters})


def _evaluateContracts(
    metrics: dict[str, float],
    healthPath: Path,
) -> dict[str, Verdict]:
    """Evaluate the controller contracts against the metrics."""
    contracts = loadControllerContracts(healthPath)
    verdicts: dict[str, Verdict] = {}
    for name, contract in contracts.items():
        verdicts[name] = contract.evaluate(metrics).verdict
    return verdicts


# ---------------------------------------------------------------------
# Component preparation (fresh vs warm-start / resume)
# ---------------------------------------------------------------------
_PreparedComponents = tuple[
    MotionController,
    MotionNormalizer,
    MotionNormalizer,
    torch.Tensor,
    torch.Tensor,
    ControllerSequenceBatch,
    torch.Tensor,
    torch.Tensor,  # clipRootTranslation on device (for rollout seed)
]


def _prepareFresh(
    config: ControllerTrainingConfig,
    clipRotation6d: torch.Tensor,
    clipRootTranslation: torch.Tensor,
    device: torch.device,
) -> _PreparedComponents:
    """Build a fresh model + normalizers fitted on the clip."""
    numBones = int(clipRotation6d.shape[-2])
    sequenceConfig = ControllerSequenceConfig(
        contextFrames=config.contextFrames,
        useAimDirection=config.useAimDirection,
        emitPhase=config.phaseMode is not PhaseMode.NONE,
        emitContacts=config.lossWeights.footContact > 0.0,
    )
    clipRootOnDevice = clipRootTranslation.to(device)
    batch = buildControllerSequences(
        clipRotation6d.to(device), clipRootOnDevice,
        sequenceConfig,
    )
    stateNormalizer, deltaNormalizer = _fitNormalizers(batch, numBones)
    model = MotionController(
        buildControllerModelConfig(config, numBones)
    ).to(device)
    controlNorm, controlMean, controlStd = _standardiseControl(batch.control)
    return (
        model,
        stateNormalizer.to(device),
        deltaNormalizer.to(device),
        controlMean,
        controlStd,
        batch,
        controlNorm,
        clipRootOnDevice,
    )


def _prepareResumed(
    config: ControllerTrainingConfig,
    clipRotation6d: torch.Tensor,
    clipRootTranslation: torch.Tensor,
    device: torch.device,
) -> _PreparedComponents:
    """Warm-start: load model + normalizers + control stats (A4 fine-tune).

    Architecture and normalization come from the checkpoint, so the
    fine-tune is numerically consistent with the resumed model.  The
    sequence layout (context / phase / aim) is derived from the loaded
    model config, not the training-config arch flags.
    """
    assert config.resumeCheckpoint is not None
    model, stateNormalizer, deltaNormalizer, controlMean, controlStd = (
        loadControllerCheckpoint(config.resumeCheckpoint, device)
    )
    model = model.to(device)
    sequenceConfig = ControllerSequenceConfig(
        contextFrames=model.config.contextFrames,
        useAimDirection=model.config.useAimDirection,
        emitPhase=model.config.phaseMode is not PhaseMode.NONE,
        emitContacts=config.lossWeights.footContact > 0.0,
    )
    clipRootOnDevice = clipRootTranslation.to(device)
    batch = buildControllerSequences(
        clipRotation6d.to(device), clipRootOnDevice,
        sequenceConfig,
    )
    controlNorm = (
        batch.control - controlMean.to(device)
    ) / controlStd.to(device)
    return (
        model,
        stateNormalizer.to(device),
        deltaNormalizer.to(device),
        controlMean,
        controlStd,
        batch,
        controlNorm,
        clipRootOnDevice,
    )


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def runControllerOverfit(
    clipRotation6d: torch.Tensor,
    clipRootTranslation: torch.Tensor,
    config: ControllerTrainingConfig,
    healthPath: Path = _DEFAULT_HEALTH_PATH,
) -> ControllerOverfitResult:
    """Overfit one clip, roll it out, and gate on the A1 contracts."""
    torch.manual_seed(config.seed)
    device = resolveControllerDevice(config.device)
    if config.resumeCheckpoint is not None:
        components = _prepareResumed(
            config, clipRotation6d, clipRootTranslation, device
        )
    else:
        components = _prepareFresh(
            config, clipRotation6d, clipRootTranslation, device
        )
    (
        model,
        stateNormalizer,
        deltaNormalizer,
        controlMean,
        controlStd,
        batch,
        controlNorm,
        clipRootOnDevice,
    ) = components
    tensors = _normalizeTensors(batch, stateNormalizer, deltaNormalizer)

    finalLoss = _trainLoop(
        model, batch, tensors, controlNorm, stateNormalizer,
        deltaNormalizer, config,
    )

    config.outputDir.mkdir(parents=True, exist_ok=True)
    metrics, verdicts, driftCurve, rollout = _evaluate(
        model,
        batch,
        tensors,
        controlNorm,
        stateNormalizer,
        deltaNormalizer,
        healthPath,
        clipRootOnDevice,
    )
    checkpointPath = saveControllerCheckpoint(
        model,
        stateNormalizer,
        deltaNormalizer,
        controlMean,
        controlStd,
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


# ---------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------
def saveControllerCheckpoint(
    model: MotionController,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    controlMean: torch.Tensor,
    controlStd: torch.Tensor,
    outputDir: Path,
) -> Path:
    """Persist the controller, normalizers and control stats."""
    path = checkpointDir(outputDir) / _CHECKPOINT_NAME
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": _controllerConfigToDict(model.config),
            "state_normalizer_config": stateNormalizer.configToDict(),
            "state_normalizer_state": stateNormalizer.state_dict(),
            "delta_normalizer_config": deltaNormalizer.configToDict(),
            "delta_normalizer_state": deltaNormalizer.state_dict(),
            "control_mean": controlMean.detach().cpu(),
            "control_std": controlStd.detach().cpu(),
        },
        path,
    )
    return path


def loadControllerCheckpoint(
    path: Path,
    device: torch.device | None = None,
) -> tuple[
    MotionController, MotionNormalizer, MotionNormalizer, torch.Tensor,
    torch.Tensor,
]:
    """Inverse of :func:`saveControllerCheckpoint`."""
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = MotionController(
        _controllerConfigFromDict(payload["model_config"])
    )
    model.load_state_dict(payload["model_state"])
    stateNormalizer = MotionNormalizer.fromConfigDict(
        payload["state_normalizer_config"]
    )
    stateNormalizer.load_state_dict(payload["state_normalizer_state"])
    deltaNormalizer = MotionNormalizer.fromConfigDict(
        payload["delta_normalizer_config"]
    )
    deltaNormalizer.load_state_dict(payload["delta_normalizer_state"])
    return (
        model,
        stateNormalizer,
        deltaNormalizer,
        payload["control_mean"],
        payload["control_std"],
    )


def _controllerConfigToDict(config: ControllerV2Config) -> dict[str, object]:
    """Serialise a :class:`ControllerV2Config` to a JSON-friendly dict."""
    return {
        "embedDim": config.embedDim,
        "numHeads": config.numHeads,
        "numLayers": config.numLayers,
        "numBones": config.numBones,
        "motionChannels": config.motionChannels,
        "globalChannels": config.globalChannels,
        "contextFrames": config.contextFrames,
        "phaseMode": config.phaseMode.value,
        "useAimDirection": config.useAimDirection,
        "useFilmConditioning": config.useFilmConditioning,
        "usePerBlockFilm": config.usePerBlockFilm,
        "filmInitStd": config.filmInitStd,
        "dropout": config.dropout,
        "maxFrames": config.maxFrames,
        "styleLatentEnabled": config.styleLatentEnabled,
        "styleLatentDim": config.styleLatentDim,
    }


def _controllerConfigFromDict(payload: dict[str, object]) -> ControllerV2Config:
    """Inverse of :func:`_controllerConfigToDict`."""
    data = dict(payload)
    data["phaseMode"] = PhaseMode(str(data["phaseMode"]))
    return ControllerV2Config(**data)  # type: ignore[arg-type]


def _saveRollout(rollout: RolloutResult, outputDir: Path) -> Path:
    """Persist the rolled-out trajectory tensors."""
    path = checkpointDir(outputDir) / _ROLLOUT_NAME
    torch.save(
        {
            "rotation6d": rollout.rotation6d.detach().cpu(),
            "rootTranslation": rollout.rootTranslation.detach().cpu(),
            "rootLocalMotion": rollout.rootLocalMotion.detach().cpu(),
        },
        path,
    )
    return path
