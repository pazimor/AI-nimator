"""Debug runner for AI-nimator v2 — fast end-to-end iteration.

Provides :func:`runDebug`, a thin orchestrator that wires the overfit
training loop and the v2 sampler into a single < 2-min pass on MPS:

    train a handful of steps (reduced model)
    → write a checkpoint
    → sample one generation
    → export a .dae artefact

The reduced architecture is declared in :class:`DebugTrainingConfig`,
which lives *beside* (not inside) the production configs — **no default
is mutated**.

Public surface
--------------
* :class:`DebugTrainingConfig` — reduced-dim frozen config for debug runs.
* :func:`buildDebugConfig` — build a :class:`V2TrainingConfig` from a
  :class:`DebugTrainingConfig`.
* :func:`runDebug` — end-to-end orchestrator (train → checkpoint →
  generate).
* :class:`DebugRunResult` — bundle of artefact paths produced.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ainimator.training.training_v2 import (
    V2TrainingConfig,
    loadDatasetSample,
    buildTrainingComponents,
    runOverfit,
)
from ainimator.core.checkpoint_io import checkpointDir
from ainimator.model.sampler_v2 import DDIMSamplerV2

LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Debug-specific constants (do NOT touch production defaults)
# ------------------------------------------------------------------

# Reduced model dims — small enough to run < 2 min on MPS.
_DEBUG_ENCODER_HIDDEN_DIM: int = 64
_DEBUG_ENCODER_NUM_LAYERS: int = 1
_DEBUG_ENCODER_NUM_HEADS: int = 4

_DEBUG_DENOISER_EMBED_DIM: int = 64
_DEBUG_DENOISER_NUM_LAYERS: int = 1
_DEBUG_DENOISER_NUM_HEADS: int = 4

_DEBUG_MAX_FRAMES: int = 32
_DEBUG_EPOCHS: int = 50
_DEBUG_DIFFUSION_STEPS_TRAINING: int = 50
_DEBUG_GENERATION_FRAMES: int = 32
_DEBUG_GENERATION_DDIM_STEPS: int = 10
_DEBUG_HEALTH_EVERY_STEPS: int = 1  # every step — max probe frequency


@dataclass(frozen=True)
class DebugTrainingConfig:
    """Reduced configuration for the debug run.

    All fields are separate from the production config so existing
    defaults are never mutated.  The debug profile is the only place
    these small dims live.

    Attributes
    ----------
    datasetRoot : Path
        Preprocessed dataset root (same as production).
    tokenizerDir : Path
        Custom BPE tokenizer directory.
    outputDir : Path
        Where to write checkpoints and the generated artefact.
    sampleLinkIndex : int
        Which (motion, text) pair to use.  Default 0.
    seed : int
        RNG seed.
    device : str
        Torch device string — ``"auto"`` delegates to
        :func:`~ainimator.training.training_v2.resolveDevice`.
    """

    datasetRoot: Path
    tokenizerDir: Path
    outputDir: Path
    sampleLinkIndex: int = 0
    seed: int = 0
    device: str = "auto"


def buildDebugConfig(
    debug: DebugTrainingConfig,
) -> V2TrainingConfig:
    """Translate a :class:`DebugTrainingConfig` to a :class:`V2TrainingConfig`.

    All reduced dims come from module-level constants so the diff
    between debug and production is visible in one place.

    Parameters
    ----------
    debug : DebugTrainingConfig
        High-level debug knobs.

    Returns
    -------
    V2TrainingConfig
        Ready-to-use training config with reduced dims and
        ``healthEverySteps=1``.
    """
    return V2TrainingConfig(
        datasetRoot=debug.datasetRoot,
        tokenizerDir=debug.tokenizerDir,
        outputDir=debug.outputDir,
        sampleLinkIndex=debug.sampleLinkIndex,
        epochs=_DEBUG_EPOCHS,
        learningRate=1e-4,
        weightDecay=0.0,
        minSnrGamma=5.0,
        velocityXyzWeight=0.0,
        diffusionStepsTraining=_DEBUG_DIFFUSION_STEPS_TRAINING,
        scheduleType="cosine",
        predictionMode="v",
        encoderHiddenDim=_DEBUG_ENCODER_HIDDEN_DIM,
        encoderNumLayers=_DEBUG_ENCODER_NUM_LAYERS,
        encoderNumHeads=_DEBUG_ENCODER_NUM_HEADS,
        denoiserEmbedDim=_DEBUG_DENOISER_EMBED_DIM,
        denoiserNumLayers=_DEBUG_DENOISER_NUM_LAYERS,
        denoiserNumHeads=_DEBUG_DENOISER_NUM_HEADS,
        maxFrames=_DEBUG_MAX_FRAMES,
        framesPerStep=0,
        seed=debug.seed,
        logEvery=10,
        device=debug.device,
        dropout=0.0,
        condMaskProb=0.0,
        useFilmConditioning=True,
        filmDropout=0.0,
        usePerBlockFilm=True,
        useNullEmbedding=True,
        healthEnabled=True,
        healthEverySteps=_DEBUG_HEALTH_EVERY_STEPS,
    )


@dataclass(frozen=True)
class DebugRunResult:
    """Bundle of artefact paths produced by :func:`runDebug`.

    Attributes
    ----------
    checkpointPath : Path
        The ``.pt`` checkpoint written after training.
    generationPath : Path
        The ``.dae`` Collada artefact from the single generation.
    healthJsonlPath : Path
        The health JSONL written during training (one line per step).
    elapsedSeconds : float
        Wall-clock time of the full run (training + generation).
    """

    checkpointPath: Path
    generationPath: Path
    healthJsonlPath: Path
    elapsedSeconds: float


def runDebug(
    debug: DebugTrainingConfig,
) -> DebugRunResult:
    """Run the full debug pipeline end-to-end.

    Steps executed:

    1. Train ``_DEBUG_EPOCHS`` steps with the reduced model (health
       probes fire at every step — ``everySteps=1``).
    2. Write the checkpoint via the standard :func:`runOverfit` path.
    3. Generate one sample from the written checkpoint and export it
       as a ``.dae`` artefact.

    Parameters
    ----------
    debug : DebugTrainingConfig
        High-level debug settings (paths, seed, device).

    Returns
    -------
    DebugRunResult
        Paths to the produced artefacts and wall-clock time.
    """
    t0 = time.time()

    config = buildDebugConfig(debug)
    LOGGER.info(
        "[debug] Starting reduced training: "
        "epochs=%d, denoiser=%dd/%dL, schedule_steps=%d, "
        "health_every=%d, device=%s.",
        config.epochs,
        config.denoiserEmbedDim,
        config.denoiserNumLayers,
        config.diffusionStepsTraining,
        config.healthEverySteps,
        config.device,
    )

    # ---- Phase 1: train + checkpoint --------------------------------
    _components, history = runOverfit(config)
    checkpointPath = (
        checkpointDir(config.outputDir) / "v2_overfit_checkpoint.pt"
    )
    if not checkpointPath.exists():
        raise RuntimeError(
            f"[debug] Expected checkpoint not found: {checkpointPath}."
        )

    finalLoss = history[-1]["loss_total"] if history else float("nan")
    LOGGER.info(
        "[debug] Training done — final loss=%.4f, checkpoint=%s.",
        finalLoss,
        checkpointPath,
    )

    # ---- Phase 2: generate one sample ------------------------------
    generationPath = debug.outputDir / "debug_sample.dae"
    _generateDebugSample(
        config=config,
        components=_components,
        generationPath=generationPath,
    )

    # ---- Phase 3: locate health JSONL ------------------------------
    healthDir = debug.outputDir / "health"
    healthJsonls = sorted(healthDir.glob("*.jsonl"))
    healthJsonlPath = healthJsonls[0] if healthJsonls else healthDir

    elapsed = time.time() - t0
    LOGGER.info(
        "[debug] Full debug run completed in %.1fs. "
        "checkpoint=%s  generation=%s  health=%s.",
        elapsed,
        checkpointPath,
        generationPath,
        healthJsonlPath,
    )
    return DebugRunResult(
        checkpointPath=checkpointPath,
        generationPath=generationPath,
        healthJsonlPath=Path(healthJsonlPath),
        elapsedSeconds=elapsed,
    )


def _generateDebugSample(
    config: V2TrainingConfig,
    components: object,
    generationPath: Path,
) -> None:
    """Sample one motion and export it as a Collada .dae file.

    Reuses the already-trained components so we don't reload from disk.
    Imports the generation helpers lazily (they live in data/ which is
    above training/ in dependency order; the CLI L5 wires them — we
    call them here only via a carefully scoped import, which is safe
    because debug_runner is itself only called from the CLI).

    Parameters
    ----------
    config : V2TrainingConfig
        The reduced config used for training.
    components : V2TrainingComponents
        Trained components bundle from :func:`runOverfit`.
    generationPath : Path
        Output path for the .dae file.
    """
    # Late import — export/ and data/builder are not in the training
    # layer.  The debug runner is only called from cli/ (L5) so this
    # is acceptable (cli → training → … is already valid).
    from ainimator.model.sampler_v2 import DDIMSamplerV2
    from ainimator.data.builder.animation_rebuilder import AnimationRebuilder
    from ainimator.core.types import (
        AnimationSample,
        DatasetBuilderConfig,
        DatasetBuilderPaths,
        DatasetBuilderProcessing,
    )
    from ainimator.geometry.quaternion import Rotation
    from ainimator.core.constants.skeletons import (
        SMPL22_BONE_ORDER,
        SMPL24_BONE_ORDER,
    )
    from ainimator.training.training_v2 import (
        V2TrainingComponents,
        EMPTY_PROMPT,
    )
    import numpy as np

    comps: V2TrainingComponents = components  # type: ignore[assignment]
    device = comps.device

    # Load sample to get the raw prompt.
    sample = loadDatasetSample(
        config.datasetRoot, linkIndex=config.sampleLinkIndex
    )

    sampler = DDIMSamplerV2(
        comps.schedule,
        predictionMode=config.predictionMode,
    )

    comps.encoder.eval()
    comps.denoiser.eval()

    with torch.no_grad():
        condEnc = comps.tokenizer.encode(sample.rawText)
        condOut = comps.encoder(
            condEnc.inputIds.to(device),
            condEnc.attentionMask.to(device),
        )
        output = sampler.sample(
            denoiser=comps.denoiser,
            textHiddenStates=condOut.hiddenStates,
            textKeyPaddingMask=condOut.keyPaddingMask,
            unconditionalTextHiddenStates=None,
            unconditionalTextKeyPaddingMask=None,
            frames=min(config.maxFrames, _DEBUG_GENERATION_FRAMES),
            numSteps=_DEBUG_GENERATION_DDIM_STEPS,
            cfgScale=1.0,
            eta=0.0,
            device=device,
            seed=config.seed,
            normalizer=comps.normalizer,
        )

    boneRotation6d = output.boneMotion[0].detach().float().cpu()
    rootTranslation = (
        output.globalMotion[0].detach().float().cpu()
        if output.globalMotion is not None else None
    )

    # rot6d → axis-angle → SMPL-24 padding.
    axisAngles = (
        Rotation(boneRotation6d, kind="rot6d")
        .axis_angle.numpy().astype(np.float32)
    )
    smpl24Count = len(SMPL24_BONE_ORDER)
    axisAngles24 = np.zeros(
        (axisAngles.shape[0], smpl24Count, 3), dtype=np.float32
    )
    smpl24Index = {
        name: idx for idx, name in enumerate(SMPL24_BONE_ORDER)
    }
    for srcIdx, name in enumerate(SMPL22_BONE_ORDER):
        tgtIdx = smpl24Index.get(name)
        if tgtIdx is not None:
            axisAngles24[:, tgtIdx, :] = axisAngles[:, srcIdx, :]
    flatAngles = axisAngles24.reshape(axisAngles24.shape[0], -1)

    extras: dict[str, object] = {
        "prompt": sample.rawText,
        "debug": True,
        "ddim_steps": _DEBUG_GENERATION_DDIM_STEPS,
        "cfg_scale": 1.0,
    }
    if rootTranslation is not None:
        extras["trans"] = rootTranslation.tolist()

    animSample = AnimationSample(
        relativePath=generationPath,
        resolvedPath=generationPath.resolve(),
        axisAngles=flatAngles,
        fps=30,
        extras=extras,
    )

    generationPath.parent.mkdir(parents=True, exist_ok=True)
    parentDir = generationPath.parent.resolve()
    paths = DatasetBuilderPaths(
        animationRoot=parentDir,
        promptRoot=parentDir,
        promptSources=[parentDir],
        indexCsv=parentDir / "missing-index.csv",
        outputRoot=parentDir,
    )
    builderConfig = DatasetBuilderConfig(
        paths=paths,
        processing=DatasetBuilderProcessing(),
    )
    rebuilder = AnimationRebuilder(builderConfig)
    rebuilder.exportCollada(animSample, generationPath)
    LOGGER.info("[debug] Generation exported to %s.", generationPath)
