"""Training utilities for AI-nimator v2 — overfit profile.

This module wires the Phase A (tokenizer, text encoder, denoiser) and
Phase B (noise schedule, losses, sampler) building blocks into a small
training loop suitable for the **overfit-1-sample sanity check**.

What this module is for
-----------------------
* Validate that the v2 stack converges end-to-end on a single sample
  (the canonical sanity check for any new diffusion architecture).
* Provide a checkpoint format the v2 generation CLI can load.
* Stay small and explicit — no chunk shuffling, no EMA, no
  early-stopping.  Those features land in Phase D when we wire the full
  training run.

What this module is NOT
-----------------------
* A production training loop.  It deliberately skips the legacy
  features (validation split, gradient accumulation, mixed precision,
  generation text cache) so the overfit run is easy to reason about.

Public surface
--------------
* :class:`V2TrainingConfig` — frozen dataclass with all knobs.
* :func:`loadDatasetSample` — read a (motion, text) pair from the
  V2 preprocessed dataset shards.
* :func:`buildTrainingComponents` — instantiate tokenizer + encoder +
  denoiser + schedule + optimizer in one call.
* :func:`trainStep` — single optimisation step (loss + backward +
  optimizer step).
* :func:`runOverfit` — top-level entry-point that loops over epochs.
* :func:`saveCheckpointV2` / :func:`loadCheckpointV2` — round-trip the
  trained state to disk.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.optim import AdamW

from ainimator.core.checkpoint_io import saveTorchObjectAtomically
from ainimator.core.resolved_config import writeResolvedConfig
from ainimator.health.hub import HealthHub, buildHealthHub
from ainimator.core.constants.preprocessed import (
    PREPROCESSED_LINK_INDEX_FILENAME,
    PREPROCESSED_MANIFEST_FILENAME,
    PREPROCESSED_SAMPLE_INDEX_FILENAME,
    PREPROCESSED_TEXT_INDEX_FILENAME,
)
from ainimator.model.denoiser_v2 import (
    MotionDenoiserV2,
    MotionDenoiserV2Config,
)
from ainimator.model.losses_v2 import (
    DEFAULT_MIN_SNR_GAMMA,
    diffusionLossV2,
    velocityXyzLossV2,
)
from ainimator.model.motion_normalizer import (
    MotionNormalizer,
)
from ainimator.diffusion.noise_schedule import (
    NoiseSchedule,
    NoiseScheduleConfig,
    PREDICTION_V,
    SUPPORTED_PREDICTIONS,
)
from ainimator.text import (
    ClipTextEncoder,
    ClipTextEncoderConfig,
    ClipTokenizer,
    CustomTextEncoder,
    CustomTextEncoderConfig,
    CustomTokenizer,
)

LOGGER = logging.getLogger(__name__)

# Empty prompt used to seed the unconditional CFG branch.  Encoded once
# at module load so callers don't pay the tokenizer cost per step.
EMPTY_PROMPT = ""


@dataclass
class TrainingRandomState:
    """Bundle of seeded RNG generators used by :func:`trainStep`.

    Two generators coexist on purpose:

    * ``cpuGenerator`` drives small index sampling (timestep, random
      crop start).  Always lives on CPU because the MPS backend rejects
      ``torch.randint`` with an MPS-bound generator on some PyTorch
      builds (e.g. 2.9.1) — the failure mode is the cryptic
      "Placeholder storage has not been allocated on MPS device".
    * ``deviceGenerator`` (when available) drives the Gaussian noise
      tensors that must live on the training device.  When MPS does
      not support generators for ``torch.randn`` either, we fall back
      to a CPU-side draw + ``.to(device)``.
    """

    cpuGenerator: torch.Generator
    deviceGenerator: torch.Generator | None

    @classmethod
    def fromSeed(cls, seed: int, device: torch.device) -> "TrainingRandomState":
        """Build the generator bundle for ``device``.

        The bundle attempts to create a device-bound generator first;
        on backends where that fails (e.g. MPS on older PyTorch
        wheels) it silently falls back to CPU draws followed by a
        ``.to(device)`` copy — slower but always correct.
        """
        cpuGenerator = torch.Generator(device=torch.device("cpu"))
        cpuGenerator.manual_seed(int(seed))

        deviceGenerator: torch.Generator | None = None
        if device.type != "cpu":
            try:
                candidate = torch.Generator(device=device)
                candidate.manual_seed(int(seed) + 1)
                # Smoke-test the generator with a tiny draw — some MPS
                # builds advertise the generator but reject randn.
                torch.randn(
                    (1,), generator=candidate, device=device
                )
                deviceGenerator = candidate
            except (RuntimeError, TypeError):
                LOGGER.warning(
                    "Device %s does not support a per-device RNG; "
                    "falling back to CPU sampling for noise tensors.",
                    device,
                )
                deviceGenerator = None
        return cls(
            cpuGenerator=cpuGenerator, deviceGenerator=deviceGenerator
        )

    def sampleNormal(
        self,
        shape: tuple[int, ...] | torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        """Draw a Gaussian tensor on ``device`` honouring the seed."""
        if self.deviceGenerator is not None and device.type != "cpu":
            return torch.randn(
                shape, generator=self.deviceGenerator, device=device
            )
        cpuTensor = torch.randn(
            shape, generator=self.cpuGenerator, device=torch.device("cpu")
        )
        return cpuTensor.to(device)


# =====================================================================
# Configuration
# =====================================================================
@dataclass(frozen=True)
class V2TrainingConfig:
    """Knobs for the v2 overfit / sanity-check training loop.

    Attributes
    ----------
    datasetRoot : Path
        Root of the V2 preprocessed dataset (the directory holding
        ``manifest.json`` and the ``sample_shards/`` / ``text_shards/``
        sub-directories).
    tokenizerDir : Path
        Directory containing ``tokenizer.json`` + ``config.json`` from
        :meth:`CustomTokenizer.save`.
    outputDir : Path
        Where checkpoints are written.
    sampleLinkIndex : int
        Index into the preprocessed link table — picks one
        ``(motion, text)`` pair to overfit on.
    epochs : int
        Number of optimisation epochs (one step per epoch in overfit).
    learningRate : float
        AdamW learning rate.  ``1e-4`` is the standard MDM-style value.
    weightDecay : float
        AdamW L2 regularisation.  Kept at 0 in overfit so the model can
        memorise.
    minSnrGamma : float
        Min-SNR clipping (0 disables; default 5.0).
    velocityXyzWeight : float
        Auxiliary FK-velocity loss weight.  ``0`` disables it.
    diffusionStepsTraining : int
        Number of timesteps in the training noise schedule.
    scheduleType : str
        ``"cosine"`` or ``"linear"``.
    predictionMode : str
        Network output target (``"v"``, ``"x0"``, ``"epsilon"``).
    encoderHiddenDim : int
        Width of the custom text encoder.
    encoderNumLayers : int
        Depth of the custom text encoder.
    encoderNumHeads : int
        Heads of the custom text encoder.
    denoiserEmbedDim : int
        Width of the v2 denoiser.
    denoiserNumLayers : int
        Depth of the v2 denoiser.
    denoiserNumHeads : int
        Heads of the v2 denoiser.
    maxFrames : int
        Frame cap for the denoiser (longer samples are truncated).
    framesPerStep : int
        How many frames to feed the denoiser per overfit step.  Setting
        this below the sample length lets a single sample produce a
        moving-window curriculum rather than a single fixed window.
    seed : int
        RNG seed used for the noise sampler — fixed for reproducibility
        of the overfit logs.
    logEvery : int
        Print a loss line every ``logEvery`` epochs.
    dropout : float
        Dropout probability applied inside encoder + denoiser.  Set to
        ``0.0`` for the canonical overfit-1-sample run (the model must
        be free to memorise the sample exactly).  Use ``0.1`` for full
        training to provide regularisation.
    condMaskProb : float
        Probability that the conditional text branch is replaced with
        the empty prompt during training (à la MDM ``cond-mask-prob``).
        ``0.0`` is fine for overfit (the model never needs to handle an
        empty prompt at inference).  **Required for any production run
        that uses ``--cfg-scale > 1`` at inference**, because CFG
        amplifies the gap between conditional and unconditional outputs;
        if the unconditional branch is untrained the gap is random and
        CFG produces noise.  ``0.1``–``0.15`` is the standard MDM range.
    """

    datasetRoot: Path
    tokenizerDir: Path
    outputDir: Path
    sampleLinkIndex: int = 0
    epochs: int = 200
    learningRate: float = 1e-4
    weightDecay: float = 0.0
    minSnrGamma: float = DEFAULT_MIN_SNR_GAMMA
    velocityXyzWeight: float = 0.0
    diffusionStepsTraining: int = 1000
    scheduleType: str = "cosine"
    predictionMode: str = PREDICTION_V
    encoderHiddenDim: int = 256
    encoderNumLayers: int = 4
    encoderNumHeads: int = 8
    denoiserEmbedDim: int = 384
    denoiserNumLayers: int = 4
    denoiserNumHeads: int = 8
    maxFrames: int = 256
    framesPerStep: int = 0  # 0 → use sample length
    seed: int = 0
    logEvery: int = 10
    device: str = "auto"
    dropout: float = 0.0  # overfit-friendly default; bump to 0.1 in prod
    condMaskProb: float = 0.0  # overfit-friendly default; 0.10–0.15 in prod
    # Phase F (2026-05-12) — exercise the SOTA conditioning path in
    # overfit too so the sanity-check actually validates what production
    # will run.  Defaults to ``True``; pass ``--no-film`` from a CLI to
    # disable for ablations.
    useFilmConditioning: bool = True
    filmDropout: float = 0.0
    usePerBlockFilm: bool = True
    useNullEmbedding: bool = True

    # --- Health monitoring (A3) ----------------------------------
    healthEnabled: bool = True
    healthEverySteps: int = 50

    def __post_init__(self) -> None:
        if self.predictionMode not in SUPPORTED_PREDICTIONS:
            raise ValueError(
                "predictionMode must be one of "
                f"{SUPPORTED_PREDICTIONS}; got {self.predictionMode!r}."
            )
        if self.scheduleType not in ("linear", "cosine"):
            raise ValueError(
                f"scheduleType must be 'linear' or 'cosine'; got "
                f"{self.scheduleType!r}."
            )
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if not (0.0 <= self.condMaskProb <= 1.0):
            raise ValueError("condMaskProb must be in [0, 1].")
        if self.healthEverySteps < 1:
            raise ValueError("healthEverySteps must be >= 1.")


# =====================================================================
# Sample loading
# =====================================================================
@dataclass(frozen=True)
class LoadedSample:
    """A single (motion, text) pair extracted from the preprocessed dataset.

    Attributes
    ----------
    rotation6d : torch.Tensor
        Per-frame rotations of shape ``(F, numBones, 6)``.
    rootTranslation : torch.Tensor
        Per-frame root translation of shape ``(F, 3)``.
    rawText : str
        Source prompt (the very same string the legacy XLM-R pipeline
        encoded into ``input_ids``).
    metadata : dict
        Loose metadata dict from the sample shard (fps, source path,
        ...).
    sampleId : int
        Numeric ID of the sample (for logging / debugging).
    textId : int
        Numeric ID of the text (for logging / debugging).
    """

    rotation6d: torch.Tensor
    rootTranslation: torch.Tensor
    rawText: str
    metadata: dict[str, Any]
    sampleId: int
    textId: int

    @property
    def numFrames(self) -> int:
        return int(self.rotation6d.shape[0])

    @property
    def numBones(self) -> int:
        return int(self.rotation6d.shape[1])


def loadDatasetSample(
    datasetRoot: Path,
    linkIndex: int = 0,
) -> LoadedSample:
    """Return the ``linkIndex``-th (motion, text) pair from the dataset.

    The function reads ``link_index.json`` to resolve the
    sample-shard / text-shard locations, then mmaps the relevant
    ``.pt`` shards.  Only the rotation6d and root_translation tensors
    are extracted from the sample shard (other components stay on
    disk).

    Raises
    ------
    FileNotFoundError
        When the dataset root, manifest, or shard files are missing.
    KeyError
        When the requested motion components are absent from the shard.
    """
    datasetRoot = Path(datasetRoot)
    if not datasetRoot.exists():
        raise FileNotFoundError(f"Dataset root not found: {datasetRoot}.")

    manifestPath = datasetRoot / PREPROCESSED_MANIFEST_FILENAME
    if not manifestPath.exists():
        raise FileNotFoundError(
            f"Missing preprocessed manifest at {manifestPath}."
        )
    manifest = json.loads(manifestPath.read_text(encoding="utf-8"))

    linkIndexPath = (
        datasetRoot
        / manifest.get("linkIndexPath", PREPROCESSED_LINK_INDEX_FILENAME)
    )
    sampleIndexPath = (
        datasetRoot
        / manifest.get(
            "sampleIndexPath", PREPROCESSED_SAMPLE_INDEX_FILENAME
        )
    )
    textIndexPath = (
        datasetRoot
        / manifest.get("textIndexPath", PREPROCESSED_TEXT_INDEX_FILENAME)
    )

    linkEntries = json.loads(linkIndexPath.read_text(encoding="utf-8"))
    if linkIndex < 0 or linkIndex >= len(linkEntries):
        raise IndexError(
            f"linkIndex={linkIndex} is out of range "
            f"(0..{len(linkEntries) - 1})."
        )
    linkEntry = linkEntries[linkIndex]
    sampleIndexEntries = json.loads(
        sampleIndexPath.read_text(encoding="utf-8")
    )
    textIndexEntries = json.loads(
        textIndexPath.read_text(encoding="utf-8")
    )
    sampleEntry = sampleIndexEntries[int(linkEntry["sampleId"])]
    textEntry = textIndexEntries[int(linkEntry["textId"])]

    sampleShardPath = (
        datasetRoot
        / manifest["sampleShards"][int(sampleEntry["shardIndex"])]["path"]
    )
    textShardPath = (
        datasetRoot
        / manifest["textShards"][int(textEntry["shardIndex"])]["path"]
    )
    sampleShard = torch.load(
        sampleShardPath, map_location="cpu", weights_only=False
    )
    textShard = torch.load(
        textShardPath, map_location="cpu", weights_only=False
    )
    samplePayload = sampleShard[int(sampleEntry["shardOffset"])]
    textPayload = textShard[int(textEntry["shardOffset"])]

    rotation = samplePayload.get("motion")
    if not isinstance(rotation, torch.Tensor):
        raise KeyError("Sample shard entry is missing the 'motion' tensor.")
    rootTranslation = samplePayload.get("root_translation")
    if not isinstance(rootTranslation, torch.Tensor):
        raise KeyError(
            "Sample shard entry is missing 'root_translation'.  "
            "Re-run preprocess_dataset with the v2 BoneDataConfig."
        )
    rawText = textPayload.get("raw_text")
    if not isinstance(rawText, str):
        raise KeyError(
            "Text shard entry is missing 'raw_text'.  Re-run "
            "preprocess_dataset to regenerate the text shards."
        )

    return LoadedSample(
        rotation6d=rotation,
        rootTranslation=rootTranslation,
        rawText=rawText,
        metadata=dict(samplePayload.get("meta", {})),
        sampleId=int(linkEntry["sampleId"]),
        textId=int(linkEntry["textId"]),
    )


# =====================================================================
# Component bundle
# =====================================================================
@dataclass
class V2TrainingComponents:
    """Bundle of objects produced by :func:`buildTrainingComponents`."""

    tokenizer: CustomTokenizer
    encoder: CustomTextEncoder
    denoiser: MotionDenoiserV2
    schedule: NoiseSchedule
    normalizer: MotionNormalizer
    optimizer: torch.optim.Optimizer
    device: torch.device
    nullTextIds: torch.Tensor = field(repr=False)
    nullAttentionMask: torch.Tensor = field(repr=False)


def resolveDevice(requested: str) -> torch.device:
    """Return a concrete :class:`torch.device` for ``requested``.

    ``"auto"`` picks ``mps`` when available, then ``cuda``, otherwise
    falls back to ``cpu``.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def buildTrainingComponents(
    config: V2TrainingConfig,
    normalizationSamples: tuple[
        Sequence[torch.Tensor], Sequence[torch.Tensor]
    ] | None = None,
) -> V2TrainingComponents:
    """Instantiate tokenizer + encoder + denoiser + schedule + normalizer + optimizer.

    The encoder ``outputDim`` is forced to ``denoiserEmbedDim`` so the
    denoiser can consume its hidden states without an extra projection.

    Parameters
    ----------
    normalizationSamples : tuple of (boneSamples, globalSamples), optional
        When provided, the normalizer is fitted on these tensors right
        after construction.  Pass the training motion(s) here so the
        diffusion process operates on a unit-variance representation.
        When ``None``, the normalizer keeps its identity init — the
        caller is then responsible for fitting it before training (or
        explicitly leaving it as identity for tests).
    """
    device = resolveDevice(config.device)

    tokenizer = CustomTokenizer.load(config.tokenizerDir)

    encoder = CustomTextEncoder(
        CustomTextEncoderConfig(
            vocabSize=tokenizer.vocabSize,
            maxLength=tokenizer.config.maxLength,
            hiddenDim=config.encoderHiddenDim,
            numLayers=config.encoderNumLayers,
            numHeads=config.encoderNumHeads,
            outputDim=config.denoiserEmbedDim,
            padTokenId=tokenizer.padTokenId,
            dropout=config.dropout,
            # Phase F — mirror the production encoder so the overfit
            # path stresses the same conditioning surface.
            useNullEmbedding=config.useNullEmbedding,
            l2NormalizeOutput=True,
        )
    ).to(device)

    denoiser = MotionDenoiserV2(
        MotionDenoiserV2Config(
            embedDim=config.denoiserEmbedDim,
            numHeads=config.denoiserNumHeads,
            numLayers=config.denoiserNumLayers,
            numBones=22,
            motionChannels=6,
            globalChannels=3,
            textEmbedDim=config.denoiserEmbedDim,  # encoder already projected
            maxFrames=config.maxFrames,
            dropout=config.dropout,
            # Phase F — keep overfit aligned with the production conditioning
            # path so the sanity-check actually verifies cfg sensitivity.
            useFilmConditioning=config.useFilmConditioning,
            filmDropout=config.filmDropout,
            usePerBlockFilm=config.usePerBlockFilm,
        )
    ).to(device)

    schedule = NoiseSchedule(
        NoiseScheduleConfig(
            numSteps=config.diffusionStepsTraining,
            scheduleType=config.scheduleType,
        )
    ).to(device)

    normalizer = MotionNormalizer(
        numBones=22,
        motionChannels=6,
        globalChannels=3,
    ).to(device)
    if normalizationSamples is not None:
        boneSamples, globalSamples = normalizationSamples
        normalizer.fitFromTensors(boneSamples, globalSamples)

    parameters = list(encoder.parameters()) + list(denoiser.parameters())
    optimizer = AdamW(
        parameters,
        lr=config.learningRate,
        weight_decay=config.weightDecay,
    )

    nullEncoded = tokenizer.encode(EMPTY_PROMPT)
    nullTextIds = nullEncoded.inputIds.to(device)
    nullAttentionMask = nullEncoded.attentionMask.to(device)

    return V2TrainingComponents(
        tokenizer=tokenizer,
        encoder=encoder,
        denoiser=denoiser,
        schedule=schedule,
        normalizer=normalizer,
        optimizer=optimizer,
        device=device,
        nullTextIds=nullTextIds,
        nullAttentionMask=nullAttentionMask,
    )


# =====================================================================
# Training step
# =====================================================================
def trainStep(
    components: V2TrainingComponents,
    sample: LoadedSample,
    config: V2TrainingConfig,
    generators: "TrainingRandomState",
) -> dict[str, float]:
    """Execute one optimisation step on ``sample``.

    Returns a logging dict with scalar values for the diffusion and
    auxiliary losses.
    """
    components.encoder.train()
    components.denoiser.train()

    device = components.device
    schedule = components.schedule

    # --- Pull motion to device ------------------------------------
    rotationRaw = sample.rotation6d.unsqueeze(0).to(device)  # (1, F, B, 6)
    rootTranslationRaw = sample.rootTranslation.unsqueeze(0).to(device)
    framesPerStep = (
        config.framesPerStep if config.framesPerStep > 0
        else min(rotationRaw.shape[1], config.maxFrames)
    )
    framesPerStep = min(framesPerStep, rotationRaw.shape[1])
    if framesPerStep < rotationRaw.shape[1]:
        # Random crop window for curriculum diversity.  Indices are
        # always sampled on CPU because some MPS generators reject the
        # randint kernel — only the noise tensors live on `device`.
        maxStart = rotationRaw.shape[1] - framesPerStep
        start = int(
            torch.randint(
                0,
                maxStart + 1,
                (1,),
                generator=generators.cpuGenerator,
                device=torch.device("cpu"),
            ).item()
        )
    else:
        start = 0
    rotationRaw = rotationRaw[:, start:start + framesPerStep]
    rootTranslationRaw = rootTranslationRaw[:, start:start + framesPerStep]

    # --- Z-normalize so the diffusion sees unit-variance x0 -------
    # The whole point of the v1 stabilisation was to bring x0 onto the
    # same scale as the Gaussian noise injected by the forward
    # process; otherwise the schedule mixes raw motion (rotation6d in
    # ~[-1,1], root_translation in metres) with N(0, 1) and the
    # denoiser sees a distribution it can never invert.  v2 used to
    # skip this step — the symptom was "sampler outputs pure noise".
    rotation = components.normalizer.normalizeBone(rotationRaw)
    rootTranslation = components.normalizer.normalizeGlobal(
        rootTranslationRaw
    )

    # --- Encode text ---------------------------------------------
    # cond-mask-prob: replace the prompt with the empty string with
    # probability ``condMaskProb``.  Required for inference-time CFG
    # (cfgScale > 1) — the unconditional branch must learn an embedding
    # for the empty prompt, otherwise CFG amplifies a random direction
    # and the sampler drifts off the manifold.
    promptToEncode = sample.rawText
    dropPrompt = False
    if config.condMaskProb > 0.0:
        coin = torch.rand(
            (1,), generator=generators.cpuGenerator, device=torch.device("cpu")
        ).item()
        if coin < config.condMaskProb:
            dropPrompt = True
            if not config.useNullEmbedding:
                promptToEncode = EMPTY_PROMPT
    encoded = components.tokenizer.encode(promptToEncode)
    inputIds = encoded.inputIds.to(device)
    attentionMask = encoded.attentionMask.to(device)
    textOutput = components.encoder(inputIds, attentionMask)

    # Phase F — when CFG dropout fires and the encoder exposes a
    # learnable null token, bypass the encoder output for this sample.
    if dropPrompt and config.useNullEmbedding:
        nullOutput = components.encoder.forwardNull(
            batchSize=textOutput.hiddenStates.shape[0],
            device=device,
            dtype=textOutput.hiddenStates.dtype,
        )
        textOutput = nullOutput

    # --- Forward diffusion --------------------------------------
    # Sample the timestep on CPU (a 1-element long tensor) and ship it
    # to the active device.  Avoids MPS generator quirks on randint.
    timesteps = torch.randint(
        0,
        schedule.numSteps,
        (1,),
        generator=generators.cpuGenerator,
        device=torch.device("cpu"),
    ).to(device)
    noiseRotation = generators.sampleNormal(rotation.shape, device=device)
    noiseGlobal = generators.sampleNormal(
        rootTranslation.shape, device=device
    )
    xtRotation, _ = schedule.qSample(
        rotation, timesteps, noise=noiseRotation
    )
    xtGlobal, _ = schedule.qSample(
        rootTranslation, timesteps, noise=noiseGlobal
    )
    targetRotation = schedule.predictionTarget(
        rotation, noiseRotation, timesteps, config.predictionMode
    )
    targetGlobal = schedule.predictionTarget(
        rootTranslation, noiseGlobal, timesteps, config.predictionMode
    )

    # --- Denoiser forward ---------------------------------------
    output = components.denoiser(
        noisyMotion=xtRotation,
        timesteps=timesteps,
        textHiddenStates=textOutput.hiddenStates,
        textKeyPaddingMask=textOutput.keyPaddingMask,
        noisyGlobalFeatures=xtGlobal,
    )

    # --- Losses --------------------------------------------------
    boneLoss = diffusionLossV2(
        prediction=output.boneOutput,
        target=targetRotation,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=config.predictionMode,
        gamma=config.minSnrGamma,
    )
    assert output.globalOutput is not None
    globalLoss = diffusionLossV2(
        prediction=output.globalOutput,
        target=targetGlobal,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=config.predictionMode,
        gamma=config.minSnrGamma,
    )
    total = boneLoss + globalLoss

    velLossValue = 0.0
    if config.velocityXyzWeight > 0.0:
        # Velocity-XYZ uses FK on the predicted x0 — for that we need
        # the rotation in raw rotation6d space, not normalized space.
        x0PredNorm = schedule.x0FromPrediction(
            output.boneOutput, xtRotation, timesteps, config.predictionMode
        )
        x0PredRaw = components.normalizer.denormalizeBone(x0PredNorm)
        velLoss = velocityXyzLossV2(
            predictedRotation6d=x0PredRaw,
            targetRotation6d=rotationRaw,
            timesteps=timesteps,
            alphasCumprod=schedule.alphasCumprod,
        )
        total = total + config.velocityXyzWeight * velLoss
        velLossValue = float(velLoss.detach().item())

    # --- Backward + step ---------------------------------------
    components.optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        list(components.encoder.parameters())
        + list(components.denoiser.parameters()),
        max_norm=1.0,
    )
    components.optimizer.step()

    return {
        "loss_total": float(total.detach().item()),
        "loss_bone": float(boneLoss.detach().item()),
        "loss_global": float(globalLoss.detach().item()),
        "loss_vel_xyz": velLossValue,
        "timestep": int(timesteps.item()),
    }


# =====================================================================
# Top-level training loop
# =====================================================================
def runOverfit(
    config: V2TrainingConfig,
) -> tuple[V2TrainingComponents, list[dict[str, float]]]:
    """Run the overfit-1-sample sanity-check training loop.

    Returns the final ``V2TrainingComponents`` (with trained weights)
    and the list of per-epoch log dicts.
    """
    config.outputDir.mkdir(parents=True, exist_ok=True)
    writeResolvedConfig(config, config.outputDir)
    sample = loadDatasetSample(
        config.datasetRoot, linkIndex=config.sampleLinkIndex
    )
    LOGGER.info(
        "Overfitting on linkIdx=%d (sampleId=%d, textId=%d, "
        "frames=%d, prompt=%r).",
        config.sampleLinkIndex,
        sample.sampleId,
        sample.textId,
        sample.numFrames,
        sample.rawText,
    )

    # Compute normalization statistics from the (sole) overfit sample
    # so the diffusion sees ~zero-mean, unit-variance x0.
    boneSamples = [sample.rotation6d]
    globalSamples = [sample.rootTranslation]
    components = buildTrainingComponents(
        config,
        normalizationSamples=(boneSamples, globalSamples),
    )
    LOGGER.info(
        "Normalizer fitted from overfit sample "
        "(bone std range=[%.3f, %.3f], global std=%s).",
        components.normalizer.boneStd.min().item(),
        components.normalizer.boneStd.max().item(),
        components.normalizer.globalStd.flatten().tolist()
        if components.normalizer.hasGlobalBranch
        else "n/a",
    )
    encoderParams = sum(p.numel() for p in components.encoder.parameters())
    denoiserParams = sum(
        p.numel() for p in components.denoiser.parameters()
    )
    LOGGER.info(
        "Trainable params: encoder=%d, denoiser=%d, total=%d.",
        encoderParams,
        denoiserParams,
        encoderParams + denoiserParams,
    )

    generators = TrainingRandomState.fromSeed(
        seed=config.seed, device=components.device
    )

    # --- Health hub (A3) ----------------------------------------
    healthHub: HealthHub | None = None
    if config.healthEnabled:
        healthHub = buildHealthHub(config.outputDir)
        healthHub._everySteps = config.healthEverySteps
        healthHub.attach(components.denoiser)

    history: list[dict[str, float]] = []
    startTime = time.time()
    for epoch in range(1, config.epochs + 1):
        metrics = trainStep(components, sample, config, generators)
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        if healthHub is not None:
            healthHub.step(globalStep=epoch, metrics=metrics)
        if epoch == 1 or epoch % config.logEvery == 0 or epoch == config.epochs:
            elapsed = time.time() - startTime
            LOGGER.info(
                "epoch=%d  loss=%.4f  bone=%.4f  global=%.4f  vel=%.4f  "
                "t=%d  elapsed=%.1fs",
                epoch,
                metrics["loss_total"],
                metrics["loss_bone"],
                metrics["loss_global"],
                metrics["loss_vel_xyz"],
                int(metrics["timestep"]),
                elapsed,
            )

    if healthHub is not None:
        healthHub.detach()
        healthHub.close()

    checkpointPath = config.outputDir / "v2_overfit_checkpoint.pt"
    saveCheckpointV2(components, sample, config, checkpointPath)
    LOGGER.info("Checkpoint saved to %s.", checkpointPath)

    return components, history


# =====================================================================
# Checkpoint I/O
# =====================================================================
def saveCheckpointV2(
    components: V2TrainingComponents,
    sample: LoadedSample,
    config: V2TrainingConfig,
    path: Path,
) -> None:
    """Persist the v2 training state in a single ``.pt`` file."""
    payload = {
        "version": 3,  # bump: v3 adds normalizer state
        "encoder_state_dict": components.encoder.state_dict(),
        "encoder_config": _encoderConfigToDict(components.encoder.config),
        "denoiser_state_dict": components.denoiser.state_dict(),
        "denoiser_config": _denoiserConfigToDict(components.denoiser.config),
        "schedule_config": _scheduleConfigToDict(components.schedule.config),
        "schedule_state_dict": components.schedule.state_dict(),
        "normalizer_config": components.normalizer.configToDict(),
        "normalizer_state_dict": components.normalizer.state_dict(),
        "tokenizer_dir": str(config.tokenizerDir.resolve()),
        "training_config": _trainingConfigToDict(config),
        "training_sample": {
            "sampleId": sample.sampleId,
            "textId": sample.textId,
            "rawText": sample.rawText,
            "frames": sample.numFrames,
        },
    }
    saveTorchObjectAtomically(payload, path)


def loadCheckpointV2(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[
    CustomTokenizer | ClipTokenizer,
    CustomTextEncoder | ClipTextEncoder,
    MotionDenoiserV2,
    NoiseSchedule,
    MotionNormalizer,
    dict[str, Any],
]:
    """Load a v2 checkpoint and rebuild every component.

    Returns ``(tokenizer, encoder, denoiser, schedule, normalizer,
    payload)`` — the normalizer is mandatory in v3+ checkpoints; for
    v2 (pre-normalizer) checkpoints we fall back to an identity
    normalizer (caller-side responsibility to interpret correctly).
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    version = int(payload.get("version", 0))
    if version not in (2, 3):
        raise ValueError(
            f"Unsupported checkpoint version: {payload.get('version')}."
        )

    # Phase 2 — the text encoder is either the custom BPE transformer
    # or the frozen CLIP tower.  ``text_encoder_type`` defaults to
    # "custom" so pre-Phase-2 checkpoints keep loading unchanged.
    textEncoderType = str(payload.get("text_encoder_type", "custom"))
    if textEncoderType == "clip":
        clipEncoderConfig = _clipEncoderConfigFromDict(
            payload["encoder_config"]
        )
        tokenizer = ClipTokenizer(
            modelName=clipEncoderConfig.modelName,
            maxLength=clipEncoderConfig.maxLength,
        )
        encoder = ClipTextEncoder(clipEncoderConfig).to(device)
        # The frozen CLIP tower (``clip.*`` keys) is filtered out at
        # save time to keep checkpoints small — its weights are
        # reloaded from the HF hub by ``ClipTextEncoder.__init__``.
        # Only the trainable projection + null embedding are persisted,
        # so ``strict=False`` tolerates the missing ``clip.*`` keys.
        encoder.load_state_dict(
            payload["encoder_state_dict"], strict=False
        )
    else:
        tokenizerDir = Path(payload["tokenizer_dir"])
        tokenizer = CustomTokenizer.load(tokenizerDir)

        encoderConfig = _encoderConfigFromDict(payload["encoder_config"])
        encoder = CustomTextEncoder(encoderConfig).to(device)
        encoder.load_state_dict(payload["encoder_state_dict"])

    denoiserConfig = _denoiserConfigFromDict(payload["denoiser_config"])
    # Backward-compat: checkpoints saved before the alignmentEnabled /
    # useFilmConditioning fields were serialised (commits between the
    # Phase D module ship and the matching serialiser fix) carry the
    # corresponding state-dict keys but lack the flag.  Detect that
    # mismatch from the state-dict shape and re-enable the flags.
    denoiserState = payload["denoiser_state_dict"]
    keyNames = list(denoiserState.keys())
    hasAlignmentKeys = any(
        key.startswith("alignmentHead.") for key in keyNames
    )
    hasFilmKeys = any(
        key.startswith("filmConditioning.") for key in keyNames
    )
    hasPerBlockFilmKeys = any(
        ".adaln." in key for key in keyNames
    )
    hasAuxPoolAlignmentKeys = any(
        key.startswith("auxPoolAlignment.") for key in keyNames
    )
    hasX0AlignmentKeys = any(
        key.startswith("x0AlignmentEncoder.") for key in keyNames
    )
    hasSelfCondKeys = any(
        key.startswith("selfCondBoneProj.")
        or key.startswith("selfCondGlobalProj.")
        for key in keyNames
    )
    if (
        hasAlignmentKeys and not denoiserConfig.alignmentEnabled
    ) or (
        hasFilmKeys and not denoiserConfig.useFilmConditioning
    ) or (
        hasPerBlockFilmKeys and not denoiserConfig.usePerBlockFilm
    ) or (
        hasAuxPoolAlignmentKeys and not denoiserConfig.auxPoolAlignmentEnabled
    ) or (
        hasX0AlignmentKeys and not denoiserConfig.x0AlignmentEnabled
    ) or (
        hasSelfCondKeys and not denoiserConfig.useSelfConditioning
    ):
        from dataclasses import replace as _replace
        LOGGER.warning(
            "Checkpoint denoiser config does not record all Phase D/E/F "
            "module flags (alignmentEnabled=%s, useFilmConditioning=%s, "
            "usePerBlockFilm=%s, auxPoolAlignmentEnabled=%s) but the saved "
            "state_dict carries the corresponding keys — re-enabling them "
            "for load.",
            denoiserConfig.alignmentEnabled,
            denoiserConfig.useFilmConditioning,
            denoiserConfig.usePerBlockFilm,
            denoiserConfig.auxPoolAlignmentEnabled,
        )
        denoiserConfig = _replace(
            denoiserConfig,
            alignmentEnabled=denoiserConfig.alignmentEnabled
            or hasAlignmentKeys,
            useFilmConditioning=denoiserConfig.useFilmConditioning
            or hasFilmKeys,
            usePerBlockFilm=denoiserConfig.usePerBlockFilm
            or hasPerBlockFilmKeys,
            auxPoolAlignmentEnabled=denoiserConfig.auxPoolAlignmentEnabled
            or hasAuxPoolAlignmentKeys,
            x0AlignmentEnabled=denoiserConfig.x0AlignmentEnabled
            or hasX0AlignmentKeys,
            useSelfConditioning=denoiserConfig.useSelfConditioning
            or hasSelfCondKeys,
        )
    denoiser = MotionDenoiserV2(denoiserConfig).to(device)
    denoiser.load_state_dict(denoiserState)

    scheduleConfig = _scheduleConfigFromDict(payload["schedule_config"])
    schedule = NoiseSchedule(scheduleConfig).to(device)
    schedule.load_state_dict(payload["schedule_state_dict"])

    if version >= 3:
        normalizer = MotionNormalizer.fromConfigDict(
            payload["normalizer_config"]
        ).to(device)
        normalizer.load_state_dict(payload["normalizer_state_dict"])
    else:
        # Legacy v2 checkpoint without normalizer: fall back to identity
        # transform.  This will only produce sensible output when the
        # checkpoint was trained with the (broken) raw-space pipeline,
        # which is documented to produce noise — so this branch exists
        # mainly to allow legacy checkpoints to load and surface a
        # warning rather than silently corrupting downstream code.
        LOGGER.warning(
            "Loading a v2 checkpoint without a normalizer; output will "
            "match the legacy behaviour (raw-space sampling)."
        )
        normalizer = MotionNormalizer(
            numBones=denoiserConfig.numBones,
            motionChannels=denoiserConfig.motionChannels,
            globalChannels=denoiserConfig.globalChannels,
        ).to(device)

    return tokenizer, encoder, denoiser, schedule, normalizer, payload


# ---------------------------------------------------------------------
# Internal serialization helpers
# ---------------------------------------------------------------------
def _trainingConfigToDict(config: V2TrainingConfig) -> dict[str, Any]:
    return {
        "datasetRoot": str(config.datasetRoot),
        "tokenizerDir": str(config.tokenizerDir),
        "outputDir": str(config.outputDir),
        "sampleLinkIndex": config.sampleLinkIndex,
        "epochs": config.epochs,
        "learningRate": config.learningRate,
        "weightDecay": config.weightDecay,
        "minSnrGamma": config.minSnrGamma,
        "velocityXyzWeight": config.velocityXyzWeight,
        "diffusionStepsTraining": config.diffusionStepsTraining,
        "scheduleType": config.scheduleType,
        "predictionMode": config.predictionMode,
        "encoderHiddenDim": config.encoderHiddenDim,
        "encoderNumLayers": config.encoderNumLayers,
        "encoderNumHeads": config.encoderNumHeads,
        "denoiserEmbedDim": config.denoiserEmbedDim,
        "denoiserNumLayers": config.denoiserNumLayers,
        "denoiserNumHeads": config.denoiserNumHeads,
        "maxFrames": config.maxFrames,
        "framesPerStep": config.framesPerStep,
        "seed": config.seed,
        "logEvery": config.logEvery,
        "device": config.device,
    }


def _encoderConfigToDict(config: CustomTextEncoderConfig) -> dict[str, Any]:
    return {
        "vocabSize": config.vocabSize,
        "maxLength": config.maxLength,
        "hiddenDim": config.hiddenDim,
        "numLayers": config.numLayers,
        "numHeads": config.numHeads,
        "ffnDim": config.ffnDim,
        "dropout": config.dropout,
        "padTokenId": config.padTokenId,
        "outputDim": config.outputDim,
    }


def _encoderConfigFromDict(payload: dict[str, Any]) -> CustomTextEncoderConfig:
    return CustomTextEncoderConfig(
        vocabSize=int(payload["vocabSize"]),
        maxLength=int(payload["maxLength"]),
        hiddenDim=int(payload["hiddenDim"]),
        numLayers=int(payload["numLayers"]),
        numHeads=int(payload["numHeads"]),
        ffnDim=int(payload["ffnDim"]),
        dropout=float(payload["dropout"]),
        padTokenId=int(payload["padTokenId"]),
        outputDim=int(payload["outputDim"]),
    )


def _clipEncoderConfigFromDict(
    payload: dict[str, Any],
) -> ClipTextEncoderConfig:
    """Rebuild a :class:`ClipTextEncoderConfig` from a checkpoint payload."""
    return ClipTextEncoderConfig(
        modelName=str(payload["modelName"]),
        maxLength=int(payload["maxLength"]),
        outputDim=int(payload["outputDim"]),
        clipHiddenDim=int(payload["clipHiddenDim"]),
        dropout=float(payload["dropout"]),
        useNullEmbedding=bool(payload["useNullEmbedding"]),
        l2NormalizeOutput=bool(payload["l2NormalizeOutput"]),
    )


def _denoiserConfigToDict(config: MotionDenoiserV2Config) -> dict[str, Any]:
    return {
        "embedDim": config.embedDim,
        "numHeads": config.numHeads,
        "numLayers": config.numLayers,
        "numBones": config.numBones,
        "motionChannels": config.motionChannels,
        "globalChannels": config.globalChannels,
        "textEmbedDim": config.textEmbedDim,
        "maxFrames": config.maxFrames,
        "dropout": config.dropout,
        # Phase D.1 — must round-trip so a checkpoint trained with the
        # alignment head can be re-loaded with the matching architecture
        # (the head adds extra state-dict keys; rebuilding the denoiser
        # without it would crash ``load_state_dict``).
        "alignmentEnabled": config.alignmentEnabled,
        # Phase D.1 (Levier B) — projection dim of the SimCLR-style
        # alignment head.  Older checkpoints (pre-Levier-B) may not
        # have this field; the loader falls back to the default.
        "alignmentProjectionDim": config.alignmentProjectionDim,
        # Phase D Levier D — FiLM conditioning shortcut.  Defaults
        # below match the new field defaults so pre-Levier-D
        # checkpoints round-trip without the FiLM module.
        "useFilmConditioning": config.useFilmConditioning,
        "filmDropout": config.filmDropout,
        # Phase E (Levier E) — per-block AdaLN.  Older checkpoints
        # without this field load with ``usePerBlockFilm=False``.
        "usePerBlockFilm": config.usePerBlockFilm,
        # Phase F iter-2 — auxiliary raw-pool alignment.
        "auxPoolAlignmentEnabled": config.auxPoolAlignmentEnabled,
        # 2026-06-01 — generated-motion (x0) alignment encoder.  Must
        # round-trip so the extra submodule state-dict keys rebuild on
        # load; pre-dating checkpoints fall back to False (no module).
        "x0AlignmentEnabled": config.x0AlignmentEnabled,
        "x0AlignmentEmbedDim": config.x0AlignmentEmbedDim,
        "x0AlignmentNumLayers": config.x0AlignmentNumLayers,
        "x0AlignmentNumHeads": config.x0AlignmentNumHeads,
        # 2026-06-02 — self-conditioning input projections.
        "useSelfConditioning": config.useSelfConditioning,
    }


def _denoiserConfigFromDict(payload: dict[str, Any]) -> MotionDenoiserV2Config:
    return MotionDenoiserV2Config(
        embedDim=int(payload["embedDim"]),
        numHeads=int(payload["numHeads"]),
        numLayers=int(payload["numLayers"]),
        numBones=int(payload["numBones"]),
        motionChannels=int(payload["motionChannels"]),
        globalChannels=int(payload["globalChannels"]),
        textEmbedDim=int(payload["textEmbedDim"]),
        maxFrames=int(payload["maxFrames"]),
        dropout=float(payload["dropout"]),
        # Default ``False`` keeps backward-compatibility with pre-D.1
        # checkpoints that pre-date the field.
        alignmentEnabled=bool(payload.get("alignmentEnabled", False)),
        alignmentProjectionDim=int(
            payload.get("alignmentProjectionDim", 128)
        ),
        useFilmConditioning=bool(
            payload.get("useFilmConditioning", False)
        ),
        filmDropout=float(payload.get("filmDropout", 0.0)),
        usePerBlockFilm=bool(payload.get("usePerBlockFilm", False)),
        auxPoolAlignmentEnabled=bool(
            payload.get("auxPoolAlignmentEnabled", False)
        ),
        x0AlignmentEnabled=bool(payload.get("x0AlignmentEnabled", False)),
        x0AlignmentEmbedDim=int(payload.get("x0AlignmentEmbedDim", 256)),
        x0AlignmentNumLayers=int(payload.get("x0AlignmentNumLayers", 2)),
        x0AlignmentNumHeads=int(payload.get("x0AlignmentNumHeads", 4)),
        useSelfConditioning=bool(
            payload.get("useSelfConditioning", False)
        ),
    )


def _scheduleConfigToDict(config: NoiseScheduleConfig) -> dict[str, Any]:
    return {
        "numSteps": config.numSteps,
        "scheduleType": config.scheduleType,
        "linearBetaStart": config.linearBetaStart,
        "linearBetaEnd": config.linearBetaEnd,
        "cosineS": config.cosineS,
    }


def _scheduleConfigFromDict(payload: dict[str, Any]) -> NoiseScheduleConfig:
    return NoiseScheduleConfig(
        numSteps=int(payload["numSteps"]),
        scheduleType=str(payload["scheduleType"]),
        linearBetaStart=float(payload["linearBetaStart"]),
        linearBetaEnd=float(payload["linearBetaEnd"]),
        cosineS=float(payload["cosineS"]),
    )
