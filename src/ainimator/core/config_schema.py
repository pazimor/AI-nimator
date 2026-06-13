"""Strict Pydantic v2 schemas for AI-nimator v2 training configs.

All models use ``extra="forbid"`` so unknown YAML keys raise a
``ValidationError`` that names the offending field — this is the
primary guard against silent misconfiguration.

The schema covers:

* :class:`V2FullTrainingConfigSchema` — the full multi-sample run.
* :class:`V2TrainingConfigSchema` — the overfit/sanity-check run.
* :class:`MotionDenoiserV2ConfigSchema` — denoiser architecture.
* :class:`TextEncoderConfigSchema` — text encoder architecture.
* :class:`DiffusionConfigSchema` — schedule and prediction mode.
* :class:`LossesConfigSchema` — every loss weight.

Defaults must match the runtime dataclass defaults in
``full_training_v2.py`` and ``training_v2.py`` exactly — any
divergence is a bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, field_validator, model_validator


# =====================================================================
# Sub-schemas
# =====================================================================

_PREDICTION_MODES = Literal["v", "x0", "epsilon"]
_SCHEDULE_TYPES = Literal["linear", "cosine"]
_ENCODER_TYPES = Literal["custom", "clip"]
_BEST_METRICS = Literal[
    "loss_total",
    "loss_diffusion",
    "loss_bone",
    "loss_global",
    "loss_vel_xyz",
    "loss_joint_xyz",
    "loss_foot_contact",
    "loss_clip_guidance",
    "loss_clip_aux_pool",
    "loss_x0_contrastive",
]


class DiffusionConfigSchema(BaseModel, extra="forbid"):
    """Noise schedule and prediction-target settings.

    Attributes
    ----------
    diffusionStepsTraining : int
        Number of timesteps in the training noise schedule.
    scheduleType : str
        Beta schedule shape (``"cosine"`` or ``"linear"``).
    predictionMode : str
        Network output target (``"v"``, ``"x0"``, ``"epsilon"``).
    minSnrGamma : float
        Min-SNR-γ clipping threshold (0 disables; default 5.0).
    """

    diffusionStepsTraining: int = 1000
    scheduleType: _SCHEDULE_TYPES = "cosine"
    predictionMode: _PREDICTION_MODES = "v"
    minSnrGamma: float = 5.0


class TextEncoderConfigSchema(BaseModel, extra="forbid"):
    """Architecture knobs for the custom BPE text encoder.

    Attributes
    ----------
    encoderHiddenDim : int
        Embedding width of the transformer encoder.
    encoderNumLayers : int
        Number of transformer layers.
    encoderNumHeads : int
        Number of attention heads.
    encoderType : str
        ``"custom"`` (default BPE) or ``"clip"`` (frozen CLIP tower).
    clipModelName : str
        HuggingFace model id (only when encoderType=="clip").
    clipMaxLength : int
        Max token length for CLIP tokenizer (1-77).
    encoderLrMultiplier : float
        Text encoder LR = base LR * this multiplier.
    useNullEmbedding : bool
        Use a learnable null embedding for CFG dropout.
    encoderArtifactPath : Optional[Path]
        Phase A7 — path to a standalone encoder artifact directory
        produced by :func:`ainimator.text.artifact.saveEncoderArtifact`.
        When set, the generation run loads the encoder FROM this
        artifact instead of constructing it inline.  The artifact
        must be compatible with ``encoderType``.
    encoderTrainable : bool
        Phase A7 — when ``True`` (default) the encoder is fine-tuned
        jointly with the denoiser and its weights are saved in the
        generation checkpoint.  When ``False``, the encoder is frozen
        and its weights are omitted from the checkpoint (reference +
        artifact hash only).
    """

    encoderHiddenDim: int = 256
    encoderNumLayers: int = 4
    encoderNumHeads: int = 8
    encoderType: _ENCODER_TYPES = "custom"
    clipModelName: str = "openai/clip-vit-base-patch32"
    clipMaxLength: int = 32
    encoderLrMultiplier: float = 3.0
    useNullEmbedding: bool = True
    # Phase A7 — standalone encoder artifact (§2.9).
    encoderArtifactPath: Optional[Path] = None
    encoderTrainable: bool = True

    @field_validator("clipMaxLength")
    @classmethod
    def _checkClipMaxLength(cls, value: int) -> int:
        """Ensure clipMaxLength is within CLIP's 77-token limit."""
        if not (1 <= value <= 77):
            raise ValueError(
                f"clipMaxLength must be in [1, 77]; got {value}."
            )
        return value


class MotionDenoiserV2ConfigSchema(BaseModel, extra="forbid"):
    """Architecture knobs for MotionDenoiserV2.

    Attributes
    ----------
    denoiserEmbedDim : int
        Width of the denoiser transformer.
    denoiserNumLayers : int
        Number of denoiser blocks.
    denoiserNumHeads : int
        Number of attention heads.
    maxFrames : int
        Frame cap (longer samples are truncated).
    dropout : float
        Dropout probability inside encoder + denoiser.
    useFilmConditioning : bool
        Add a global FiLM conditioning shortcut.
    filmDropout : float
        Dropout inside the FiLM MLP.
    filmInitStd : float
        Std of the FiLM/AdaLN projection init.
    usePerBlockFilm : bool
        Enable per-block AdaLN-style FiLM modulation.
    useSelfConditioning : bool
        Enable self-conditioning (Analog Bits).
    selfConditioningProb : float
        Probability of running the self-cond estimate pass.
    """

    denoiserEmbedDim: int = 384
    denoiserNumLayers: int = 4
    denoiserNumHeads: int = 8
    maxFrames: int = 256
    dropout: float = 0.0
    useFilmConditioning: bool = True
    filmDropout: float = 0.0
    filmInitStd: float = 0.1
    usePerBlockFilm: bool = True
    useSelfConditioning: bool = False
    selfConditioningProb: float = 0.5


class LossesConfigSchema(BaseModel, extra="forbid"):
    """Weights for every loss component.

    Attributes
    ----------
    velocityXyzWeight : float
        FK-velocity loss weight (0 disables).
    jointPositionWeight : float
        MDM-style FK joint-position loss weight (0 disables).
    footContactWeight : float
        Anti-foot-skating loss weight (0 disables).
    clipGuidanceWeight : float
        Text-motion InfoNCE contrastive loss weight.
    contrastiveTemperature : float
        InfoNCE softmax temperature.
    contrastiveBankSize : int
        FIFO memory bank size for extra negatives (0 disables).
    clipGuidanceWeightStart : float
        Initial contrastive weight for linear warmup.
    clipGuidanceWarmupEpochs : int
        Epochs to linearly anneal the contrastive weight.
    auxPoolContrastiveWeight : float
        Auxiliary InfoNCE on the raw encoder pool (0 disables).
    x0ContrastiveWeight : float
        Generated-motion (x0) InfoNCE contrastive (0 disables).
    """

    velocityXyzWeight: float = 0.0
    jointPositionWeight: float = 0.0
    footContactWeight: float = 0.0
    clipGuidanceWeight: float = 0.6
    contrastiveTemperature: float = 0.1
    contrastiveBankSize: int = 256
    clipGuidanceWeightStart: float = 1.0
    clipGuidanceWarmupEpochs: int = 15
    auxPoolContrastiveWeight: float = 0.5
    x0ContrastiveWeight: float = 0.0


class HealthConfigSchema(BaseModel, extra="forbid"):
    """Health monitoring configuration.

    Attributes
    ----------
    enabled : bool
        Enable HealthHub step() calls during training.
    everySteps : int
        Capture health metrics every N optimiser steps.
    """

    enabled: bool = True
    everySteps: int = 50

    @field_validator("everySteps")
    @classmethod
    def _checkEverySteps(cls, value: int) -> int:
        """Ensure everySteps is positive."""
        if value < 1:
            raise ValueError("everySteps must be >= 1.")
        return value


class ValidationConfigSchema(BaseModel, extra="forbid"):
    """Validation and checkpoint-selection settings.

    Attributes
    ----------
    validationFraction : float
        Fraction of the filtered links reserved for validation.
    validationSeed : int
        RNG seed used for the deterministic train/val split.
    validateEveryEpochs : int
        Run a validation pass every N epochs.
    bestMetric : str
        Validation key driving the best-checkpoint selection.
    bestImprovementMin : float
        Minimum absolute drop on bestMetric for a real improvement.
    stagnationPatience : int
        Log a warning after this many epochs without improvement.
    """

    validationFraction: float = 0.10
    validationSeed: int = 42
    validateEveryEpochs: int = 1
    bestMetric: _BEST_METRICS = "loss_total"
    bestImprovementMin: float = 0.0
    stagnationPatience: int = 0


class RegularisationConfigSchema(BaseModel, extra="forbid"):
    """Regularisation and augmentation settings.

    Attributes
    ----------
    condMaskProb : float
        CFG dropout — probability of zeroing the text embedding.
    mirrorProb : float
        SMPL-22 left/right mirror augmentation probability.
    emaDecay : float
        EMA decay on encoder + denoiser (0 disables).
    emaUseWarmup : bool
        Ramp the EMA decay during the first steps.
    weightDecay : float
        AdamW L2 regularisation weight decay.
    """

    condMaskProb: float = 0.20
    mirrorProb: float = 0.0
    emaDecay: float = 0.0
    emaUseWarmup: bool = True
    weightDecay: float = 3e-4


# =====================================================================
# Top-level schemas
# =====================================================================

class V2FullTrainingConfigSchema(BaseModel, extra="forbid"):
    """Complete schema for :class:`V2FullTrainingConfig`.

    All fields are optional (have defaults) except ``datasetRoot``,
    ``tokenizerDir``, and ``outputDir`` which must be supplied.

    Attributes
    ----------
    datasetRoot : Path
        Root of the preprocessed dataset.
    tokenizerDir : Path
        Directory holding ``tokenizer.json`` + ``config.json``.
    outputDir : Path
        Where to write checkpoints and ``resolved_config.yaml``.
    datasetFolders : Optional[Tuple[str, ...]]
        Folder filter (``None`` = all folders).
    sampleLinkIndices : Optional[Tuple[int, ...]]
        Small-N controlled mode (exact link indices).
    epochs : int
        Number of training epochs.
    batchSize : int
        Batch size.
    gradientAccumulation : int
        Micro-batches accumulated before an optimiser step.
    learningRate : float
        Base AdamW learning rate.
    normalizerFitMaxSamples : int
        Cap for the streaming normalizer fit.
    maxSamplesPerEpoch : int
        Random sub-sample cap per epoch (0 = no cap).
    logEvery : int
        Log every N optimisation steps.
    seed : int
        RNG seed.
    device : str
        Torch device (``"auto"``, ``"mps"``, ``"cuda"``, ``"cpu"``).
    resumeCheckpoint : Optional[Path]
        Checkpoint to resume from.
    diffusion : DiffusionConfigSchema
        Noise schedule and prediction settings.
    encoder : TextEncoderConfigSchema
        Text encoder architecture.
    denoiser : MotionDenoiserV2ConfigSchema
        Motion denoiser architecture.
    losses : LossesConfigSchema
        Loss component weights.
    validation : ValidationConfigSchema
        Validation and checkpoint-selection settings.
    regularisation : RegularisationConfigSchema
        Regularisation and augmentation settings.
    """

    datasetRoot: Path
    tokenizerDir: Path
    outputDir: Path
    datasetFolders: Optional[Tuple[str, ...]] = None
    sampleLinkIndices: Optional[Tuple[int, ...]] = None

    epochs: int = 100
    batchSize: int = 8
    gradientAccumulation: int = 1
    learningRate: float = 1e-4
    normalizerFitMaxSamples: int = 2000
    maxSamplesPerEpoch: int = 5000
    logEvery: int = 50
    seed: int = 0
    device: str = "auto"
    resumeCheckpoint: Optional[Path] = None

    diffusion: DiffusionConfigSchema = DiffusionConfigSchema()
    encoder: TextEncoderConfigSchema = TextEncoderConfigSchema()
    denoiser: MotionDenoiserV2ConfigSchema = MotionDenoiserV2ConfigSchema(
        dropout=0.1
    )
    losses: LossesConfigSchema = LossesConfigSchema()
    validation: ValidationConfigSchema = ValidationConfigSchema()
    regularisation: RegularisationConfigSchema = RegularisationConfigSchema()
    health: HealthConfigSchema = HealthConfigSchema()

    @model_validator(mode="after")
    def _validateConsistency(self) -> "V2FullTrainingConfigSchema":
        """Cross-field consistency checks."""
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if self.batchSize < 1:
            raise ValueError("batchSize must be >= 1.")
        if self.gradientAccumulation < 1:
            raise ValueError("gradientAccumulation must be >= 1.")
        if self.maxSamplesPerEpoch < 0:
            raise ValueError("maxSamplesPerEpoch must be >= 0.")
        if not (0.0 < self.validation.validationFraction < 1.0):
            raise ValueError(
                "validationFraction must be in (0, 1) exclusive."
            )
        if self.regularisation.emaDecay < 0.0 or (
            self.regularisation.emaDecay >= 1.0
        ):
            raise ValueError("emaDecay must be in [0, 1).")
        if self.losses.clipGuidanceWeightStart < self.losses.clipGuidanceWeight:
            raise ValueError(
                "clipGuidanceWeightStart must be >= clipGuidanceWeight."
            )
        return self


class V2TrainingConfigSchema(BaseModel, extra="forbid"):
    """Complete schema for :class:`V2TrainingConfig` (overfit loop).

    Attributes
    ----------
    datasetRoot : Path
        Root of the preprocessed dataset.
    tokenizerDir : Path
        Directory holding ``tokenizer.json`` + ``config.json``.
    outputDir : Path
        Where to write the checkpoint and ``resolved_config.yaml``.
    sampleLinkIndex : int
        Index into the link table (picks the overfit sample).
    epochs : int
        Number of optimisation epochs.
    learningRate : float
        AdamW learning rate.
    weightDecay : float
        AdamW weight decay.
    framesPerStep : int
        Sliding-window size (0 = full sample).
    seed : int
        RNG seed.
    logEvery : int
        Print a loss line every N epochs.
    device : str
        Torch device.
    dropout : float
        Dropout probability.
    condMaskProb : float
        CFG dropout probability.
    diffusion : DiffusionConfigSchema
        Noise schedule and prediction settings.
    encoder : TextEncoderConfigSchema
        Text encoder architecture.
    denoiser : MotionDenoiserV2ConfigSchema
        Motion denoiser architecture.
    """

    datasetRoot: Path
    tokenizerDir: Path
    outputDir: Path
    sampleLinkIndex: int = 0
    epochs: int = 200
    learningRate: float = 1e-4
    weightDecay: float = 0.0
    framesPerStep: int = 0
    seed: int = 0
    logEvery: int = 10
    device: str = "auto"
    dropout: float = 0.0
    condMaskProb: float = 0.0
    diffusion: DiffusionConfigSchema = DiffusionConfigSchema()
    encoder: TextEncoderConfigSchema = TextEncoderConfigSchema()
    denoiser: MotionDenoiserV2ConfigSchema = MotionDenoiserV2ConfigSchema()
    health: HealthConfigSchema = HealthConfigSchema()

    @model_validator(mode="after")
    def _validateConsistency(self) -> "V2TrainingConfigSchema":
        """Cross-field consistency checks."""
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if not (0.0 <= self.condMaskProb <= 1.0):
            raise ValueError("condMaskProb must be in [0, 1].")
        return self
