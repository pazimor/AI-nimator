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
* :class:`ControllerProfileDataSchema` — data knobs for a controller
  profile.
* :class:`ControllerProfileArchSchema` — arch knobs for a controller
  profile.
* :class:`ControllerProfileTrainingSchema` — training knobs for a
  controller profile.
* :class:`ControllerProfileSchema` — one named controller profile.

Defaults must match the runtime dataclass defaults in
``full_training_v2.py`` and ``training_v2.py`` exactly — any
divergence is a bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


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
# Goal A — deterministic controller schemas (ROADMAP_DETERMINIST §3.2)
# =====================================================================

_PHASE_MODES = Literal["none", "explicit", "learned"]
_MODEL_TYPES = Literal["diffusion", "controller"]


class ControllerLossesSchema(BaseModel, extra="forbid", populate_by_name=True):
    """Loss toggles for the controller (kebab-case in YAML).

    Attributes
    ----------
    velocityLoss : bool
        L2 loss on regressed velocities (flag shared with v2).
    footContactLoss : bool
        FK foot-contact supervision (anti-skating).
    geodesicRotation : bool
        Geodesic loss on the 6D rotations.
    """

    velocityLoss: bool = Field(True, alias="velocity-loss")
    footContactLoss: bool = Field(True, alias="foot-contact-loss")
    geodesicRotation: bool = Field(True, alias="geodesic-rotation")


class ControllerTrainingSchema(BaseModel, extra="forbid", populate_by_name=True):
    """Controller training knobs (kebab-case in YAML).

    Attributes
    ----------
    scheduledSampling : float
        Scheduled-sampling target probability (ramped 0 → target in
        A4).  Stays ``0.0`` until short-rollout validation passes.
    rolloutLossHorizon : int
        ``> 0`` adds a closed-loop rollout-loss window of that many
        transitions per optimizer step (2026-07-05 long-horizon
        divergence fix).  ``0`` (default) disables it.
    rolloutLossWeight : float
        Weight of the rollout-loss term when ``rolloutLossHorizon > 0``.
    """

    scheduledSampling: float = Field(0.0, alias="scheduled-sampling")
    rolloutLossHorizon: int = Field(0, alias="rollout-loss-horizon")
    rolloutLossWeight: float = Field(1.0, alias="rollout-loss-weight")

    @field_validator("scheduledSampling")
    @classmethod
    def _checkScheduledSampling(cls, value: float) -> float:
        """Scheduled-sampling probability must be a valid probability."""
        if not (0.0 <= value <= 1.0):
            raise ValueError("scheduled-sampling must be in [0, 1].")
        return value

    @field_validator("rolloutLossHorizon")
    @classmethod
    def _checkRolloutLossHorizon(cls, value: int) -> int:
        """The rollout-loss horizon is a transition count (0 = off)."""
        if value < 0:
            raise ValueError("rollout-loss-horizon must be >= 0.")
        return value

    @field_validator("rolloutLossWeight")
    @classmethod
    def _checkRolloutLossWeight(cls, value: float) -> float:
        """A negative rollout-loss weight would reward divergence."""
        if value < 0.0:
            raise ValueError("rollout-loss-weight must be >= 0.")
        return value


class ControllerConfigSchema(BaseModel, extra="forbid", populate_by_name=True):
    """Strict schema for the ``v2.generation.controller`` YAML block.

    ``extra="forbid"`` makes any unknown key raise a ``ValidationError``
    that names the offending field — the A0 acceptance guard.

    Attributes
    ----------
    autoregressive : bool
        Whether the controller runs autoregressively (always ``True``).
    phase : str
        Locomotor phase regime (``none`` | ``explicit`` | ``learned``).
    styleLatent : bool
        DEFERRED (A3) — inject a style latent.  Must stay ``False``
        until a style-labelled dataset exists.
    contextFrames : int
        Number of past frames seen per forward (A1 → 1).
    losses : ControllerLossesSchema
        Loss toggles.
    training : ControllerTrainingSchema
        Training knobs.
    """

    autoregressive: bool = True
    phase: _PHASE_MODES = "explicit"
    styleLatent: bool = Field(False, alias="style-latent")
    contextFrames: int = Field(1, alias="context-frames")
    losses: ControllerLossesSchema = ControllerLossesSchema()
    training: ControllerTrainingSchema = ControllerTrainingSchema()

    @field_validator("contextFrames")
    @classmethod
    def _checkContextFrames(cls, value: int) -> int:
        """Context window must be at least one frame."""
        if value < 1:
            raise ValueError("context-frames must be >= 1.")
        return value

    @model_validator(mode="after")
    def _checkStyleDeferred(self) -> "ControllerConfigSchema":
        """Guard against enabling style latents before A3 ships."""
        if self.styleLatent:
            raise ValueError(
                "style-latent must stay false: the current dataset has "
                "no style labels (ROADMAP_DETERMINIST §1/§7). Enabling "
                "it would train a style lever on data that cannot carry "
                "it. Phase A3 wires this once a labelled dataset exists."
            )
        return self


class GenerationModelSelectorSchema(
    BaseModel, extra="forbid", populate_by_name=True
):
    """Engine selector read from ``v2.generation`` (§3.2).

    Only the two Goal A fields are validated here; the diffusion
    ``generation`` keys are parsed elsewhere and intentionally ignored.

    Attributes
    ----------
    modelType : str
        ``diffusion`` (default) or ``controller``.
    controller : Optional[ControllerConfigSchema]
        Controller block; required when ``modelType == "controller"``.
    """

    modelType: _MODEL_TYPES = Field("diffusion", alias="model-type")
    controller: Optional[ControllerConfigSchema] = None

    @model_validator(mode="after")
    def _requireControllerBlock(self) -> "GenerationModelSelectorSchema":
        """A controller run must carry a controller block."""
        if self.modelType == "controller" and self.controller is None:
            raise ValueError(
                "model-type=controller requires a 'controller' block "
                "under v2.generation."
            )
        return self


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


# =====================================================================
# Goal A — controller *profile* schemas (G-PROFILES / ROADMAP_DETERMINIST §3.2)
# =====================================================================

class ControllerProfileDataSchema(
    BaseModel, extra="forbid", populate_by_name=True
):
    """Data-selection knobs for a controller profile.

    Attributes
    ----------
    sampleIndex : int
        Link index of the single clip used by ``overfit`` / ``debug``.
    minFrames : int
        Minimum frame count for clip selection (``full`` only).
    numClips : int
        Total clips to load for ``full`` profile.
    heldOutClips : int
        Clips reserved as held-out for ``full`` profile.
    clipBatchSize : int
        Clips per optimiser mini-batch for ``full`` profile.
    evalSampleClips : int
        Clips sampled for metrics evaluation in ``full`` profile.
    captionFilter : str or None
        Optional regex (case-insensitive, ``re.search``) applied to each
        candidate clip's caption (``raw_text``); only matching clips are
        eligible for selection (``full`` profile only).  ``None``
        (default) disables filtering -- unchanged behaviour.
    """

    sampleIndex: int = Field(0, alias="sample-index")
    minFrames: int = Field(0, alias="min-frames")
    numClips: int = Field(256, alias="num-clips")
    heldOutClips: int = Field(32, alias="held-out-clips")
    clipBatchSize: int = Field(8, alias="clip-batch-size")
    evalSampleClips: int = Field(32, alias="eval-sample-clips")
    captionFilter: Optional[str] = Field(None, alias="caption-filter")


class ControllerProfileArchSchema(
    BaseModel, extra="forbid", populate_by_name=True
):
    """Architecture knobs for a controller profile.

    Attributes
    ----------
    embedDim : int
        Transformer embedding width.
    numHeads : int
        Number of attention heads.
    numLayers : int
        Number of transformer blocks.
    contextFrames : int
        Autoregressive context-window length.
    phase : str
        Locomotor phase regime (``none`` | ``explicit`` | ``learned``).
    aimDirection : bool
        Append aim-direction control channels.
    """

    embedDim: int = Field(128, alias="embed-dim")
    numHeads: int = Field(4, alias="num-heads")
    numLayers: int = Field(3, alias="num-layers")
    contextFrames: int = Field(1, alias="context-frames")
    phase: _PHASE_MODES = "none"
    aimDirection: bool = Field(False, alias="aim-direction")


class ControllerProfileTrainingSchema(
    BaseModel, extra="forbid", populate_by_name=True
):
    """Training hyper-parameters for a controller profile.

    Attributes
    ----------
    epochs : int
        Number of optimisation epochs.
    learningRate : float
        AdamW learning rate.
    weightDecay : float
        AdamW L2 weight decay.
    seed : int
        RNG seed (reproducibility).
    device : str
        Torch device (``auto`` | ``cpu`` | ``mps`` | ``cuda``).
    logEvery : int
        Log a loss line every N epochs.
    scheduledSampling : float
        Target scheduled-sampling probability (A4, ``0`` = off).
    rolloutLossHorizon : int
        Closed-loop rollout-loss window length per optimizer step
        (2026-07-05 divergence fix; ``0`` = off).
    rolloutLossWeight : float
        Weight of the rollout-loss term when the horizon is > 0.
    footContactWeight : float
        Anti-skating foot-contact loss weight (``0`` = off).
    """

    epochs: int = 300
    learningRate: float = Field(1e-3, alias="learning-rate")
    weightDecay: float = Field(0.0, alias="weight-decay")
    seed: int = 0
    device: str = "auto"
    logEvery: int = Field(50, alias="log-every")
    scheduledSampling: float = Field(0.0, alias="scheduled-sampling")
    rolloutLossHorizon: int = Field(0, alias="rollout-loss-horizon")
    rolloutLossWeight: float = Field(1.0, alias="rollout-loss-weight")
    footContactWeight: float = Field(0.0, alias="foot-contact-weight")


class ControllerProfileTextSchema(
    BaseModel, extra="forbid", populate_by_name=True
):
    """Text-conditioning knobs for a controller profile.

    Attributes
    ----------
    promptEmbChannels : int
        Width of the prompt embedding (must match text_encoder output-dim).
        ``0`` disables text conditioning.
    condDropoutProb : float
        Probability of replacing a prompt embedding with the null embedding
        during training (classifier-free conditioning dropout).
    encoderArtifactPath : str or None
        Path to the frozen encoder artifact directory.  ``None`` means the
        path is supplied at runtime via ``--encoder-artifact``.
    """

    promptEmbChannels: int = Field(0, alias="prompt-emb-channels")
    condDropoutProb: float = Field(0.1, alias="cond-dropout-prob")
    encoderArtifactPath: Optional[str] = Field(
        None, alias="encoder-artifact-path"
    )


class ControllerProfileSchema(
    BaseModel, extra="forbid", populate_by_name=True
):
    """One named controller profile (delta on top of base defaults).

    Attributes
    ----------
    data : ControllerProfileDataSchema
        Data-selection knobs.
    arch : ControllerProfileArchSchema
        Architecture knobs.
    training : ControllerProfileTrainingSchema
        Training hyper-parameters.
    text : ControllerProfileTextSchema
        Text-conditioning knobs (``promptEmbChannels=0`` = off by default).
    """

    data: ControllerProfileDataSchema = ControllerProfileDataSchema()
    arch: ControllerProfileArchSchema = ControllerProfileArchSchema()
    training: ControllerProfileTrainingSchema = (
        ControllerProfileTrainingSchema()
    )
    text: ControllerProfileTextSchema = ControllerProfileTextSchema()


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
