"""Generation-specific dataclasses shared across features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

PREDICTION_TARGET_X0 = "x0"


@dataclass(frozen=True)
class GenerationTrainingPaths:
    """
    Filesystem locations used during generation training.

    Attributes
    ----------
    datasetRoot : Path
        Root directory containing prompt and animation files.
    clipCheckpoint : Path
        Path to the pre-trained CLIP model checkpoint.
    checkpointDir : Path
        Directory to save generation model checkpoints.
    validationIndices : Optional[Path]
        Optional path for fixed validation indices.
    datasetFolders : Optional[List[str]]
        Optional top-level folders to include during training.
    """

    datasetRoot: Path
    clipCheckpoint: Path
    checkpointDir: Path
    validationIndices: Optional[Path] = None
    datasetFolders: Optional[List[str]] = None


@dataclass(frozen=True)
class GenerationTrainingHyperparameters:
    """
    Hyperparameters controlling the generation training loop.

    Attributes
    ----------
    batchSize : int
        Batch size passed to the dataloader.
    epochs : int
        Number of full epochs to run.
    device : str
        Requested device backend ("auto", "cuda", "cpu", "mps").
    validationSplit : float
        Fraction of dataset reserved for validation (0.0-1.0).
    earlyStoppingPatience : int
        Number of epochs without improvement before stopping.
    maxPromptLength : int
        Maximum token length for text prompts.
    modelName : str
        Hugging Face identifier for the XLM-Roberta tokenizer.
    resumeCheckpoint : Optional[Path]
        Path to a checkpoint file to resume training from.
    maxSamplesPerEpoch : Optional[int]
        Optional cap for samples per epoch.
    fixedTrainChunk : bool
        When True, reuse the same training chunk each epoch.
    overfitSamples : Optional[str]
        Optional overfit selector. Accepts either a fixed subset size
        ("16") or a 1-based inclusive range ("3:6").
    clearMpsCache : bool
        When False, skip explicit calls to torch.mps.empty_cache().
    deterministicCorruption : bool
        When True, reuse a stable timestep/noise pair per sample to make
        overfit runs deterministic across epochs.
    disableDropout : bool
        When True, force dropout modules to probability 0 during training.
    
    Learning Rate
    -------------
    learningRate : float
        Constant learning rate used during training.

    Loss Configuration
    -----------------
    xyzWeight : float
        Base XYZ reconstruction loss weight.
    xyzWeightSchedule : str
        Schedule mode for XYZ weighting.
    velXyzWeight : float
        Velocity matching weight in joint XYZ space.
    diffusionWeight : float
        Weight for the main x0 diffusion loss.
    accelerationWeight : float
        Weight for acceleration regularization loss.
    clipGuidanceWeight : float
        Weight for the auxiliary CLIP text-motion alignment guidance loss.
    """

    batchSize: int
    epochs: int
    device: str = "auto"
    validationSplit: float = 0.1
    earlyStoppingPatience: int = 5
    maxPromptLength: int = 64
    modelName: str = "xlm-roberta-base"
    resumeCheckpoint: Optional[Path] = None
    MM_memoryLimitGB: float = 0.0  # Memory limit in GB (0 = disabled)
    gradientAccumulation: int = 1  # Accumulate gradients over N batches
    maxSamplesPerEpoch: Optional[int] = None
    fixedTrainChunk: bool = False
    overfitSamples: Optional[str] = None
    clearMpsCache: bool = True
    deterministicCorruption: bool = False
    disableDropout: bool = False
    
    # Learning Rate
    learningRate: float = 0.001
    xyzWeight: float = 0.1
    xyzWeightSchedule: str = "none"
    velXyzWeight: float = 0.01
    velXyzWeightSchedule: str = "none"
    diffusionWeight: float = 1.0
    accelerationWeight: float = 0.0
    clipGuidanceWeight: float = 0.0
    footSkatingWeight: float = 0.0
    minSnrGamma: float = 5.0


@dataclass(frozen=True)
class GenerationTrainingConfig:
    """
    Configuration loaded from the generation training YAML file.

    Attributes
    ----------
    paths : GenerationTrainingPaths
        Filesystem layout for dataset and checkpoints.
    training : GenerationTrainingHyperparameters
        Hyperparameters controlling optimization.
    networkConfigPath : Optional[Path]
        Path to network.yaml for architecture configuration.
    """

    paths: GenerationTrainingPaths
    training: GenerationTrainingHyperparameters
    networkConfigPath: Optional[Path] = None


@dataclass(frozen=True)
class GenerationInferenceConfig:
    """
    Configuration for motion generation inference.

    Attributes
    ----------
    checkpoint : Path
        Path to the trained generation model checkpoint.
    prompt : str
        Text prompt describing the motion.
    frames : int
        Number of frames to generate.
    output : Path
        Path to save the generated motion.
    device : str
        Device backend for inference.
    ddimSteps : int
        Number of DDIM sampling steps (can be less than training steps).
    """

    checkpoint: Path
    prompt: str
    frames: int
    output: Path
    device: str = "auto"
    ddimSteps: int = 50


@dataclass(frozen=True)
class GenerationModelSettings:
    """
    Settings required to build a generation model for inference.

    Attributes
    ----------
    modelName : str
        Hugging Face identifier for the text encoder.
    clipCheckpoint : Optional[Path]
        Optional path to the CLIP checkpoint weights.
    networkConfigPath : Optional[Path]
        Optional path to the network configuration file.
    profile : Optional[str]
        Optional network profile name to load.
    maxPromptLength : int
        Maximum tokenizer prompt length used during generation.
    """

    modelName: str
    clipCheckpoint: Optional[Path]
    networkConfigPath: Optional[Path] = None
    profile: Optional[str] = None
    maxPromptLength: int = 64


@dataclass(frozen=True)
class GenerationOutputOptions:
    """
    Output locations and export options for generated animations.

    Attributes
    ----------
    jsonPath : Path
        Destination path for the JSON animation payload.
    daePath : Path
        Destination path for the Collada export.
    fps : Optional[int]
        Optional frames per second override.
    colladaInterpolation : str
        Interpolation mode written in Collada samplers ("linear" or "step").
    zeroRootTranslation : bool
        Zero root translation during Collada export.
    anchorRootTranslation : bool
        Anchor root translation during Collada export.
    """

    jsonPath: Path
    daePath: Path
    fps: Optional[int] = None
    colladaInterpolation: str = "linear"
    zeroRootTranslation: bool = False
    anchorRootTranslation: bool = False


@dataclass(frozen=True)
class GenerationTrainingResult:
    """
    Outcome of a generation training run.

    Attributes
    ----------
    epochsRun : int
        Number of epochs executed.
    finalLoss : float
        Loss value obtained at the end of training.
    device : str
        Resolved device identifier.
    """

    epochsRun: int
    finalLoss: float
    device: str
