"""Generation-specific dataclasses shared across features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Exhaustive list of valid tags for motion generation
VALID_TAGS: List[str] = [
    "Dance",
    "Combat",
    "Déplacement",
    "Idle",
    "Gesture",
    "Acrobatie",
    "Sport",
    "Dégâts subit",
    "Monture ou Véhicule",
]


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
    """

    datasetRoot: Path
    clipCheckpoint: Path
    checkpointDir: Path
    validationIndices: Optional[Path] = None


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
    
    Learning Rate Configuration
    ---------------------------
    learningRate : float
        Initial/base learning rate.
    lrMin : float
        Minimum learning rate floor.
    lrWarmupEpochs : int
        Number of warmup epochs (0 to disable).
    lrSchedule : str
        Schedule type: "constant", "cosine", "linear", "step".
    lrDecayEpochs : Optional[int]
        Decay phase length (default: epochs - warmup).

    Loss Configuration
    -----------------
    geodesicWeight : float
        Base geodesic loss weight.
    geodesicWeightSchedule : str
        Schedule mode for geodesic weighting.
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
    
    # Learning Rate Configuration
    learningRate: float = 0.001
    lrMin: float = 1e-7
    lrWarmupEpochs: int = 0
    lrSchedule: str = "cosine"
    lrDecayEpochs: Optional[int] = None
    geodesicWeight: float = 0.1
    geodesicWeightSchedule: str = "none"


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
    tag : Optional[str]
        Categorical tag from the exhaustive list, if provided.
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
    tag: Optional[str]
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
    """

    modelName: str
    clipCheckpoint: Optional[Path]
    networkConfigPath: Optional[Path] = None
    profile: Optional[str] = None


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
    zeroRootTranslation : bool
        Zero root translation during Collada export.
    anchorRootTranslation : bool
        Anchor root translation during Collada export.
    """

    jsonPath: Path
    daePath: Path
    fps: Optional[int] = None
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


def validateTag(tag: str) -> str:
    """
    Validate that a tag is in the exhaustive list.

    Parameters
    ----------
    tag : str
        Tag to validate.

    Returns
    -------
    str
        The validated tag.

    Raises
    ------
    ValueError
        Raised when the tag is not in the valid list.
    """
    if tag not in VALID_TAGS:
        validList = ", ".join(f'"{t}"' for t in VALID_TAGS)
        raise ValueError(
            f'Invalid tag "{tag}". Must be one of: {validList}'
        )
    return tag
