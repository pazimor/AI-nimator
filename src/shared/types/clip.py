"""CLIP-specific dataclasses shared across features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

##TODO: certaines classes font doublons avec d'autres types 
@dataclass(frozen=True)
class PromptFileDescriptor:
    """
    Describe a prompt asset on disk.

    Attributes
    ----------
    path : Path
        Filesystem path to the prompt JSON file.
    tag : str
        High-level label attached to the prompt file.
    """

    path: Path
    tag: str


@dataclass(frozen=True)
class AnimationFileDescriptor:
    """
    Describe an animation asset on disk.

    Attributes
    ----------
    path : Path
        Filesystem path to the animation payload.
    metadata : Dict[str, object]
        Metadata attached to the animation file.
    """

    path: Path
    metadata: Dict[str, object]


@dataclass
class ClipPromptSegment:
    """
    Prompt slice aligned with a motion interval.

    Attributes
    ----------
    startFrame : int
        First frame included in the segment (inclusive).
    endFrame : int
        First frame excluded from the segment (exclusive).
    text : str
        Natural language description of the motion slice.
    sourceFile : str
        Original prompt source identifier.
    tag : str
        Dataset-level tag attached to the prompt file.
    metadata : Dict[str, object]
        Optional metadata extracted from the prompt file.
    """

    startFrame: int
    endFrame: int
    text: str
    sourceFile: str
    tag: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ClipDatasetRecord:
    """
    Index entry connecting a prompt segment to its animation.

    Attributes
    ----------
    promptText : str
        Text that will be tokenized for CLIP alignment.
    tag : str
        Dataset-level tag for display or conditioning.
    animationPath : Path
        Path to the animation payload associated with this record.
    startFrame : int
        First frame included in the motion slice (inclusive).
    endFrame : int
        First frame excluded from the motion slice (exclusive).
    sourceFile : str
        Identifier for traceability back to the original dataset asset.
    metadata : Dict[str, object]
        Optional metadata propagated to the dataset item.
    """

    promptText: str
    tag: str
    animationPath: Path
    startFrame: int
    endFrame: int
    sourceFile: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class MotionTextSample:
    """
    Dataset sample combining text and motion.

    Attributes
    ----------
    text : str
        User-facing textual description.
    tag : str
        Optional dataset tag.
    startFrame : int
        First frame index included.
    endFrame : int
        First frame index excluded.
    motion : torch.Tensor
        Motion payload with shape (frames, bones, 6).
    meta : Dict[str, object]
        Additional metadata propagated from the source files.
    sourceFile : Optional[Path]
        Optional original animation identifier.
    """

    text: str
    tag: str
    startFrame: int
    endFrame: int
    motion: torch.Tensor
    meta: Dict[str, object] = field(default_factory=dict)
    sourceFile: Optional[Path] = None


@dataclass(frozen=True)
class ClipTrainingPaths:
    """
    Filesystem locations used during CLIP training.

    Attributes
    ----------
    datasetRoot : Path
        Root directory containing preprocessed dataset shards.
    """

    datasetRoot: Path


@dataclass(frozen=True)
class ClipTrainingHyperparameters:
    """
    Hyperparameters controlling the CLIP training loop.

    Attributes
    ----------
    batchSize : int
        Batch size passed to the dataloader.
    maxPromptLength : int
        Maximum token length used by the tokenizer.
    modelName : str
        Hugging Face identifier for the tokenizer/model.
    epochs : int
        Number of full epochs to run.
    device : str
        Requested device backend ("auto", "cuda", "cpu", "mps").
    validationSplit : float
        Fraction of dataset reserved for validation (0.0-1.0).
    earlyStoppingPatience : int
        Number of epochs without improvement before stopping.
    checkpointDir : Optional[Path]
        Directory to save model checkpoints.
    resumeCheckpoint : Optional[Path]
        Path to a checkpoint file to resume training from.
    gradientAccumulation : int
        Number of batches to accumulate before optimizer step.
    MM_memoryLimitGB : float
        Maximum memory usage in GB before triggering cleanup (0 to disable).
    weightDecay : float
        Weight decay for regularization.
    
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
    """

    batchSize: int
    maxPromptLength: int
    modelName: str
    epochs: int
    device: str = "auto"
    validationSplit: float = 0.1
    earlyStoppingPatience: int = 3
    checkpointDir: Optional[Path] = None
    resumeCheckpoint: Optional[Path] = None
    gradientAccumulation: int = 1
    MM_memoryLimitGB: float = 0.0
    weightDecay: float = 0.0
    
    # Learning Rate Configuration
    learningRate: float = 0.001
    lrMin: float = 1e-7
    lrWarmupEpochs: int = 0
    lrSchedule: str = "cosine"
    lrDecayEpochs: Optional[int] = None


@dataclass(frozen=True)
class ClipTrainingConfig:
    """
    Configuration loaded from the CLIP training YAML file.

    Attributes
    ----------
    paths : ClipTrainingPaths
        Filesystem layout for prompts and animations.
    training : ClipTrainingHyperparameters
        Hyperparameters controlling optimization.
    networkConfigPath : Optional[Path]
        Path to network.yaml for architecture configuration.
    """

    paths: ClipTrainingPaths
    training: ClipTrainingHyperparameters
    networkConfigPath: Optional[Path] = None


@dataclass(frozen=True)
class ClipTrainingResult:
    """
    Outcome of a CLIP training run.

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
