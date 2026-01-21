"""Shared type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .clip import (
    AnimationFileDescriptor,
    ClipDatasetRecord,
    ClipPromptSegment,
    ClipTrainingConfig,
    ClipTrainingHyperparameters,
    ClipTrainingPaths,
    ClipTrainingResult,
    MotionTextSample,
    PromptFileDescriptor,
)
from .datasets import (
    DatasetBuildOptions,
    DatasetBuildReport,
    DatasetBuilderConfig,
    DatasetBuilderPaths,
    DatasetBuilderProcessing,
    DatasetPaths,
    AnimationSample,
    ConvertedPrompt,
    PreprocessDatasetConfig,
    PreprocessDatasetPaths,
    PreprocessDatasetProcessing,
    PreprocessedDatasetManifest,
    PreprocessedDatasetShardInfo,
    PreprocessedSampleIndex,
    PromptData,
    PromptRecord,
    PromptSegment,
    PromptSample,
)
from .generation import (
    VALID_TAGS,
    GenerationInferenceConfig,
    GenerationModelSettings,
    GenerationOutputOptions,
    GenerationTrainingConfig,
    GenerationTrainingHyperparameters,
    GenerationTrainingPaths,
    GenerationTrainingResult,
    validateTag,
)
from .network import (
    ClipNetworkConfig,
    GenerationNetworkConfig,
    LearningRateHyperparameters,
    NetworkConfig,
)


@dataclass(frozen=True)
class DeviceSelectionOptions:
    """
    CLI-facing options used to resolve a `torch.device`.

    Attributes
    ----------
    requestedDevice : Optional[str]
        Desired backend ("cuda", "dml", "mps", "cpu" or "auto").
    allowCuda : bool
        True when CUDA backends can be used.
    allowDirectML : bool
        True when DirectML backend can be used.
    allowMps : bool
        True when Apple MPS backend can be used.
    requireGpu : bool
        When True an exception is raised if no GPU backend is available.
    """

    requestedDevice: Optional[str] = None
    allowCuda: bool = True
    allowDirectML: bool = True
    allowMps: bool = True
    requireGpu: bool = False


__all__ = [
    "DatasetBuildOptions",
    "DatasetBuildReport",
    "DatasetBuilderConfig",
    "DatasetBuilderPaths",
    "DatasetBuilderProcessing",
    "DatasetPaths",
    "AnimationFileDescriptor",
    "ClipDatasetRecord",
    "ClipPromptSegment",
    "ClipTrainingConfig",
    "ClipTrainingHyperparameters",
    "ClipTrainingPaths",
    "ClipTrainingResult",
    "MotionTextSample",
    "PromptFileDescriptor",
    "DeviceSelectionOptions",
    "PromptData",
    "PromptRecord",
    "PromptSegment",
    "PromptSample",
    "ConvertedPrompt",
    "AnimationSample",
    "PreprocessDatasetConfig",
    "PreprocessDatasetPaths",
    "PreprocessDatasetProcessing",
    "PreprocessedDatasetManifest",
    "PreprocessedDatasetShardInfo",
    "PreprocessedSampleIndex",
    "VALID_TAGS",
    "GenerationInferenceConfig",
    "GenerationModelSettings",
    "GenerationOutputOptions",
    "GenerationTrainingConfig",
    "GenerationTrainingHyperparameters",
    "GenerationTrainingPaths",
    "GenerationTrainingResult",
    "validateTag",
    # Network and LR types
    "ClipNetworkConfig",
    "GenerationNetworkConfig",
    "LearningRateHyperparameters",
    "NetworkConfig",
]
