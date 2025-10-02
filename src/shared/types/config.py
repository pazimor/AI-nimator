"""Configuration dataclasses shared across features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DeviceSelectionOptions:
    """Selection constraints for choosing the runtime device."""

    requestedBackend: Optional[str]
    allowCuda: bool = True
    allowDirectML: bool = True
    allowMps: bool = True
    requireGpu: bool = False


@dataclass
class TrainingConfiguration:
    """Configuration parameters for training the diffusion model."""

    dataDirectory: Path
    saveDirectory: Path
    experimentName: str
    epochs: int
    batchSize: int
    learningRate: float
    sequenceFrames: int
    contextHistory: int
    contextTrainMode: str
    contextTrainRatio: float
    modelDimension: int
    layerCount: int
    expertCount: int
    expertTopK: int
    checkpointInterval: int
    resumePath: Optional[Path]
    maximumTokens: int
    textModelName: str
    prepareQat: bool
    randomSeed: int
    validationSplit: float
    validationInterval: int
    successThresholdDegrees: float
    targetSuccessRate: float
    cacheOnDevice: bool
    maximumValidationSamples: int
    recacheEveryEpoch: bool
    deviceOptions: DeviceSelectionOptions


@dataclass
class SamplingConfiguration:
    """Configuration parameters for inference sampling."""

    checkpointPath: Path
    promptsPath: Path
    outputPath: Path
    frameCount: int
    steps: int
    guidance: float
    contextJsons: List[Path]
    boneNames: List[str]
    omitMetadata: bool
    textModelName: Optional[str]
    deviceOptions: DeviceSelectionOptions
