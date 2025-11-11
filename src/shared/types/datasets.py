"""Dataset-related dataclasses shared across features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetPaths:
    """
    Bundle every filesystem path required to build datasets.

    Attributes
    ----------
    convertedDataset : Path
        Directory containing the current converted dataset (may be missing).
    baseDataset : Path
        Root directory containing the original AMASS assets.
    outputDataset : Path
        Directory where the refreshed dataset will be written.
    humanMl3dMapping : Path
        CSV file providing the HumanML3D prompts mapping.
    """

    convertedDataset: Path
    baseDataset: Path
    outputDataset: Path
    humanMl3dMapping: Path


@dataclass(frozen=True)
class DatasetBuildOptions:
    """
    Options controlling dataset build behaviour.

    Attributes
    ----------
    debugMode : bool
        When True the builder raises immediately on the first error.
    progressStyle : str
        Either "auto", "rich", "tqdm" or "none" to control progress display.
    maxSamples : Optional[int]
        Optional hard limit used for dry-runs or debugging sessions.
    """

    debugMode: bool = False
    progressStyle: str = "auto"
    maxSamples: Optional[int] = None


@dataclass
class PromptData:
    """
    Canonical prompt representation reused everywhere.

    Attributes
    ----------
    text : str
        Primary textual description.
    metadata : Dict[str, str]
        Additional metadata fields.
    humanMl3d : Dict[str, str]
        Enriched mapping extracted from the HumanML3D CSV.
    """

    text: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    humanMl3d: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetBuildReport:
    """
    Summary describing a dataset build run outcome.

    Attributes
    ----------
    processedSamples : int
        Number of samples written to the output dataset.
    failedSamples : List[str]
        Sample identifiers that could not be processed.
    outputDirectory : Path
        Directory containing the refreshed dataset.
    """

    processedSamples: int
    failedSamples: List[str]
    outputDirectory: Path


@dataclass(frozen=True)
class DatasetBuilderPaths:
    """
    Filesystem locations used by the dataset builder.

    Attributes
    ----------
    animationRoot : Path
        Directory containing AMASS-style animation files.
    promptRoot : Path
        Primary directory containing HumanML3D prompt assets.
    promptSources : List[Path]
        Additional roots searched when resolving prompt files.
    indexCsv : Path
        CSV mapping animation files to prompt segments.
    outputRoot : Path
        Destination directory receiving the rebuilt dataset.
    """

    animationRoot: Path
    animationRoots: List[Path]
    promptRoot: Path
    promptSources: List[Path]
    customPromptFiles: List[Path]
    indexCsv: Path
    outputRoot: Path
    convertedRoot: Optional[Path] = None


@dataclass(frozen=True)
class DatasetBuilderProcessing:
    """
    Behavioural settings for dataset rebuilding.

    Attributes
    ----------
    animationExtension : str
        Extension enforced when locating animation files.
    promptTextExtension : str
        Extension enforced when loading textual prompts.
    fallbackFps : int
        Default FPS used when the source animation misses one.
    """

    animationExtension: str = ".npz"
    promptTextExtension: str = ".txt"
    fallbackFps: int = 60


@dataclass(frozen=True)
class DatasetBuilderConfig:
    """
    Full configuration loaded from the YAML file.

    Attributes
    ----------
    paths : DatasetBuilderPaths
        Group of filesystem paths.
    processing : DatasetBuilderProcessing
        Processing directives for the builder pipeline.
    """

    paths: DatasetBuilderPaths
    processing: DatasetBuilderProcessing = DatasetBuilderProcessing()
