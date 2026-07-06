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
    includeCustomPrompts : bool
        When True include converted custom prompts (#Simple/#Advanced).
    """

    debugMode: bool = False
    includeCustomPrompts: bool = True


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
    promptRoot: Path
    promptSources: List[Path]
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
    includeCustomPrompts : bool
        When True include converted custom prompts (#Simple/#Advanced).
    """

    animationExtension: str = ".npz"
    promptTextExtension: str = ".txt"
    fallbackFps: int = 60
    includeCustomPrompts: bool = True


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


@dataclass(frozen=True)
class PreprocessDatasetPaths:
    """
    Filesystem locations used by dataset preprocessing.

    Attributes
    ----------
    inputRoot : Path
        Root directory containing prompt.json and animation.json files.
    outputRoot : Path
        Destination directory for the preprocessed dataset.
    includeFolders : Optional[List[str]]
        Optional top-level folders to include (e.g., KIT, CMU).
    networkConfigPath : Optional[Path]
        Optional path to the shared network.yaml feature toggles.
    """

    inputRoot: Path
    outputRoot: Path
    includeFolders: Optional[List[str]] = None
    networkConfigPath: Optional[Path] = None


@dataclass(frozen=True)
class PreprocessDatasetProcessing:
    """
    Behavioural settings for dataset preprocessing.

    Attributes
    ----------
    modelName : str
        Hugging Face tokenizer identifier used for preprocessing.
    maxPromptLength : int
        Token length used for preprocessing.
    sampleShardSize : int
        Number of motion samples stored per shard.
    textShardSize : int
        Number of unique text entries stored per shard.
    textBatchSize : int
        Number of unique texts encoded per batch while building pooled cache.
    splitFrames : Optional[int]
        Split segments into windows of this size (None to disable).
    downsampleTargetFrames : Optional[int]
        Downsample each segment to this frame count (None to disable).
    maxSegmentFrames : Optional[int]
        Drop samples whose final frame count exceeds this limit.
    """

    modelName: str
    maxPromptLength: int
    sampleShardSize: int
    textShardSize: int = 2048
    textBatchSize: int = 64
    splitFrames: Optional[int] = None
    downsampleTargetFrames: Optional[int] = None
    maxSegmentFrames: Optional[int] = None

    @property
    def shardSize(self) -> int:
        """Legacy alias for the motion shard size."""
        return self.sampleShardSize


@dataclass(frozen=True)
class PreprocessDatasetConfig:
    """
    Configuration used by the dataset preprocessor CLI.

    Attributes
    ----------
    paths : PreprocessDatasetPaths
        Filesystem layout for the preprocessor.
    processing : PreprocessDatasetProcessing
        Preprocessing options for motions and prompts.
    """

    paths: PreprocessDatasetPaths
    processing: PreprocessDatasetProcessing


@dataclass(frozen=True)
class PreprocessedDatasetShardInfo:
    """
    Manifest entry describing a single shard.

    Attributes
    ----------
    path : str
        Relative path to the shard file from the dataset root.
    sampleCount : int
        Number of samples stored in the shard.
    """

    path: str
    sampleCount: int


@dataclass(frozen=True)
class PreprocessedSampleIndexV2:
    """Index entry pointing to a preprocessed motion sample."""

    sampleId: int
    shardIndex: int
    shardOffset: int
    frames: int
    sampleBytes: int
    datasetFolder: str
    sourceFile: str
    startFrame: int
    endFrame: int


@dataclass(frozen=True)
class PreprocessedTextIndexV2:
    """Index entry pointing to a unique text payload."""

    textId: int
    shardIndex: int
    shardOffset: int
    usageCount: int


@dataclass(frozen=True)
class PreprocessedLinkIndexV2:
    """Index entry linking one text and one motion sample."""

    linkId: int
    sampleId: int
    textId: int
    datasetFolder: str
    sourceFile: str
    frames: int
    pairBytes: int


@dataclass(frozen=True)
class PreprocessedDatasetManifestV2:
    """Dataset-level metadata for the V2 preprocessed dataset format."""

    version: int
    modelName: str
    maxPromptLength: int
    splitFrames: Optional[int]
    downsampleTargetFrames: Optional[int]
    maxSegmentFrames: Optional[int]
    sampleShardSize: int
    textShardSize: int
    totalSamples: int
    totalTexts: int
    totalLinks: int
    averageSampleBytes: float
    maxSampleBytes: int
    averagePairBytes: float
    averageFrames: float
    maxFrames: int
    sampleShards: List[PreprocessedDatasetShardInfo]
    textShards: List[PreprocessedDatasetShardInfo]
    sampleIndexPath: str
    textIndexPath: str
    linkIndexPath: str
    enabledComponents: List[str] = field(default_factory=list)
    pooledTextDim: int = 0


@dataclass(frozen=True)
class GenerationTextCacheManifest:
    """Manifest describing a checkpoint-specific cached text embedding set."""

    version: int
    clipFingerprint: str
    embedDim: int
    totalTexts: int
    shardSize: int
    shards: List[PreprocessedDatasetShardInfo]


# Backward-compatible aliases used by existing imports.
PreprocessedTextIndex = PreprocessedTextIndexV2
PreprocessedLinkIndex = PreprocessedLinkIndexV2
PreprocessedSampleIndex = PreprocessedSampleIndexV2
PreprocessedDatasetManifest = PreprocessedDatasetManifestV2

@dataclass(frozen=True)
class PromptRecord:
    """Single row from the HumanML3D index CSV."""

    animationRelativePath: Path
    startFrame: int
    endFrame: int
    promptFile: str


@dataclass(frozen=True)
class PromptSegment:
    """Prompt slice covering a contiguous frame range."""

    startFrame: int
    endFrame: int
    text: str
    sourceFile: str


@dataclass(frozen=True)
class PromptSample:
    """Bundled prompt records for a single animation."""

    relativePath: Path
    records: List[PromptRecord]


@dataclass(frozen=True)
class ConvertedPrompt:
    """Custom prompt exported from a converted dataset sample."""

    simple: str
    advanced: str
    promptIdentifier: str


@dataclass(frozen=True)
class AnimationSample:
    """Subset of fields loaded from an AMASS npz file."""

    relativePath: Path
    resolvedPath: Path
    axisAngles: np.ndarray
    fps: int
    extras: Dict[str, object]
