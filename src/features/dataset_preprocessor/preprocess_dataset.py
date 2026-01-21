"""Preprocess converted datasets into shardable tensor files."""

from __future__ import annotations

import gc
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import XLMRobertaTokenizerFast

from src.features.dataset_builder.dataset_reader import (
    loadAnimationPayload,
    loadPromptSegments,
)
from src.features.dataset_builder.progress import TqdmProgressReporter
from src.shared.constants.preprocessed import (
    PREPROCESSED_INDEX_FILENAME,
    PREPROCESSED_MANIFEST_FILENAME,
    PREPROCESSED_MANIFEST_VERSION,
    PREPROCESSED_MIN_FRAME_COUNT,
    PREPROCESSED_PROMPT_FILENAME,
    PREPROCESSED_SHARDS_DIRNAME,
    PREPROCESSED_TAG_PREFIX_TEMPLATE,
)
from src.shared.types import (
    PreprocessDatasetConfig,
    PreprocessedDatasetManifest,
    PreprocessedDatasetShardInfo,
    PreprocessedSampleIndex,
)

PROMPT_FILENAME = PREPROCESSED_PROMPT_FILENAME
MANIFEST_FILENAME = PREPROCESSED_MANIFEST_FILENAME
INDEX_FILENAME = PREPROCESSED_INDEX_FILENAME
SHARDS_DIRNAME = PREPROCESSED_SHARDS_DIRNAME
MANIFEST_VERSION = PREPROCESSED_MANIFEST_VERSION
TAG_PREFIX_TEMPLATE = PREPROCESSED_TAG_PREFIX_TEMPLATE
MIN_FRAME_COUNT = PREPROCESSED_MIN_FRAME_COUNT


class DatasetPreprocessor:
    """Convert a converted dataset into a shardable tensor dataset."""

    def __init__(self, config: PreprocessDatasetConfig) -> None:
        """
        Initialize the preprocessor.

        Parameters
        ----------
        config : PreprocessDatasetConfig
            Parsed preprocessing configuration.
        """
        self.config = config
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            config.processing.modelName,
        )
        self._writer = ShardWriter(
            outputRoot=config.paths.outputRoot,
            shardSize=config.processing.shardSize,
        )

    def run(self) -> None:
        """
        Execute preprocessing and write shards + manifest.

        Returns
        -------
        None
            Results are written to disk.
        """
        promptFiles = self._listPromptFiles(self.config.paths.inputRoot)
        reporter = TqdmProgressReporter(
            total=len(promptFiles),
            description="Dataset preprocess",
        )
        for promptPath in promptFiles:
            self._processPromptFile(promptPath)
            reporter.advance(promptPath.as_posix())
        reporter.close()
        self._writer.finalize(self._buildManifest())

    def _listPromptFiles(self, root: Path) -> List[Path]:
        """
        Return sorted prompt.json files under a root.

        Parameters
        ----------
        root : Path
            Dataset root directory.

        Returns
        -------
        List[Path]
            Sorted list of prompt.json files.
        """
        return sorted(root.rglob(PROMPT_FILENAME))

    def _processPromptFile(self, promptPath: Path) -> None:
        """
        Load a prompt file, its animation, and write preprocessed samples.

        Parameters
        ----------
        promptPath : Path
            Path to the prompt.json file.
        """
        tag, promptMeta, segments = loadPromptSegments(promptPath)
        animationPath = _resolveAnimationPath(
            promptPath,
            self.config.paths.inputRoot,
        )
        motion, motionMeta = loadAnimationPayload(animationPath)
        try:
            for segment in segments:
                self._processSegment(
                    segmentText=segment.text,
                    segmentStart=segment.startFrame,
                    segmentEnd=segment.endFrame,
                    tag=tag,
                    promptMeta=promptMeta,
                    motion=motion,
                    motionMeta=motionMeta,
                    sourceFile=segment.sourceFile,
                )
        finally:
            del motion
            gc.collect()

    def _processSegment(
        self,
        segmentText: str,
        segmentStart: int,
        segmentEnd: int,
        tag: str,
        promptMeta: Dict[str, object],
        motion: torch.Tensor,
        motionMeta: Dict[str, object],
        sourceFile: str,
    ) -> None:
        """
        Split, filter, and serialize one prompt segment.

        Parameters
        ----------
        segmentText : str
            Prompt text to tokenize.
        segmentStart : int
            Start frame for the segment.
        segmentEnd : int
            End frame for the segment.
        tag : str
            Dataset tag string.
        promptMeta : Dict[str, object]
            Prompt-level metadata.
        motion : torch.Tensor
            Full motion tensor for the animation.
        motionMeta : Dict[str, object]
            Metadata attached to the motion payload.
        sourceFile : str
            Source identifier for traceability.
        """
        windows = _buildWindows(
            segmentStart,
            segmentEnd,
            self.config.processing.splitFrames,
        )
        for windowStart, windowEnd in windows:
            motionSlice = motion[windowStart:windowEnd]
            motionSlice = _downsampleMotion(
                motionSlice,
                self.config.processing.downsampleTargetFrames,
            )
            if not _isFramesValid(
                motionSlice.shape[0],
                self.config.processing.maxSegmentFrames,
            ):
                continue
            sample = self._buildSample(
                text=segmentText,
                tag=tag,
                motionSlice=motionSlice,
                windowStart=windowStart,
                windowEnd=windowEnd,
                mergedMeta=motionMeta | promptMeta,
            )
            sampleBytes = _estimateSampleBytes(sample)
            self._writer.addSample(
                sample=sample,
                frames=motionSlice.shape[0],
                sampleBytes=sampleBytes,
                tag=tag,
                sourceFile=sourceFile,
            )

    def _buildSample(
        self,
        text: str,
        tag: str,
        motionSlice: torch.Tensor,
        windowStart: int,
        windowEnd: int,
        mergedMeta: Dict[str, object],
    ) -> Dict[str, object]:
        """
        Build a single preprocessed sample dictionary.

        Parameters
        ----------
        text : str
            Raw prompt text.
        tag : str
            Dataset tag.
        motionSlice : torch.Tensor
            Motion tensor for the window.
        windowStart : int
            Start frame for the window.
        windowEnd : int
            End frame for the window.
        mergedMeta : Dict[str, object]
            Combined metadata to store with the sample.

        Returns
        -------
        Dict[str, object]
            Sample dictionary with tensors.
        """
        encoded = _tokenizeText(
            self.tokenizer,
            text,
            tag,
            self.config.processing.maxPromptLength,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "motion": motionSlice,
            "time": torch.tensor([windowStart, windowEnd]),
            "tag": tag,
            "meta": mergedMeta,
        }

    def _buildManifest(self) -> PreprocessedDatasetManifest:
        """
        Build the manifest metadata after preprocessing.

        Returns
        -------
        PreprocessedDatasetManifest
            Manifest populated with dataset statistics.
        """
        return self._writer.buildManifest(
            modelName=self.config.processing.modelName,
            maxPromptLength=self.config.processing.maxPromptLength,
            splitFrames=self.config.processing.splitFrames,
            downsampleTargetFrames=(
                self.config.processing.downsampleTargetFrames
            ),
            maxSegmentFrames=self.config.processing.maxSegmentFrames,
        )


class ShardWriter:
    """Write preprocessed samples into shard files with an index."""

    def __init__(self, outputRoot: Path, shardSize: int) -> None:
        """
        Initialize the shard writer.

        Parameters
        ----------
        outputRoot : Path
            Root directory for output shards.
        shardSize : int
            Maximum samples per shard file.
        """
        self.outputRoot = outputRoot
        self.shardSize = shardSize
        self.shardsDir = outputRoot / SHARDS_DIRNAME
        self.shardsDir.mkdir(parents=True, exist_ok=True)
        self._currentSamples: List[Dict[str, object]] = []
        self._indexEntries: List[PreprocessedSampleIndex] = []
        self._shardInfos: List[PreprocessedDatasetShardInfo] = []
        self._sampleBytesSum = 0
        self._sampleCount = 0
        self._framesSum = 0
        self._maxSampleBytes = 0
        self._maxFrames = 0

    def addSample(
        self,
        sample: Dict[str, object],
        frames: int,
        sampleBytes: int,
        tag: str,
        sourceFile: str,
    ) -> None:
        """
        Queue a sample for shard writing.

        Parameters
        ----------
        sample : Dict[str, object]
            Preprocessed sample dictionary.
        frames : int
            Frame count for the sample.
        sampleBytes : int
            Estimated sample size in bytes.
        tag : str
            Dataset tag for the sample.
        sourceFile : str
            Source identifier for traceability.
        """
        shardIndex = len(self._shardInfos)
        shardOffset = len(self._currentSamples)
        self._currentSamples.append(sample)
        self._indexEntries.append(
            PreprocessedSampleIndex(
                shardIndex=shardIndex,
                shardOffset=shardOffset,
                frames=frames,
                sampleBytes=sampleBytes,
                tag=tag,
                sourceFile=sourceFile,
            ),
        )
        self._updateStats(frames, sampleBytes)
        if len(self._currentSamples) >= self.shardSize:
            self._flushShard()

    def finalize(self, manifest: PreprocessedDatasetManifest) -> None:
        """
        Write pending samples, index, and manifest to disk.

        Parameters
        ----------
        manifest : PreprocessedDatasetManifest
            Dataset manifest to serialize.
        """
        if self._currentSamples:
            self._flushShard()
        self._writeIndex()
        self._writeManifest(manifest)

    def buildManifest(
        self,
        modelName: str,
        maxPromptLength: int,
        splitFrames: Optional[int],
        downsampleTargetFrames: Optional[int],
        maxSegmentFrames: Optional[int],
    ) -> PreprocessedDatasetManifest:
        """
        Create a manifest dataclass with computed stats.

        Returns
        -------
        PreprocessedDatasetManifest
            Manifest with dataset-level statistics.
        """
        averageSampleBytes = _safeAverage(
            self._sampleBytesSum,
            self._sampleCount,
        )
        averageFrames = _safeAverage(self._framesSum, self._sampleCount)
        return PreprocessedDatasetManifest(
            version=MANIFEST_VERSION,
            modelName=modelName,
            maxPromptLength=maxPromptLength,
            splitFrames=splitFrames,
            downsampleTargetFrames=downsampleTargetFrames,
            maxSegmentFrames=maxSegmentFrames,
            shardSize=self.shardSize,
            totalSamples=self._sampleCount,
            averageSampleBytes=averageSampleBytes,
            maxSampleBytes=self._maxSampleBytes,
            averageFrames=averageFrames,
            maxFrames=self._maxFrames,
            shards=self._shardInfos,
            indexPath=INDEX_FILENAME,
        )

    def _updateStats(self, frames: int, sampleBytes: int) -> None:
        """
        Update rolling dataset statistics.

        Parameters
        ----------
        frames : int
            Frame count for the sample.
        sampleBytes : int
            Estimated sample size.
        """
        self._sampleCount += 1
        self._framesSum += frames
        self._sampleBytesSum += sampleBytes
        self._maxFrames = max(self._maxFrames, frames)
        self._maxSampleBytes = max(self._maxSampleBytes, sampleBytes)

    def _flushShard(self) -> None:
        """
        Write the current shard to disk and reset the buffer.
        """
        shardPath = self._shardPath(len(self._shardInfos))
        torch.save(self._currentSamples, shardPath)
        self._shardInfos.append(
            PreprocessedDatasetShardInfo(
                path=str(shardPath.relative_to(self.outputRoot)),
                sampleCount=len(self._currentSamples),
            ),
        )
        self._currentSamples = []
        gc.collect()

    def _writeIndex(self) -> None:
        """Serialize the sample index to disk."""
        indexPayload = [asdict(entry) for entry in self._indexEntries]
        self._writeJson(self.outputRoot / INDEX_FILENAME, indexPayload)

    def _writeManifest(self, manifest: PreprocessedDatasetManifest) -> None:
        """
        Serialize the manifest to disk.

        Parameters
        ----------
        manifest : PreprocessedDatasetManifest
            Manifest dataclass to serialize.
        """
        payload = asdict(manifest)
        payload["shards"] = [asdict(entry) for entry in manifest.shards]
        self._writeJson(self.outputRoot / MANIFEST_FILENAME, payload)

    def _writeJson(self, path: Path, payload: object) -> None:
        """
        Write JSON payload to disk with ASCII encoding.

        Parameters
        ----------
        path : Path
            Destination JSON path.
        payload : object
            Serializable payload.
        """
        serialized = json.dumps(payload, ensure_ascii=True, indent=2)
        path.write_text(serialized, encoding="utf-8")

    def _shardPath(self, shardIndex: int) -> Path:
        """
        Return the shard file path for a given shard index.

        Parameters
        ----------
        shardIndex : int
            Shard index in the dataset.
        """
        filename = f"shard_{shardIndex:05d}.pt"
        return self.shardsDir / filename


def _resolveAnimationPath(promptPath: Path, datasetRoot: Path) -> Path:
    """
    Resolve animation path next to the prompt file.

    Parameters
    ----------
    promptPath : Path
        Path to the prompt.json file.
    datasetRoot : Path
        Root directory for prompts and animations.

    Returns
    -------
    Path
        Resolved animation file path.
    """
    try:
        relativeDirectory = promptPath.parent.relative_to(datasetRoot)
    except ValueError:
        relativeDirectory = promptPath.parent
    targetDirectory = datasetRoot / relativeDirectory
    candidates = [
        targetDirectory / "animation.json",
        targetDirectory / "animation.js",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No animation file found next to {promptPath}")


def _buildWindows(
    startFrame: int,
    endFrame: int,
    splitFrames: Optional[int],
) -> List[Tuple[int, int]]:
    """
    Build window ranges for a segment.

    Parameters
    ----------
    startFrame : int
        Segment start frame.
    endFrame : int
        Segment end frame.
    splitFrames : Optional[int]
        Window size for splitting.

    Returns
    -------
    List[Tuple[int, int]]
        Window ranges for the segment.
    """
    if not _isFrameCountValid(splitFrames):
        return [(startFrame, endFrame)]
    windowStarts = list(range(startFrame, endFrame, splitFrames))
    return [
        (windowStart, min(windowStart + splitFrames, endFrame))
        for windowStart in windowStarts
        if windowStart < endFrame
    ]


def _isFrameCountValid(value: Optional[int]) -> bool:
    """
    Return True when a frame count is valid.

    Parameters
    ----------
    value : Optional[int]
        Frame count to validate.
    """
    return value is not None and value >= MIN_FRAME_COUNT


def _downsampleMotion(
    motion: torch.Tensor,
    targetFrames: Optional[int],
) -> torch.Tensor:
    """
    Downsample a motion tensor to the target frame count.

    Parameters
    ----------
    motion : torch.Tensor
        Motion tensor shaped (frames, bones, channels).
    targetFrames : Optional[int]
        Target frame count.

    Returns
    -------
    torch.Tensor
        Downsampled motion tensor.
    """
    if not _isFrameCountValid(targetFrames):
        return motion
    currentFrames = motion.shape[0]
    if currentFrames <= targetFrames:
        return motion
    stride = math.ceil(currentFrames / targetFrames)
    return motion[::max(stride, MIN_FRAME_COUNT)]


def _isFramesValid(frames: int, maxFrames: Optional[int]) -> bool:
    """
    Check whether frame count passes the filter.

    Parameters
    ----------
    frames : int
        Frame count to validate.
    maxFrames : Optional[int]
        Maximum allowed frame count.
    """
    if frames < MIN_FRAME_COUNT:
        return False
    if maxFrames is None:
        return True
    return frames <= maxFrames


def _tokenizeText(
    tokenizer: XLMRobertaTokenizerFast,
    text: str,
    tag: str,
    maxPromptLength: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize text with an optional tag prefix.

    Parameters
    ----------
    tokenizer : XLMRobertaTokenizerFast
        Tokenizer instance.
    text : str
        Prompt text.
    tag : str
        Dataset tag.
    maxPromptLength : int
        Token length for truncation and padding.
    """
    composed = _composeText(tag, text)
    encoded = tokenizer(
        composed,
        padding="max_length",
        truncation=True,
        max_length=maxPromptLength,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
    }


def _composeText(tag: str, text: str) -> str:
    """
    Compose prompt text with optional tag prefix.

    Parameters
    ----------
    tag : str
        Dataset tag.
    text : str
        Prompt text.
    """
    if tag:
        return f"{TAG_PREFIX_TEMPLATE.format(tag=tag)}{text}"
    return text


def _estimateSampleBytes(sample: Dict[str, object]) -> int:
    """
    Estimate the memory footprint of a sample.

    Parameters
    ----------
    sample : Dict[str, object]
        Sample dictionary with tensors.
    """
    inputIds = sample["input_ids"]
    attentionMask = sample["attention_mask"]
    motion = sample["motion"]
    totalBytes = (
        _tensorBytes(inputIds)
        + _tensorBytes(attentionMask)
        + _tensorBytes(motion)
    )
    return int(totalBytes)


def _tensorBytes(tensor: torch.Tensor) -> int:
    """
    Return bytes consumed by a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to inspect.
    """
    return int(tensor.numel() * tensor.element_size())


def _safeAverage(total: int, count: int) -> float:
    """
    Return a safe average.

    Parameters
    ----------
    total : int
        Total sum.
    count : int
        Count of entries.
    """
    if count <= 0:
        return 0.0
    return float(total) / float(count)
