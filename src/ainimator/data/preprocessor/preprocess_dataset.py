"""Preprocess converted datasets into the V2 shardable tensor format."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast

from ainimator.data.builder.dataset_reader import (
    loadAnimationPayloadWithExtras,
    loadPromptSegments,
)
from ainimator.data.builder.progress import TqdmProgressReporter
from ainimator.core.config_loader import loadNetworkConfig
from ainimator.core.constants.preprocessed import (
    PREPROCESSED_GENERATION_TEXT_CACHE_DIRNAME,
    PREPROCESSED_LINK_INDEX_FILENAME,
    PREPROCESSED_MANIFEST_FILENAME,
    PREPROCESSED_MANIFEST_VERSION,
    PREPROCESSED_MIN_FRAME_COUNT,
    PREPROCESSED_PROMPT_FILENAME,
    PREPROCESSED_SAMPLE_INDEX_FILENAME,
    PREPROCESSED_SAMPLE_SHARDS_DIRNAME,
    PREPROCESSED_TEXT_INDEX_FILENAME,
    PREPROCESSED_TEXT_SHARDS_DIRNAME,
)
from ainimator.geometry.components import (
    buildEnabledComponents,
    buildMotionFeatureTensors,
)
from ainimator.geometry.components.feature_builder import (
    canonicalizeMotionUpright,
)

# Orientation canonicalisation toggle.  2026-06-17 — upgraded from the
# yaw-only facing fix to the full upright canonicalisation
# (canonicalizeMotionUpright): the AMASS→rot6d conversion stored bodies
# lying down in arbitrary directions (raw Z-up root orientation applied
# to the Y-up rest skeleton).  When set (any non-empty value), every
# window is rotated so frame 0 stands upright (Y-up) and faces +Z before
# features are built.  This changes the data distribution, so a new
# normaliser fit and a full from-scratch retrain are required.
_CANONICALIZE_FACING_ENV = "AINIMATOR_CANONICALIZE_FACING"
from ainimator.geometry.components.base import MotionComponent
from ainimator.core.types import (
    PreprocessDatasetConfig,
    PreprocessedDatasetManifestV2,
    PreprocessedDatasetShardInfo,
    PreprocessedLinkIndexV2,
    PreprocessedSampleIndexV2,
    PreprocessedTextIndexV2,
)

PROMPT_FILENAME = PREPROCESSED_PROMPT_FILENAME
MANIFEST_FILENAME = PREPROCESSED_MANIFEST_FILENAME
SAMPLE_INDEX_FILENAME = PREPROCESSED_SAMPLE_INDEX_FILENAME
TEXT_INDEX_FILENAME = PREPROCESSED_TEXT_INDEX_FILENAME
LINK_INDEX_FILENAME = PREPROCESSED_LINK_INDEX_FILENAME
SAMPLE_SHARDS_DIRNAME = PREPROCESSED_SAMPLE_SHARDS_DIRNAME
TEXT_SHARDS_DIRNAME = PREPROCESSED_TEXT_SHARDS_DIRNAME
GENERATION_CACHE_DIRNAME = PREPROCESSED_GENERATION_TEXT_CACHE_DIRNAME
MANIFEST_VERSION = PREPROCESSED_MANIFEST_VERSION
MIN_FRAME_COUNT = PREPROCESSED_MIN_FRAME_COUNT
LOGGER = logging.getLogger("dataset.preprocess")


@dataclass
class _PendingTextRecord:
    """Mutable unique-text registry entry used before the text shards exist."""

    textId: int
    rawText: str
    usageCount: int = 0
    inputIds: Optional[torch.Tensor] = None
    attentionMask: Optional[torch.Tensor] = None
    pooledText: Optional[torch.Tensor] = None


@dataclass
class _PendingLinkRecord:
    """Mutable link entry finalized once text bytes are known."""

    sampleId: int
    textId: int
    datasetFolder: str
    sourceFile: str
    frames: int


class DatasetPreprocessor:
    """Convert a converted dataset into a V2 preprocessed dataset."""

    def __init__(self, config: PreprocessDatasetConfig) -> None:
        self.config = config
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            config.processing.modelName,
        )
        self.textEncoder = XLMRobertaModel.from_pretrained(
            config.processing.modelName,
            low_cpu_mem_usage=True,
        )
        self.textEncoder.eval()
        for parameter in self.textEncoder.parameters():
            parameter.requires_grad = False

        self.networkConfig = loadNetworkConfig(
            configPath=config.paths.networkConfigPath,
            profile="v2",
        )
        self.enabledComponents = _mergeEnabledComponents(
            buildEnabledComponents(self.networkConfig.generation.boneData),
            (
                buildEnabledComponents(self.networkConfig.clip.boneData)
                if self.networkConfig.clip.boneData is not None
                else ()
            ),
        )
        self.enabledComponentKeys = [
            component.key for component in self.enabledComponents
        ]
        if "rotation6d" not in self.enabledComponentKeys:
            raise ValueError(
                "The V2 preprocessing pipeline still requires `rotation6d` "
                "because `motion` remains the base training tensor."
            )
        LOGGER.info(
            "Preprocess enabled motion components: %s",
            ", ".join(self.enabledComponentKeys),
        )
        self._writer = DatasetV2Writer(
            outputRoot=config.paths.outputRoot,
            sampleShardSize=config.processing.sampleShardSize,
            textShardSize=config.processing.textShardSize,
        )
        self._textIdByValue: dict[str, int] = {}
        self._textRecords: list[_PendingTextRecord] = []

    def run(self, includeFolders: Optional[List[str]] = None) -> None:
        """Execute preprocessing and write shards + indices + manifest."""
        folders = includeFolders or self.config.paths.includeFolders
        promptFiles = self._listPromptFiles(
            self.config.paths.inputRoot,
            includeFolders=folders,
        )
        reporter = TqdmProgressReporter(
            total=len(promptFiles),
            description="Dataset preprocess",
        )
        for promptPath in promptFiles:
            self._processPromptFile(promptPath)
            reporter.advance(promptPath.as_posix())
        reporter.close()

        pooledTextDim = self._materializeUniqueTexts()
        self._writer.finalize(
            modelName=self.config.processing.modelName,
            maxPromptLength=self.config.processing.maxPromptLength,
            splitFrames=self.config.processing.splitFrames,
            downsampleTargetFrames=self.config.processing.downsampleTargetFrames,
            maxSegmentFrames=self.config.processing.maxSegmentFrames,
            enabledComponents=self.enabledComponentKeys,
            pooledTextDim=pooledTextDim,
        )

    def _listPromptFiles(
        self,
        root: Path,
        includeFolders: Optional[List[str]] = None,
    ) -> List[Path]:
        """Return sorted prompt.json files under a root."""
        promptFiles = sorted(root.rglob(PROMPT_FILENAME))
        if not includeFolders:
            return promptFiles
        allowed = {
            folder.strip().lower()
            for folder in includeFolders
            if folder.strip()
        }
        if not allowed:
            return promptFiles
        filtered: List[Path] = []
        for promptPath in promptFiles:
            topLevelFolder = _extractTopLevelFolder(promptPath, root)
            if topLevelFolder.lower() in allowed:
                filtered.append(promptPath)
        return filtered

    def _processPromptFile(self, promptPath: Path) -> None:
        """Load one prompt file, its animation, and register its links."""
        promptMeta, segments = loadPromptSegments(promptPath)
        datasetFolder = _extractTopLevelFolder(
            promptPath,
            self.config.paths.inputRoot,
        )
        animationPath = _resolveAnimationPath(
            promptPath,
            self.config.paths.inputRoot,
        )
        motion, motionMeta, motionExtras = loadAnimationPayloadWithExtras(
            animationPath,
        )
        try:
            for segment in segments:
                self._processSegment(
                    segmentText=segment.text,
                    segmentStart=segment.startFrame,
                    segmentEnd=segment.endFrame,
                    datasetFolder=datasetFolder,
                    promptMeta=promptMeta,
                    motion=motion,
                    motionMeta=motionMeta,
                    motionExtras=motionExtras,
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
        datasetFolder: str,
        promptMeta: Dict[str, object],
        motion: torch.Tensor,
        motionMeta: Dict[str, object],
        motionExtras: Dict[str, object],
        sourceFile: str,
    ) -> None:
        """Split, filter, and serialize one prompt segment."""
        textId = self._registerText(segmentText)
        windows = _buildWindows(
            segmentStart,
            segmentEnd,
            self.config.processing.splitFrames,
        )
        for windowStart, windowEnd in windows:
            motionSlice = motion[windowStart:windowEnd]
            motionSlice, downsampleIndices = _downsampleMotion(
                motionSlice,
                self.config.processing.downsampleTargetFrames,
            )
            slicedExtras = _sliceTemporalExtras(
                extras=motionExtras,
                start=windowStart,
                end=windowEnd,
                downsampleIndices=downsampleIndices,
            )
            if not _isFramesValid(
                motionSlice.shape[0],
                self.config.processing.maxSegmentFrames,
            ):
                continue
            if os.environ.get(_CANONICALIZE_FACING_ENV):
                motionSlice, slicedExtras = canonicalizeMotionUpright(
                    motion=motionSlice,
                    extras=slicedExtras,
                )
            featureTensors = buildMotionFeatureTensors(
                motion=motionSlice,
                extras=slicedExtras,
                enabledComponents=self.enabledComponents,
            )
            sample = {
                "motion": motionSlice,
                "time": torch.tensor(
                    [windowStart, windowEnd],
                    dtype=torch.long,
                ),
                "meta": motionMeta | promptMeta,
            }
            sample.update(featureTensors)
            sampleBytes = _estimateSampleBytes(sample)
            sampleId = self._writer.addSample(
                sample=sample,
                frames=motionSlice.shape[0],
                sampleBytes=sampleBytes,
                datasetFolder=datasetFolder,
                sourceFile=sourceFile,
                startFrame=windowStart,
                endFrame=windowEnd,
            )
            self._writer.addLink(
                sampleId=sampleId,
                textId=textId,
                datasetFolder=datasetFolder,
                sourceFile=sourceFile,
                frames=motionSlice.shape[0],
            )

    def _registerText(self, rawText: str) -> int:
        """Deduplicate and register one text string."""
        normalized = rawText.strip()
        existingId = self._textIdByValue.get(normalized)
        if existingId is not None:
            self._textRecords[existingId].usageCount += 1
            return existingId
        textId = len(self._textRecords)
        self._textIdByValue[normalized] = textId
        self._textRecords.append(
            _PendingTextRecord(
                textId=textId,
                rawText=normalized,
                usageCount=1,
            )
        )
        return textId

    def _materializeUniqueTexts(self) -> int:
        """Tokenize + encode every unique text, then write text shards."""
        if not self._textRecords:
            pooledTextDim = int(self.textEncoder.config.hidden_size)
            self._writer.writeTexts(self._textRecords)
            return pooledTextDim

        pooledTextDim = int(self.textEncoder.config.hidden_size)
        batchSize = self.config.processing.textBatchSize
        for start in range(0, len(self._textRecords), batchSize):
            chunk = self._textRecords[start:start + batchSize]
            texts = [record.rawText for record in chunk]
            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.config.processing.maxPromptLength,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.textEncoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
            pooled = _maskedMean(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            pooledTextDim = int(pooled.shape[-1])
            for index, record in enumerate(chunk):
                record.inputIds = encoded["input_ids"][index].detach().cpu()
                record.attentionMask = (
                    encoded["attention_mask"][index].detach().cpu()
                )
                record.pooledText = pooled[index].detach().cpu()
            del outputs, pooled, encoded
            gc.collect()

        self._writer.writeTexts(self._textRecords)
        return pooledTextDim


class DatasetV2Writer:
    """Write preprocessed samples, texts, links, and the V2 manifest."""

    def __init__(
        self,
        outputRoot: Path,
        sampleShardSize: int,
        textShardSize: int,
    ) -> None:
        self.outputRoot = outputRoot
        self.sampleShardSize = sampleShardSize
        self.textShardSize = textShardSize
        self.sampleShardsDir = outputRoot / SAMPLE_SHARDS_DIRNAME
        self.textShardsDir = outputRoot / TEXT_SHARDS_DIRNAME
        self.sampleShardsDir.mkdir(parents=True, exist_ok=True)
        self.textShardsDir.mkdir(parents=True, exist_ok=True)
        (outputRoot / GENERATION_CACHE_DIRNAME).mkdir(parents=True, exist_ok=True)

        self._currentSamples: list[dict[str, object]] = []
        self._currentTexts: list[dict[str, object]] = []
        self._sampleIndexEntries: list[PreprocessedSampleIndexV2] = []
        self._textIndexEntries: list[PreprocessedTextIndexV2] = []
        self._pendingLinks: list[_PendingLinkRecord] = []
        self._linkEntries: list[PreprocessedLinkIndexV2] = []
        self._sampleShards: list[PreprocessedDatasetShardInfo] = []
        self._textShards: list[PreprocessedDatasetShardInfo] = []

        self._sampleBytesById: list[int] = []
        self._textBytesById: list[int] = []
        self._sampleBytesSum = 0
        self._pairBytesSum = 0
        self._framesSum = 0
        self._maxSampleBytes = 0
        self._maxFrames = 0

    def addSample(
        self,
        sample: dict[str, object],
        frames: int,
        sampleBytes: int,
        datasetFolder: str,
        sourceFile: str,
        startFrame: int,
        endFrame: int,
    ) -> int:
        """Queue one motion sample for shard writing."""
        sampleId = len(self._sampleIndexEntries)
        shardIndex = len(self._sampleShards)
        shardOffset = len(self._currentSamples)
        self._currentSamples.append(sample)
        self._sampleIndexEntries.append(
            PreprocessedSampleIndexV2(
                sampleId=sampleId,
                shardIndex=shardIndex,
                shardOffset=shardOffset,
                frames=frames,
                sampleBytes=sampleBytes,
                datasetFolder=datasetFolder,
                sourceFile=sourceFile,
                startFrame=startFrame,
                endFrame=endFrame,
            )
        )
        self._sampleBytesById.append(sampleBytes)
        self._sampleBytesSum += sampleBytes
        self._framesSum += frames
        self._maxSampleBytes = max(self._maxSampleBytes, sampleBytes)
        self._maxFrames = max(self._maxFrames, frames)
        if len(self._currentSamples) >= self.sampleShardSize:
            self._flushSampleShard()
        return sampleId

    def addLink(
        self,
        sampleId: int,
        textId: int,
        datasetFolder: str,
        sourceFile: str,
        frames: int,
    ) -> None:
        """Register one training link between a sample and a text."""
        self._pendingLinks.append(
            _PendingLinkRecord(
                sampleId=sampleId,
                textId=textId,
                datasetFolder=datasetFolder,
                sourceFile=sourceFile,
                frames=frames,
            )
        )

    def writeTexts(
        self,
        records: Sequence[_PendingTextRecord],
    ) -> None:
        """Write every unique text payload and finalize link byte estimates."""
        for record in records:
            if (
                record.inputIds is None
                or record.attentionMask is None
                or record.pooledText is None
            ):
                raise ValueError(
                    f"Text record {record.textId} is missing encoded tensors."
                )
            payload = {
                "text_id": record.textId,
                "raw_text": record.rawText,
                "input_ids": record.inputIds,
                "attention_mask": record.attentionMask,
                "pooled_text": record.pooledText,
            }
            textBytes = _estimateSampleBytes(payload)
            self._textBytesById.append(textBytes)
            shardIndex = len(self._textShards)
            shardOffset = len(self._currentTexts)
            self._currentTexts.append(payload)
            self._textIndexEntries.append(
                PreprocessedTextIndexV2(
                    textId=record.textId,
                    shardIndex=shardIndex,
                    shardOffset=shardOffset,
                    usageCount=record.usageCount,
                )
            )
            if len(self._currentTexts) >= self.textShardSize:
                self._flushTextShard()

        if self._currentTexts:
            self._flushTextShard()

        self._linkEntries = []
        self._pairBytesSum = 0
        for index, link in enumerate(self._pendingLinks):
            pairBytes = (
                self._sampleBytesById[link.sampleId]
                + self._textBytesById[link.textId]
            )
            self._pairBytesSum += pairBytes
            self._linkEntries.append(
                PreprocessedLinkIndexV2(
                    linkId=index,
                    sampleId=link.sampleId,
                    textId=link.textId,
                    datasetFolder=link.datasetFolder,
                    sourceFile=link.sourceFile,
                    frames=link.frames,
                    pairBytes=pairBytes,
                )
            )

    def finalize(
        self,
        modelName: str,
        maxPromptLength: int,
        splitFrames: Optional[int],
        downsampleTargetFrames: Optional[int],
        maxSegmentFrames: Optional[int],
        enabledComponents: list[str],
        pooledTextDim: int,
    ) -> None:
        """Flush shards, write indices, and serialize the V2 manifest."""
        if self._currentSamples:
            self._flushSampleShard()
        if self._pendingLinks and not self._linkEntries:
            raise RuntimeError(
                "Links were registered before text shards were materialized."
            )
        self._writeJson(
            self.outputRoot / SAMPLE_INDEX_FILENAME,
            [asdict(entry) for entry in self._sampleIndexEntries],
        )
        self._writeJson(
            self.outputRoot / TEXT_INDEX_FILENAME,
            [asdict(entry) for entry in self._textIndexEntries],
        )
        self._writeJson(
            self.outputRoot / LINK_INDEX_FILENAME,
            [asdict(entry) for entry in self._linkEntries],
        )
        manifest = PreprocessedDatasetManifestV2(
            version=MANIFEST_VERSION,
            modelName=modelName,
            maxPromptLength=maxPromptLength,
            splitFrames=splitFrames,
            downsampleTargetFrames=downsampleTargetFrames,
            maxSegmentFrames=maxSegmentFrames,
            sampleShardSize=self.sampleShardSize,
            textShardSize=self.textShardSize,
            totalSamples=len(self._sampleIndexEntries),
            totalTexts=len(self._textIndexEntries),
            totalLinks=len(self._linkEntries),
            averageSampleBytes=_safeAverage(
                self._sampleBytesSum,
                len(self._sampleIndexEntries),
            ),
            maxSampleBytes=self._maxSampleBytes,
            averagePairBytes=_safeAverage(
                self._pairBytesSum,
                len(self._linkEntries),
            ),
            averageFrames=_safeAverage(
                self._framesSum,
                len(self._sampleIndexEntries),
            ),
            maxFrames=self._maxFrames,
            sampleShards=self._sampleShards,
            textShards=self._textShards,
            sampleIndexPath=SAMPLE_INDEX_FILENAME,
            textIndexPath=TEXT_INDEX_FILENAME,
            linkIndexPath=LINK_INDEX_FILENAME,
            enabledComponents=list(enabledComponents),
            pooledTextDim=pooledTextDim,
        )
        payload = asdict(manifest)
        payload["sampleShards"] = [asdict(entry) for entry in manifest.sampleShards]
        payload["textShards"] = [asdict(entry) for entry in manifest.textShards]
        self._writeJson(self.outputRoot / MANIFEST_FILENAME, payload)

    def _flushSampleShard(self) -> None:
        """Write the buffered motion samples to disk."""
        shardPath = self._sampleShardPath(len(self._sampleShards))
        torch.save(self._currentSamples, shardPath)
        self._sampleShards.append(
            PreprocessedDatasetShardInfo(
                path=str(shardPath.relative_to(self.outputRoot)),
                sampleCount=len(self._currentSamples),
            )
        )
        self._currentSamples = []
        gc.collect()

    def _flushTextShard(self) -> None:
        """Write the buffered unique texts to disk."""
        shardPath = self._textShardPath(len(self._textShards))
        torch.save(self._currentTexts, shardPath)
        self._textShards.append(
            PreprocessedDatasetShardInfo(
                path=str(shardPath.relative_to(self.outputRoot)),
                sampleCount=len(self._currentTexts),
            )
        )
        self._currentTexts = []
        gc.collect()

    def _sampleShardPath(self, shardIndex: int) -> Path:
        filename = f"sample_shard_{shardIndex:05d}.pt"
        return self.sampleShardsDir / filename

    def _textShardPath(self, shardIndex: int) -> Path:
        filename = f"text_shard_{shardIndex:05d}.pt"
        return self.textShardsDir / filename

    def _writeJson(self, path: Path, payload: object) -> None:
        serialized = json.dumps(payload, ensure_ascii=True, indent=2)
        path.write_text(serialized, encoding="utf-8")


def _mergeEnabledComponents(
    primary: Sequence[MotionComponent],
    secondary: Sequence[MotionComponent],
) -> tuple[MotionComponent, ...]:
    """Merge component sequences while preserving first-seen order."""
    merged: dict[str, MotionComponent] = {}
    for component in (*primary, *secondary):
        merged.setdefault(component.key, component)
    return tuple(merged.values())


def _resolveAnimationPath(promptPath: Path, datasetRoot: Path) -> Path:
    """Resolve animation path next to the prompt file."""
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


def _extractTopLevelFolder(path: Path, root: Path) -> str:
    """Return the first directory component relative to `root`."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    if not relative.parts:
        return ""
    return str(relative.parts[0])


def _buildWindows(
    startFrame: int,
    endFrame: int,
    splitFrames: Optional[int],
) -> List[Tuple[int, int]]:
    """Build window ranges for a segment."""
    if not _isFrameCountValid(splitFrames):
        return [(startFrame, endFrame)]
    windowStarts = list(range(startFrame, endFrame, splitFrames))
    return [
        (windowStart, min(windowStart + splitFrames, endFrame))
        for windowStart in windowStarts
        if windowStart < endFrame
    ]


def _isFrameCountValid(value: Optional[int]) -> bool:
    """Return True when a frame count is valid."""
    return value is not None and value >= MIN_FRAME_COUNT


def _downsampleMotion(
    motion: torch.Tensor,
    targetFrames: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample a motion tensor to the target frame count."""
    currentFrames = motion.shape[0]
    indices = _buildDownsampleIndices(currentFrames, targetFrames)
    if indices.numel() == currentFrames:
        return motion, indices
    return motion.index_select(0, indices), indices


def _buildDownsampleIndices(
    frameCount: int,
    targetFrames: Optional[int],
) -> torch.Tensor:
    """Return the temporal indices kept after downsampling."""
    if frameCount <= 0:
        return torch.zeros(0, dtype=torch.long)
    if not _isFrameCountValid(targetFrames):
        return torch.arange(frameCount, dtype=torch.long)
    if frameCount <= targetFrames:
        return torch.arange(frameCount, dtype=torch.long)
    stride = math.ceil(frameCount / targetFrames)
    return torch.arange(0, frameCount, max(stride, MIN_FRAME_COUNT))


def _sliceTemporalExtras(
    extras: Dict[str, object],
    start: int,
    end: int,
    downsampleIndices: torch.Tensor,
) -> Dict[str, object]:
    """Slice and downsample temporal extras to match a motion window."""
    sliced: Dict[str, object] = {}
    rawTranslation = extras.get("trans")
    if rawTranslation is None:
        return sliced
    translation = torch.as_tensor(rawTranslation, dtype=torch.float32)
    if translation.dim() != 2 or translation.shape[1] != 3:
        LOGGER.warning(
            "Ignoring root translation with unexpected shape %s.",
            tuple(translation.shape),
        )
        return sliced
    translation = translation[start:end]
    if translation.shape[0] != downsampleIndices.numel():
        translation = translation.index_select(0, downsampleIndices)
    sliced["trans"] = translation
    return sliced


def _isFramesValid(frames: int, maxFrames: Optional[int]) -> bool:
    """Check whether frame count passes the filter."""
    if frames < MIN_FRAME_COUNT:
        return False
    if maxFrames is None:
        return True
    return frames <= maxFrames


def _maskedMean(
    sequenceOutput: torch.Tensor,
    attentionMask: torch.Tensor,
) -> torch.Tensor:
    """Compute a masked mean pooling over the sequence dimension."""
    expandedMask = attentionMask.unsqueeze(-1).expand_as(sequenceOutput).float()
    safeDenominator = expandedMask.sum(dim=1).clamp(min=1e-6)
    maskedSum = (sequenceOutput * expandedMask).sum(dim=1)
    return maskedSum / safeDenominator


def _estimateSampleBytes(sample: Dict[str, object]) -> int:
    """Estimate the memory footprint of a sample-like payload."""
    totalBytes = 0
    for value in sample.values():
        if isinstance(value, torch.Tensor):
            totalBytes += _tensorBytes(value)
    return int(totalBytes)


def _tensorBytes(tensor: torch.Tensor) -> int:
    """Return bytes consumed by a tensor."""
    return int(tensor.numel() * tensor.element_size())


def _safeAverage(total: int | float, count: int) -> float:
    """Return a safe average."""
    if count <= 0:
        return 0.0
    return float(total) / float(count)
