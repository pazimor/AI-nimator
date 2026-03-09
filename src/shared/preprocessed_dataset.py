"""Dataset loader for the V2 preprocessed link-based dataset format."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.shared.constants.preprocessed import (
    PREPROCESSED_GENERATION_TEXT_CACHE_DIRNAME,
    PREPROCESSED_GENERATION_TEXT_CACHE_MANIFEST_FILENAME,
    PREPROCESSED_GENERATION_TEXT_CACHE_VERSION,
    PREPROCESSED_LINK_INDEX_FILENAME,
    PREPROCESSED_MANIFEST_FILENAME,
    PREPROCESSED_MANIFEST_VERSION,
    PREPROCESSED_SAMPLE_INDEX_FILENAME,
    PREPROCESSED_TEXT_INDEX_FILENAME,
)
from src.shared.types import (
    GenerationTextCacheManifest,
    PreprocessedDatasetManifestV2,
    PreprocessedDatasetShardInfo,
    PreprocessedLinkIndexV2,
    PreprocessedSampleIndexV2,
    PreprocessedTextIndexV2,
)

LOGGER = logging.getLogger("shared.preprocessed_dataset")


def computeClipCheckpointFingerprint(checkpointPath: Path) -> str:
    """Return a stable SHA-256 fingerprint for a CLIP checkpoint file."""
    digest = hashlib.sha256()
    with checkpointPath.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolveGenerationTextCacheDir(
    datasetRoot: Path,
    clipFingerprint: str,
) -> Path:
    """Return the directory holding cached generation embeddings."""
    return datasetRoot / PREPROCESSED_GENERATION_TEXT_CACHE_DIRNAME / clipFingerprint


class PreprocessedLinkDataset(Dataset[Dict[str, object]]):
    """Dataset loading preprocessed V2 link records on demand."""

    def __init__(
        self,
        datasetRoot: Path,
        generationCacheCheckpoint: Optional[Path] = None,
        includeTokenizedText: bool = True,
        sampleShardCacheSize: int = 4,
        textShardCacheSize: int = 8,
        generationShardCacheSize: int = 4,
    ) -> None:
        self.datasetRoot = datasetRoot
        self.includeTokenizedText = includeTokenizedText
        self.sampleShardCacheSize = max(1, int(sampleShardCacheSize))
        self.textShardCacheSize = max(1, int(textShardCacheSize))
        self.generationShardCacheSize = max(1, int(generationShardCacheSize))
        self.manifest = _loadManifest(datasetRoot)
        self.sampleIndexEntries = _loadSampleIndex(
            datasetRoot,
            self.manifest.sampleIndexPath,
        )
        self.textIndexEntries = _loadTextIndex(
            datasetRoot,
            self.manifest.textIndexPath,
        )
        self.linkEntries = _loadLinkIndex(
            datasetRoot,
            self.manifest.linkIndexPath,
        )
        self.indexEntries = self.linkEntries
        self._maxPairBytes = max(
            (entry.pairBytes for entry in self.linkEntries),
            default=0,
        )

        self._sampleShardCache: OrderedDict[int, List[Dict[str, object]]] = (
            OrderedDict()
        )
        self._textShardCache: OrderedDict[int, List[Dict[str, object]]] = (
            OrderedDict()
        )
        self._generationShardCache: OrderedDict[int, List[Dict[str, object]]] = (
            OrderedDict()
        )

        self.generationCacheCheckpoint = (
            generationCacheCheckpoint.expanduser()
            if generationCacheCheckpoint is not None
            else None
        )
        self.generationClipFingerprint: Optional[str] = None
        self.generationCacheDir: Optional[Path] = None
        self.generationCacheManifest: Optional[GenerationTextCacheManifest] = None
        if self.generationCacheCheckpoint is not None:
            self._loadGenerationCache(self.generationCacheCheckpoint)

    def __len__(self) -> int:
        return len(self.linkEntries)

    def __getitem__(self, index: int) -> Dict[str, object]:
        linkEntry = self.linkEntries[index]
        sampleEntry = self.sampleIndexEntries[linkEntry.sampleId]
        textEntry = self.textIndexEntries[linkEntry.textId]
        samplePayload = self._loadSampleShard(sampleEntry.shardIndex)[
            sampleEntry.shardOffset
        ]
        textPayload = self._loadTextShard(textEntry.shardIndex)[
            textEntry.shardOffset
        ]
        payload: Dict[str, object] = {
            "sample_id": int(linkEntry.sampleId),
            "text_id": int(linkEntry.textId),
            "pooled_text": textPayload["pooled_text"],
            "motion": samplePayload["motion"],
            "time": samplePayload["time"],
            "meta": samplePayload["meta"],
        }
        if self.includeTokenizedText:
            payload["input_ids"] = textPayload["input_ids"]
            payload["attention_mask"] = textPayload["attention_mask"]
        for key, value in samplePayload.items():
            if key in {"motion", "time", "meta"}:
                continue
            payload[key] = value

        if self.generationCacheManifest is not None:
            generationPayload = self._loadGenerationShard(textEntry.shardIndex)[
                textEntry.shardOffset
            ]
            payload["generation_text_embedding"] = generationPayload[
                "text_embedding"
            ]
        return payload

    def getAverageSampleBytes(self) -> float:
        return self.manifest.averagePairBytes

    def getMaxSampleBytes(self) -> int:
        return self._maxPairBytes

    def getMaxFrames(self) -> int:
        return self.manifest.maxFrames

    def validateCompatibility(
        self,
        modelName: str,
        maxPromptLength: int,
        requiredComponents: Optional[List[str]] = None,
        expectedPooledTextDim: Optional[int] = None,
        expectedGenerationEmbedDim: Optional[int] = None,
        requireGenerationTextEmbedding: bool = False,
    ) -> None:
        if self.manifest.modelName != modelName:
            raise ValueError(
                "Preprocessed dataset was built with "
                f"{self.manifest.modelName} but training expects {modelName}."
            )
        if self.manifest.maxPromptLength != maxPromptLength:
            raise ValueError(
                "Preprocessed dataset was built with max-length "
                f"{self.manifest.maxPromptLength} but training expects "
                f"{maxPromptLength}."
            )
        if self.manifest.pooledTextDim <= 0:
            raise ValueError(
                "Preprocessed dataset is missing pooled_text metadata. "
                "Re-run preprocess_dataset to rebuild the V2 dataset."
            )
        if (
            expectedPooledTextDim is not None
            and self.manifest.pooledTextDim != expectedPooledTextDim
        ):
            raise ValueError(
                "Preprocessed dataset pooled_text dimension "
                f"{self.manifest.pooledTextDim} does not match the current "
                f"text encoder hidden size {expectedPooledTextDim}."
            )
        if requiredComponents:
            available = set(self.manifest.enabledComponents or ["rotation6d"])
            missing = [
                component
                for component in requiredComponents
                if component not in available
            ]
            if missing:
                raise ValueError(
                    "Preprocessed dataset is missing required motion "
                    f"components: {', '.join(missing)}. Re-run "
                    "preprocess_dataset with matching network-config toggles."
                )
        if requireGenerationTextEmbedding and self.generationCacheManifest is None:
            raise ValueError(
                "Generation text cache is missing. Run "
                "`python -m src.cli.precompute_generation_text_cache "
                f"--dataset-root {self.datasetRoot} --clip-checkpoint "
                f"{self.generationCacheCheckpoint}` before train_generation."
            )
        if (
            expectedGenerationEmbedDim is not None
            and self.generationCacheManifest is not None
            and self.generationCacheManifest.embedDim != expectedGenerationEmbedDim
        ):
            raise ValueError(
                "Generation text cache embed dim "
                f"{self.generationCacheManifest.embedDim} does not match the "
                f"current CLIP embed dim {expectedGenerationEmbedDim}."
            )

    def clearCache(self) -> None:
        self._sampleShardCache.clear()
        self._textShardCache.clear()
        self._generationShardCache.clear()

    def preloadIndices(self, indices: List[int]) -> List[Dict[str, object]]:
        """
        Materialize a subset with shard-aware locality.

        The returned sample set is identical to iterating over ``indices`` with
        ``__getitem__``, but the access order is re-arranged to minimize shard
        cache thrashing during epoch preload.
        """
        self._primeTextShardCaches(indices)
        ordered = sorted(indices, key=self._preloadSortKey)
        return [self[index] for index in ordered]

    def _loadGenerationCache(self, checkpointPath: Path) -> None:
        clipFingerprint = computeClipCheckpointFingerprint(checkpointPath)
        cacheDir = resolveGenerationTextCacheDir(self.datasetRoot, clipFingerprint)
        manifestPath = (
            cacheDir / PREPROCESSED_GENERATION_TEXT_CACHE_MANIFEST_FILENAME
        )
        if not manifestPath.exists():
            from src.shared.generation_text_cache import ensureGenerationTextCache

            LOGGER.info(
                "Generation text cache missing for %s; building it now.",
                checkpointPath,
            )
            ensureGenerationTextCache(self.datasetRoot, checkpointPath)
        manifest = _loadGenerationCacheManifest(manifestPath)
        if manifest.clipFingerprint != clipFingerprint:
            raise ValueError(
                "Generation text cache fingerprint does not match the "
                f"requested checkpoint {checkpointPath}."
            )
        if manifest.totalTexts != self.manifest.totalTexts:
            raise ValueError(
                "Generation text cache was built for a different dataset "
                f"(cache texts={manifest.totalTexts}, dataset texts="
                f"{self.manifest.totalTexts}). Re-run "
                "precompute_generation_text_cache."
            )
        if len(manifest.shards) != len(self.manifest.textShards):
            raise ValueError(
                "Generation text cache shard layout does not match the "
                "dataset text shards. Re-run precompute_generation_text_cache."
            )
        for index, shardInfo in enumerate(manifest.shards):
            expectedCount = self.manifest.textShards[index].sampleCount
            if shardInfo.sampleCount != expectedCount:
                raise ValueError(
                    "Generation text cache shard counts do not match the "
                    "dataset text shards. Re-run "
                    "precompute_generation_text_cache."
                )
        self.generationClipFingerprint = clipFingerprint
        self.generationCacheDir = cacheDir
        self.generationCacheManifest = manifest

    def _loadSampleShard(self, shardIndex: int) -> List[Dict[str, object]]:
        cachedShard = self._sampleShardCache.get(shardIndex)
        if cachedShard is not None:
            self._sampleShardCache.move_to_end(shardIndex)
            return cachedShard
        shardInfo = self.manifest.sampleShards[shardIndex]
        shardPath = self.datasetRoot / shardInfo.path
        samples = torch.load(shardPath, map_location="cpu")
        self._rememberShard(
            cache=self._sampleShardCache,
            shardIndex=shardIndex,
            payload=samples,
            maxEntries=self.sampleShardCacheSize,
        )
        return samples

    def _loadTextShard(self, shardIndex: int) -> List[Dict[str, object]]:
        cachedShard = self._textShardCache.get(shardIndex)
        if cachedShard is not None:
            self._textShardCache.move_to_end(shardIndex)
            return cachedShard
        shardInfo = self.manifest.textShards[shardIndex]
        shardPath = self.datasetRoot / shardInfo.path
        texts = torch.load(shardPath, map_location="cpu")
        self._rememberShard(
            cache=self._textShardCache,
            shardIndex=shardIndex,
            payload=texts,
            maxEntries=self.textShardCacheSize,
        )
        return texts

    def _loadGenerationShard(self, shardIndex: int) -> List[Dict[str, object]]:
        if self.generationCacheManifest is None or self.generationCacheDir is None:
            raise RuntimeError("Generation text cache is not loaded.")
        cachedShard = self._generationShardCache.get(shardIndex)
        if cachedShard is not None:
            self._generationShardCache.move_to_end(shardIndex)
            return cachedShard
        shardInfo = self.generationCacheManifest.shards[shardIndex]
        shardPath = self.generationCacheDir / shardInfo.path
        texts = torch.load(shardPath, map_location="cpu")
        self._rememberShard(
            cache=self._generationShardCache,
            shardIndex=shardIndex,
            payload=texts,
            maxEntries=self.generationShardCacheSize,
        )
        return texts

    def _rememberShard(
        self,
        cache: OrderedDict[int, List[Dict[str, object]]],
        shardIndex: int,
        payload: List[Dict[str, object]],
        maxEntries: int,
    ) -> None:
        cache[shardIndex] = payload
        cache.move_to_end(shardIndex)
        while len(cache) > maxEntries:
            cache.popitem(last=False)

    def _preloadSortKey(self, index: int) -> tuple[int, int, int, int]:
        """Return a shard-locality key for bulk preload ordering."""
        linkEntry = self.linkEntries[index]
        sampleEntry = self.sampleIndexEntries[linkEntry.sampleId]
        textEntry = self.textIndexEntries[linkEntry.textId]
        return (
            sampleEntry.shardIndex,
            textEntry.shardIndex,
            sampleEntry.shardOffset,
            textEntry.shardOffset,
        )

    def _primeTextShardCaches(self, indices: List[int]) -> None:
        """Warm and retain every text shard needed for the current preload."""
        if not indices:
            return
        textShardIndices = sorted(
            {
                self.textIndexEntries[self.linkEntries[index].textId].shardIndex
                for index in indices
            }
        )
        requiredTextCacheSize = max(self.textShardCacheSize, len(textShardIndices))
        originalTextCacheSize = self.textShardCacheSize
        self.textShardCacheSize = requiredTextCacheSize
        try:
            for shardIndex in textShardIndices:
                self._loadTextShard(shardIndex)
            if (
                self.generationCacheManifest is not None
                and self.generationCacheDir is not None
            ):
                originalGenerationCacheSize = self.generationShardCacheSize
                self.generationShardCacheSize = max(
                    self.generationShardCacheSize,
                    len(textShardIndices),
                )
                try:
                    for shardIndex in textShardIndices:
                        self._loadGenerationShard(shardIndex)
                finally:
                    self.generationShardCacheSize = originalGenerationCacheSize
        finally:
            self.textShardCacheSize = originalTextCacheSize


def _loadManifest(datasetRoot: Path) -> PreprocessedDatasetManifestV2:
    manifestPath = datasetRoot / PREPROCESSED_MANIFEST_FILENAME
    if not manifestPath.exists():
        raise FileNotFoundError(f"Missing preprocessed manifest: {manifestPath}")
    payload = json.loads(manifestPath.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0))
    if version != PREPROCESSED_MANIFEST_VERSION:
        raise ValueError(
            "Unsupported preprocessed dataset version "
            f"{version}. This code expects V{PREPROCESSED_MANIFEST_VERSION}. "
            "Re-run preprocess_dataset to rebuild the dataset in V2 format."
        )
    sampleShards = _loadShardInfos(payload.get("sampleShards"))
    textShards = _loadShardInfos(payload.get("textShards"))
    return PreprocessedDatasetManifestV2(
        version=version,
        modelName=str(payload.get("modelName", "")),
        maxPromptLength=int(payload.get("maxPromptLength", 0)),
        splitFrames=_optionalInt(payload, "splitFrames"),
        downsampleTargetFrames=_optionalInt(payload, "downsampleTargetFrames"),
        maxSegmentFrames=_optionalInt(payload, "maxSegmentFrames"),
        sampleShardSize=int(payload.get("sampleShardSize", 0)),
        textShardSize=int(payload.get("textShardSize", 0)),
        totalSamples=int(payload.get("totalSamples", 0)),
        totalTexts=int(payload.get("totalTexts", 0)),
        totalLinks=int(payload.get("totalLinks", 0)),
        averageSampleBytes=float(payload.get("averageSampleBytes", 0.0)),
        maxSampleBytes=int(payload.get("maxSampleBytes", 0)),
        averagePairBytes=float(payload.get("averagePairBytes", 0.0)),
        averageFrames=float(payload.get("averageFrames", 0.0)),
        maxFrames=int(payload.get("maxFrames", 0)),
        sampleShards=sampleShards,
        textShards=textShards,
        sampleIndexPath=str(
            payload.get("sampleIndexPath", PREPROCESSED_SAMPLE_INDEX_FILENAME)
        ),
        textIndexPath=str(
            payload.get("textIndexPath", PREPROCESSED_TEXT_INDEX_FILENAME)
        ),
        linkIndexPath=str(
            payload.get("linkIndexPath", PREPROCESSED_LINK_INDEX_FILENAME)
        ),
        enabledComponents=_enabledComponents(payload),
        pooledTextDim=int(payload.get("pooledTextDim", 0)),
    )


def _loadSampleIndex(
    datasetRoot: Path,
    indexPath: str,
) -> List[PreprocessedSampleIndexV2]:
    payload = _loadJsonList(datasetRoot / indexPath, "sample index")
    return [
        PreprocessedSampleIndexV2(
            sampleId=int(entry.get("sampleId", index)),
            shardIndex=int(entry["shardIndex"]),
            shardOffset=int(entry["shardOffset"]),
            frames=int(entry.get("frames", 0)),
            sampleBytes=int(entry.get("sampleBytes", 0)),
            datasetFolder=str(entry.get("datasetFolder", "")),
            sourceFile=str(entry.get("sourceFile", "")),
            startFrame=int(entry.get("startFrame", 0)),
            endFrame=int(entry.get("endFrame", 0)),
        )
        for index, entry in enumerate(payload)
    ]


def _loadTextIndex(
    datasetRoot: Path,
    indexPath: str,
) -> List[PreprocessedTextIndexV2]:
    payload = _loadJsonList(datasetRoot / indexPath, "text index")
    return [
        PreprocessedTextIndexV2(
            textId=int(entry.get("textId", index)),
            shardIndex=int(entry["shardIndex"]),
            shardOffset=int(entry["shardOffset"]),
            usageCount=int(entry.get("usageCount", 0)),
        )
        for index, entry in enumerate(payload)
    ]


def _loadLinkIndex(
    datasetRoot: Path,
    indexPath: str,
) -> List[PreprocessedLinkIndexV2]:
    payload = _loadJsonList(datasetRoot / indexPath, "link index")
    return [
        PreprocessedLinkIndexV2(
            linkId=int(entry.get("linkId", index)),
            sampleId=int(entry["sampleId"]),
            textId=int(entry["textId"]),
            datasetFolder=str(entry.get("datasetFolder", "")),
            sourceFile=str(entry.get("sourceFile", "")),
            frames=int(entry.get("frames", 0)),
            pairBytes=int(entry.get("pairBytes", 0)),
        )
        for index, entry in enumerate(payload)
    ]


def _loadGenerationCacheManifest(path: Path) -> GenerationTextCacheManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0))
    if version != PREPROCESSED_GENERATION_TEXT_CACHE_VERSION:
        raise ValueError(
            "Unsupported generation text cache version "
            f"{version}. Re-run precompute_generation_text_cache."
        )
    return GenerationTextCacheManifest(
        version=version,
        clipFingerprint=str(payload.get("clipFingerprint", "")),
        embedDim=int(payload.get("embedDim", 0)),
        totalTexts=int(payload.get("totalTexts", 0)),
        shardSize=int(payload.get("shardSize", 0)),
        shards=_loadShardInfos(payload.get("shards")),
    )


def _loadJsonList(path: Path, label: str) -> List[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessed {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid preprocessed {label}: expected a JSON list.")
    return payload


def _loadShardInfos(rawValue: object) -> List[PreprocessedDatasetShardInfo]:
    if not isinstance(rawValue, list):
        return []
    shardInfos: List[PreprocessedDatasetShardInfo] = []
    for entry in rawValue:
        if not isinstance(entry, dict):
            continue
        shardInfos.append(
            PreprocessedDatasetShardInfo(
                path=str(entry.get("path", "")),
                sampleCount=int(entry.get("sampleCount", 0)),
            )
        )
    return shardInfos


def _enabledComponents(payload: Dict[str, object]) -> List[str]:
    rawValue = payload.get("enabledComponents")
    if not isinstance(rawValue, list):
        return ["rotation6d"]
    normalized = [
        str(component).strip()
        for component in rawValue
        if str(component).strip()
    ]
    return normalized or ["rotation6d"]


def _optionalInt(payload: Dict[str, object], key: str) -> Optional[int]:
    rawValue = payload.get(key)
    if rawValue is None:
        return None
    return int(rawValue)


# Temporary alias to reduce churn in callers that still import the legacy name.
PreprocessedMotionDataset = PreprocessedLinkDataset
