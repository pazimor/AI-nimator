"""Dataset loader for preprocessed shard-based datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.shared.constants.preprocessed import (
    PREPROCESSED_INDEX_FILENAME,
    PREPROCESSED_MANIFEST_FILENAME,
    PREPROCESSED_MANIFEST_VERSION,
)
from src.shared.types import (
    PreprocessedDatasetManifest,
    PreprocessedDatasetShardInfo,
    PreprocessedSampleIndex,
)


class PreprocessedMotionDataset(Dataset[Dict[str, object]]):
    """Dataset loading preprocessed shard files on demand."""

    def __init__(self, datasetRoot: Path) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        datasetRoot : Path
            Root directory of the preprocessed dataset.
        """
        self.datasetRoot = datasetRoot
        self.manifest = _loadManifest(datasetRoot)
        self.indexEntries = _loadIndex(datasetRoot, self.manifest.indexPath)
        self._cachedShardIndex: Optional[int] = None
        self._cachedSamples: Optional[List[Dict[str, object]]] = None

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self.indexEntries)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """
        Load a sample by global index.

        Parameters
        ----------
        index : int
            Global sample index.

        Returns
        -------
        Dict[str, object]
            Sample dictionary with tensors.
        """
        entry = self.indexEntries[index]
        samples = self._loadShard(entry.shardIndex)
        return samples[entry.shardOffset]

    def getAverageSampleBytes(self) -> float:
        """
        Return the average sample size from the manifest.

        Returns
        -------
        float
            Average sample size in bytes.
        """
        return self.manifest.averageSampleBytes

    def getMaxSampleBytes(self) -> int:
        """
        Return the maximum sample size from the manifest.

        Returns
        -------
        int
            Maximum sample size in bytes.
        """
        return self.manifest.maxSampleBytes

    def getMaxFrames(self) -> int:
        """
        Return the maximum frame count from the manifest.

        Returns
        -------
        int
            Maximum frame count.
        """
        return self.manifest.maxFrames

    def validateCompatibility(
        self,
        modelName: str,
        maxPromptLength: int,
    ) -> None:
        """
        Validate dataset compatibility with training settings.

        Parameters
        ----------
        modelName : str
            Expected tokenizer name.
        maxPromptLength : int
            Expected token length.
        """
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

    def clearCache(self) -> None:
        """
        Clear the cached shard.
        """
        self._cachedShardIndex = None
        self._cachedSamples = None

    def _loadShard(self, shardIndex: int) -> List[Dict[str, object]]:
        """
        Load a shard into memory.

        Parameters
        ----------
        shardIndex : int
            Index of the shard to load.

        Returns
        -------
        List[Dict[str, object]]
            List of samples stored in the shard.
        """
        if self._cachedShardIndex == shardIndex:
            if self._cachedSamples is None:
                raise RuntimeError("Shard cache is empty.")
            return self._cachedSamples
        shardInfo = self.manifest.shards[shardIndex]
        shardPath = self.datasetRoot / shardInfo.path
        samples = torch.load(shardPath, map_location="cpu")
        self._cachedShardIndex = shardIndex
        self._cachedSamples = samples
        return samples


def _loadManifest(datasetRoot: Path) -> PreprocessedDatasetManifest:
    """
    Load the dataset manifest from disk.

    Parameters
    ----------
    datasetRoot : Path
        Dataset root containing the manifest file.
    """
    manifestPath = datasetRoot / PREPROCESSED_MANIFEST_FILENAME
    if not manifestPath.exists():
        raise FileNotFoundError(
            f"Missing preprocessed manifest: {manifestPath}"
        )
    payload = json.loads(manifestPath.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0))
    if version != PREPROCESSED_MANIFEST_VERSION:
        raise ValueError(
            "Unsupported manifest version "
            f"{version} (expected {PREPROCESSED_MANIFEST_VERSION})."
        )
    shards = [
        PreprocessedDatasetShardInfo(
            path=entry["path"],
            sampleCount=int(entry["sampleCount"]),
        )
        for entry in payload.get("shards", [])
    ]
    return PreprocessedDatasetManifest(
        version=version,
        modelName=str(payload.get("modelName", "")),
        maxPromptLength=int(payload.get("maxPromptLength", 0)),
        splitFrames=_optionalInt(payload, "splitFrames"),
        downsampleTargetFrames=_optionalInt(payload, "downsampleTargetFrames"),
        maxSegmentFrames=_optionalInt(payload, "maxSegmentFrames"),
        shardSize=int(payload.get("shardSize", 0)),
        totalSamples=int(payload.get("totalSamples", 0)),
        averageSampleBytes=float(payload.get("averageSampleBytes", 0.0)),
        maxSampleBytes=int(payload.get("maxSampleBytes", 0)),
        averageFrames=float(payload.get("averageFrames", 0.0)),
        maxFrames=int(payload.get("maxFrames", 0)),
        shards=shards,
        indexPath=str(payload.get("indexPath", PREPROCESSED_INDEX_FILENAME)),
    )


def _loadIndex(
    datasetRoot: Path,
    indexPath: str,
) -> List[PreprocessedSampleIndex]:
    """
    Load the sample index from disk.

    Parameters
    ----------
    datasetRoot : Path
        Dataset root containing the index file.
    """
    resolvedPath = datasetRoot / indexPath
    if not resolvedPath.exists():
        raise FileNotFoundError(f"Missing preprocessed index: {resolvedPath}")
    payload = json.loads(resolvedPath.read_text(encoding="utf-8"))
    entries: List[PreprocessedSampleIndex] = []
    for entry in payload:
        entries.append(
            PreprocessedSampleIndex(
                shardIndex=int(entry["shardIndex"]),
                shardOffset=int(entry["shardOffset"]),
                frames=int(entry.get("frames", 0)),
                sampleBytes=int(entry.get("sampleBytes", 0)),
                tag=str(entry.get("tag", "")),
                sourceFile=str(entry.get("sourceFile", "")),
            ),
        )
    return entries


def _optionalInt(payload: Dict[str, object], key: str) -> Optional[int]:
    """
    Extract an optional integer from a payload.

    Parameters
    ----------
    payload : Dict[str, object]
        JSON payload dictionary.
    key : str
        Key to read from the payload.
    """
    value = payload.get(key)
    if value in (None, "null"):
        return None
    return int(value)
