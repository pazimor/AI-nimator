"""Unified dataset management for training workflows."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.shared.model.clip.data import motionTextCollate
from src.shared.preprocessed_dataset import PreprocessedLinkDataset

LOGGER = logging.getLogger("shared.dataset_manager")

BYTES_PER_GB = 1024 * 1024 * 1024
AUTO_MEMORY_FRACTION = 0.6
AUTO_MIN_SAMPLE_MULTIPLIER = 4
MIN_SAMPLES = 1
MODEL_MEMORY_MULTIPLIER = 3
DEFAULT_VALIDATION_SEED = 42
DEFAULT_CHUNK_SELECTION_SEED = 42
VALIDATION_INDICES_KEY = "indices"
VALIDATION_METADATA_KEY = "metadata"
VALIDATION_META_TOTAL = "total_size"
VALIDATION_META_SPLIT = "validation_split"
VALIDATION_META_SEED = "seed"
VALIDATION_META_FOLDERS = "dataset_folders"


def _normalizeFolderName(value: str) -> str:
    """Normalize folder labels for stable comparisons."""
    return value.replace("\\", "/").strip().lower()


@dataclass
class MemoryManagerConfig:
    """
    Configuration for memory management.

    Attributes
    ----------
    MM_memoryLimitGB : float
        Maximum memory usage in GB before triggering cleanup.
        Set to 0 to disable memory-based cleanup.
    clearMpsCache : bool
        When False, skip torch.mps.empty_cache() during cleanup.
    """

    MM_memoryLimitGB: float = 0.0
    clearMpsCache: bool = True

    @property
    def MM_memoryLimitBytes(self) -> int:
        """Return memory limit in bytes."""
        return int(self.MM_memoryLimitGB * BYTES_PER_GB)


class MemoryManager:
    """
    Manages memory usage during training.

    Uses a threshold-based approach instead of interval-based GC.
    Monitors both CPU and GPU (MPS/CUDA) memory.
    """

    def __init__(
        self,
        config: MemoryManagerConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize memory manager.

        Parameters
        ----------
        config : MemoryManagerConfig
            Memory management configuration.
        device : Optional[torch.device]
            Target device for GPU memory management.
        """
        self.config = config
        self.device = device
        self._lastCheckBatch = 0
        self._checkInterval = 10

    def getMemoryUsageGB(self) -> float:
        """
        Return current process memory usage in GB.

        Returns
        -------
        float
            Memory usage in gigabytes.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / BYTES_PER_GB

    def getGPUMemoryUsageGB(self) -> Optional[float]:
        """
        Return current GPU memory usage in GB.

        Returns
        -------
        Optional[float]
            GPU memory usage in GB, or None if not available.
        """
        if self.device is None:
            return None

        if self.device.type == "mps":
            try:
                allocated = torch.mps.current_allocated_memory() / BYTES_PER_GB
                return allocated
            except Exception:
                return None
        if self.device.type == "cuda":
            try:
                allocated = (
                    torch.cuda.memory_allocated(self.device) / BYTES_PER_GB
                )
                return allocated
            except Exception:
                return None

        return None

    def checkAndCleanup(self, batchIndex: int, force: bool = False) -> bool:
        """
        Check memory usage and trigger cleanup if needed.

        Parameters
        ----------
        batchIndex : int
            Current batch index.
        force : bool
            Force cleanup regardless of threshold.

        Returns
        -------
        bool
            True if cleanup was performed.
        """
        if self.config.MM_memoryLimitGB <= 0 and not force:
            return False

        if not force and (
            batchIndex - self._lastCheckBatch
        ) < self._checkInterval:
            return False

        self._lastCheckBatch = batchIndex

        cpuMemGB = self.getMemoryUsageGB()
        gpuMemGB = self.getGPUMemoryUsageGB()

        currentUsageGB = cpuMemGB
        if gpuMemGB is not None:
            currentUsageGB = max(currentUsageGB, gpuMemGB)

        if currentUsageGB > self.config.MM_memoryLimitGB or force:
            self._performCleanup()
            LOGGER.debug(
                "MM: Memory cleanup triggered at %.2f GB (limit: %.2f GB)",
                currentUsageGB,
                self.config.MM_memoryLimitGB,
            )
            return True

        return False

    def _performCleanup(self) -> None:
        """Perform garbage collection and GPU cache cleanup."""
        gc.collect()

        if self.device is not None:
            if self.device.type == "mps":
                if (
                    self.config.clearMpsCache
                    and hasattr(torch.mps, "empty_cache")
                ):
                    torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

    def logMemoryStatus(self, context: str = "") -> None:
        """
        Log current memory usage.

        Parameters
        ----------
        context : str
            Optional context string for the log message.
        """
        cpuMemGB = self.getMemoryUsageGB()
        gpuMemGB = self.getGPUMemoryUsageGB()

        parts = [f"CPU: {cpuMemGB:.2f} GB"]
        if gpuMemGB is not None:
            parts.append(f"GPU: {gpuMemGB:.2f} GB")

        memStr = ", ".join(parts)
        contextStr = f" ({context})" if context else ""
        LOGGER.info("MM: Memory usage%s: %s", contextStr, memStr)


class DatasetManager:
    """
    Unified dataset management for CLIP and Generation training.

    Handles:
    - Preprocessed dataset loading
    - Automatic chunk sizing based on memory
    - Dataset rotation across epochs
    - Memory management integration
    """

    def __init__(
        self,
        datasetRoot: Path,
        batchSize: int,
        validationSplit: float,
        modelMemoryBytes: int,
        memoryConfig: Optional[MemoryManagerConfig] = None,
        device: Optional[torch.device] = None,
        validationIndicesPath: Optional[Path] = None,
        maxSamplesPerEpoch: Optional[int] = None,
        datasetFolders: Optional[List[str]] = None,
        fixedSampleRange: Optional[Tuple[int, int]] = None,
        generationCacheCheckpoint: Optional[Path] = None,
        includeTokenizedText: bool = True,
        preloadEpochChunks: bool = False,
    ) -> None:
        self.datasetRoot = datasetRoot
        self.batchSize = batchSize
        self.validationSplit = validationSplit
        self.modelMemoryBytes = modelMemoryBytes
        self.device = device
        self.validationIndicesPath = validationIndicesPath
        self.maxSamplesPerEpoch = maxSamplesPerEpoch
        self.datasetFolders = datasetFolders
        self.fixedSampleRange = fixedSampleRange
        self.generationCacheCheckpoint = generationCacheCheckpoint
        self.includeTokenizedText = includeTokenizedText
        self.preloadEpochChunks = preloadEpochChunks

        self.memoryConfig = memoryConfig or MemoryManagerConfig()
        self.memoryManager = MemoryManager(self.memoryConfig, device)

        self._dataset: Optional[PreprocessedLinkDataset] = None
        self._totalSize: Optional[int] = None
        self._activeIndices: Optional[List[int]] = None
        self._activeIndexSet: Optional[set[int]] = None
        self._maxSamples: Optional[int] = None
        self._fixedValidationIndices: Optional[List[int]] = None
        self._fixedValidationIndexSet: Optional[set[int]] = None
        self._fixedValidationLoader: Optional[DataLoader] = None
        self._fixedSampleIndices: Optional[List[int]] = None
        self._fixedSampleChunkInfo: Optional[str] = None
        self._shuffledCycleIndex: Optional[int] = None
        self._shuffledCycleOrder: Optional[List[int]] = None

    def _ensureDataset(self) -> PreprocessedLinkDataset:
        """
        Lazy-load dataset metadata.

        Returns
        -------
        PreprocessedLinkDataset
            Loaded preprocessed dataset.
        """
        if self._dataset is None:
            LOGGER.info(
                "MM: Loading preprocessed dataset from %s",
                self.datasetRoot,
            )
            self._dataset = PreprocessedLinkDataset(
                self.datasetRoot,
                generationCacheCheckpoint=self.generationCacheCheckpoint,
                includeTokenizedText=self.includeTokenizedText,
            )
            self._activeIndices = self._buildActiveIndices(self._dataset)
            self._activeIndexSet = set(self._activeIndices)
            self._totalSize = len(self._activeIndices)
            self._maxSamples = None
            LOGGER.info(
                "MM: Dataset indexed: %d total samples",
                self._totalSize,
            )
        return self._dataset

    @property
    def dataset(self) -> PreprocessedLinkDataset:
        """Return the underlying dataset instance."""
        return self._ensureDataset()

    @property
    def totalSize(self) -> int:
        """Get total dataset size."""
        self._ensureDataset()
        return self._totalSize or 0

    def _buildActiveIndices(self, dataset: PreprocessedLinkDataset) -> List[int]:
        """Build the list of sample indices active for this training run."""
        totalIndices = list(range(len(dataset)))
        if not self.datasetFolders:
            return totalIndices
        allowedFolders = {
            self._normalizeFolderName(folder)
            for folder in self.datasetFolders
            if folder and folder.strip()
        }
        if not allowedFolders:
            return totalIndices
        filtered: List[int] = []
        for index, entry in enumerate(dataset.indexEntries):
            folder = self._normalizeFolderName(entry.datasetFolder)
            if folder in allowedFolders:
                filtered.append(index)
        if not filtered:
            requested = ", ".join(sorted(allowedFolders))
            raise ValueError(
                "No samples found for dataset folders: "
                f"{requested}. Rebuild/preprocess dataset with folder metadata."
            )
        LOGGER.info(
            "MM: Folder filter active (%d folders): %s -> %d samples",
            len(allowedFolders),
            ", ".join(sorted(allowedFolders)),
            len(filtered),
        )
        return filtered

    def _normalizeFolderName(self, value: str) -> str:
        """Normalize folder labels for stable comparisons."""
        return _normalizeFolderName(value)

    @property
    def effectiveSamplesPerEpoch(self) -> int:
        """Get effective samples used per epoch."""
        maxSamples = self._getMaxSamples()
        return min(maxSamples, self.totalSize)

    @property
    def epochsForFullCoverage(self) -> int:
        """Get number of epochs needed to see full dataset."""
        maxSamples = self._getMaxSamples()
        if maxSamples <= 0:
            return 1
        return math.ceil(self.totalSize / maxSamples)

    def getDataloadersForEpoch(
        self,
        epochIndex: int,
    ) -> Tuple[DataLoader, Optional[DataLoader], str]:
        """
        Get dataloaders for a specific epoch.

        Parameters
        ----------
        epochIndex : int
            Zero-based epoch index.

        Returns
        -------
        Tuple[DataLoader, Optional[DataLoader], str]
            Train dataloader, optional validation dataloader, and chunk info.
        """
        self._ensureDataset()
        return self._getRotatingDataloaders(epochIndex)

    def getEpochSampleIndices(
        self,
        epochIndex: int,
    ) -> Tuple[List[int], str]:
        """
        Return the raw dataset indices selected for an epoch.

        Parameters
        ----------
        epochIndex : int
            Zero-based epoch index.

        Returns
        -------
        Tuple[List[int], str]
            Selected dataset indices and a human-readable chunk label.
        """
        self._ensureDataset()
        return self._resolveEpochIndices(epochIndex)

    def _getRotatingDataloaders(
        self,
        epochIndex: int,
    ) -> Tuple[DataLoader, Optional[DataLoader], str]:
        """Get dataloaders with rotating chunk selection."""
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Dataset not initialized.")
        valIndexSet = self._getFixedValidationIndexSet()

        if epochIndex > 0:
            self.clearCache()

        indices, chunkInfo = self._resolveEpochIndices(epochIndex)
        totalSize = self._totalSize or len(indices)
        if self.fixedSampleRange is not None or len(indices) < totalSize:
            LOGGER.info(
                "MM: Loading chunk for epoch %d: %s",
                epochIndex + 1,
                chunkInfo,
            )
        return self._buildDataloaders(indices, chunkInfo, valIndexSet)

    def _resolveEpochIndices(
        self,
        epochIndex: int,
    ) -> Tuple[List[int], str]:
        """Resolve raw dataset indices for a given epoch."""
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Dataset not initialized.")
        activeIndices = self._activeIndices
        if activeIndices is None:
            activeIndices = list(range(len(dataset)))
        totalSize = self._totalSize or len(activeIndices)

        if self.fixedSampleRange is not None:
            return self._getFixedSampleSelection(activeIndices, totalSize)

        chunkSize = self._getMaxSamples()
        if chunkSize >= totalSize:
            return activeIndices, f"all {totalSize} samples"

        cycleEpochs = _estimateCoverage(totalSize, chunkSize)
        cycleIndex = epochIndex // cycleEpochs
        chunkIndex = epochIndex % cycleEpochs
        cycleOrder = self._getShuffledCycleOrder(totalSize, cycleIndex)
        startIndex = chunkIndex * chunkSize
        endIndex = min(startIndex + chunkSize, totalSize)
        indices = cycleOrder[startIndex:endIndex]
        mappedIndices = [activeIndices[idx] for idx in indices]
        chunkInfo = _formatShuffledChunkInfo(
            chunkIndex=chunkIndex,
            cycleEpochs=cycleEpochs,
            chunkSize=len(indices),
            totalSize=totalSize,
        )
        return mappedIndices, chunkInfo

    def _getFixedSampleSelection(
        self,
        activeIndices: List[int],
        totalSize: int,
    ) -> Tuple[List[int], str]:
        """Return the cached fixed sample selection."""
        if (
            self._fixedSampleIndices is not None
            and self._fixedSampleChunkInfo is not None
        ):
            return self._fixedSampleIndices, self._fixedSampleChunkInfo
        if self.fixedSampleRange is None:
            raise RuntimeError("Fixed sample range is not configured.")
        startPos, endPos = self.fixedSampleRange
        if startPos <= 0 or endPos <= 0:
            raise ValueError(
                "Fixed sample range must use positive 1-based positions."
            )
        if startPos > endPos:
            raise ValueError(
                "Fixed sample range start must be <= end."
            )
        if endPos > totalSize:
            raise ValueError(
                "Fixed sample range "
                f"{startPos}:{endPos} exceeds active dataset size "
                f"({totalSize} samples)."
            )
        selected = activeIndices[startPos - 1:endPos]
        uniqueCount = len(selected)

        # When only a handful of samples are selected (typical overfit use-case
        # with a single sample), the DataLoader would produce at most
        # ``uniqueCount`` batches per epoch -- far too few gradient steps to
        # memorise the target.  Repeat the selected indices so the epoch fills
        # up to ``_getMaxSamples()`` entries.  Each repeated copy still
        # receives different noise/timestep because deterministic corruption
        # is seeded by (sampleId × stepCounter), so this is NOT redundant.
        targetN = max(self._getMaxSamples(), self.batchSize or 1)
        if uniqueCount > 0 and targetN > uniqueCount:
            repeats = math.ceil(targetN / uniqueCount)
            selected = (selected * repeats)[:targetN]

        self._fixedSampleIndices = selected
        self._fixedSampleChunkInfo = (
            f"samples {startPos}-{endPos}/{totalSize} "
            f"(repeated {len(selected)}×)"
        )
        return self._fixedSampleIndices, self._fixedSampleChunkInfo

    def _buildDataloaders(
        self,
        indices: List[int],
        chunkInfo: str,
        valIndexSet: Optional[set[int]] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], str]:
        """Build train/val dataloaders from indices."""
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Dataset not initialized.")
        if valIndexSet:
            trainIndices = self._filterTrainingIndices(indices, valIndexSet)
            trainDataset = self._buildEpochDataset(
                trainIndices,
                label=f"train {chunkInfo}",
            )
            trainLoader = self._makeDataloader(
                trainDataset,
                shuffle=True,
            )
            valLoader = self._buildFixedValidationLoader()
            return trainLoader, valLoader, chunkInfo
        chunkDataset = self._buildEpochDataset(indices, label=chunkInfo)

        if self.validationSplit <= 0.0 or len(indices) < 2:
            trainLoader = self._makeDataloader(chunkDataset, shuffle=True)
            return trainLoader, None, chunkInfo

        valSize = max(1, int(len(indices) * self.validationSplit))
        trainSize = len(indices) - valSize
        trainSubset, valSubset = random_split(chunkDataset, [trainSize, valSize])

        trainLoader = self._makeDataloader(trainSubset, shuffle=True)
        valLoader = self._makeDataloader(valSubset, shuffle=False)

        return trainLoader, valLoader, chunkInfo

    def _buildEpochDataset(
        self,
        indices: List[int],
        label: str,
    ) -> Dataset[Dict[str, object]]:
        """Build a dataset view for the current epoch selection."""
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Dataset not initialized.")
        if not self.preloadEpochChunks:
            return Subset(dataset, indices)
        if not indices:
            return _InMemorySampleDataset([])
        LOGGER.info(
            "MM: Preloading %d samples in RAM for %s",
            len(indices),
            label,
        )
        if isinstance(dataset, PreprocessedLinkDataset):
            samples = dataset.preloadIndices(indices)
        else:
            preloadIndices = sorted(indices)
            samples = [dataset[index] for index in preloadIndices]
        dataset.clearCache()
        return _InMemorySampleDataset(samples)

    def _getShuffledCycleOrder(
        self,
        totalSize: int,
        cycleIndex: int,
    ) -> List[int]:
        """Return the cached shuffled order for a full dataset-coverage cycle."""
        if (
            self._shuffledCycleOrder is not None
            and self._shuffledCycleIndex == cycleIndex
            and len(self._shuffledCycleOrder) == totalSize
        ):
            return self._shuffledCycleOrder
        order = list(range(totalSize))
        generator = random.Random(DEFAULT_CHUNK_SELECTION_SEED + cycleIndex)
        generator.shuffle(order)
        self._shuffledCycleIndex = cycleIndex
        self._shuffledCycleOrder = order
        return order

    def _makeDataloader(
        self,
        dataset: Dataset[Dict[str, object]],
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a dataloader with custom collation."""
        pinMemory = self.device is not None and self.device.type == "cuda"
        return DataLoader(
            dataset,
            batch_size=self.batchSize,
            shuffle=shuffle,
            collate_fn=motionTextCollate,
            num_workers=0,
            pin_memory=pinMemory,
        )

    def checkMemory(self, batchIndex: int, force: bool = False) -> bool:
        """
        Check memory and cleanup if needed.

        Parameters
        ----------
        batchIndex : int
            Current batch index.
        force : bool
            Force cleanup regardless of threshold.

        Returns
        -------
        bool
            True if cleanup was performed.
        """
        return self.memoryManager.checkAndCleanup(batchIndex, force)

    def clearCache(self) -> None:
        """
        Clear dataset cache and memory.

        Call this between dataset rotations or when memory needs to be freed.
        """
        if self._dataset is not None:
            self._dataset.clearCache()
        self.memoryManager._performCleanup()
        LOGGER.info("MM: Dataset cache and memory cleared")

    def logMemoryStatus(self, context: str = "") -> None:
        """Log current memory usage."""
        self.memoryManager.logMemoryStatus(context)

    def _getMaxSamples(self) -> int:
        """
        Return auto-computed max samples per epoch.

        Returns
        -------
        int
            Auto-computed chunk size.
        """
        if self._maxSamples is None:
            autoSamples = _computeAutoChunkSize(
                dataset=self.dataset,
                batchSize=self.batchSize,
                modelMemoryBytes=self.modelMemoryBytes,
            )
            LOGGER.info(
                "MM: Auto chunk size: %d samples per epoch",
                autoSamples,
            )
            self._maxSamples = autoSamples
            if self.maxSamplesPerEpoch is not None:
                if self.maxSamplesPerEpoch > 0:
                    cappedSamples = min(
                        self._maxSamples,
                        self.maxSamplesPerEpoch,
                    )
                    if cappedSamples != self._maxSamples:
                        LOGGER.info(
                            "MM: Capping samples per epoch to %d",
                            cappedSamples,
                        )
                    self._maxSamples = cappedSamples
            coverage = _estimateCoverage(self.totalSize, self._maxSamples)
            LOGGER.info(
                "MM: Full dataset coverage every %d epochs",
                coverage,
            )
        return max(self._maxSamples, MIN_SAMPLES)

    def _filterTrainingIndices(
        self,
        indices: List[int],
        valIndexSet: set[int],
    ) -> List[int]:
        """
        Remove validation indices from the training set.
        """
        filtered = [idx for idx in indices if idx not in valIndexSet]
        if filtered:
            return filtered
        LOGGER.warning(
            "MM: Fixed validation set overlaps entire chunk; "
            "falling back to full chunk for training.",
        )
        return indices

    def _buildFixedValidationLoader(self) -> Optional[DataLoader]:
        """
        Build a fixed validation loader from stored indices.
        """
        if self._fixedValidationLoader is not None:
            return self._fixedValidationLoader
        valIndices = self._getFixedValidationIndices()
        if valIndices is None:
            return None
        valSubset = self._buildEpochDataset(
            valIndices,
            label="fixed validation",
        )
        self._fixedValidationLoader = self._makeDataloader(
            valSubset,
            shuffle=False,
        )
        LOGGER.info(
            "MM: Using fixed validation set: %d samples",
            len(valIndices),
        )
        return self._fixedValidationLoader

    def _getFixedValidationIndexSet(self) -> Optional[set[int]]:
        """
        Return validation indices as a set for fast filtering.
        """
        if self._fixedValidationIndexSet is not None:
            return self._fixedValidationIndexSet
        valIndices = self._getFixedValidationIndices()
        if valIndices is None:
            return None
        self._fixedValidationIndexSet = set(valIndices)
        return self._fixedValidationIndexSet

    def _getFixedValidationIndices(self) -> Optional[List[int]]:
        """
        Load or create fixed validation indices if configured.
        """
        if self.validationIndicesPath is None:
            return None
        if self.validationSplit <= 0.0:
            return None
        if self._fixedValidationIndices is not None:
            return self._fixedValidationIndices
        loaded = self._loadValidationIndices()
        if loaded is None:
            loaded = self._createValidationIndices()
            self._saveValidationIndices(loaded)
        self._fixedValidationIndices = loaded
        return loaded

    def _loadValidationIndices(self) -> Optional[List[int]]:
        """
        Load validation indices from disk if present and valid.
        """
        if self.validationIndicesPath is None:
            return None
        if not self.validationIndicesPath.exists():
            return None
        try:
            payload = json.loads(
                self.validationIndicesPath.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            LOGGER.warning(
                "MM: Invalid validation indices file, regenerating: %s",
                self.validationIndicesPath,
            )
            return None
        indices = payload.get(VALIDATION_INDICES_KEY)
        if not isinstance(indices, list):
            return None
        if not indices:
            return None
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Dataset not initialized.")
        activeIndices = self._activeIndices
        if activeIndices is None:
            raise RuntimeError("Active dataset indices are not initialized.")
        totalSize = len(activeIndices)
        metadata = payload.get(VALIDATION_METADATA_KEY)
        if not _validationMetadataMatches(
            metadata=metadata,
            totalSize=totalSize,
            validationSplit=self.validationSplit,
            datasetFolders=self.datasetFolders,
        ):
            LOGGER.warning(
                "MM: Validation indices metadata mismatch, regenerating: %s",
                self.validationIndicesPath,
            )
            return None
        if any(
            not isinstance(idx, int)
            or idx < 0
            or idx >= len(dataset)
            for idx in indices
        ):
            LOGGER.warning(
                "MM: Validation indices out of range, regenerating."
            )
            return None
        activeIndexSet = self._activeIndexSet
        if activeIndexSet is None:
            return indices
        filtered = sorted([idx for idx in indices if idx in activeIndexSet])
        if not filtered:
            LOGGER.warning(
                "MM: Validation indices do not overlap selected dataset folders."
            )
            return None
        return filtered

    def _createValidationIndices(self) -> List[int]:
        """
        Create a deterministic validation split for the full dataset.
        """
        activeIndices = self._activeIndices
        if activeIndices is None:
            return []
        totalSize = len(activeIndices)
        if totalSize <= 0:
            return []
        valSize = max(int(totalSize * self.validationSplit), MIN_SAMPLES)
        if totalSize > 1:
            valSize = min(valSize, totalSize - 1)
        generator = random.Random(DEFAULT_VALIDATION_SEED)
        positions = list(range(totalSize))
        generator.shuffle(positions)
        selectedPositions = sorted(positions[:valSize])
        return sorted([activeIndices[pos] for pos in selectedPositions])

    def _saveValidationIndices(self, indices: List[int]) -> None:
        """
        Persist validation indices to disk for reuse.
        """
        if self.validationIndicesPath is None:
            return
        self.validationIndicesPath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            VALIDATION_METADATA_KEY: {
                VALIDATION_META_TOTAL: self.totalSize,
                VALIDATION_META_SPLIT: self.validationSplit,
                VALIDATION_META_SEED: DEFAULT_VALIDATION_SEED,
                VALIDATION_META_FOLDERS: _normalizeValidationFolders(
                    self.datasetFolders
                ),
            },
            VALIDATION_INDICES_KEY: indices,
        }
        self.validationIndicesPath.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def _normalizeValidationFolders(
    datasetFolders: Optional[List[str]],
) -> Optional[List[str]]:
    """Return a stable folder signature for validation-split cache files."""
    if not datasetFolders:
        return None
    normalized = sorted(
        {
            _normalizeFolderName(folder)
            for folder in datasetFolders
            if folder and folder.strip()
        }
    )
    return normalized or None


def _validationMetadataMatches(
    metadata: object,
    totalSize: int,
    validationSplit: float,
    datasetFolders: Optional[List[str]],
) -> bool:
    """Return True when cached validation indices match the active run."""
    if not isinstance(metadata, dict):
        return False
    expectedFolders = _normalizeValidationFolders(datasetFolders)
    cachedTotal = metadata.get(VALIDATION_META_TOTAL)
    cachedSplit = metadata.get(VALIDATION_META_SPLIT)
    cachedFolders = metadata.get(VALIDATION_META_FOLDERS)
    if cachedTotal != totalSize:
        return False
    try:
        cachedSplitValue = float(cachedSplit)
    except (TypeError, ValueError):
        return False
    if not math.isclose(
        cachedSplitValue,
        validationSplit,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return False
    if expectedFolders is None:
        return cachedFolders is None
    if not isinstance(cachedFolders, list):
        return False
    normalizedCachedFolders = sorted(
        str(folder).strip().lower()
        for folder in cachedFolders
        if str(folder).strip()
    )
    return normalizedCachedFolders == expectedFolders


def estimateModelBytes(model: torch.nn.Module) -> int:
    """
    Estimate the memory footprint of a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance to inspect.

    Returns
    -------
    int
        Estimated bytes including training overhead.
    """
    parameterBytes = sum(
        param.numel() * param.element_size() for param in model.parameters()
    )
    bufferBytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    totalBytes = parameterBytes + bufferBytes
    return int(totalBytes * MODEL_MEMORY_MULTIPLIER)


def _computeAutoChunkSize(
    dataset: PreprocessedLinkDataset,
    batchSize: int,
    modelMemoryBytes: int,
) -> int:
    """
    Compute an automatic chunk size from memory and model size.

    Parameters
    ----------
    dataset : PreprocessedLinkDataset
        Dataset providing average sample size.
    batchSize : int
        Training batch size.
    modelMemoryBytes : int
        Estimated model memory footprint.
    """
    availableBytes = psutil.virtual_memory().available
    budgetBytes = int(availableBytes * AUTO_MEMORY_FRACTION) - modelMemoryBytes
    if budgetBytes <= 0:
        return max(batchSize, MIN_SAMPLES)
    averageSampleBytes = dataset.getAverageSampleBytes()
    if averageSampleBytes <= 0:
        return max(batchSize, MIN_SAMPLES)
    rawMaxSamples = int(budgetBytes / averageSampleBytes)
    minSamples = max(batchSize * AUTO_MIN_SAMPLE_MULTIPLIER, MIN_SAMPLES)
    boundedSamples = min(rawMaxSamples, len(dataset))
    return max(boundedSamples, minSamples)


def _estimateCoverage(totalSize: int, chunkSize: int) -> int:
    """
    Estimate epochs needed for full dataset coverage.

    Parameters
    ----------
    totalSize : int
        Total number of samples in the dataset.
    chunkSize : int
        Samples per epoch.
    """
    if chunkSize <= 0:
        return 1
    return math.ceil(totalSize / chunkSize)


def _selectShuffledChunkIndices(
    totalSize: int,
    chunkSize: int,
    chunkIndex: int,
    cycleIndex: int,
) -> List[int]:
    """
    Select indices for a shuffled dataset chunk.

    Parameters
    ----------
    totalSize : int
        Total number of samples in the dataset.
    chunkSize : int
        Number of samples per epoch.
    chunkIndex : int
        Zero-based chunk index inside the current full-coverage cycle.
    cycleIndex : int
        Zero-based index of the current full-coverage cycle.
    """
    order = list(range(totalSize))
    generator = random.Random(DEFAULT_CHUNK_SELECTION_SEED + cycleIndex)
    generator.shuffle(order)
    startIndex = chunkIndex * chunkSize
    endIndex = min(startIndex + chunkSize, totalSize)
    return order[startIndex:endIndex]


def _formatShuffledChunkInfo(
    chunkIndex: int,
    cycleEpochs: int,
    chunkSize: int,
    totalSize: int,
) -> str:
    """
    Format a human-readable chunk description.

    Parameters
    ----------
    chunkIndex : int
        Zero-based chunk index inside the current full-coverage cycle.
    cycleEpochs : int
        Number of epochs required to cover the whole dataset once.
    chunkSize : int
        Chunk size.
    totalSize : int
        Total dataset size.
    """
    return (
        f"shuffled chunk {chunkIndex + 1}/{cycleEpochs} "
        f"({chunkSize}/{totalSize} samples)"
    )


class _InMemorySampleDataset(Dataset[Dict[str, object]]):
    """Dataset wrapper for a chunk fully materialized in RAM."""

    def __init__(self, samples: List[Dict[str, object]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.samples[index]
