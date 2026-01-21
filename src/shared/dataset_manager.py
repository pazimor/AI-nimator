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
from src.shared.preprocessed_dataset import PreprocessedMotionDataset

LOGGER = logging.getLogger("shared.dataset_manager")

BYTES_PER_GB = 1024 * 1024 * 1024
AUTO_MEMORY_FRACTION = 0.6
AUTO_MIN_SAMPLE_MULTIPLIER = 4
DEFAULT_NUM_WORKERS_DIVISOR = 2
DEFAULT_PREFETCH_FACTOR = 2
MAX_NUM_WORKERS = 8
MIN_SAMPLES = 1
MODEL_MEMORY_MULTIPLIER = 3
DEFAULT_VALIDATION_SEED = 42
VALIDATION_INDICES_KEY = "indices"
VALIDATION_METADATA_KEY = "metadata"
VALIDATION_META_TOTAL = "total_size"
VALIDATION_META_SPLIT = "validation_split"
VALIDATION_META_SEED = "seed"


@dataclass
class MemoryManagerConfig:
    """
    Configuration for memory management.

    Attributes
    ----------
    MM_memoryLimitGB : float
        Maximum memory usage in GB before triggering cleanup.
        Set to 0 to disable memory-based cleanup.
    """

    MM_memoryLimitGB: float = 0.0

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
                if hasattr(torch.mps, "empty_cache"):
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
    ) -> None:
        self.datasetRoot = datasetRoot
        self.batchSize = batchSize
        self.validationSplit = validationSplit
        self.modelMemoryBytes = modelMemoryBytes
        self.device = device
        self.validationIndicesPath = validationIndicesPath
        self.maxSamplesPerEpoch = maxSamplesPerEpoch

        self.memoryConfig = memoryConfig or MemoryManagerConfig()
        self.memoryManager = MemoryManager(self.memoryConfig, device)

        self._dataset: Optional[PreprocessedMotionDataset] = None
        self._totalSize: Optional[int] = None
        self._maxSamples: Optional[int] = None
        self._fixedValidationIndices: Optional[List[int]] = None
        self._fixedValidationIndexSet: Optional[set[int]] = None
        self._fixedValidationLoader: Optional[DataLoader] = None

    def _ensureDataset(self) -> PreprocessedMotionDataset:
        """
        Lazy-load dataset metadata.

        Returns
        -------
        PreprocessedMotionDataset
            Loaded preprocessed dataset.
        """
        if self._dataset is None:
            LOGGER.info(
                "MM: Loading preprocessed dataset from %s",
                self.datasetRoot,
            )
            self._dataset = PreprocessedMotionDataset(self.datasetRoot)
            self._totalSize = len(self._dataset)
            self._maxSamples = None
            LOGGER.info(
                "MM: Dataset indexed: %d total samples",
                self._totalSize,
            )
        return self._dataset

    @property
    def dataset(self) -> PreprocessedMotionDataset:
        """Return the underlying dataset instance."""
        return self._ensureDataset()

    @property
    def totalSize(self) -> int:
        """Get total dataset size."""
        self._ensureDataset()
        return self._totalSize or 0

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

    def _getRotatingDataloaders(
        self,
        epochIndex: int,
    ) -> Tuple[DataLoader, Optional[DataLoader], str]:
        """Get dataloaders with rotating chunk selection."""
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Dataset not initialized.")
        totalSize = self._totalSize or len(dataset)
        chunkSize = self._getMaxSamples()
        valIndexSet = self._getFixedValidationIndexSet()

        if epochIndex > 0:
            self.clearCache()

        if chunkSize >= totalSize:
            indices = list(range(totalSize))
            chunkInfo = f"all {totalSize} samples"
            return self._buildDataloaders(
                indices,
                chunkInfo,
                valIndexSet,
            )

        startIndex, indices = _selectChunkIndices(
            totalSize=totalSize,
            chunkSize=chunkSize,
            epochIndex=epochIndex,
        )
        chunkInfo = _formatChunkInfo(startIndex, chunkSize, totalSize)
        LOGGER.info(
            "MM: Loading chunk for epoch %d: %s",
            epochIndex + 1,
            chunkInfo,
        )
        return self._buildDataloaders(indices, chunkInfo, valIndexSet)

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
            trainLoader = self._makeDataloader(
                Subset(dataset, trainIndices),
                shuffle=True,
            )
            valLoader = self._buildFixedValidationLoader()
            return trainLoader, valLoader, chunkInfo
        chunkSubset = Subset(dataset, indices)

        if self.validationSplit <= 0.0 or len(indices) < 2:
            trainLoader = self._makeDataloader(chunkSubset, shuffle=True)
            return trainLoader, None, chunkInfo

        valSize = max(1, int(len(indices) * self.validationSplit))
        trainSize = len(indices) - valSize
        trainSubset, valSubset = random_split(chunkSubset, [trainSize, valSize])

        trainLoader = self._makeDataloader(trainSubset, shuffle=True)
        valLoader = self._makeDataloader(valSubset, shuffle=False)

        return trainLoader, valLoader, chunkInfo

    def _makeDataloader(
        self,
        dataset: Dataset[Dict[str, object]],
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a dataloader with custom collation."""
        numWorkers = _resolveNumWorkers()
        pinMemory = self.device is not None and self.device.type == "cuda"
        persistentWorkers = numWorkers > 0
        prefetchFactor = DEFAULT_PREFETCH_FACTOR if numWorkers > 0 else None
        return DataLoader(
            dataset,
            batch_size=self.batchSize,
            shuffle=shuffle,
            collate_fn=motionTextCollate,
            num_workers=numWorkers,
            pin_memory=pinMemory,
            persistent_workers=persistentWorkers,
            prefetch_factor=prefetchFactor,
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
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Dataset not initialized.")
        valSubset = Subset(dataset, valIndices)
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
        totalSize = self.totalSize
        if any(
            not isinstance(idx, int)
            or idx < 0
            or idx >= totalSize
            for idx in indices
        ):
            LOGGER.warning(
                "MM: Validation indices out of range, regenerating."
            )
            return None
        return indices

    def _createValidationIndices(self) -> List[int]:
        """
        Create a deterministic validation split for the full dataset.
        """
        totalSize = self.totalSize
        if totalSize <= 0:
            return []
        valSize = max(int(totalSize * self.validationSplit), MIN_SAMPLES)
        if totalSize > 1:
            valSize = min(valSize, totalSize - 1)
        generator = random.Random(DEFAULT_VALIDATION_SEED)
        indices = list(range(totalSize))
        generator.shuffle(indices)
        return sorted(indices[:valSize])

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
            },
            VALIDATION_INDICES_KEY: indices,
        }
        self.validationIndicesPath.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


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
    dataset: PreprocessedMotionDataset,
    batchSize: int,
    modelMemoryBytes: int,
) -> int:
    """
    Compute an automatic chunk size from memory and model size.

    Parameters
    ----------
    dataset : PreprocessedMotionDataset
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


def _selectChunkIndices(
    totalSize: int,
    chunkSize: int,
    epochIndex: int,
) -> Tuple[int, List[int]]:
    """
    Select indices for a rotating dataset chunk.

    Parameters
    ----------
    totalSize : int
        Total number of samples in the dataset.
    chunkSize : int
        Number of samples per epoch.
    epochIndex : int
        Zero-based epoch index.
    """
    startIndex = (epochIndex * chunkSize) % totalSize
    endIndex = min(startIndex + chunkSize, totalSize)
    if startIndex + chunkSize > totalSize:
        indices = list(range(startIndex, totalSize)) + list(
            range(0, (startIndex + chunkSize) % totalSize)
        )
    else:
        indices = list(range(startIndex, endIndex))
    return startIndex, indices


def _formatChunkInfo(startIndex: int, chunkSize: int, totalSize: int) -> str:
    """
    Format a human-readable chunk description.

    Parameters
    ----------
    startIndex : int
        Chunk start index.
    chunkSize : int
        Chunk size.
    totalSize : int
        Total dataset size.
    """
    endIndex = min(startIndex + chunkSize, totalSize)
    return f"samples {startIndex + 1}-{endIndex}/{totalSize}"


def _resolveNumWorkers() -> int:
    """
    Resolve default number of dataloader workers.

    Returns
    -------
    int
        Resolved worker count.
    """
    cpuCount = os.cpu_count() or 1
    suggested = max(cpuCount // DEFAULT_NUM_WORKERS_DIVISOR, 1)
    return min(suggested, MAX_NUM_WORKERS)
