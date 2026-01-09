"""Unified dataset management for training workflows.

This module provides a reusable DatasetManager class that handles:
- Dataset loading with file logging
- Max samples limiting
- Dataset rotation across epochs
- Memory management with threshold-based GC
"""

from __future__ import annotations

import gc
import logging
import math
import os
import psutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from transformers import XLMRobertaTokenizerFast

from src.shared.model.clip.data import MotionTextClipDataset, motionTextCollate
from src.shared.types import ClipDatasetRecord

LOGGER = logging.getLogger("shared.dataset_manager")

ZERO_FRAME_COUNT = 0
MIN_FRAME_COUNT = 1


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
        return int(self.MM_memoryLimitGB * 1024 * 1024 * 1024)


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
        self._checkInterval = 10  # Check memory every N batches
        
    def getMemoryUsageGB(self) -> float:
        """
        Return current process memory usage in GB.
        
        Returns
        -------
        float
            Memory usage in gigabytes.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    
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
                # MPS memory tracking
                allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
                return allocated
            except Exception:
                return None
        elif self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
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
            
        # Only check every N batches to avoid overhead
        if not force and (batchIndex - self._lastCheckBatch) < self._checkInterval:
            return False
            
        self._lastCheckBatch = batchIndex
        
        # Get current memory usage
        cpuMemGB = self.getMemoryUsageGB()
        gpuMemGB = self.getGPUMemoryUsageGB()
        
        # Check if we exceed the limit
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


@dataclass(frozen=True)
class MotionProcessingConfig:
    """
    Configuration for motion length processing.

    Attributes
    ----------
    motionSplitFrames : Optional[int]
        Maximum frames per sample after splitting (None to disable).
    motionDownsampleTargetFrames : Optional[int]
        Target frame count for temporal downsampling (None to disable).
    """

    motionSplitFrames: Optional[int] = None
    motionDownsampleTargetFrames: Optional[int] = None

    def hasSplit(self) -> bool:
        """Return True when splitting is enabled."""
        return _isFrameCountValid(self.motionSplitFrames)

    def hasDownsample(self) -> bool:
        """Return True when downsampling is enabled."""
        return _isFrameCountValid(self.motionDownsampleTargetFrames)


class MotionWindowDataset(Dataset[Dict[str, object]]):
    """
    Dataset wrapper applying motion splitting and downsampling.

    Uses the underlying MotionTextClipDataset for caching and tokenization.
    """

    def __init__(
        self,
        baseDataset: MotionTextClipDataset,
        processingConfig: MotionProcessingConfig,
    ) -> None:
        """
        Initialize the wrapper dataset.

        Parameters
        ----------
        baseDataset : MotionTextClipDataset
            Base dataset providing motion loading and tokenization.
        processingConfig : MotionProcessingConfig
            Motion length processing configuration.
        """
        self.baseDataset = baseDataset
        self.processingConfig = processingConfig
        self.records = _splitRecords(
            baseDataset.records,
            processingConfig.motionSplitFrames,
        )

    def __len__(self) -> int:
        """Return the number of samples after processing."""
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Return a processed sample from the dataset."""
        record = self.records[index]
        motionSlice, meta = self.baseDataset._sliceMotion(record)
        motionSlice = _downsampleMotion(
            motionSlice,
            self.processingConfig.motionDownsampleTargetFrames,
        )
        return _buildSample(record, motionSlice, meta, self.baseDataset)

    def clearCache(self) -> None:
        """Clear the underlying motion cache."""
        self.baseDataset.clearCache()

    def getCacheStats(self) -> dict:
        """Return cache hit/miss statistics."""
        return self.baseDataset.getCacheStats()


def _isFrameCountValid(value: Optional[int]) -> bool:
    """
    Return True when a frame count is valid.

    Parameters
    ----------
    value : Optional[int]
        Frame count to validate.
    """
    return value is not None and value >= MIN_FRAME_COUNT


def _getRecordLength(record: ClipDatasetRecord) -> int:
    """
    Return the number of frames in a record.

    Parameters
    ----------
    record : ClipDatasetRecord
        Dataset record to inspect.
    """
    return max(record.endFrame - record.startFrame, ZERO_FRAME_COUNT)


def _splitRecords(
    records: List[ClipDatasetRecord],
    splitFrames: Optional[int],
) -> List[ClipDatasetRecord]:
    """
    Split records into fixed-length windows when requested.

    Parameters
    ----------
    records : List[ClipDatasetRecord]
        Dataset records to split.
    splitFrames : Optional[int]
        Maximum frames per split sample.
    """
    if not _isFrameCountValid(splitFrames):
        return list(records)
    return _expandRecords(records, splitFrames)


def _expandRecords(
    records: List[ClipDatasetRecord],
    splitFrames: int,
) -> List[ClipDatasetRecord]:
    """
    Expand records using fixed-length splitting.

    Parameters
    ----------
    records : List[ClipDatasetRecord]
        Dataset records to split.
    splitFrames : int
        Maximum frames per split sample.
    """
    expandedRecords: List[ClipDatasetRecord] = []
    for record in records:
        expandedRecords.extend(_splitRecord(record, splitFrames))
    return expandedRecords


def _splitRecord(
    record: ClipDatasetRecord,
    splitFrames: int,
) -> List[ClipDatasetRecord]:
    """
    Split a single record into fixed-length windows.

    Parameters
    ----------
    record : ClipDatasetRecord
        Dataset record to split.
    splitFrames : int
        Maximum frames per split sample.
    """
    recordLength = _getRecordLength(record)
    if recordLength <= splitFrames:
        return [record]
    return _buildSplitRecords(record, splitFrames)


def _buildSplitRecords(
    record: ClipDatasetRecord,
    splitFrames: int,
) -> List[ClipDatasetRecord]:
    """
    Build split records using contiguous windows.

    Parameters
    ----------
    record : ClipDatasetRecord
        Dataset record to split.
    splitFrames : int
        Maximum frames per split sample.
    """
    splitRecords: List[ClipDatasetRecord] = []
    startFrame = record.startFrame
    while startFrame < record.endFrame:
        endFrame = min(startFrame + splitFrames, record.endFrame)
        if endFrame <= startFrame:
            break
        splitRecords.append(_cloneRecord(record, startFrame, endFrame))
        startFrame = endFrame
    return splitRecords


def _cloneRecord(
    record: ClipDatasetRecord,
    startFrame: int,
    endFrame: int,
) -> ClipDatasetRecord:
    """
    Clone a record with new frame bounds.

    Parameters
    ----------
    record : ClipDatasetRecord
        Source dataset record.
    startFrame : int
        New start frame.
    endFrame : int
        New end frame.
    """
    return ClipDatasetRecord(
        promptText=record.promptText,
        tag=record.tag,
        animationPath=record.animationPath,
        startFrame=startFrame,
        endFrame=endFrame,
        sourceFile=record.sourceFile,
        metadata=record.metadata,
    )


def _computeDownsampleStride(currentFrames: int, targetFrames: int) -> int:
    """
    Compute stride to reduce frames to the target count.

    Parameters
    ----------
    currentFrames : int
        Current motion frame count.
    targetFrames : int
        Desired target frame count.
    """
    stride = math.ceil(currentFrames / targetFrames)
    return max(stride, MIN_FRAME_COUNT)


def _downsampleMotion(
    motion: torch.Tensor,
    targetFrames: Optional[int],
) -> torch.Tensor:
    """
    Downsample a motion tensor to a target frame count.

    Parameters
    ----------
    motion : torch.Tensor
        Motion tensor shaped (frames, bones, channels).
    targetFrames : Optional[int]
        Desired target frame count.
    """
    if not _isFrameCountValid(targetFrames):
        return motion
    currentFrames = motion.shape[0]
    if currentFrames <= targetFrames:
        return motion
    stride = _computeDownsampleStride(currentFrames, targetFrames)
    return motion[::stride]


def _buildSample(
    record: ClipDatasetRecord,
    motionSlice: torch.Tensor,
    meta: Dict[str, object],
    dataset: MotionTextClipDataset,
) -> Dict[str, object]:
    """
    Build a dataset sample dictionary.

    Parameters
    ----------
    record : ClipDatasetRecord
        Dataset record metadata.
    motionSlice : torch.Tensor
        Motion slice for the record.
    meta : Dict[str, object]
        Metadata from the motion payload.
    dataset : MotionTextClipDataset
        Dataset used for tokenization.
    """
    encoded = dataset._tokenize(record.tag, record.promptText)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "motion": motionSlice,
        "time": torch.tensor([record.startFrame, record.endFrame]),
        "tag": record.tag,
        "meta": meta,
    }


class DatasetManager:
    """
    Unified dataset management for CLIP and Generation training.
    
    Handles:
    - Dataset loading with file logging
    - Max samples limiting
    - Dataset rotation across epochs
    - Memory management integration
    
    Parameters
    ----------
    datasetRoot : Path
        Root directory containing prompt and animation files.
    maxLength : int
        Maximum token length for prompts.
    batchSize : int
        Batch size for dataloaders.
    modelName : str
        Hugging Face model name for tokenizer.
    validationSplit : float
        Fraction of data for validation.
    maxSamples : Optional[int]
        Maximum samples per epoch (None = use all).
    rotateDataset : bool
        Whether to rotate through dataset chunks each epoch.
    motionSplitFrames : Optional[int]
        Maximum frames per sample after splitting (None = use all).
    motionDownsampleTargetFrames : Optional[int]
        Target frame count for temporal downsampling (None = disable).
    cacheMotion : bool
        Whether to cache motion payloads for reuse.
    memoryConfig : Optional[MemoryManagerConfig]
        Memory management configuration.
    device : Optional[torch.device]
        Target device for memory management.
    promptRoot : Optional[Path]
        Separate prompt root (defaults to datasetRoot).
    animationRoot : Optional[Path]
        Separate animation root (defaults to datasetRoot).
    """
    
    def __init__(
        self,
        datasetRoot: Path,
        maxLength: int,
        batchSize: int,
        modelName: str,
        validationSplit: float,
        maxSamples: Optional[int] = None,
        rotateDataset: bool = False,
        motionSplitFrames: Optional[int] = None,
        motionDownsampleTargetFrames: Optional[int] = None,
        cacheMotion: bool = True,
        memoryConfig: Optional[MemoryManagerConfig] = None,
        device: Optional[torch.device] = None,
        promptRoot: Optional[Path] = None,
        animationRoot: Optional[Path] = None,
    ) -> None:
        self.datasetRoot = datasetRoot
        self.promptRoot = promptRoot or datasetRoot
        self.animationRoot = animationRoot or datasetRoot
        self.maxLength = maxLength
        self.batchSize = batchSize
        self.modelName = modelName
        self.validationSplit = validationSplit
        self.maxSamples = maxSamples
        self.rotateDataset = rotateDataset
        self.motionProcessingConfig = MotionProcessingConfig(
            motionSplitFrames=motionSplitFrames,
            motionDownsampleTargetFrames=motionDownsampleTargetFrames,
        )
        self.cacheMotion = cacheMotion
        self.currentOffset = 0
        
        # Memory management
        self.memoryConfig = memoryConfig or MemoryManagerConfig()
        self.memoryManager = MemoryManager(self.memoryConfig, device)
        
        # Lazy-loaded dataset
        self._dataset: Optional[Dataset[Dict[str, object]]] = None
        self._baseDataset: Optional[MotionTextClipDataset] = None
        self._totalSize: Optional[int] = None
        self._allIndices: Optional[List[int]] = None
        self._staticDataloaders: Optional[
            Tuple[DataLoader, Optional[DataLoader], str]
        ] = None
        
    def _ensureDataset(self) -> Dataset[Dict[str, object]]:
        """
        Lazy-load dataset metadata.
        
        Returns
        -------
        Dataset[Dict[str, object]]
            The processed dataset.
        """
        if self._dataset is None:
            LOGGER.info("MM: Loading dataset from %s", self.promptRoot)
            self._baseDataset = self._buildBaseDataset()
            self._dataset = self._wrapDataset(self._baseDataset)
            self._totalSize = len(self._dataset)
            self._allIndices = list(range(self._totalSize))
            self._staticDataloaders = None
            LOGGER.info("MM: Dataset indexed: %d total samples", self._totalSize)
            
        return self._dataset

    def _buildBaseDataset(self) -> MotionTextClipDataset:
        """
        Build the base dataset instance.

        Returns
        -------
        MotionTextClipDataset
            Dataset instance without motion length transforms.
        """
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.modelName)
        return MotionTextClipDataset(
            rootPrompts=self.promptRoot,
            rootAnimations=self.animationRoot,
            tokenizer=tokenizer,
            maxLength=self.maxLength,
            cacheMotion=self.cacheMotion,
        )

    def _wrapDataset(
        self,
        baseDataset: MotionTextClipDataset,
    ) -> Dataset[Dict[str, object]]:
        """
        Apply motion length processing when configured.

        Parameters
        ----------
        baseDataset : MotionTextClipDataset
            Dataset providing motion and tokenization logic.
        """
        if not self.motionProcessingConfig.hasSplit() and not (
            self.motionProcessingConfig.hasDownsample()
        ):
            return baseDataset
        return MotionWindowDataset(baseDataset, self.motionProcessingConfig)
    
    @property
    def totalSize(self) -> int:
        """Get total dataset size."""
        self._ensureDataset()
        return self._totalSize or 0
    
    @property
    def effectiveSamplesPerEpoch(self) -> int:
        """Get effective samples used per epoch."""
        if self.maxSamples is not None:
            return min(self.maxSamples, self.totalSize)
        return self.totalSize
    
    @property
    def epochsForFullCoverage(self) -> int:
        """Get number of epochs needed to see full dataset."""
        if not self.rotateDataset or self.maxSamples is None:
            return 1
        return math.ceil(self.totalSize / self.maxSamples)
    
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
            Train dataloader, optional validation dataloader, and chunk info string.
        """
        dataset = self._ensureDataset()
        
        if self.rotateDataset and self.maxSamples is not None:
            return self._getRotatingDataloaders(epochIndex)
        else:
            return self._getStaticDataloaders()
    
    def _getRotatingDataloaders(
        self,
        epochIndex: int,
    ) -> Tuple[DataLoader, Optional[DataLoader], str]:
        """Get dataloaders with rotating chunk selection."""
        dataset = self._dataset
        totalSize = self._totalSize or len(dataset)
        chunkSize = self.maxSamples or totalSize
        
        # Clear cache from previous chunk to free memory
        if epochIndex > 0:
            self.clearMotionCache()
        
        # Calculate chunk boundaries
        startIdx = (epochIndex * chunkSize) % totalSize
        endIdx = min(startIdx + chunkSize, totalSize)
        
        # Handle wrap-around
        if startIdx + chunkSize > totalSize:
            indices = list(range(startIdx, totalSize)) + list(
                range(0, (startIdx + chunkSize) % totalSize)
            )
        else:
            indices = list(range(startIdx, endIdx))
        
        chunkInfo = f"samples {startIdx + 1}-{min(startIdx + chunkSize, totalSize)}/{totalSize}"
        
        LOGGER.info(
            "MM: Loading chunk for epoch %d: %s",
            epochIndex + 1,
            chunkInfo,
        )
        
        return self._buildDataloaders(indices, chunkInfo)
    
    def _getStaticDataloaders(self) -> Tuple[DataLoader, Optional[DataLoader], str]:
        """Get static dataloaders (all samples or limited)."""
        if self._staticDataloaders is not None:
            return self._staticDataloaders
        dataset = self._dataset
        totalSize = self._totalSize or len(dataset)
        
        if self.maxSamples is not None and self.maxSamples < totalSize:
            indices = list(range(self.maxSamples))
            chunkInfo = f"limited to {self.maxSamples}/{totalSize}"
            LOGGER.info("MM: Using limited dataset: %s", chunkInfo)
        else:
            indices = list(range(totalSize))
            chunkInfo = f"all {totalSize} samples"
            
        self._staticDataloaders = self._buildDataloaders(indices, chunkInfo)
        return self._staticDataloaders
    
    def _buildDataloaders(
        self,
        indices: List[int],
        chunkInfo: str,
    ) -> Tuple[DataLoader, Optional[DataLoader], str]:
        """Build train/val dataloaders from indices."""
        dataset = self._dataset
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
        return DataLoader(
            dataset,
            batch_size=self.batchSize,
            shuffle=shuffle,
            collate_fn=motionTextCollate,
            num_workers=0,
            pin_memory=False,
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
    
    def clearMotionCache(self) -> None:
        """
        Clear the motion cache.
        
        Call this between dataset rotations or when memory needs to be freed.
        Also triggers garbage collection.
        """
        if self._baseDataset is not None:
            self._baseDataset.clearCache()
        self.memoryManager._performCleanup()
        LOGGER.info("MM: Motion cache and memory cleared")
    
    def getCacheStats(self) -> dict:
        """Return cache hit/miss statistics."""
        if self._baseDataset is not None:
            return self._baseDataset.getCacheStats()
        return {"hits": 0, "misses": 0, "hitRate": 0.0}
    
    def logMemoryStatus(self, context: str = "") -> None:
        """Log current memory usage."""
        self.memoryManager.logMemoryStatus(context)
