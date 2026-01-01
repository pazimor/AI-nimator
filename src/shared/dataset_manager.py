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
from typing import Optional, Tuple, List

import torch
from torch.utils.data import DataLoader, Subset, random_split
from transformers import XLMRobertaTokenizerFast

from src.shared.model.clip.data import MotionTextClipDataset, motionTextCollate

LOGGER = logging.getLogger("shared.dataset_manager")


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
        self.currentOffset = 0
        
        # Memory management
        self.memoryConfig = memoryConfig or MemoryManagerConfig()
        self.memoryManager = MemoryManager(self.memoryConfig, device)
        
        # Lazy-loaded dataset
        self._dataset: Optional[MotionTextClipDataset] = None
        self._totalSize: Optional[int] = None
        self._allIndices: Optional[List[int]] = None
        
    def _ensureDataset(self) -> MotionTextClipDataset:
        """
        Lazy-load dataset metadata.
        
        Returns
        -------
        MotionTextClipDataset
            The loaded dataset.
        """
        if self._dataset is None:
            LOGGER.info("MM: Loading dataset from %s", self.promptRoot)
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.modelName)
            self._dataset = MotionTextClipDataset(
                rootPrompts=self.promptRoot,
                rootAnimations=self.animationRoot,
                tokenizer=tokenizer,
                maxLength=self.maxLength,
                cacheMotion=True,  # Enable LRU cache (single-entry, auto-clears)
            )
            self._totalSize = len(self._dataset)
            self._allIndices = list(range(self._totalSize))
            LOGGER.info("MM: Dataset indexed: %d total samples", self._totalSize)
            
        return self._dataset
    
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
        dataset = self._dataset
        totalSize = self._totalSize or len(dataset)
        
        if self.maxSamples is not None and self.maxSamples < totalSize:
            indices = list(range(self.maxSamples))
            chunkInfo = f"limited to {self.maxSamples}/{totalSize}"
            LOGGER.info("MM: Using limited dataset: %s", chunkInfo)
        else:
            indices = list(range(totalSize))
            chunkInfo = f"all {totalSize} samples"
            
        return self._buildDataloaders(indices, chunkInfo)
    
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
        dataset: MotionTextClipDataset,
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
        if self._dataset is not None:
            self._dataset.clearCache()
        self.memoryManager._performCleanup()
        LOGGER.info("MM: Motion cache and memory cleared")
    
    def getCacheStats(self) -> dict:
        """Return cache hit/miss statistics."""
        if self._dataset is not None:
            return self._dataset.getCacheStats()
        return {"hits": 0, "misses": 0, "hitRate": 0.0}
    
    def logMemoryStatus(self, context: str = "") -> None:
        """Log current memory usage."""
        self.memoryManager.logMemoryStatus(context)
