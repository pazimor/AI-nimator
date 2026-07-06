"""Learning rate scheduling utilities for training workflows.

This module provides a configurable LearningRateScheduler class that supports:
- Linear warmup phase
- Cosine decay phase
- Constant phase (optional)
- Minimum learning rate floor
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

LOGGER = logging.getLogger("shared.learning_rate")


class LRScheduleType(Enum):
    """Available learning rate schedule types."""
    
    CONSTANT = "constant"  # No scheduling, constant LR
    COSINE = "cosine"  # Cosine decay (with optional warmup)
    LINEAR = "linear"  # Linear decay (with optional warmup)
    STEP = "step"  # Step decay at specific epochs


@dataclass(frozen=True)
class LearningRateConfig:
    """
    Configuration for learning rate scheduling.
    
    Attributes
    ----------
    initialLR : float
        Initial learning rate (before warmup).
    minLR : float
        Minimum learning rate floor.
    warmupEpochs : int
        Number of warmup epochs (0 to disable warmup).
    scheduleType : str
        Type of schedule: "constant", "cosine", "linear", "step".
    decayEpochs : Optional[int]
        Number of epochs for decay phase. If None, uses (totalEpochs - warmupEpochs).
    stepSize : int
        For step schedule: decay every N epochs.
    stepGamma : float
        For step schedule: multiply LR by this factor at each step.
    """
    
    initialLR: float = 0.001
    minLR: float = 1e-7
    warmupEpochs: int = 0
    scheduleType: str = "cosine"
    decayEpochs: Optional[int] = None
    stepSize: int = 30
    stepGamma: float = 0.1


class LearningRateScheduler:
    """
    Unified learning rate scheduler for training workflows.
    
    Provides consistent LR scheduling for CLIP and Generation training
    with configurable warmup, decay, and minimum LR floor.
    
    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer to schedule.
    config : LearningRateConfig
        Learning rate configuration.
    totalEpochs : int
        Total number of training epochs.
    
    Examples
    --------
    >>> config = LearningRateConfig(initialLR=0.001, warmupEpochs=10, scheduleType="cosine")
    >>> scheduler = LearningRateScheduler(optimizer, config, totalEpochs=100)
    >>> for epoch in range(100):
    ...     train_one_epoch()
    ...     scheduler.step()
    ...     print(f"LR: {scheduler.getCurrentLR():.6f}")
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config: LearningRateConfig,
        totalEpochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.totalEpochs = totalEpochs
        self._currentEpoch = 0
        
        # Calculate decay epochs
        self.decayEpochs = config.decayEpochs
        if self.decayEpochs is None:
            self.decayEpochs = max(1, totalEpochs - config.warmupEpochs)
        
        # Store base LR from optimizer
        self.baseLR = config.initialLR
        
        # Create the underlying scheduler
        self._scheduler = self._createScheduler()
        
        LOGGER.info(
            "LR Scheduler: type=%s, initial=%.6f, min=%.2e, warmup=%d epochs",
            config.scheduleType,
            config.initialLR,
            config.minLR,
            config.warmupEpochs,
        )
    
    def _createScheduler(self) -> LambdaLR:
        """Create the underlying LambdaLR scheduler."""
        return LambdaLR(self.optimizer, lr_lambda=self._lrLambda)
    
    def _lrLambda(self, epoch: int) -> float:
        """
        Compute the learning rate multiplier for a given epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch (0-based).
            
        Returns
        -------
        float
            Learning rate multiplier (relative to base LR).
        """
        warmupEpochs = self.config.warmupEpochs
        minMultiplier = self.config.minLR / self.baseLR
        
        # Phase 1: Warmup
        if epoch < warmupEpochs:
            # Linear warmup from minLR to baseLR
            warmupProgress = (epoch + 1) / max(1, warmupEpochs)
            multiplier = minMultiplier + (1.0 - minMultiplier) * warmupProgress
            return multiplier
        
        # Phase 2: Decay
        decayEpoch = epoch - warmupEpochs
        decayEpochs = self.decayEpochs
        
        scheduleType = self.config.scheduleType.lower()
        
        if scheduleType == "constant":
            return 1.0
            
        elif scheduleType == "cosine":
            # Cosine decay from baseLR to minLR
            progress = min(1.0, decayEpoch / max(1, decayEpochs))
            cosineDecay = 0.5 * (1.0 + math.cos(math.pi * progress))
            multiplier = minMultiplier + (1.0 - minMultiplier) * cosineDecay
            return max(minMultiplier, multiplier)
            
        elif scheduleType == "linear":
            # Linear decay from baseLR to minLR
            progress = min(1.0, decayEpoch / max(1, decayEpochs))
            multiplier = 1.0 - (1.0 - minMultiplier) * progress
            return max(minMultiplier, multiplier)
            
        elif scheduleType == "step":
            # Step decay
            numSteps = decayEpoch // self.config.stepSize
            multiplier = self.config.stepGamma ** numSteps
            return max(minMultiplier, multiplier)
        
        else:
            LOGGER.warning("Unknown schedule type: %s, using constant", scheduleType)
            return 1.0
    
    def step(self) -> None:
        """Advance the scheduler by one epoch."""
        self._scheduler.step()
        self._currentEpoch += 1
    
    def getCurrentLR(self) -> float:
        """
        Get the current learning rate.
        
        Returns
        -------
        float
            Current learning rate.
        """
        return self.optimizer.param_groups[0]["lr"]
    
    def getLastLR(self) -> float:
        """
        Get the last computed learning rate.
        
        Returns
        -------
        float
            Last learning rate from scheduler.
        """
        return self._scheduler.get_last_lr()[0]
    
    def logStatus(self) -> None:
        """Log current learning rate status."""
        currentLR = self.getCurrentLR()
        phase = "warmup" if self._currentEpoch < self.config.warmupEpochs else "decay"
        LOGGER.info(
            "LR: %.6f (epoch %d, phase: %s)",
            currentLR,
            self._currentEpoch,
            phase,
        )


def buildLearningRateScheduler(
    optimizer: Optimizer,
    totalEpochs: int,
    initialLR: float = 0.001,
    minLR: float = 1e-7,
    warmupEpochs: int = 0,
    scheduleType: str = "cosine",
    decayEpochs: Optional[int] = None,
) -> LearningRateScheduler:
    """
    Build a learning rate scheduler with the specified configuration.
    
    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer to schedule.
    totalEpochs : int
        Total number of training epochs.
    initialLR : float
        Initial learning rate.
    minLR : float
        Minimum learning rate floor.
    warmupEpochs : int
        Number of warmup epochs.
    scheduleType : str
        Type of schedule ("constant", "cosine", "linear", "step").
    decayEpochs : Optional[int]
        Number of epochs for decay phase.
        
    Returns
    -------
    LearningRateScheduler
        Configured scheduler.
    """
    config = LearningRateConfig(
        initialLR=initialLR,
        minLR=minLR,
        warmupEpochs=warmupEpochs,
        scheduleType=scheduleType,
        decayEpochs=decayEpochs,
    )
    return LearningRateScheduler(optimizer, config, totalEpochs)
