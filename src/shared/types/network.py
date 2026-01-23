"""Network architecture and learning rate configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.shared.constants.rotation import (
    DEFAULT_ROTATION_REPR,
    ROTATION_CHANNELS_ROT6D,
)

SPATIOTEMPORAL_MODE_FLAT = "flat"
SPATIOTEMPORAL_MODE_FACTORIZED = "factorized"
DEFAULT_SPATIOTEMPORAL_MODE = SPATIOTEMPORAL_MODE_FLAT


@dataclass(frozen=True)
class LearningRateHyperparameters:
    """
    Learning rate scheduling configuration.
    
    Attributes
    ----------
    initialLR : float
        Initial/base learning rate.
    minLR : float
        Minimum learning rate floor (prevents LR from going too low).
    warmupEpochs : int
        Number of epochs for linear warmup phase (0 to disable).
    scheduleType : str
        Type of LR schedule: "constant", "cosine", "linear", "step".
    decayEpochs : Optional[int]
        Number of epochs for decay phase. If None, uses (totalEpochs - warmupEpochs).
    """
    
    initialLR: float = 0.001
    minLR: float = 1e-7
    warmupEpochs: int = 0
    scheduleType: str = "cosine"
    decayEpochs: Optional[int] = None


@dataclass(frozen=True)
class ClipNetworkConfig:
    """
    CLIP motion encoder architecture configuration.
    
    Attributes
    ----------
    motionNumHeads : int
        Number of attention heads in motion encoder.
    motionNumLayers : int
        Number of transformer layers in motion encoder.
    """
    
    motionNumHeads: int = 4
    motionNumLayers: int = 2


@dataclass(frozen=True)
class GenerationNetworkConfig:
    """
    Generation denoiser architecture configuration.
    
    Attributes
    ----------
    numHeads : int
        Number of attention heads in the denoiser.
    numLayers : int
        Number of denoising transformer layers.
    numSpatialLayers : int
        Number of spatial GCN blocks applied per frame.
    motionChannels : int
        Motion channels per bone (depends on rotation representation).
    rotationRepr : str
        Rotation representation used by the model.
    spatiotemporalMode : str
        Spatio-temporal strategy ("flat" or "factorized").
    numBones : int
        Number of skeleton bones in the dataset.
    diffusionSteps : int
        Number of diffusion timesteps.
    """
    
    numHeads: int = 4
    numLayers: int = 6
    numSpatialLayers: int = 1
    motionChannels: int = ROTATION_CHANNELS_ROT6D
    rotationRepr: str = DEFAULT_ROTATION_REPR
    spatiotemporalMode: str = DEFAULT_SPATIOTEMPORAL_MODE
    numBones: int = 22
    diffusionSteps: int = 1000


@dataclass(frozen=True)
class NetworkConfig:
    """
    Complete network architecture configuration.
    
    Loaded from network.yaml and shared between CLIP and Generation.
    
    Attributes
    ----------
    embedDim : int
        Shared embedding dimension (must match between CLIP and Generation).
    clip : ClipNetworkConfig
        CLIP motion encoder configuration.
    generation : GenerationNetworkConfig
        Generation denoiser configuration.
    """
    
    embedDim: int
    clip: ClipNetworkConfig
    generation: GenerationNetworkConfig
