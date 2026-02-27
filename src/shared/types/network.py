"""Network architecture and learning rate configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    embedDim : int
        Hidden width of the generation denoiser.
    numHeads : int
        Number of attention heads in the denoiser.
    numLayers : int
        Number of denoising transformer layers.
    numBones : int
        Number of skeleton bones in the dataset.
    diffusionSteps : int
        Number of diffusion timesteps.
    numSpatialLayers : int
        Number of spatial GCN blocks near bone split.
    numHierarchyLayers : int
        Number of directed hierarchy blocks near bone split.
    numSpatioTemporalLayers : int
        Number of local spatio-temporal blocks near bone split.
    """
    
    embedDim: int = 128
    numHeads: int = 4
    numLayers: int = 6
    numBones: int = 22
    diffusionSteps: int = 1000
    numSpatialLayers: int = 1
    numHierarchyLayers: int = 1
    numSpatioTemporalLayers: int = 1


@dataclass(frozen=True)
class NetworkConfig:
    """
    Complete network architecture configuration.
    
    Loaded from network.yaml.
    
    Attributes
    ----------
    embedDim : int
        CLIP embedding dimension used by the frozen text encoder.
    clip : ClipNetworkConfig
        CLIP motion encoder configuration.
    generation : GenerationNetworkConfig
        Generation denoiser configuration, including its own width.
    """
    
    embedDim: int
    clip: ClipNetworkConfig
    generation: GenerationNetworkConfig
