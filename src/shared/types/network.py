"""Network architecture and learning rate configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
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
class BoneDataConfig:
    """
    Toggle which motion features are prepared and consumed.

    Attributes
    ----------
    rotation6d : bool
        Local 6D joint rotations.
    footContact : bool
        Foot contact channels (MDM-style lower-body contact labels).
    handContact : bool
        Hand contact channels (custom extension, useful for crawl).
    rootTranslation : bool
        Absolute or anchored root translation.
    rootVelocity : bool
        Root linear velocity.
    rootYaw : bool
        Root heading angle.
    rootYawVelocity : bool
        Root angular velocity around the up axis.
    jointXyz : bool
        Global joint positions from forward kinematics.
    jointVelocity : bool
        Global joint velocities from forward kinematics.
    endEffectorVelocity : bool
        Velocities for wrists/ankles end-effectors.
    pelvisHeight : bool
        Root joint height above the floor.
    """

    rotation6d: bool = True
    footContact: bool = False
    handContact: bool = False
    rootTranslation: bool = False
    rootVelocity: bool = False
    rootYaw: bool = False
    rootYawVelocity: bool = False
    jointXyz: bool = False
    jointVelocity: bool = False
    endEffectorVelocity: bool = False
    pelvisHeight: bool = False


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
    boneData : Optional[BoneDataConfig]
        Optional motion feature layout used by the CLIP motion encoder.
        When omitted, CLIP falls back to the legacy rotation-only input.
    """

    motionNumHeads: int = 4
    motionNumLayers: int = 2
    boneData: Optional[BoneDataConfig] = None


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
    numSpatioTemporalLayers : int
        Number of local spatio-temporal blocks near bone split.
    boneData : BoneDataConfig
        Feature toggles shared by preprocessing and training.
    """
    
    embedDim: int = 128
    numHeads: int = 4
    numLayers: int = 6
    numBones: int = 22
    diffusionSteps: int = 1000
    numSpatialLayers: int = 1
    numSpatioTemporalLayers: int = 1
    boneData: BoneDataConfig = field(default_factory=BoneDataConfig)


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
