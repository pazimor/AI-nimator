"""Generate motion from a checkpoint and export animation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
import yaml

from src.features.dataset_builder.animation_rebuilder import AnimationRebuilder
from src.features.generation.train_generation import loadCheckpoint
from src.shared.config_loader import loadBuilderConfig, loadNetworkConfig
from src.shared.constants.clip import DEFAULT_MODEL_NAME
from src.shared.constants.skeletons import SMPL22_BONE_ORDER, SMPL24_BONE_ORDER
from src.shared.model.generation.motion_generator import MotionGenerator
from src.shared.quaternion import Rotation
from src.shared.types import (
    AnimationSample,
    DatasetBuilderConfig,
    GenerationInferenceConfig,
    GenerationModelSettings,
    GenerationOutputOptions,
)
from src.shared.types.network import NetworkConfig

AXIS_ANGLE_CHANNELS = 3
JSON_INDENT = 2
SMPL22_BONE_COUNT = len(SMPL22_BONE_ORDER)
SMPL24_BONE_COUNT = len(SMPL24_BONE_ORDER)

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"

EXTRA_PROMPT_KEY = "prompt"
EXTRA_TAG_KEY = "tag"
EXTRA_CHECKPOINT_KEY = "checkpoint"
EXTRA_MODEL_NAME_KEY = "modelName"
EXTRA_DDIM_STEPS_KEY = "ddimSteps"

YAML_PATHS_KEY = "paths"
YAML_TRAINING_KEY = "training"
YAML_CLIP_CHECKPOINT_KEY = "clip-checkpoint"
YAML_NETWORK_CONFIG_KEY = "network-config"
YAML_MODEL_NAME_KEY = "model-name"


def loadYamlPayload(configPath: Path) -> dict[str, object]:
    """
    Load a YAML file into a dictionary payload.

    Parameters
    ----------
    configPath : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, object]
        Parsed YAML payload (empty if file is empty).
    """
    if not configPath.exists():
        raise FileNotFoundError(f"Missing config file: {configPath}")
    content = configPath.read_text(encoding="utf-8")
    return yaml.safe_load(content) or {}


def ensureDict(value: object) -> dict[str, object]:
    """
    Return a dictionary if the input is a dict, else an empty dict.

    Parameters
    ----------
    value : object
        Value to check.

    Returns
    -------
    dict[str, object]
        Input cast to dict when possible, or empty dict.
    """
    if isinstance(value, dict):
        return value
    return {}


def optionalString(value: object) -> Optional[str]:
    """
    Return a string when the input is a non-empty string.

    Parameters
    ----------
    value : object
        Input value to normalize.

    Returns
    -------
    Optional[str]
        Normalized string or None.
    """
    if isinstance(value, str) and value:
        return value
    return None


def resolveOptionalPath(
    configPath: Path,
    rawPath: Optional[str],
) -> Optional[Path]:
    """
    Resolve a path string relative to a config file location.

    Parameters
    ----------
    configPath : Path
        Configuration file used as base directory.
    rawPath : Optional[str]
        Raw path string from configuration.

    Returns
    -------
    Optional[Path]
        Resolved path or None when input is empty.
    """
    if not rawPath:
        return None
    return (configPath.parent / rawPath).expanduser().resolve()


def loadInferenceSettings(
    configPath: Path,
    profile: Optional[str],
) -> tuple[Optional[Path], Optional[Path], str]:
    """
    Extract inference-related settings from a training config file.

    Parameters
    ----------
    configPath : Path
        Path to the generation training config.
    profile : Optional[str]
        Optional profile name to read (defaults to "training").

    Returns
    -------
    tuple[Optional[Path], Optional[Path], str]
        Clip checkpoint path, network config path, and model name.
    """
    payload = loadYamlPayload(configPath)
    pathsSection = ensureDict(payload.get(YAML_PATHS_KEY, {}))
    sectionName = profile or YAML_TRAINING_KEY
    trainingSection = ensureDict(payload.get(sectionName, {}))
    modelName = str(
        trainingSection.get(YAML_MODEL_NAME_KEY, DEFAULT_MODEL_NAME)
    )
    clipCheckpoint = resolveOptionalPath(
        configPath,
        optionalString(pathsSection.get(YAML_CLIP_CHECKPOINT_KEY)),
    )
    networkConfig = resolveOptionalPath(
        configPath,
        optionalString(pathsSection.get(YAML_NETWORK_CONFIG_KEY)),
    )
    return clipCheckpoint, networkConfig, modelName


def requireExistingPath(path: Path, label: str) -> Path:
    """
    Ensure a filesystem path exists.

    Parameters
    ----------
    path : Path
        Path to validate.
    label : str
        Human-readable label for error messages.

    Returns
    -------
    Path
        The validated path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def validateOptionalPath(path: Optional[Path], label: str) -> Optional[Path]:
    """
    Ensure an optional path exists when provided.

    Parameters
    ----------
    path : Optional[Path]
        Optional path to validate.
    label : str
        Label for error messages.

    Returns
    -------
    Optional[Path]
        The validated path or None.
    """
    if path is None:
        return None
    return requireExistingPath(path, label)


def resolveDevice(deviceName: str) -> torch.device:
    """
    Resolve a torch.device from a CLI string.

    Parameters
    ----------
    deviceName : str
        Requested device ("auto", "cuda", "mps", "cpu").

    Returns
    -------
    torch.device
        Resolved device instance.
    """
    if deviceName == DEVICE_AUTO:
        if torch.cuda.is_available():
            return torch.device(DEVICE_CUDA)
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device(DEVICE_MPS)
        return torch.device(DEVICE_CPU)
    if deviceName == DEVICE_CUDA:
        return torch.device(DEVICE_CUDA)
    if deviceName == DEVICE_MPS:
        return torch.device(DEVICE_MPS)
    return torch.device(deviceName)


def selectBoneOrder(numBones: int) -> list[str]:
    """
    Choose the skeleton bone order based on the model bone count.

    Parameters
    ----------
    numBones : int
        Number of bones expected by the model.

    Returns
    -------
    list[str]
        Bone name ordering matching the model output.
    """
    if numBones == SMPL22_BONE_COUNT:
        return SMPL22_BONE_ORDER
    if numBones == SMPL24_BONE_COUNT:
        return SMPL24_BONE_ORDER
    raise ValueError(f"Unsupported bone count: {numBones}")


def buildMotionGenerator(
    networkConfig: NetworkConfig,
    modelName: str,
    clipCheckpointPath: Optional[Path],
) -> MotionGenerator:
    """
    Instantiate a motion generator from network config.

    Parameters
    ----------
    networkConfig : NetworkConfig
        Loaded network architecture configuration.
    modelName : str
        Hugging Face model identifier for the text encoder.
    clipCheckpointPath : Optional[Path]
        Path to CLIP checkpoint, if available.

    Returns
    -------
    MotionGenerator
        Initialized generation model.
    """
    return MotionGenerator(
        embedDim=networkConfig.embedDim,
        numHeads=networkConfig.generation.numHeads,
        numLayers=networkConfig.generation.numLayers,
        numBones=networkConfig.generation.numBones,
        diffusionSteps=networkConfig.generation.diffusionSteps,
        modelName=modelName,
        clipCheckpoint=clipCheckpointPath,
    )


def loadModelCheckpoint(
    checkpointPath: Path,
    model: MotionGenerator,
) -> None:
    """
    Load a generation checkpoint into the model.

    Parameters
    ----------
    checkpointPath : Path
        Path to the generation checkpoint.
    model : MotionGenerator
        Model instance to load.
    """
    loadCheckpoint(checkpointPath=checkpointPath, model=model)


def generateMotionQuat(
    model: MotionGenerator,
    inferenceConfig: GenerationInferenceConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Run text-conditioned motion generation.

    Parameters
    ----------
    model : MotionGenerator
        Generation model.
    inferenceConfig : GenerationInferenceConfig
        Prompt, tag, and sampling configuration.
    device : torch.device
        Device for inference.

    Returns
    -------
    torch.Tensor
        Generated quaternion motion tensor.
    """
    return model.generate(
        prompt=inferenceConfig.prompt,
        tag=inferenceConfig.tag,
        numFrames=inferenceConfig.frames,
        ddimSteps=inferenceConfig.ddimSteps,
        device=device,
    )


def convertQuaternionToAxisAngles(motionQuat: torch.Tensor) -> np.ndarray:
    """
    Convert quaternion motion to axis-angle representation.

    Parameters
    ----------
    motionQuat : torch.Tensor
        Quaternion motion tensor in (w, x, y, z) order.

    Returns
    -------
    np.ndarray
        Axis-angle rotations shaped (frames, bones, 3).
    """
    quatReordered = torch.cat(
        (motionQuat[..., 1:], motionQuat[..., :1]),
        dim=-1,
    )
    axisAngles = Rotation(quatReordered, kind="quat").axis_angle
    axisAngles = axisAngles.squeeze(0).detach().cpu().numpy()
    return axisAngles.astype(np.float32)


def mapAxisAnglesToSmpl24(
    axisAngles: np.ndarray,
    boneOrder: list[str],
) -> np.ndarray:
    """
    Map axis-angles to SMPL-24 ordering with missing bones filled.

    Parameters
    ----------
    axisAngles : np.ndarray
        Axis-angle rotations shaped (frames, bones, 3).
    boneOrder : list[str]
        Bone order for the incoming tensor.

    Returns
    -------
    np.ndarray
        Axis-angles shaped (frames, 24, 3) in SMPL-24 order.
    """
    boneCount = axisAngles.shape[1]
    if boneCount != len(boneOrder):
        raise ValueError(
            "Bone count mismatch: "
            f"{boneCount} != {len(boneOrder)}"
        )
    frameCount = axisAngles.shape[0]
    axisAnglesFull = np.zeros(
        (frameCount, SMPL24_BONE_COUNT, AXIS_ANGLE_CHANNELS),
        dtype=np.float32,
    )
    boneIndex = {
        boneName: index
        for index, boneName in enumerate(SMPL24_BONE_ORDER)
    }
    for sourceIndex, boneName in enumerate(boneOrder):
        targetIndex = boneIndex[boneName]
        axisAnglesFull[:, targetIndex, :] = axisAngles[:, sourceIndex, :]
    return axisAnglesFull


def buildAnimationSample(
    axisAnglesFull: np.ndarray,
    fps: int,
    outputJsonPath: Path,
    extras: dict[str, object],
) -> AnimationSample:
    """
    Build an AnimationSample from axis-angles.

    Parameters
    ----------
    axisAnglesFull : np.ndarray
        Axis-angles in SMPL-24 order shaped (frames, 24, 3).
    fps : int
        Frames per second for the animation.
    outputJsonPath : Path
        Output JSON path used as sample source.
    extras : dict[str, object]
        Extra metadata to store with the sample.

    Returns
    -------
    AnimationSample
        Sample ready for JSON and Collada export.
    """
    frameCount = axisAnglesFull.shape[0]
    flatAngles = axisAnglesFull.reshape(frameCount, -1)
    return AnimationSample(
        relativePath=outputJsonPath,
        resolvedPath=outputJsonPath.resolve(),
        axisAngles=flatAngles.astype(np.float32),
        fps=fps,
        extras=extras,
    )


def buildExtras(
    inferenceConfig: GenerationInferenceConfig,
    modelSettings: GenerationModelSettings,
) -> dict[str, object]:
    """
    Build metadata extras for generated animation payloads.

    Parameters
    ----------
    inferenceConfig : GenerationInferenceConfig
        Prompt and sampling configuration.
    modelSettings : GenerationModelSettings
        Model settings used for generation.

    Returns
    -------
    dict[str, object]
        Extras dictionary for animation payloads.
    """
    extras = {
        EXTRA_PROMPT_KEY: inferenceConfig.prompt,
        EXTRA_CHECKPOINT_KEY: inferenceConfig.checkpoint.as_posix(),
        EXTRA_MODEL_NAME_KEY: modelSettings.modelName,
        EXTRA_DDIM_STEPS_KEY: inferenceConfig.ddimSteps,
    }
    if inferenceConfig.tag is not None:
        extras[EXTRA_TAG_KEY] = inferenceConfig.tag
    return extras


def resolveFps(
    builderConfig: DatasetBuilderConfig,
    fps: Optional[int],
) -> int:
    """
    Resolve FPS from CLI value or dataset config fallback.

    Parameters
    ----------
    builderConfig : DatasetBuilderConfig
        Builder config used for fallback FPS.
    fps : Optional[int]
        CLI-provided FPS value.

    Returns
    -------
    int
        Frames per second to use for output.
    """
    if fps is not None:
        return fps
    return builderConfig.processing.fallbackFps


def writeJsonPayload(
    payload: dict[str, object],
    outputJsonPath: Path,
) -> None:
    """
    Write the animation payload to disk.

    Parameters
    ----------
    payload : dict[str, object]
        JSON-serializable animation payload.
    outputJsonPath : Path
        Destination path for the JSON file.
    """
    outputJsonPath.parent.mkdir(parents=True, exist_ok=True)
    outputJsonPath.write_text(
        json.dumps(payload, indent=JSON_INDENT),
        encoding="utf-8",
    )


def validateGenerationPaths(
    inferenceConfig: GenerationInferenceConfig,
    datasetConfigPath: Path,
    modelSettings: GenerationModelSettings,
) -> GenerationModelSettings:
    """
    Validate required paths for generation and update model settings.

    Parameters
    ----------
    inferenceConfig : GenerationInferenceConfig
        Generation settings including checkpoint path.
    datasetConfigPath : Path
        Dataset builder configuration path.
    modelSettings : GenerationModelSettings
        Model settings containing optional checkpoint paths.

    Returns
    -------
    GenerationModelSettings
        Model settings with validated checkpoint paths.
    """
    requireExistingPath(inferenceConfig.checkpoint, "checkpoint")
    requireExistingPath(datasetConfigPath, "dataset config")
    validatedClip = validateOptionalPath(
        modelSettings.clipCheckpoint,
        "CLIP checkpoint",
    )
    return GenerationModelSettings(
        modelName=modelSettings.modelName,
        clipCheckpoint=validatedClip,
        networkConfigPath=modelSettings.networkConfigPath,
        profile=modelSettings.profile,
    )


def prepareOutputContext(
    datasetConfigPath: Path,
    inferenceConfig: GenerationInferenceConfig,
    modelSettings: GenerationModelSettings,
    outputOptions: GenerationOutputOptions,
) -> tuple[AnimationRebuilder, int, dict[str, object]]:
    """
    Prepare the output context for JSON and Collada export.

    Parameters
    ----------
    datasetConfigPath : Path
        Dataset builder configuration path.
    inferenceConfig : GenerationInferenceConfig
        Prompt and sampling configuration.
    modelSettings : GenerationModelSettings
        Model settings used for generation.
    outputOptions : GenerationOutputOptions
        Output paths and export options.

    Returns
    -------
    tuple[AnimationRebuilder, int, dict[str, object]]
        Rebuilder, resolved FPS, and extras dictionary.
    """
    builderConfig = loadBuilderConfig(datasetConfigPath)
    resolvedFps = resolveFps(builderConfig, outputOptions.fps)
    rebuilder = AnimationRebuilder(builderConfig)
    extras = buildExtras(inferenceConfig, modelSettings)
    return rebuilder, resolvedFps, extras


def prepareGenerationState(
    inferenceConfig: GenerationInferenceConfig,
    modelSettings: GenerationModelSettings,
) -> tuple[MotionGenerator, torch.device, list[str]]:
    """
    Prepare model, device, and bone order for generation.

    Parameters
    ----------
    inferenceConfig : GenerationInferenceConfig
        Sampling configuration and device request.
    modelSettings : GenerationModelSettings
        Model settings for CLIP and network configuration.

    Returns
    -------
    tuple[MotionGenerator, torch.device, list[str]]
        Model, device, and bone order list.
    """
    device = resolveDevice(inferenceConfig.device)
    networkConfig = loadNetworkConfig(
        configPath=modelSettings.networkConfigPath,
        profile=modelSettings.profile,
    )
    boneOrder = selectBoneOrder(networkConfig.generation.numBones)
    model = buildMotionGenerator(
        networkConfig=networkConfig,
        modelName=modelSettings.modelName,
        clipCheckpointPath=modelSettings.clipCheckpoint,
    )
    loadModelCheckpoint(inferenceConfig.checkpoint, model)
    model = model.to(device)
    return model, device, boneOrder


def buildSampleFromPrompt(
    model: MotionGenerator,
    device: torch.device,
    boneOrder: list[str],
    inferenceConfig: GenerationInferenceConfig,
    fps: int,
    outputJsonPath: Path,
    extras: dict[str, object],
) -> AnimationSample:
    """
    Generate motion from a prompt and build an AnimationSample.

    Parameters
    ----------
    model : MotionGenerator
        Initialized generation model.
    device : torch.device
        Device for inference.
    boneOrder : list[str]
        Bone order matching the model output.
    inferenceConfig : GenerationInferenceConfig
        Prompt and sampling configuration.
    fps : int
        Frames per second for the output.
    outputJsonPath : Path
        Output JSON path used as the sample source.
    extras : dict[str, object]
        Additional metadata to attach to the sample.

    Returns
    -------
    AnimationSample
        Generated animation sample.
    """
    motionQuat = generateMotionQuat(model, inferenceConfig, device)
    axisAngles = convertQuaternionToAxisAngles(motionQuat)
    axisAnglesFull = mapAxisAnglesToSmpl24(axisAngles, boneOrder)
    return buildAnimationSample(
        axisAnglesFull=axisAnglesFull,
        fps=fps,
        outputJsonPath=outputJsonPath,
        extras=extras,
    )


def exportAnimationOutputs(
    sample: AnimationSample,
    rebuilder: AnimationRebuilder,
    outputJsonPath: Path,
    outputDaePath: Path,
    zeroRootTranslation: bool,
    anchorRootTranslation: bool,
) -> None:
    """
    Export JSON and Collada files for a generated animation.

    Parameters
    ----------
    sample : AnimationSample
        Generated animation data.
    rebuilder : AnimationRebuilder
        Rebuilder instance for JSON and Collada export.
    outputJsonPath : Path
        Destination JSON path.
    outputDaePath : Path
        Destination Collada path.
    zeroRootTranslation : bool
        Zero root translation when exporting Collada.
    anchorRootTranslation : bool
        Anchor root translation when exporting Collada.
    """
    payload = rebuilder.buildPayload(sample)
    writeJsonPayload(payload, outputJsonPath)
    outputDaePath.parent.mkdir(parents=True, exist_ok=True)
    rebuilder.exportCollada(
        sample,
        outputDaePath,
        zeroRootTranslation=zeroRootTranslation,
        anchorRootTranslation=anchorRootTranslation,
    )


def generateAndExport(
    inferenceConfig: GenerationInferenceConfig,
    modelSettings: GenerationModelSettings,
    outputOptions: GenerationOutputOptions,
    rebuilder: AnimationRebuilder,
    resolvedFps: int,
    extras: dict[str, object],
) -> None:
    """
    Generate a sample and export JSON/Collada outputs.

    Parameters
    ----------
    inferenceConfig : GenerationInferenceConfig
        Prompt and sampling configuration.
    modelSettings : GenerationModelSettings
        Model settings for generation.
    outputOptions : GenerationOutputOptions
        Output paths and export options.
    rebuilder : AnimationRebuilder
        Rebuilder instance for JSON and Collada export.
    resolvedFps : int
        Resolved frames per second value.
    extras : dict[str, object]
        Extras to include in the animation payload.
    """
    model, device, boneOrder = prepareGenerationState(
        inferenceConfig, modelSettings
    )
    sample = buildSampleFromPrompt(
        model, device, boneOrder, inferenceConfig, resolvedFps,
        outputOptions.jsonPath, extras
    )
    exportAnimationOutputs(
        sample, rebuilder, outputOptions.jsonPath,
        outputOptions.daePath,
        outputOptions.zeroRootTranslation,
        outputOptions.anchorRootTranslation,
    )


def createAnimationFromCheckpoint(
    inferenceConfig: GenerationInferenceConfig,
    modelSettings: GenerationModelSettings,
    outputOptions: GenerationOutputOptions,
    datasetConfigPath: Path,
) -> None:
    """
    Run the full generation pipeline from checkpoint to outputs.

    Parameters
    ----------
    inferenceConfig : GenerationInferenceConfig
        Prompt and sampling configuration.
    modelSettings : GenerationModelSettings
        Model settings for generation.
    outputOptions : GenerationOutputOptions
        Output paths and export options.
    datasetConfigPath : Path
        Dataset builder configuration path.
    """
    modelSettings = validateGenerationPaths(
        inferenceConfig, datasetConfigPath, modelSettings
    )
    rebuilder, resolvedFps, extras = prepareOutputContext(
        datasetConfigPath, inferenceConfig, modelSettings, outputOptions
    )
    generateAndExport(
        inferenceConfig, modelSettings, outputOptions, rebuilder,
        resolvedFps, extras
    )
