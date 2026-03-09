"""Generate motion from a checkpoint and export animation outputs."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Optional
import json
import logging

import numpy as np
import torch
import yaml

from src.features.dataset_builder.animation_rebuilder import AnimationRebuilder
from src.features.generation.train_generation import loadCheckpoint
from src.shared.config_loader import (
    loadBuilderConfig,
    loadGenerationConfig,
    loadNetworkConfig,
)
from src.shared.constants.clip import DEFAULT_MODEL_NAME
from src.shared.constants.skeletons import SMPL22_BONE_ORDER, SMPL24_BONE_ORDER
from src.shared.dataset_manager import DatasetManager
from src.shared.model.components import buildEnabledComponents
from src.shared.model.generation.motion_generator import MotionGenerator
from src.shared.preprocessed_dataset import PreprocessedLinkDataset
from src.shared.quaternion import Rotation
from src.shared.types import (
    AnimationSample,
    DatasetBuilderConfig,
    GenerationTrainingConfig,
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
LOGGER = logging.getLogger("generation.infer")

EXTRA_PROMPT_KEY = "prompt"
EXTRA_CHECKPOINT_KEY = "checkpoint"
EXTRA_MODEL_NAME_KEY = "modelName"
EXTRA_DDIM_STEPS_KEY = "ddimSteps"
EXTRA_MODE_KEY = "mode"
EXTRA_REFERENCE_DATASET_INDEX_KEY = "referenceDatasetIndex"
EXTRA_REFERENCE_SELECTION_KEY = "referenceSelection"
EXTRA_REFERENCE_SOURCE_FILE_KEY = "referenceSourceFile"
EXTRA_REFERENCE_START_FRAME_KEY = "referenceStartFrame"
EXTRA_REFERENCE_END_FRAME_KEY = "referenceEndFrame"
EXTRA_REFERENCE_FPS_KEY = "referenceFps"
EXTRA_REFERENCE_FRAMES_KEY = "referenceFrames"

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
) -> tuple[Optional[Path], Optional[Path], str, int]:
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
    tuple[Optional[Path], Optional[Path], str, int]
        Clip checkpoint path, network config path, model name, and
        tokenizer max length.
    """
    payload = loadYamlPayload(configPath)
    pathsSection = ensureDict(payload.get(YAML_PATHS_KEY, {}))
    sectionName = profile or YAML_TRAINING_KEY
    trainingSection = ensureDict(payload.get(sectionName, {}))
    modelName = str(
        trainingSection.get(YAML_MODEL_NAME_KEY, DEFAULT_MODEL_NAME)
    )
    maxPromptLength = int(trainingSection.get("max-length", 64))
    clipCheckpoint = resolveOptionalPath(
        configPath,
        optionalString(pathsSection.get(YAML_CLIP_CHECKPOINT_KEY)),
    )
    networkConfig = resolveOptionalPath(
        configPath,
        optionalString(pathsSection.get(YAML_NETWORK_CONFIG_KEY)),
    )
    return (
        clipCheckpoint,
        networkConfig,
        modelName,
        maxPromptLength,
    )


def parsePositiveInt(value: str, label: str) -> int:
    """Parse a strictly positive integer from text."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer.") from error
    if parsed <= 0:
        raise ValueError(f"{label} must be strictly positive.")
    return parsed


def parseOverfitSelection(
    rawValue: Optional[str],
) -> tuple[Optional[tuple[int, int]], Optional[int], Optional[str]]:
    """
    Parse an overfit selector into either a fixed range or a sample count.

    Parameters
    ----------
    rawValue : Optional[str]
        Raw selector from the generation config.

    Returns
    -------
    tuple[Optional[tuple[int, int]], Optional[int], Optional[str]]
        Fixed 1-based inclusive range, sample count, and normalized raw value.
    """
    if rawValue is None:
        return None, None, None
    normalized = str(rawValue).strip()
    if not normalized or normalized.lower() == "null":
        return None, None, None
    if ":" not in normalized:
        return None, parsePositiveInt(normalized, label="overfit-samples"), normalized

    parts = normalized.split(":")
    if len(parts) != 2:
        raise ValueError(
            "overfit-samples range must use the format start:end "
            "(example: 3:6)."
        )
    start = parsePositiveInt(
        parts[0].strip(),
        label="overfit-samples range start",
    )
    end = parsePositiveInt(
        parts[1].strip(),
        label="overfit-samples range end",
    )
    if start > end:
        raise ValueError(
            "overfit-samples range start must be <= end "
            f"(got {normalized!r})."
        )
    return (start, end), None, normalized


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


def asTensor(value: object, label: str) -> torch.Tensor:
    """
    Convert an arbitrary payload value into a float tensor.

    Parameters
    ----------
    value : object
        Value to normalize.
    label : str
        Label used in error messages.

    Returns
    -------
    torch.Tensor
        Float32 tensor stored on CPU.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    try:
        return torch.as_tensor(value, dtype=torch.float32)
    except Exception as error:
        raise TypeError(f"{label} must be tensor-convertible.") from error


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
    maxPromptLength: int,
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
    maxPromptLength : int
        Tokenizer max length used during inference tokenization.

    Returns
    -------
    MotionGenerator
        Initialized generation model.
    """
    clipMotionComponents = (
        buildEnabledComponents(networkConfig.clip.boneData)
        if networkConfig.clip.boneData is not None
        else ()
    )
    generationMotionComponents = buildEnabledComponents(
        networkConfig.generation.boneData
    )
    if networkConfig.clip.boneData is not None and not clipMotionComponents:
        raise ValueError(
            "clip.bone-data is configured but enables no motion features. "
            "Enable at least one component or remove clip.bone-data to keep "
            "the legacy rotation-only CLIP input."
        )
    return MotionGenerator(
        embedDim=networkConfig.embedDim,
        generationEmbedDim=networkConfig.generation.embedDim,
        numHeads=networkConfig.generation.numHeads,
        numLayers=networkConfig.generation.numLayers,
        numBones=networkConfig.generation.numBones,
        diffusionSteps=networkConfig.generation.diffusionSteps,
        numSpatialLayers=networkConfig.generation.numSpatialLayers,
        numSpatioTemporalLayers=networkConfig.generation.numSpatioTemporalLayers,
        modelName=modelName,
        clipCheckpoint=clipCheckpointPath,
        maxPromptLength=maxPromptLength,
        clipMotionNumHeads=networkConfig.clip.motionNumHeads,
        clipMotionNumLayers=networkConfig.clip.motionNumLayers,
        clipMotionComponents=clipMotionComponents,
        generationMotionComponents=generationMotionComponents,
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
    applyPostProcessing: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Run text-conditioned motion generation.

    Parameters
    ----------
    model : MotionGenerator
        Generation model.
    inferenceConfig : GenerationInferenceConfig
        Prompt and sampling configuration.
    device : torch.device
        Device for inference.

    Returns
    -------
    dict[str, torch.Tensor]
        Generated motion sample with quaternion and optional extras.
    """
    return model.generateSample(
        prompt=inferenceConfig.prompt,
        numFrames=inferenceConfig.frames,
        ddimSteps=inferenceConfig.ddimSteps,
        device=device,
        applyPostProcessing=applyPostProcessing,
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
    return extras


def resolveOverfitReplay(
    generationConfigPath: Optional[Path],
    profile: Optional[str],
) -> Optional[
    tuple[
        PreprocessedLinkDataset,
        int,
        str,
        GenerationTrainingConfig,
        str,
    ]
]:
    """
    Resolve the exact dataset sample used by a single-sample overfit profile.

    Parameters
    ----------
    generationConfigPath : Optional[Path]
        Path to the generation training config file.
    profile : Optional[str]
        Requested config profile.

    Returns
    -------
    Optional[tuple[PreprocessedLinkDataset, int, str, GenerationTrainingConfig, str]]
        Dataset instance, selected dataset index, chunk info, loaded training
        config, and normalized selector string.
    """
    if generationConfigPath is None or profile != "overfit":
        return None
    generationConfig = loadGenerationConfig(
        generationConfigPath,
        profile=profile,
    )
    fixedSampleRange, maxSamplesPerEpoch, selectionRaw = parseOverfitSelection(
        generationConfig.training.overfitSamples
    )
    if selectionRaw is None:
        return None
    datasetManager = DatasetManager(
        datasetRoot=generationConfig.paths.datasetRoot,
        batchSize=1,
        validationSplit=0.0,
        modelMemoryBytes=0,
        datasetFolders=generationConfig.paths.datasetFolders,
        fixedSampleRange=fixedSampleRange,
        maxSamplesPerEpoch=maxSamplesPerEpoch,
        includeTokenizedText=False,
    )
    selectedIndices, chunkInfo = datasetManager.getEpochSampleIndices(0)
    if len(selectedIndices) != 1:
        LOGGER.info(
            "Deterministic overfit replay disabled: selector %s resolved to %d samples.",
            selectionRaw,
            len(selectedIndices),
        )
        return None
    return (
        datasetManager.dataset,
        selectedIndices[0],
        chunkInfo,
        generationConfig,
        selectionRaw,
    )


def resolveReferenceSampleFps(
    samplePayload: dict[str, object],
    requestedFps: Optional[int],
) -> int:
    """
    Resolve the FPS used for deterministic reference replay.

    Parameters
    ----------
    samplePayload : dict[str, object]
        Preprocessed sample payload.
    requestedFps : Optional[int]
        CLI FPS override.

    Returns
    -------
    int
        Native FPS from the sample when available, else requested FPS, else 24.
    """
    meta = samplePayload.get("meta")
    if isinstance(meta, dict):
        fpsValue = meta.get("fps")
        if isinstance(fpsValue, (int, float)) and fpsValue > 0:
            sampleFps = int(round(float(fpsValue)))
            if requestedFps is not None and requestedFps != sampleFps:
                LOGGER.info(
                    "Overfit replay forcing native FPS %d instead of requested %d.",
                    sampleFps,
                    requestedFps,
                )
            return sampleFps
    if requestedFps is not None:
        return requestedFps
    return 24


def buildSampleFromReferenceReplay(
    dataset: PreprocessedLinkDataset,
    datasetIndex: int,
    selectionRaw: str,
    chunkInfo: str,
    inferenceConfig: GenerationInferenceConfig,
    outputOptions: GenerationOutputOptions,
    outputJsonPath: Path,
    extras: dict[str, object],
) -> AnimationSample:
    """
    Export the exact overfit training sample instead of sampling diffusion.

    Parameters
    ----------
    dataset : PreprocessedLinkDataset
        Dataset holding the selected reference sample.
    datasetIndex : int
        Raw dataset index to replay.
    selectionRaw : str
        Normalized overfit selector string.
    chunkInfo : str
        Human-readable selection summary.
    inferenceConfig : GenerationInferenceConfig
        CLI inference settings.
    outputOptions : GenerationOutputOptions
        Output export settings.
    outputJsonPath : Path
        Target JSON path.
    extras : dict[str, object]
        Base extras payload.

    Returns
    -------
    AnimationSample
        Animation sample rebuilt from the exact preprocessed reference window.
    """
    linkEntry = dataset.indexEntries[datasetIndex]
    sampleEntry = dataset.sampleIndexEntries[linkEntry.sampleId]
    samplePayload = dataset[datasetIndex]
    motion6d = asTensor(samplePayload.get("motion"), label="motion")
    axisAngles = Rotation(motion6d, kind="rot6d").axis_angle
    axisAnglesNumpy = axisAngles.detach().cpu().numpy().astype(np.float32)
    axisAnglesFull = mapAxisAnglesToSmpl24(
        axisAnglesNumpy,
        SMPL22_BONE_ORDER,
    )
    resolvedFps = resolveReferenceSampleFps(
        samplePayload,
        outputOptions.fps,
    )
    if inferenceConfig.frames != sampleEntry.frames:
        LOGGER.info(
            "Overfit replay forcing native frame count %d instead of requested %d.",
            sampleEntry.frames,
            inferenceConfig.frames,
        )
    LOGGER.info(
        "Deterministic overfit replay active: %s -> dataset index %d (%s).",
        selectionRaw,
        datasetIndex,
        chunkInfo,
    )
    sampleExtras = dict(extras)
    sampleExtras[EXTRA_DDIM_STEPS_KEY] = 0
    sampleExtras[EXTRA_MODE_KEY] = "overfit-reference-replay"
    sampleExtras[EXTRA_REFERENCE_DATASET_INDEX_KEY] = datasetIndex
    sampleExtras[EXTRA_REFERENCE_SELECTION_KEY] = selectionRaw
    sampleExtras[EXTRA_REFERENCE_SOURCE_FILE_KEY] = sampleEntry.sourceFile
    sampleExtras[EXTRA_REFERENCE_START_FRAME_KEY] = sampleEntry.startFrame
    sampleExtras[EXTRA_REFERENCE_END_FRAME_KEY] = sampleEntry.endFrame
    sampleExtras[EXTRA_REFERENCE_FPS_KEY] = resolvedFps
    sampleExtras[EXTRA_REFERENCE_FRAMES_KEY] = sampleEntry.frames
    rootTranslation = samplePayload.get("root_translation")
    if rootTranslation is not None:
        sampleExtras["trans"] = asTensor(
            rootTranslation,
            label="root_translation",
        ).tolist()
    return buildAnimationSample(
        axisAnglesFull=axisAnglesFull,
        fps=resolvedFps,
        outputJsonPath=outputJsonPath,
        extras=sampleExtras,
    )


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
        maxPromptLength=modelSettings.maxPromptLength,
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
) -> tuple[MotionGenerator, torch.device, list[str], GenerationInferenceConfig, bool]:
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
    tuple[MotionGenerator, torch.device, list[str], GenerationInferenceConfig, bool]
        Model, device, bone order list, resolved inference config, and whether
        inference post-processing should run.
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
        maxPromptLength=modelSettings.maxPromptLength,
    )
    loadModelCheckpoint(inferenceConfig.checkpoint, model)
    model = model.to(device)
    resolvedInferenceConfig = inferenceConfig
    applyPostProcessing = True
    if modelSettings.profile == "overfit":
        applyPostProcessing = False
        if inferenceConfig.ddimSteps < model.diffusionSteps:
            resolvedInferenceConfig = replace(
                inferenceConfig,
                ddimSteps=model.diffusionSteps,
            )
        LOGGER.info(
            "Overfit inference enabled: applyPostProcessing=%s, ddimSteps=%d",
            applyPostProcessing,
            resolvedInferenceConfig.ddimSteps,
        )
    return (
        model,
        device,
        boneOrder,
        resolvedInferenceConfig,
        applyPostProcessing,
    )


def buildSampleFromPrompt(
    model: MotionGenerator,
    device: torch.device,
    boneOrder: list[str],
    inferenceConfig: GenerationInferenceConfig,
    fps: int,
    outputJsonPath: Path,
    extras: dict[str, object],
    applyPostProcessing: bool = True,
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
    generatedSample = generateMotionQuat(
        model,
        inferenceConfig,
        device,
        applyPostProcessing=applyPostProcessing,
    )
    motionQuat = generatedSample["motion_quat"]
    sampleExtras = dict(extras)
    rootTranslation = generatedSample.get("root_translation")
    if isinstance(rootTranslation, torch.Tensor):
        sampleExtras["trans"] = (
            rootTranslation.squeeze(0).detach().cpu().tolist()
        )
    axisAngles = convertQuaternionToAxisAngles(motionQuat)
    axisAnglesFull = mapAxisAnglesToSmpl24(axisAngles, boneOrder)
    return buildAnimationSample(
        axisAnglesFull=axisAnglesFull,
        fps=fps,
        outputJsonPath=outputJsonPath,
        extras=sampleExtras,
    )


def exportAnimationOutputs(
    sample: AnimationSample,
    rebuilder: AnimationRebuilder,
    outputJsonPath: Path,
    outputDaePath: Path,
    colladaInterpolation: str,
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
    colladaInterpolation : str
        Collada interpolation mode ("linear" or "step").
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
        interpolation=colladaInterpolation,
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
    generationConfigPath: Optional[Path] = None,
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
    generationConfigPath : Optional[Path]
        Training config path used to resolve deterministic overfit replay.
    """
    exportExtras = dict(extras)
    overfitReplay = resolveOverfitReplay(
        generationConfigPath=generationConfigPath,
        profile=modelSettings.profile,
    )
    if overfitReplay is not None:
        dataset, datasetIndex, chunkInfo, _, selectionRaw = overfitReplay
        sample = buildSampleFromReferenceReplay(
            dataset=dataset,
            datasetIndex=datasetIndex,
            selectionRaw=selectionRaw,
            chunkInfo=chunkInfo,
            inferenceConfig=inferenceConfig,
            outputOptions=outputOptions,
            outputJsonPath=outputOptions.jsonPath,
            extras=exportExtras,
        )
    else:
        (
            model,
            device,
            boneOrder,
            resolvedInferenceConfig,
            applyPostProcessing,
        ) = prepareGenerationState(
            inferenceConfig, modelSettings
        )
        exportExtras[EXTRA_DDIM_STEPS_KEY] = resolvedInferenceConfig.ddimSteps
        sample = buildSampleFromPrompt(
            model,
            device,
            boneOrder,
            resolvedInferenceConfig,
            resolvedFps,
            outputOptions.jsonPath,
            exportExtras,
            applyPostProcessing=applyPostProcessing,
        )
    exportAnimationOutputs(
        sample, rebuilder, outputOptions.jsonPath,
        outputOptions.daePath,
        outputOptions.colladaInterpolation,
        outputOptions.zeroRootTranslation,
        outputOptions.anchorRootTranslation,
    )


def createAnimationFromCheckpoint(
    inferenceConfig: GenerationInferenceConfig,
    modelSettings: GenerationModelSettings,
    outputOptions: GenerationOutputOptions,
    datasetConfigPath: Path,
    generationConfigPath: Optional[Path] = None,
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
    generationConfigPath : Optional[Path]
        Generation training config used to resolve overfit replay.
    """
    modelSettings = validateGenerationPaths(
        inferenceConfig, datasetConfigPath, modelSettings
    )
    rebuilder, resolvedFps, extras = prepareOutputContext(
        datasetConfigPath, inferenceConfig, modelSettings, outputOptions
    )
    generateAndExport(
        inferenceConfig, modelSettings, outputOptions, rebuilder,
        resolvedFps, extras, generationConfigPath=generationConfigPath
    )
