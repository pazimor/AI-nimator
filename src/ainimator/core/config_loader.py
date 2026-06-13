"""Unified configuration loader for all training workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from ainimator.core.constants.clip import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_MAX_LENGTH,
)
from ainimator.core.types import (
    BoneDataConfig,
    DatasetBuilderConfig,
    DatasetBuilderPaths,
    DatasetBuilderProcessing,
    PreprocessDatasetConfig,
    PreprocessDatasetPaths,
    PreprocessDatasetProcessing,
    GenerationTrainingConfig,
    GenerationTrainingHyperparameters,
    GenerationTrainingPaths,
)
from ainimator.core.types.network import (
    ClipNetworkConfig,
    GenerationNetworkConfig,
    LearningRateHyperparameters,
    NetworkConfig,
)

# Generation default values
GENERATION_DEFAULT_BATCH_SIZE = 8
GENERATION_DEFAULT_LEARNING_RATE = 0.0001
GENERATION_DEFAULT_EPOCHS = 500
GENERATION_DEFAULT_VALIDATION_SPLIT = 0.1
GENERATION_DEFAULT_EARLY_STOPPING_PATIENCE = 5
GENERATION_DEFAULT_MAX_LENGTH = 64
GENERATION_DEFAULT_MODEL_NAME = "xlm-roberta-base"
GENERATION_DEFAULT_XYZ_WEIGHT = 0.1
GENERATION_DEFAULT_XYZ_SCHEDULE = "none"
GENERATION_DEFAULT_VEL_XYZ_WEIGHT = 0.01
GENERATION_DEFAULT_VEL_XYZ_SCHEDULE = "none"
GENERATION_DEFAULT_DIFFUSION_WEIGHT = 1.0
GENERATION_DEFAULT_ACCELERATION_WEIGHT = 0.0
GENERATION_DEFAULT_CLIP_GUIDANCE_WEIGHT = 0.0
PREPROCESS_DEFAULT_SAMPLE_SHARD_SIZE = 256
PREPROCESS_DEFAULT_TEXT_SHARD_SIZE = 2048
PREPROCESS_DEFAULT_TEXT_BATCH_SIZE = 64

LOGGER = logging.getLogger("shared.config.network")

DEFAULT_NETWORK_CONFIG_PATH = Path("src/configs/network.yaml")


def loadNetworkConfig(
    configPath: Optional[Path] = None,
    profile: Optional[str] = None,
) -> NetworkConfig:
    """
    Load network architecture configuration from YAML file.
    
    Parameters
    ----------
    configPath : Optional[Path]
        Path to network.yaml. If None, uses default path.
    profile : Optional[str]
        Profile name to load (e.g., "spark"). If None, uses "default".
        
    Returns
    -------
    NetworkConfig
        Loaded network configuration.
    """
    resolved = (configPath or DEFAULT_NETWORK_CONFIG_PATH)
    resolved = resolved.expanduser().resolve()
    
    if not resolved.exists():
        LOGGER.warning(
            "Network config not found at %s, using defaults",
            resolved,
        )
        return _defaultNetworkConfig()
    
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    
    # Select profile
    profileName = profile or "default"
    section = payload.get(profileName)
    
    if section is None:
        LOGGER.warning(
            "Profile '%s' not found in network config, trying 'default'",
            profileName,
        )
        section = payload.get("default", {})
    
    return _parseNetworkConfig(section)


def _parseNetworkConfig(section: Dict[str, Any]) -> NetworkConfig:
    """Parse network configuration from YAML section."""
    clipSection = section.get("clip", {})
    generationSection = section.get("generation", {})
    clipBoneDataSection = clipSection.get("bone-data")
    boneDataSection = generationSection.get("bone-data", {})
    clipEmbedDim = int(section.get("embed-dim", 128))
    generationEmbedDim = int(
        generationSection.get("embed-dim", clipEmbedDim)
    )
    generationNumHeads = int(generationSection.get("num-heads", 4))
    if generationEmbedDim % generationNumHeads != 0:
        raise ValueError(
            "generation.embed-dim must be divisible by generation.num-heads "
            f"(embed-dim={generationEmbedDim}, num-heads={generationNumHeads})."
        )

    return NetworkConfig(
        embedDim=clipEmbedDim,
        clip=ClipNetworkConfig(
            motionNumHeads=int(clipSection.get("motion-num-heads", 4)),
            motionNumLayers=int(clipSection.get("motion-num-layers", 2)),
            boneData=(
                _parseBoneDataConfig(clipBoneDataSection)
                if isinstance(clipBoneDataSection, dict)
                else None
            ),
        ),
        generation=GenerationNetworkConfig(
            embedDim=generationEmbedDim,
            numHeads=generationNumHeads,
            numLayers=int(generationSection.get("num-layers", 6)),
            numBones=int(generationSection.get("num-bones", 22)),
            diffusionSteps=int(generationSection.get("diffusion-steps", 1000)),
            numSpatialLayers=int(
                generationSection.get("num-spatial-layers", 1)
            ),
            numSpatioTemporalLayers=int(
                generationSection.get("num-spatio-temporal-layers", 1)
            ),
            boneData=_parseBoneDataConfig(boneDataSection),
        ),
    )


def _parseBoneDataConfig(section: Dict[str, Any]) -> BoneDataConfig:
    """Parse optional motion feature toggles from network.yaml."""
    return BoneDataConfig(
        rotation6d=_bool(section, "rotation6d", True),
        footContact=_bool(section, "foot-contact", False),
        handContact=_bool(section, "hand-contact", False),
        rootTranslation=_bool(section, "root-translation", False),
        rootVelocity=_bool(section, "root-velocity", False),
        rootYaw=_bool(section, "root-yaw", False),
        rootYawVelocity=_bool(section, "root-yaw-velocity", False),
        jointXyz=_bool(section, "joint-xyz", False),
        jointVelocity=_bool(section, "joint-velocity", False),
        endEffectorVelocity=_bool(
            section,
            "end-effector-velocity",
            False,
        ),
        pelvisHeight=_bool(section, "pelvis-height", False),
    )



def _defaultNetworkConfig() -> NetworkConfig:
    """Return default network configuration."""
    return NetworkConfig(
        embedDim=128,
        clip=ClipNetworkConfig(),
        generation=GenerationNetworkConfig(embedDim=128),
    )


def loadLearningRateConfig(
    section: Dict[str, Any],
    defaultInitialLR: float = 0.001,
) -> LearningRateHyperparameters:
    """
    Load learning rate configuration from a YAML training section.
    
    Parameters
    ----------
    section : Dict[str, Any]
        Training section from YAML config.
    defaultInitialLR : float
        Default initial learning rate if not specified.
        
    Returns
    -------
    LearningRateHyperparameters
        Learning rate configuration.
    """
    return LearningRateHyperparameters(
        initialLR=float(section.get("learning-rate", defaultInitialLR)),
        minLR=float(section.get("lr-min", 1e-7)),
        warmupEpochs=int(section.get("lr-warmup-epochs", 0)),
        scheduleType=str(section.get("lr-schedule", "cosine")),
        decayEpochs=_optionalInt(section, "lr-decay-epochs"),
    )


def _optionalInt(section: Dict[str, Any], key: str) -> Optional[int]:
    """Get optional integer from section."""
    value = section.get(key)
    if value is None or value == "null":
        return None
    return int(value)


def _optionalString(section: Dict[str, Any], key: str) -> Optional[str]:
    """Get optional string from section."""
    value = section.get(key)
    if value is None or value == "null":
        return None
    return str(value).strip()


# ==============================================================================
# GENERATION TRAINING CONFIG
# ==============================================================================


def loadGenerationConfig(
    configPath: Path,
    profile: Optional[str] = None,
) -> GenerationTrainingConfig:
    """
    Parse the generation training configuration YAML file.

    Parameters
    ----------
    configPath : Path
        Filesystem path to the YAML file.
    profile : Optional[str]
        Optional profile name to load (e.g., "spark"). If None, uses "training".

    Returns
    -------
    GenerationTrainingConfig
        Fully-populated configuration dataclass.
    """
    resolved = configPath.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Generation training config missing: {resolved}"
        )

    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    pathsSection = payload.get("paths", {})
    
    # Select training section based on profile
    sectionName = profile if profile else "training"
    trainingSection = payload.get(sectionName, {})
    if not trainingSection:
        raise ValueError(f"Profile '{sectionName}' not found in config file")

    paths = _loadGenerationPaths(resolved, pathsSection)
    
    # Get network config path (must exist if provided)
    networkConfigPath = _optionalExistingPath(
        resolved,
        pathsSection.get("network-config"),
    )
    xyzWeight = _float(
        trainingSection,
        "xyz-weight",
        GENERATION_DEFAULT_XYZ_WEIGHT,
    )
    xyzSchedule = str(
        trainingSection.get(
            "xyz-weight-schedule",
            GENERATION_DEFAULT_XYZ_SCHEDULE,
        ),
    )
    rootTranslationWeight = _float(
        trainingSection,
        "root-translation-weight",
        1.0,
    )
    jointXyzWeight = _float(
        trainingSection,
        "joint-xyz-weight",
        1.0,
    )
    pelvisHeightWeight = _float(
        trainingSection,
        "pelvis-height-weight",
        1.0,
    )
    condMaskProb = _float(
        trainingSection,
        "cond-mask-prob",
        0.1,
    )
    weightDecay = _float(
        trainingSection,
        "weight-decay",
        0.0,
    )
    emaEnabled = _bool(
        trainingSection,
        "ema-enabled",
        False,
    )
    emaDecay = _float(
        trainingSection,
        "ema-decay",
        0.9999,
    )
    emaWarmup = _bool(
        trainingSection,
        "ema-warmup",
        True,
    )
    velXyzWeight = _float(
        trainingSection,
        "vel-xyz-weight",
        GENERATION_DEFAULT_VEL_XYZ_WEIGHT,
    )
    velXyzSchedule = str(
        trainingSection.get(
            "vel-xyz-weight-schedule",
            GENERATION_DEFAULT_VEL_XYZ_SCHEDULE,
        ),
    )
    diffusionWeight = _float(
        trainingSection,
        "diffusion-weight",
        GENERATION_DEFAULT_DIFFUSION_WEIGHT,
    )
    accelerationWeight = _float(
        trainingSection,
        "acceleration-weight",
        GENERATION_DEFAULT_ACCELERATION_WEIGHT,
    )
    clipGuidanceWeight = _float(
        trainingSection,
        "clip-guidance-weight",
        GENERATION_DEFAULT_CLIP_GUIDANCE_WEIGHT,
    )
    # Foot-skating penalty is disabled by default: when active it evaluates
    # FK on every noisy rot6d prediction and multiplies the resulting foot
    # velocity by the ground-truth contact mask.  At high diffusion timesteps
    # this is pure FK-of-noise, producing wild gradients that push the
    # denoiser toward a degenerate mean pose.  Enable it (e.g. 1.0) only
    # once the base diffusion loss has settled.
    footSkatingWeight = _float(
        trainingSection,
        "foot-skating-weight",
        0.0,
    )
    
    hyperparameters = GenerationTrainingHyperparameters(
        batchSize=_int(
            trainingSection,
            "batch-size",
            GENERATION_DEFAULT_BATCH_SIZE,
        ),
        epochs=_int(trainingSection, "epochs", GENERATION_DEFAULT_EPOCHS),
        device=str(trainingSection.get("device", "auto")),
        validationSplit=_float(
            trainingSection,
            "validation-split",
            GENERATION_DEFAULT_VALIDATION_SPLIT,
        ),
        earlyStoppingPatience=_int(
            trainingSection,
            "early-stopping-patience",
            GENERATION_DEFAULT_EARLY_STOPPING_PATIENCE,
        ),
        maxPromptLength=_int(
            trainingSection,
            "max-length",
            GENERATION_DEFAULT_MAX_LENGTH,
        ),
        modelName=str(
            trainingSection.get("model-name", GENERATION_DEFAULT_MODEL_NAME)
        ),
        resumeCheckpoint=_optionalExistingPath(
            resolved,
            trainingSection.get("resume-checkpoint"),
            strict=False,
            label="resume-checkpoint",
        ),
        gradientAccumulation=_int(trainingSection, "gradient-accumulation", 1),
        maxSamplesPerEpoch=_optionalInt(
            trainingSection,
            "max-samples-per-epoch",
        ),
        fixedTrainChunk=_bool(
            trainingSection,
            "fixed-train-chunk",
            False,
        ),
        overfitSamples=_optionalString(
            trainingSection,
            "overfit-samples",
        ),
        clearMpsCache=_bool(
            trainingSection,
            "clear-mps-cache",
            True,
        ),
        deterministicCorruption=_bool(
            trainingSection,
            "deterministic-corruption",
            False,
        ),
        disableDropout=_bool(
            trainingSection,
            "disable-dropout",
            False,
        ),
        # Learning Rate
        learningRate=_float(
            trainingSection,
            "learning-rate",
            GENERATION_DEFAULT_LEARNING_RATE,
        ),
        lrSchedule=str(trainingSection.get("lr-schedule", "constant")),
        lrMin=_float(trainingSection, "lr-min", 1e-7),
        lrWarmupEpochs=_int(trainingSection, "lr-warmup-epochs", 0),
        lrDecayEpochs=_optionalInt(trainingSection, "lr-decay-epochs"),
        xyzWeight=xyzWeight,
        xyzWeightSchedule=xyzSchedule,
        rootTranslationWeight=rootTranslationWeight,
        jointXyzWeight=jointXyzWeight,
        pelvisHeightWeight=pelvisHeightWeight,
        condMaskProb=condMaskProb,
        weightDecay=weightDecay,
        emaEnabled=emaEnabled,
        emaDecay=emaDecay,
        emaWarmup=emaWarmup,
        velXyzWeight=velXyzWeight,
        velXyzWeightSchedule=velXyzSchedule,
        diffusionWeight=diffusionWeight,
        accelerationWeight=accelerationWeight,
        clipGuidanceWeight=clipGuidanceWeight,
        footSkatingWeight=footSkatingWeight,
        minSnrGamma=_float(trainingSection, "min-snr-gamma", 5.0),
        mirrorProbability=_float(trainingSection, "mirror-probability", 0.0),
    )

    return GenerationTrainingConfig(
        paths=paths,
        training=hyperparameters,
        networkConfigPath=networkConfigPath,
    )


def _loadGenerationPaths(
    configPath: Path,
    section: Dict[str, Any],
) -> GenerationTrainingPaths:
    """Load paths section from config."""
    datasetRoot = _resolveExistingPath(
        configPath,
        _require(section, "dataset-root"),
        "dataset-root",
    )
    clipCheckpoint = _resolveExistingPath(
        configPath,
        _require(section, "clip-checkpoint"),
        "clip-checkpoint",
    )
    checkpointDir = _optionalPath(configPath, section.get("checkpoint-dir"))
    if checkpointDir is None:
        checkpointDir = Path("output/generation_checkpoints")
        checkpointDir.mkdir(parents=True, exist_ok=True)
    validationIndices = _optionalResolvedPath(
        configPath,
        section.get("validation-indices"),
    )
    if validationIndices is None:
        validationIndices = checkpointDir / "validation_indices.json"
    datasetFolders = _optionalStringList(section.get("dataset-folders"))

    return GenerationTrainingPaths(
        datasetRoot=datasetRoot,
        clipCheckpoint=clipCheckpoint,
        checkpointDir=checkpointDir,
        validationIndices=validationIndices,
        datasetFolders=datasetFolders,
    )


# ==============================================================================
# DATASET BUILDER CONFIG
# ==============================================================================


def loadBuilderConfig(configPath: Path) -> DatasetBuilderConfig:
    """
    Load the builder configuration from a YAML file.

    Parameters
    ----------
    configPath : Path
        Configuration file to parse.

    Returns
    -------
    DatasetBuilderConfig
        Fully-populated configuration dataclass.
    """

    if not configPath.exists():
        raise FileNotFoundError(f"Missing configuration file: {configPath}")
    payload = yaml.safe_load(configPath.read_text(encoding="utf-8")) or {}
    
    pathsSection = payload.get("paths", {})
    processingSection = payload.get("processing", {})
    
    animationRootRaw = _checkValue(pathsSection, "amass-root")
    indexCsvRaw = _checkValue(pathsSection, "humanml3d-mapping")
    promptRootRaw = _checkValue(pathsSection, "converted-root")
    outputRootRaw = _checkValue(pathsSection, "output-root")
    convertedRootRaw = _checkValue(pathsSection, "converted-root")
    
    animationRoot = _resolvePath(configPath, animationRootRaw)
    
    indexCsv = _resolvePath(configPath, indexCsvRaw)
    promptRoot = (
        _resolvePath(configPath, promptRootRaw)
        if promptRootRaw else indexCsv.parent
    )
    convertedRoot = (
        _resolvePath(configPath, convertedRootRaw)
        if convertedRootRaw else None
    )
    promptSourcesRaw = pathsSection.get("prompt-sources", [])
    promptSources = _resolvePath(configPath, promptSourcesRaw)

    outputRoot = _resolvePath(configPath, outputRootRaw or "output")

    paths = DatasetBuilderPaths(
        indexCsv=indexCsv,
        animationRoot=animationRoot,
        promptRoot=promptRoot,
        promptSources=promptSources,
        outputRoot=outputRoot,
        convertedRoot=convertedRoot,
    )

    animationExtensionRaw = _checkValue(
        processingSection,
        "animation-extension",
    )
    promptExtensionRaw = _checkValue(processingSection, "prompt-text-extension")
    fallbackFpsRaw = _checkValue(processingSection, "fallback-fps")

    processing = DatasetBuilderProcessing(
        animationExtension=str(animationExtensionRaw or ".npz"),
        promptTextExtension=str(promptExtensionRaw or ".txt"),
        fallbackFps=int(fallbackFpsRaw) if fallbackFpsRaw is not None else 60,
        includeCustomPrompts=_bool(
            processingSection,
            "include-custom-prompts",
            True,
        ),
    )
    return DatasetBuilderConfig(paths=paths, processing=processing)


# ==============================================================================
# PREPROCESS DATASET CONFIG
# ==============================================================================


def loadPreprocessConfig(configPath: Path) -> PreprocessDatasetConfig:
    """
    Load the dataset preprocessing configuration from a YAML file.

    Parameters
    ----------
    configPath : Path
        Configuration file to parse.

    Returns
    -------
    PreprocessDatasetConfig
        Parsed preprocessing configuration.
    """
    if not configPath.exists():
        raise FileNotFoundError(f"Missing configuration file: {configPath}")
    payload = yaml.safe_load(configPath.read_text(encoding="utf-8")) or {}
    pathsSection = payload.get("paths", {})
    processingSection = payload.get("processing", {})

    inputRoot = _resolvePath(configPath, _require(pathsSection, "input-root"))
    outputRoot = _resolvePath(configPath, _require(pathsSection, "output-root"))
    outputRoot.mkdir(parents=True, exist_ok=True)
    includeFolders = _optionalStringList(pathsSection.get("include-folders"))
    networkConfigPath = _optionalExistingPath(
        configPath,
        pathsSection.get("network-config"),
        strict=False,
        label="network-config",
    )

    processing = PreprocessDatasetProcessing(
        modelName=str(
            processingSection.get("model-name", DEFAULT_MODEL_NAME),
        ),
        maxPromptLength=_int(
            processingSection,
            "max-length",
            DEFAULT_PROMPT_MAX_LENGTH,
        ),
        sampleShardSize=_int(
            processingSection,
            "sample-shard-size",
            _int(
                processingSection,
                "shard-size",
                PREPROCESS_DEFAULT_SAMPLE_SHARD_SIZE,
            ),
        ),
        textShardSize=_int(
            processingSection,
            "text-shard-size",
            PREPROCESS_DEFAULT_TEXT_SHARD_SIZE,
        ),
        textBatchSize=_int(
            processingSection,
            "text-batch-size",
            PREPROCESS_DEFAULT_TEXT_BATCH_SIZE,
        ),
        splitFrames=_optionalInt(processingSection, "split-frames"),
        downsampleTargetFrames=_optionalInt(
            processingSection,
            "downsample-target-frames",
        ),
        maxSegmentFrames=_optionalInt(
            processingSection,
            "max-segment-frames",
        ),
    )
    _validatePreprocessSettings(processing)
    return PreprocessDatasetConfig(
        paths=PreprocessDatasetPaths(
            inputRoot=inputRoot,
            outputRoot=outputRoot,
            includeFolders=includeFolders,
            networkConfigPath=networkConfigPath,
        ),
        processing=processing,
    )


def _validatePreprocessSettings(
    processing: PreprocessDatasetProcessing,
) -> None:
    """
    Validate preprocessing settings for conflicting options.

    Parameters
    ----------
    processing : PreprocessDatasetProcessing
        Processing settings to validate.
    """
    if processing.splitFrames and processing.downsampleTargetFrames:
        raise ValueError(
            "Only one of split-frames or downsample-target-frames may be set."
        )
    if processing.sampleShardSize <= 0:
        raise ValueError("sample-shard-size must be strictly positive.")
    if processing.textShardSize <= 0:
        raise ValueError("text-shard-size must be strictly positive.")
    if processing.textBatchSize <= 0:
        raise ValueError("text-batch-size must be strictly positive.")


def _resolvePath(configPath: Path, rawValue: str) -> Path:
    if not rawValue:
        return configPath.parent
    candidate = Path(rawValue)
    if candidate.is_absolute():
        return candidate
    return (configPath.parent / candidate).resolve()


def _checkValue(section: Dict[str, Any], key: str) -> Optional[str]:
    if key in section and section[key] not in (None, ""):
        return str(section[key])
    else:
        raise ValueError(f"Configuration is missing: {key}")


# ==============================================================================
# SHARED HELPER FUNCTIONS
# ==============================================================================


def _require(section: Dict[str, Any], key: str) -> str:
    """Require a key to be present."""
    value = section.get(key)
    if value in (None, ""):
        raise ValueError(f"Training config missing required field: {key}")
    return str(value)


def _int(section: Dict[str, Any], key: str, default: int) -> int:
    """Get an integer value with default."""
    if key not in section:
        return int(default)
    return int(section[key])


def _float(section: Dict[str, Any], key: str, default: float) -> float:
    """Get a float value with default."""
    if key not in section:
        return float(default)
    return float(section[key])


def _bool(section: Dict[str, Any], key: str, default: bool) -> bool:
    """Get a boolean value with default."""
    if key not in section:
        return bool(default)
    value = section[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _optionalStringList(rawValue: Any) -> Optional[List[str]]:
    """Normalize a raw config value into a list of non-empty strings."""
    if rawValue in (None, ""):
        return None
    if isinstance(rawValue, str):
        values = [item.strip() for item in rawValue.split(",")]
    elif isinstance(rawValue, list):
        values = [str(item).strip() for item in rawValue]
    else:
        values = [str(rawValue).strip()]
    normalized = [item for item in values if item]
    if not normalized:
        return None
    return normalized


def _optionalPath(configPath: Path, rawValue: Optional[str]) -> Optional[Path]:
    """
    Resolve an optional path, creating the directory if it does not exist.

    Parameters
    ----------
    configPath : Path
        Path to the configuration file used as an anchor.
    rawValue : Optional[str]
        User-provided path value from the YAML file.

    Returns
    -------
    Optional[Path]
        Resolved path or None if rawValue is empty.
    """
    if rawValue in (None, ""):
        return None
    candidate = Path(rawValue).expanduser()
    if not candidate.is_absolute():
        candidate = (configPath.parent / candidate).resolve()
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _optionalResolvedPath(
    configPath: Path,
    rawValue: Optional[str],
) -> Optional[Path]:
    """
    Resolve an optional path without creating directories.
    """
    if rawValue in (None, ""):
        return None
    candidate = Path(rawValue).expanduser()
    if not candidate.is_absolute():
        candidate = (configPath.parent / candidate).resolve()
    return candidate


def _optionalExistingPath(
    configPath: Path,
    rawValue: Optional[str],
    strict: bool = True,
    label: str = "path",
) -> Optional[Path]:
    """
    Resolve an optional path that must exist if provided.

    Parameters
    ----------
    configPath : Path
        Path to the configuration file used as an anchor.
    rawValue : Optional[str]
        User-provided path value from the YAML file.

    Returns
    -------
    Optional[Path]
        Resolved path or None if rawValue is empty.

    Raises
    ------
    FileNotFoundError
        Raised when the specified path does not exist and strict=True.
    """
    if rawValue in (None, ""):
        return None
    candidates = list(_candidatePaths(configPath.parent, str(rawValue)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not strict:
        attempted = ", ".join(str(c) for c in candidates)
        LOGGER.warning(
            "Configured %s does not exist; ignoring. Tried: %s",
            label,
            attempted,
        )
        return None
    attempted = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Configured {label} does not exist. Tried: {attempted}",
    )


def _resolveExistingPath(
    configPath: Path,
    rawValue: str,
    label: str,
) -> Path:
    """
    Resolve a path string against likely roots and ensure it exists.

    Parameters
    ----------
    configPath : Path
        Path to the configuration file used as an anchor.
    rawValue : str
        User-provided path value from the YAML file.
    label : str
        Field name used for error messages.

    Returns
    -------
    Path
        First existing resolved path.

    Raises
    ------
    FileNotFoundError
        Raised when no candidate path could be resolved.
    """
    candidates = _candidatePaths(configPath.parent, rawValue)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Configured {label} does not exist. Tried: {attempted}",
    )


def _candidatePaths(baseDir: Path, rawValue: str) -> Iterable[Path]:
    candidate = Path(rawValue).expanduser()
    if candidate.is_absolute():
        return (candidate,)
    configRelative = (baseDir / candidate).resolve()
    cwdRelative = (Path.cwd() / candidate).resolve()
    if configRelative == cwdRelative:
        return (configRelative,)
    return (configRelative, cwdRelative)
