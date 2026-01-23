"""Unified configuration loader for all training workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from src.shared.constants.clip import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_MAX_LENGTH,
    DEFAULT_VALIDATION_SPLIT,
)
from src.shared.constants.rotation import (
    ROTATION_CHANNELS_ROT6D,
    normalizeRotationRepr,
    resolveRotationChannels,
)
from src.shared.types import (
    ClipTrainingConfig,
    ClipTrainingHyperparameters,
    ClipTrainingPaths,
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
from src.shared.types.network import (
    ClipNetworkConfig,
    GenerationNetworkConfig,
    LearningRateHyperparameters,
    NetworkConfig,
    DEFAULT_SPATIOTEMPORAL_MODE,
    SPATIOTEMPORAL_MODE_FACTORIZED,
    SPATIOTEMPORAL_MODE_FLAT,
)

# Generation default values
GENERATION_DEFAULT_BATCH_SIZE = 8
GENERATION_DEFAULT_LEARNING_RATE = 0.0001
GENERATION_DEFAULT_EPOCHS = 500
GENERATION_DEFAULT_VALIDATION_SPLIT = 0.1
GENERATION_DEFAULT_EARLY_STOPPING_PATIENCE = 5
GENERATION_DEFAULT_MAX_LENGTH = 64
GENERATION_DEFAULT_MODEL_NAME = "xlm-roberta-base"
GENERATION_DEFAULT_GEODESIC_WEIGHT = 0.1
GENERATION_DEFAULT_GEODESIC_SCHEDULE = "none"
PREPROCESS_DEFAULT_SHARD_SIZE = 256

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
    rotationRepr = normalizeRotationRepr(
        generationSection.get("rotation-repr")
    )
    motionChannels = _resolveMotionChannels(
        generationSection,
        rotationRepr,
    )
    spatiotemporalMode = _normalizeSpatiotemporalMode(
        generationSection.get("spatiotemporal-mode")
    )
    
    return NetworkConfig(
        embedDim=int(section.get("embed-dim", 128)),
        clip=ClipNetworkConfig(
            motionNumHeads=int(clipSection.get("motion-num-heads", 4)),
            motionNumLayers=int(clipSection.get("motion-num-layers", 2)),
        ),
        generation=GenerationNetworkConfig(
            numHeads=int(generationSection.get("num-heads", 4)),
            numLayers=int(generationSection.get("num-layers", 6)),
            numSpatialLayers=int(
                generationSection.get("num-spatial-layers", 1)
            ),
            motionChannels=motionChannels,
            rotationRepr=rotationRepr,
            spatiotemporalMode=spatiotemporalMode,
            numBones=int(generationSection.get("num-bones", 22)),
            diffusionSteps=int(generationSection.get("diffusion-steps", 1000)),
        ),
    )


def _resolveMotionChannels(
    generationSection: Dict[str, Any],
    rotationRepr: str,
) -> int:
    """Resolve motion channels from config and rotation representation."""
    defaultChannels = resolveRotationChannels(
        rotationRepr,
        ROTATION_CHANNELS_ROT6D,
    )
    channelsValue = generationSection.get("motion-channels")
    if channelsValue is None:
        return defaultChannels
    motionChannels = int(channelsValue)
    expected = resolveRotationChannels(rotationRepr, motionChannels)
    if motionChannels != expected:
        LOGGER.warning(
            "motion-channels=%s does not match rotation-repr=%s (using %s)",
            motionChannels,
            rotationRepr,
            expected,
        )
        return expected
    return motionChannels


def _normalizeSpatiotemporalMode(mode: Any) -> str:
    """Normalize spatio-temporal mode."""
    if mode is None:
        return DEFAULT_SPATIOTEMPORAL_MODE
    normalized = str(mode).strip().lower()
    if normalized in (
        SPATIOTEMPORAL_MODE_FLAT,
        SPATIOTEMPORAL_MODE_FACTORIZED,
    ):
        return normalized
    LOGGER.warning(
        "Unknown spatiotemporal mode '%s', using default",
        normalized,
    )
    return DEFAULT_SPATIOTEMPORAL_MODE


def _defaultNetworkConfig() -> NetworkConfig:
    """Return default network configuration."""
    return NetworkConfig(
        embedDim=128,
        clip=ClipNetworkConfig(),
        generation=GenerationNetworkConfig(),
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


# ==============================================================================
# CLIP TRAINING CONFIG
# ==============================================================================


def loadTrainingConfig(
    configPath: Path,
    profile: Optional[str] = None,
) -> ClipTrainingConfig:
    """
    Parse the CLIP training configuration YAML file.

    Parameters
    ----------
    configPath : Path
        Filesystem path to the YAML file.
    profile : Optional[str]
        Name of the profile to load (e.g., "spark"). If None, uses "training".

    Returns
    -------
    ClipTrainingConfig
        Fully-populated configuration dataclass.
    """
    resolved = configPath.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"CLIP training config missing: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    pathsSection = payload.get("paths", {})
    
    # Profile selection: default to "training".
    # Fallback to legacy "training" key.
    profileName = profile or "training"
    if profileName in payload:
        trainingSection = payload.get(profileName, {})
    else:
        # Fallback for legacy configs without profiles
        trainingSection = payload.get("training", {})
    
    paths = _loadClipPaths(resolved, pathsSection)
    
    # Get network config path (must exist if provided)
    networkConfigPath = _optionalExistingPath(
        resolved,
        pathsSection.get("network-config"),
    )
    
    hyperparameters = ClipTrainingHyperparameters(
        batchSize=_int(trainingSection, "batch-size", DEFAULT_BATCH_SIZE),
        maxPromptLength=_int(
            trainingSection,
            "max-length",
            DEFAULT_PROMPT_MAX_LENGTH,
        ),
        modelName=str(
            trainingSection.get("model-name", DEFAULT_MODEL_NAME),
        ),
        epochs=_int(trainingSection, "epochs", 1),
        device=str(trainingSection.get("device", "auto")),
        validationSplit=_float(
            trainingSection,
            "validation-split",
            DEFAULT_VALIDATION_SPLIT,
        ),
        earlyStoppingPatience=_int(
            trainingSection,
            "early-stopping-patience",
            DEFAULT_EARLY_STOPPING_PATIENCE,
        ),
        checkpointDir=_optionalPath(
            resolved,
            pathsSection.get("checkpoint-dir"),
        ),
        resumeCheckpoint=_optionalExistingPathOrNone(
            resolved,
            trainingSection.get("resume-checkpoint"),
            "resume-checkpoint",
        ),
        gradientAccumulation=_int(trainingSection, "gradient-accumulation", 1),
        MM_memoryLimitGB=_float(trainingSection, "MM-memory-limit-gb", 0.0),
        weightDecay=_float(trainingSection, "weight-decay", 0.0),
        # Learning Rate Configuration
        learningRate=_float(
            trainingSection,
            "learning-rate",
            DEFAULT_LEARNING_RATE,
        ),
        lrMin=_float(trainingSection, "lr-min", 1e-7),
        lrWarmupEpochs=_int(trainingSection, "lr-warmup-epochs", 0),
        lrSchedule=str(trainingSection.get("lr-schedule", "cosine")),
        lrDecayEpochs=_optionalInt(trainingSection, "lr-decay-epochs"),
        geodesicWeight=_float(
            trainingSection,
            "geodesic-weight",
            GENERATION_DEFAULT_GEODESIC_WEIGHT,
        ),
        geodesicWeightSchedule=str(
            trainingSection.get(
                "geodesic-weight-schedule",
                GENERATION_DEFAULT_GEODESIC_SCHEDULE,
            )
        ),
    )
    return ClipTrainingConfig(
        paths=paths,
        training=hyperparameters,
        networkConfigPath=networkConfigPath,
    )


def _loadClipPaths(
    configPath: Path,
    section: Dict[str, Any],
) -> ClipTrainingPaths:
    resolvedDataset = _resolveExistingPath(
        configPath,
        _require(section, "dataset-root"),
        "dataset-root",
    )
    return ClipTrainingPaths(datasetRoot=resolvedDataset)


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
        resumeCheckpoint=_optionalExistingPathOrNone(
            resolved,
            trainingSection.get("resume-checkpoint"),
            "resume-checkpoint",
        ),
        MM_memoryLimitGB=_float(trainingSection, "MM-memory-limit-gb", 0.0),
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
        # Learning Rate Configuration
        learningRate=_float(
            trainingSection,
            "learning-rate",
            GENERATION_DEFAULT_LEARNING_RATE,
        ),
        lrMin=_float(trainingSection, "lr-min", 1e-7),
        lrWarmupEpochs=_int(trainingSection, "lr-warmup-epochs", 0),
        lrSchedule=str(trainingSection.get("lr-schedule", "cosine")),
        lrDecayEpochs=_optionalInt(trainingSection, "lr-decay-epochs"),
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

    return GenerationTrainingPaths(
        datasetRoot=datasetRoot,
        clipCheckpoint=clipCheckpoint,
        checkpointDir=checkpointDir,
        validationIndices=validationIndices,
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

    processing = PreprocessDatasetProcessing(
        modelName=str(
            processingSection.get("model-name", DEFAULT_MODEL_NAME),
        ),
        maxPromptLength=_int(
            processingSection,
            "max-length",
            DEFAULT_PROMPT_MAX_LENGTH,
        ),
        shardSize=_int(
            processingSection,
            "shard-size",
            PREPROCESS_DEFAULT_SHARD_SIZE,
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
        Raised when the specified path does not exist.
    """
    if rawValue in (None, ""):
        return None
    candidates = list(_candidatePaths(configPath.parent, str(rawValue)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Configured resume-checkpoint does not exist. Tried: {attempted}",
    )



def _optionalExistingPathOrNone(
    configPath: Path,
    rawValue: Optional[str],
    label: str,
) -> Optional[Path]:
    """
    Resolve an optional path and return None when missing.

    Parameters
    ----------
    configPath : Path
        Path to the configuration file used as an anchor.
    rawValue : Optional[str]
        User-provided path value from the YAML file.
    label : str
        Label for warning messages.

    Returns
    -------
    Optional[Path]
        Resolved path or None when missing.
    """
    if rawValue in (None, ""):
        return None
    candidates = list(
        _candidatePaths(configPath.parent, str(rawValue))
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = ", ".join(str(c) for c in candidates)
    LOGGER.warning(
        "Optional %s does not exist. Tried: %s",
        label,
        attempted,
    )
    return None


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
