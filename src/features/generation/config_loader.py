"""Training configuration loader for generation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from src.shared.types import (
    GenerationTrainingConfig,
    GenerationTrainingHyperparameters,
    GenerationTrainingPaths,
)


# Default values
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_EPOCHS = 500
DEFAULT_EMBED_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_BONES = 22
DEFAULT_DIFFUSION_STEPS = 1000
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_MAX_LENGTH = 64
DEFAULT_MODEL_NAME = "xlm-roberta-base"


def loadGenerationConfig(configPath: Path) -> GenerationTrainingConfig:
    """
    Parse the generation training configuration YAML file.

    Parameters
    ----------
    configPath : Path
        Filesystem path to the YAML file.

    Returns
    -------
    GenerationTrainingConfig
        Fully-populated configuration dataclass.
    """
    resolved = configPath.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Generation training config missing: {resolved}")

    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    pathsSection = payload.get("paths", {})
    trainingSection = payload.get("training", {})

    paths = _loadPaths(resolved, pathsSection)
    hyperparameters = GenerationTrainingHyperparameters(
        batchSize=_int(trainingSection, "batch-size", DEFAULT_BATCH_SIZE),
        learningRate=_float(trainingSection, "learning-rate", DEFAULT_LEARNING_RATE),
        epochs=_int(trainingSection, "epochs", DEFAULT_EPOCHS),
        embedDim=_int(trainingSection, "embed-dim", DEFAULT_EMBED_DIM),
        numHeads=_int(trainingSection, "num-heads", DEFAULT_NUM_HEADS),
        numLayers=_int(trainingSection, "num-layers", DEFAULT_NUM_LAYERS),
        numBones=_int(trainingSection, "num-bones", DEFAULT_NUM_BONES),
        diffusionSteps=_int(trainingSection, "diffusion-steps", DEFAULT_DIFFUSION_STEPS),
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
        maxPromptLength=_int(trainingSection, "max-length", DEFAULT_MAX_LENGTH),
        modelName=str(trainingSection.get("model-name", DEFAULT_MODEL_NAME)),
        resumeCheckpoint=_optionalExistingPath(
            resolved,
            trainingSection.get("resume-checkpoint"),
        ),
    )

    return GenerationTrainingConfig(paths=paths, training=hyperparameters)


def _loadPaths(configPath: Path, section: Dict[str, Any]) -> GenerationTrainingPaths:
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

    return GenerationTrainingPaths(
        datasetRoot=datasetRoot,
        clipCheckpoint=clipCheckpoint,
        checkpointDir=checkpointDir,
    )


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


def _optionalPath(configPath: Path, rawValue: Optional[str]) -> Optional[Path]:
    """Resolve an optional path, creating directory if needed."""
    if rawValue in (None, ""):
        return None
    candidate = Path(rawValue).expanduser()
    if not candidate.is_absolute():
        candidate = (configPath.parent / candidate).resolve()
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _optionalExistingPath(
    configPath: Path,
    rawValue: Optional[str],
) -> Optional[Path]:
    """Resolve an optional path that must exist if provided."""
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


def _resolveExistingPath(
    configPath: Path,
    rawValue: str,
    label: str,
) -> Path:
    """Resolve a path that must exist."""
    candidates = _candidatePaths(configPath.parent, rawValue)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Configured {label} does not exist. Tried: {attempted}",
    )


def _candidatePaths(baseDir: Path, rawValue: str) -> Iterable[Path]:
    """Generate candidate paths for resolution."""
    candidate = Path(rawValue).expanduser()
    if candidate.is_absolute():
        return (candidate,)
    configRelative = (baseDir / candidate).resolve()
    cwdRelative = (Path.cwd() / candidate).resolve()
    if configRelative == cwdRelative:
        return (configRelative,)
    return (configRelative, cwdRelative)
