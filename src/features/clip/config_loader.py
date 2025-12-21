"""Training configuration loader for CLIP workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from src.shared.constants.clip import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EMBED_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_MAX_LENGTH,
    DEFAULT_VALIDATION_SPLIT,
)
from src.shared.types import (
    ClipTrainingConfig,
    ClipTrainingHyperparameters,
    ClipTrainingPaths,
)


def loadTrainingConfig(configPath: Path) -> ClipTrainingConfig:
    """
    Parse the CLIP training configuration YAML file.

    Parameters
    ----------
    configPath : Path
        Filesystem path to the YAML file.

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
    trainingSection = payload.get("training", {})
    paths = _loadPaths(resolved, pathsSection)
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
        learningRate=_float(
            trainingSection,
            "learning-rate",
            DEFAULT_LEARNING_RATE,
        ),
        epochs=_int(trainingSection, "epochs", 1),
        device=str(trainingSection.get("device", "auto")),
        embedDim=_int(trainingSection, "embed-dim", DEFAULT_EMBED_DIM),
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
        resumeCheckpoint=_optionalExistingPath(
            resolved,
            trainingSection.get("resume-checkpoint"),
        ),
    )
    return ClipTrainingConfig(paths=paths, training=hyperparameters)


def _loadPaths(configPath: Path, section: Dict[str, Any]) -> ClipTrainingPaths:
    datasetRoot = section.get("dataset-root")
    if datasetRoot:
        resolvedDataset = _resolveExistingPath(
            configPath,
            str(datasetRoot),
            "dataset-root",
        )
        return ClipTrainingPaths(
            promptRoot=resolvedDataset,
            animationRoot=resolvedDataset,
        )
    return ClipTrainingPaths(
        promptRoot=_resolveExistingPath(
            configPath,
            _require(section, "prompt-root"),
            "prompt-root",
        ),
        animationRoot=_resolveExistingPath(
            configPath,
            _require(section, "animation-root"),
            "animation-root",
        ),
    )


def _require(section: Dict[str, Any], key: str) -> str:
    value = section.get(key)
    if value in (None, ""):
        raise ValueError(f"Training config missing required field: {key}")
    return str(value)


def _int(section: Dict[str, Any], key: str, default: int) -> int:
    if key not in section:
        return int(default)
    return int(section[key])


def _float(section: Dict[str, Any], key: str, default: float) -> float:
    if key not in section:
        return float(default)
    return float(section[key])


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
