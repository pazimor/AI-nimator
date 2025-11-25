"""YAML-backed configuration loader for the dataset builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from src.shared.types import (
    DatasetBuilderConfig,
    DatasetBuilderPaths,
    DatasetBuilderProcessing,
)


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
    
    animationRootRaw = _check_value(pathsSection, "amass-root")
    indexCsvRaw = _check_value(pathsSection, "humanml3d-mapping")
    promptRootRaw = _check_value(pathsSection, "converted-root")
    # ?? what need too check
    outputRootRaw = convertedRootRaw = _check_value(pathsSection, "converted-root" )
    # ?? same 
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
        promptRoot=promptRoot, ## This is the HumanML3D prompts
        promptSources=promptSources, ## This is the "GPT-5 made" prompts
        outputRoot=outputRoot, ## changer le systeme de recuperation
        convertedRoot=convertedRoot, ## todo: c'est le meme que prompt root dont on le vire
    )

    animationExtensionRaw = _check_value(processingSection, "animation-extension")
    promptExtensionRaw = _check_value(processingSection, "prompt-text-extension")
    fallbackFpsRaw = _check_value(processingSection, "fallback-fps")

    processing = DatasetBuilderProcessing(
        animationExtension=str(animationExtensionRaw or ".npz"),
        promptTextExtension=str(promptExtensionRaw or ".txt"),
        fallbackFps=int(fallbackFpsRaw) if fallbackFpsRaw is not None else 60,
    )
    return DatasetBuilderConfig(paths=paths, processing=processing)


def _resolvePath(configPath: Path, rawValue: str) -> Path:
    if not rawValue:
        return configPath.parent
    candidate = Path(rawValue)
    if candidate.is_absolute():
        return candidate
    return (configPath.parent / candidate).resolve()


def _check_value(section: Dict[str, Any], key: str) -> Optional[str]:
    if key in section and section[key] not in (None, ""):
        return str(section[key])
    else:
        ## need to find how to place section name here
        raise ValueError(f"Configuration is missing: {key}")
