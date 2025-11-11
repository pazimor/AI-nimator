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
    animationRootRaw = _first_value(
        pathsSection,
        ["animation_root", "amass_root", "amass-root", "amassRoot"],
    )
    indexCsvRaw = _first_value(
        pathsSection,
        ["index_csv", "index-csv", "humanml3d_mapping", "humanml3d-mapping", "humanMl3dMapping"],
    )
    if not indexCsvRaw:
        raise ValueError(
            "Configuration is missing `paths.index_csv` "
            "or `paths.humanml3d-mapping`.",
        )
    promptRootRaw = _first_value(
        pathsSection,
        ["prompt_root", "prompt-root", "converted_root", "converted-root", "convertedRoot"],
    )
    convertedRootRaw = _first_value(
        pathsSection,
        ["converted_root", "converted-root", "convertedRoot"],
    )
    outputRootRaw = _first_value(pathsSection, ["output_root", "output-root", "outputRoot"])
    animationRoot = _resolvePath(configPath, animationRootRaw or ".")
    animationRootsRaw = pathsSection.get("animation_roots", [])
    animationRoots = _coerce_path_list(configPath, animationRootsRaw)
    if not animationRoots:
        animationRoots = [animationRoot]
    elif animationRoot not in animationRoots:
        animationRoots.insert(0, animationRoot)
    indexCsv = _resolvePath(configPath, indexCsvRaw)
    promptRoot = (
        _resolvePath(configPath, promptRootRaw)
        if promptRootRaw
        else indexCsv.parent
    )
    convertedRoot = (
        _resolvePath(configPath, convertedRootRaw)
        if convertedRootRaw
        else None
    )
    promptSourcesRaw = pathsSection.get("prompt_sources", [])
    promptSources = _coerce_path_list(configPath, promptSourcesRaw)
    if not promptSources:
        promptSources = [promptRoot]
    elif promptRoot not in promptSources:
        promptSources.insert(0, promptRoot)
    customPromptFiles = _coerce_path_list(
        configPath,
        pathsSection.get("custom_prompt_files", []),
    )
    outputRoot = _resolvePath(configPath, outputRootRaw or "output")
    paths = DatasetBuilderPaths(
        animationRoot=animationRoot,
        animationRoots=animationRoots,
        promptRoot=promptRoot,
        promptSources=promptSources,
        customPromptFiles=customPromptFiles,
        indexCsv=indexCsv,
        outputRoot=outputRoot,
        convertedRoot=convertedRoot,
    )
    animationExtensionRaw = _first_value(
        processingSection,
        ["animation_extension", "animation-extension"],
    )
    promptExtensionRaw = _first_value(
        processingSection,
        ["prompt_text_extension", "prompt-text-extension"],
    )
    fallbackFpsRaw = _first_value(processingSection, ["fallback_fps", "fallback-fps"])
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


def _first_value(section: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in section and section[key] not in (None, ""):
            raw = section[key]
            return str(raw)
    return None


def _coerce_path_list(configPath: Path, value: Any) -> List[Path]:
    if isinstance(value, list):
        return [_resolvePath(configPath, str(entry)) for entry in value if entry]
    if isinstance(value, str) and value:
        return [_resolvePath(configPath, value)]
    return []
