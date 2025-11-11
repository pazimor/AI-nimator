"""Tests covering the YAML configuration loader."""

from __future__ import annotations

from pathlib import Path

from src.features.dataset_builder.config_loader import loadBuilderConfig


def test_loadBuilderConfig_resolves_relative_paths(tmp_path: Path) -> None:
    """
    Ensure relative paths declare in YAML are resolved from the file location.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.
    """

    configPath = tmp_path / "config" / "dataset.yaml"
    configPath.parent.mkdir(parents=True)
    configPath.write_text(
        "paths:\n"
        "  animation_root: data/animations\n"
        "  animation_roots:\n"
        "    - data/animations\n"
        "    - data/animations_extra\n"
        "  prompt_root: data/prompts\n"
        "  converted_root: data/converted\n"
        "  index_csv: data/index.csv\n"
        "  custom_prompt_files:\n"
        "    - data/prompts/custom.jsonl\n"
        "  output_root: data/output\n"
        "processing:\n"
        "  animation_extension: .npz\n"
        "  prompt_text_extension: .txt\n"
        "  fallback_fps: 75\n",
        encoding="utf-8",
    )
    config = loadBuilderConfig(configPath)
    assert config.paths.animationRoot == configPath.parent / "data/animations"
    assert config.paths.animationRoots[1] == configPath.parent / "data/animations_extra"
    assert config.paths.promptRoot == configPath.parent / "data/prompts"
    assert config.paths.promptSources == [config.paths.promptRoot]
    assert config.paths.customPromptFiles[0] == configPath.parent / "data/prompts/custom.jsonl"
    assert config.paths.convertedRoot == configPath.parent / "data/converted"
    assert config.paths.indexCsv == configPath.parent / "data/index.csv"
    assert config.paths.outputRoot == configPath.parent / "data/output"
    assert config.processing.fallbackFps == 75


def test_loadBuilderConfig_supports_aliases_and_prompt_fallback(tmp_path: Path) -> None:
    """
    Ensure kebab-case aliases are supported and prompt root falls back to the index folder.
    """

    configPath = tmp_path / "dataset.yaml"
    indexPath = tmp_path / "humanml3d" / "index.csv"
    indexPath.parent.mkdir(parents=True)
    indexPath.write_text("", encoding="utf-8")
    configPath.write_text(
        "paths:\n"
        f"  humanml3d-mapping: {indexPath.as_posix()}\n"
        f"  amass-root: {(tmp_path / 'amass').as_posix()}\n"
        f"  prompt_sources:\n"
        f"    - {(tmp_path / 'converted').as_posix()}\n"
        f"    - {(tmp_path / 'prompts').as_posix()}\n"
        f"  custom_prompt_files:\n"
        f"    - {(tmp_path / 'custom' / 'batch.jsonl').as_posix()}\n"
        "processing:\n"
        "  animation-extension: .npy\n"
        "  fallback-fps: 55\n",
        encoding="utf-8",
    )
    config = loadBuilderConfig(configPath)
    assert config.paths.indexCsv == indexPath
    assert config.paths.promptRoot == indexPath.parent
    assert config.paths.promptSources == [
        indexPath.parent,
        tmp_path / "converted",
        tmp_path / "prompts",
    ]
    assert config.paths.animationRoot == (tmp_path / "amass")
    assert config.paths.animationRoots[0] == (tmp_path / "amass")
    assert config.processing.animationExtension == ".npy"
    assert config.processing.fallbackFps == 55
    assert config.paths.customPromptFiles[0] == tmp_path / "custom" / "batch.jsonl"
    assert config.paths.convertedRoot is None
