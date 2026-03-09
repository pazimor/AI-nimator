"""CLI entry-point orchestrating dataset preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.features.dataset_preprocessor.preprocess_dataset import (
    DatasetPreprocessor,
)
from src.shared.config_loader import loadPreprocessConfig

DEFAULT_CONFIG_PATH = Path("src/configs/preprocess_dataset.yaml")


def buildArgumentParser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess converted datasets into shard files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the preprocessing YAML configuration file.",
    )
    parser.add_argument(
        "--include-folders",
        type=str,
        default=None,
        help=(
            "Comma-separated top-level folders to preprocess "
            "(example: KIT,CMU,ACCAD)."
        ),
    )
    parser.add_argument(
        "--network-config",
        type=Path,
        default=None,
        help=(
            "Optional override for the shared network.yaml file used "
            "to decide which motion features are preprocessed."
        ),
    )
    return parser


def main() -> None:
    """
    CLI entry-point.

    Returns
    -------
    None
        This function exits the process when finished.
    """
    parser = buildArgumentParser()
    arguments = parser.parse_args()
    config = loadPreprocessConfig(arguments.config)
    if arguments.network_config is not None:
        config = replace(
            config,
            paths=replace(
                config.paths,
                networkConfigPath=arguments.network_config.expanduser(),
            ),
        )
    preprocessor = DatasetPreprocessor(config)
    includeFolders = _parseFolderList(arguments.include_folders)
    preprocessor.run(includeFolders=includeFolders)
    parser.exit(
        0,
        f"Preprocessing finished. Output: {config.paths.outputRoot}\n",
    )


def _parseFolderList(rawValue: str | None) -> list[str] | None:
    """Parse comma-separated folder names from CLI."""
    if rawValue is None:
        return None
    folders = [item.strip() for item in rawValue.split(",")]
    normalized = [item for item in folders if item]
    if not normalized:
        return None
    return normalized


if __name__ == "__main__":
    main()
