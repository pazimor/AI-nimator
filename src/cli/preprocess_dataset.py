"""CLI entry-point orchestrating dataset preprocessing."""

from __future__ import annotations

import argparse
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
    preprocessor = DatasetPreprocessor(config)
    preprocessor.run()
    parser.exit(
        0,
        f"Preprocessing finished. Output: {config.paths.outputRoot}\n",
    )


if __name__ == "__main__":
    main()
