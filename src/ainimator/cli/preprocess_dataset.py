"""CLI entry-point orchestrating dataset preprocessing."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from ainimator.data.preprocessor.preprocess_dataset import (
    DatasetPreprocessor,
)
from ainimator.core.config_loader import loadPreprocessConfig

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
        "--canonicalize-facing",
        dest="canonicalizeFacing",
        action="store_true",
        help=(
            "Rotate every clip so frame 0 stands UPRIGHT (Y-up) and "
            "faces +Z (canonicalizeMotionUpright).  Fixes the AMASS Z-up "
            "root orientation applied to the Y-up rest skeleton that "
            "stored bodies lying down.  Mandatory for correct geometry; "
            "requires a full preprocessing pass + from-scratch retrain."
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
    if arguments.canonicalizeFacing:
        # The preprocessor checks this env var inside its window loop —
        # set it before constructing the DatasetPreprocessor so child
        # workers (if any) inherit it.
        os.environ["AINIMATOR_CANONICALIZE_FACING"] = "1"
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
