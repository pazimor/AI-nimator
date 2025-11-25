"""CLI entry-point orchestrating dataset conversions."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.features.dataset_builder.config_loader import loadBuilderConfig
from src.features.dataset_builder.dataset_builder import DatasetBuilder
from src.shared.types import DatasetBuildOptions


def buildArgumentParser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Convert and enrich animation datasets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/configs/dataset.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (fails on first error).",
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
    config = loadBuilderConfig(arguments.config)
    options = DatasetBuildOptions(
        debugMode=arguments.debug,
    )
    builder = DatasetBuilder(config=config, options=options)
    report = builder.buildDataset()
    parser.exit(
        0,
        f"Processed {report.processedSamples} samples with "
        f"{len(report.failedSamples)} failures. "
        f"Output: {report.outputDirectory}\n",
    )


if __name__ == "__main__":
    main()
