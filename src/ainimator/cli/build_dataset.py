"""CLI entry-point orchestrating dataset conversions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ainimator.core.config_loader import loadBuilderConfig
from ainimator.data.builder.dataset_builder import DatasetBuilder
from ainimator.core.types import DatasetBuildOptions


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
    parser.add_argument(
        "--custom-prompts",
        dest="includeCustomPrompts",
        action="store_true",
        default=None,
        help="Include converted custom prompts (#Simple/#Advanced).",
    )
    parser.add_argument(
        "--no-custom-prompts",
        dest="includeCustomPrompts",
        action="store_false",
        help="Disable converted custom prompts (#Simple/#Advanced).",
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
    includeCustomPrompts = (
        arguments.includeCustomPrompts
        if arguments.includeCustomPrompts is not None
        else config.processing.includeCustomPrompts
    )
    options = DatasetBuildOptions(
        debugMode=arguments.debug,
        includeCustomPrompts=includeCustomPrompts,
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
