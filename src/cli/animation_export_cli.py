"""CLI entry point for the animation format converter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.shared.animation_exporter import (
    AnimationExportError,
    AnimationExportFormat,
    AnimationFormatConverter,
)


def buildParser() -> argparse.ArgumentParser:
    """Return the argparse parser configured for the exporter CLI."""

    parser = argparse.ArgumentParser(
        description="Convert rotroot JSON clips into standard animation formats.",
    )
    parser.add_argument("input", help="Path to the source rotroot JSON file.")
    parser.add_argument(
        "output",
        help=(
            "Destination file path. The format can be inferred from the"
            " extension when --format is omitted."
        ),
    )
    parser.add_argument(
        "--format",
        dest="formatName",
        choices=["fb", "bvh", "collada", "fbx", "dae"],
        help="Explicit export format to use.",
    )
    return parser


def _inferFormatFromOutput(outputPath: Path) -> AnimationExportFormat:
    """Infer the export format from the requested output path."""

    suffix = outputPath.suffix.lower().lstrip(".")
    if not suffix:
        raise AnimationExportError(
            "Unable to infer format: specify --format or an output extension.",
        )
    return AnimationExportFormat.fromName(suffix)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by ``python -m``."""

    parser = buildParser()
    arguments = parser.parse_args(argv)
    try:
        inputPath = Path(arguments.input)
        outputPath = Path(arguments.output)
        if arguments.formatName:
            exportFormat = AnimationExportFormat.fromName(arguments.formatName)
        else:
            exportFormat = _inferFormatFromOutput(outputPath)
        converter = AnimationFormatConverter.fromJson(inputPath)
        converter.writeToPath(exportFormat, outputPath)
        print(f"[export] {exportFormat.name.lower()} -> {outputPath}")
        return 0
    except AnimationExportError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - unexpected failure
        print(f"[unexpected] {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
