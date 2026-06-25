"""CLI entry point for ONNX export (Phase A8) and controller bundle (A7/B0).

Usage
-----
.. code-block:: bash

    python -m ainimator.cli.export_onnx encoder \\
        --encoder-artifact path/to/encoder_artifact \\
        --output output/onnx/encoder.onnx

    python -m ainimator.cli.export_onnx denoiser \\
        --checkpoint path/to/checkpoint.pt \\
        --output output/onnx/denoiser_step.onnx

    python -m ainimator.cli.export_onnx controller \\
        --checkpoint path/to/controller.pt \\
        --output output/onnx/controller_step.onnx

    python -m ainimator.cli.export_onnx bundle \\
        --checkpoint path/to/controller.pt \\
        --output-dir output/controller_bundle

This CLI is logic-free: it parses arguments, loads the components
from existing artifacts/checkpoints, delegates to
:mod:`ainimator.export.onnx` and :mod:`ainimator.export.bundle`, and
reports success.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENCODER_COMMAND = "encoder"
_DENOISER_COMMAND = "denoiser"
_CONTROLLER_COMMAND = "controller"
_BUNDLE_COMMAND = "bundle"


def _buildParser() -> argparse.ArgumentParser:
    """Build the argument parser for the export_onnx CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m ainimator.cli.export_onnx",
        description="Export encoder or denoiser to ONNX.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- encoder sub-command -------------------------------------------
    encParser = sub.add_parser(
        _ENCODER_COMMAND,
        help="Export the text encoder (encode() path).",
    )
    encParser.add_argument(
        "--encoder-artifact",
        type=Path,
        required=True,
        metavar="DIR",
        help="Path to an encoder artifact directory "
        "(contains config.yaml + weights.pt).",
    )
    encParser.add_argument(
        "--output",
        type=Path,
        default=Path("output/onnx/encoder.onnx"),
        metavar="FILE",
        help="Destination .onnx file.  Default: output/onnx/encoder.onnx",
    )
    encParser.add_argument(
        "--batch-size", type=int, default=1, metavar="N"
    )
    encParser.add_argument(
        "--text-len", type=int, default=16, metavar="N"
    )

    # --- denoiser sub-command ------------------------------------------
    denParser = sub.add_parser(
        _DENOISER_COMMAND,
        help="Export one denoiser step (the DDIM loop stays in Python).",
    )
    denParser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to a v2 checkpoint (.pt).",
    )
    denParser.add_argument(
        "--output",
        type=Path,
        default=Path("output/onnx/denoiser_step.onnx"),
        metavar="FILE",
        help=(
            "Destination .onnx file.  "
            "Default: output/onnx/denoiser_step.onnx"
        ),
    )
    denParser.add_argument(
        "--batch-size", type=int, default=1, metavar="N"
    )
    denParser.add_argument(
        "--frames", type=int, default=32, metavar="N"
    )
    denParser.add_argument(
        "--text-len", type=int, default=16, metavar="N"
    )

    # --- controller sub-command (Goal A, phase A5) ---------------------
    ctrlParser = sub.add_parser(
        _CONTROLLER_COMMAND,
        help="Export one controller forward (rollout loop stays in C#/C++).",
    )
    ctrlParser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to a controller checkpoint (.pt).",
    )
    ctrlParser.add_argument(
        "--output",
        type=Path,
        default=Path("output/onnx/controller_step.onnx"),
        metavar="FILE",
        help=(
            "Destination .onnx file.  "
            "Default: output/onnx/controller_step.onnx"
        ),
    )
    ctrlParser.add_argument(
        "--batch-size", type=int, default=1, metavar="N"
    )

    # --- bundle sub-command (Goal A/A7, Goal B/B0) --------------------
    bundleParser = sub.add_parser(
        _BUNDLE_COMMAND,
        help=(
            "Assemble a complete controller bundle: "
            "ONNX + norm_stats.json + manifest.json + presets/."
        ),
    )
    bundleParser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to a controller checkpoint (.pt).",
    )
    bundleParser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/controller_bundle"),
        metavar="DIR",
        help=(
            "Destination bundle directory.  "
            "Default: output/controller_bundle"
        ),
    )
    bundleParser.add_argument(
        "--resolved-config",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to an existing resolved_config.yaml to include "
            "in the bundle.  Optional."
        ),
    )
    bundleParser.add_argument(
        "--batch-size", type=int, default=1, metavar="N"
    )
    return parser


def _exportEncoder(args: argparse.Namespace) -> None:
    """Load an encoder artifact and export it to ONNX.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the ``encoder`` sub-command.
    """
    from ainimator.text.artifact import loadEncoderArtifact
    from ainimator.text.custom_text_encoder import CustomTextEncoder
    from ainimator.export.onnx import exportEncoder

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    encoder, _tokenizer = loadEncoderArtifact(args.encoder_artifact)
    if not isinstance(encoder, CustomTextEncoder):
        raise TypeError(
            "export_onnx encoder only supports CustomTextEncoder; "
            f"got {type(encoder).__name__}."
        )
    encoder.eval()
    exportEncoder(
        encoder=encoder,
        outputPath=args.output,
        batchSize=args.batch_size,
        textLen=args.text_len,
    )
    print(f"Encoder exported to {args.output}")


def _exportDenoiser(args: argparse.Namespace) -> None:
    """Load a checkpoint and export one denoiser step to ONNX.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the ``denoiser`` sub-command.
    """
    from ainimator.training.training_v2 import loadCheckpointV2
    from ainimator.export.onnx import exportDenoiser

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    _tokenizer, _encoder, denoiser, _schedule, _normalizer, _meta = (
        loadCheckpointV2(args.checkpoint, device="cpu")
    )
    denoiser.eval()
    exportDenoiser(
        denoiser=denoiser,
        outputPath=args.output,
        batchSize=args.batch_size,
        frames=args.frames,
        textLen=args.text_len,
    )
    print(f"Denoiser step exported to {args.output}")


def _exportController(args: argparse.Namespace) -> None:
    """Load a controller checkpoint and export one forward to ONNX.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the ``controller`` sub-command.
    """
    from ainimator.training.controller_training_v2 import (
        loadControllerCheckpoint,
    )
    from ainimator.export.onnx import exportController

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    model, _stateNorm, _deltaNorm, _mean, _std = loadControllerCheckpoint(
        args.checkpoint, device=None
    )
    model.eval()
    exportController(
        controller=model,
        outputPath=args.output,
        batchSize=args.batch_size,
    )
    print(f"Controller forward exported to {args.output}")


def _exportBundle(args: argparse.Namespace) -> None:
    """Load a controller checkpoint and export a complete bundle.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the ``bundle`` sub-command.
    """
    from ainimator.training.controller_training_v2 import (
        loadControllerCheckpoint,
    )
    from ainimator.export.bundle import exportControllerBundle

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    model, stateNorm, deltaNorm, controlMean, controlStd = (
        loadControllerCheckpoint(args.checkpoint, device=None)
    )
    model.eval()
    bundleDir = exportControllerBundle(
        controller=model,
        stateNorm=stateNorm,
        deltaNorm=deltaNorm,
        controlMean=controlMean,
        controlStd=controlStd,
        outputDir=args.output_dir,
        resolvedConfigPath=args.resolved_config,
        batchSize=args.batch_size,
    )
    print(f"Controller bundle assembled at {bundleDir}")


def main() -> None:
    """Entry point for ``python -m ainimator.cli.export_onnx``."""
    parser = _buildParser()
    args = parser.parse_args()

    if args.command == _ENCODER_COMMAND:
        _exportEncoder(args)
    elif args.command == _DENOISER_COMMAND:
        _exportDenoiser(args)
    elif args.command == _CONTROLLER_COMMAND:
        _exportController(args)
    elif args.command == _BUNDLE_COMMAND:
        _exportBundle(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
