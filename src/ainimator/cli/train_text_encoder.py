"""Standalone text encoder training CLI — Phase A7 scaffold.

Provides the MECHANICS for standalone encoder training / pre-training.
The pre-training OBJECTIVE (contrastive text↔motion, TMR-style, etc.)
is decided experimentally by step B2 and is NOT implemented here — this
module provides the build/save/load infrastructure and a placeholder
training loop that will be filled in once B2 is settled.

Usage
-----
Build and save a fresh encoder artifact (no training — init weights only)::

    poetry run python -m ainimator.cli.train_text_encoder save \\
        --tokenizerDir output/text/custom_tokenizer \\
        --artifactDir  output/text/encoder_artifact \\
        --encoderType  custom

Load an existing artifact and report its hash::

    poetry run python -m ainimator.cli.train_text_encoder info \\
        --artifactDir output/text/encoder_artifact

Public surface
--------------
* :func:`main` — argparse entry point.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ainimator.text.artifact import (
    loadEncoderArtifact,
    readArtifactHash,
    saveEncoderArtifact,
)
from ainimator.text.clip_text_encoder import (
    ClipTextEncoder,
    ClipTextEncoderConfig,
    ClipTokenizer,
)
from ainimator.text.custom_text_encoder import (
    CustomTextEncoder,
    CustomTextEncoderConfig,
)
from ainimator.text.custom_tokenizer import CustomTokenizer

LOGGER = logging.getLogger(__name__)

_DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
_DEFAULT_CLIP_MAX_LENGTH = 32
_DEFAULT_HIDDEN_DIM = 256
_DEFAULT_NUM_LAYERS = 4
_DEFAULT_NUM_HEADS = 8
_DEFAULT_OUTPUT_DIM = 384


def _buildParser() -> argparse.ArgumentParser:
    """Return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ainimator.cli.train_text_encoder",
        description=(
            "Standalone text encoder: save / inspect an encoder artifact."
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    _addSaveSubcommand(subparsers)
    _addInfoSubcommand(subparsers)

    return parser


def _addSaveSubcommand(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> None:
    """Register the ``save`` sub-command."""
    sub = subparsers.add_parser(
        "save",
        help=(
            "Build an encoder from config + tokenizer and save its "
            "artifact.  Weights are at init (no training)."
        ),
    )
    sub.add_argument(
        "--artifactDir",
        type=Path,
        required=True,
        help="Output directory for the encoder artifact.",
    )
    sub.add_argument(
        "--encoderType",
        choices=["custom", "clip"],
        default="custom",
        help="Encoder type: 'custom' (BPE) or 'clip' (frozen CLIP).",
    )
    sub.add_argument(
        "--tokenizerDir",
        type=Path,
        default=None,
        help=(
            "Directory of a saved CustomTokenizer (required for "
            "--encoderType=custom)."
        ),
    )
    sub.add_argument(
        "--outputDim",
        type=int,
        default=_DEFAULT_OUTPUT_DIM,
        help="Encoder output dimension (should match denoiser embedDim).",
    )
    sub.add_argument(
        "--hiddenDim",
        type=int,
        default=_DEFAULT_HIDDEN_DIM,
        help="Custom encoder hidden dim (ignored for clip).",
    )
    sub.add_argument(
        "--numLayers",
        type=int,
        default=_DEFAULT_NUM_LAYERS,
        help="Custom encoder num layers (ignored for clip).",
    )
    sub.add_argument(
        "--numHeads",
        type=int,
        default=_DEFAULT_NUM_HEADS,
        help="Custom encoder num heads (ignored for clip).",
    )
    sub.add_argument(
        "--clipModelName",
        default=_DEFAULT_CLIP_MODEL,
        help="HuggingFace CLIP model id (only for --encoderType=clip).",
    )
    sub.add_argument(
        "--clipMaxLength",
        type=int,
        default=_DEFAULT_CLIP_MAX_LENGTH,
        help="Max token length for CLIP tokenizer (1-77).",
    )


def _addInfoSubcommand(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> None:
    """Register the ``info`` sub-command."""
    sub = subparsers.add_parser(
        "info",
        help="Print the stored hash and config of an encoder artifact.",
    )
    sub.add_argument(
        "--artifactDir",
        type=Path,
        required=True,
        help="Encoder artifact directory to inspect.",
    )


def _runSave(args: argparse.Namespace) -> int:
    """Execute the ``save`` sub-command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 = success).
    """
    artifactDir: Path = args.artifactDir

    if args.encoderType == "custom":
        if args.tokenizerDir is None:
            LOGGER.error(
                "--tokenizerDir is required for --encoderType=custom."
            )
            return 1
        tokenizer = CustomTokenizer.load(args.tokenizerDir)
        encoder = CustomTextEncoder(
            CustomTextEncoderConfig(
                vocabSize=tokenizer.vocabSize,
                maxLength=tokenizer.config.maxLength,
                hiddenDim=args.hiddenDim,
                numLayers=args.numLayers,
                numHeads=args.numHeads,
                outputDim=args.outputDim,
                padTokenId=tokenizer.padTokenId,
                useNullEmbedding=True,
                l2NormalizeOutput=True,
            )
        )
        digest = saveEncoderArtifact(
            encoder=encoder,
            artifactDir=artifactDir,
            tokenizer=tokenizer,
        )
    else:
        tokenizer_clip = ClipTokenizer(
            modelName=args.clipModelName,
            maxLength=args.clipMaxLength,
        )
        encoder_clip = ClipTextEncoder(
            ClipTextEncoderConfig(
                modelName=args.clipModelName,
                maxLength=args.clipMaxLength,
                outputDim=args.outputDim,
                useNullEmbedding=True,
                l2NormalizeOutput=True,
            )
        )
        digest = saveEncoderArtifact(
            encoder=encoder_clip,
            artifactDir=artifactDir,
            tokenizer=tokenizer_clip,
        )

    LOGGER.info(
        "Encoder artifact saved to %s (hash=%s…).",
        artifactDir,
        digest[:12],
    )
    return 0


def _runInfo(args: argparse.Namespace) -> int:
    """Execute the ``info`` sub-command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 = success).
    """
    artifactDir: Path = args.artifactDir
    storedHash = readArtifactHash(artifactDir)
    encoder, tokenizer = loadEncoderArtifact(artifactDir, device="cpu")
    print(f"artifact_dir : {artifactDir}")
    print(f"stored_hash  : {storedHash}")
    print(f"encoder_type : {type(encoder).__name__}")
    print(f"output_dim   : {encoder.outputDim}")
    print(f"tokenizer    : {type(tokenizer).__name__}")
    return 0


def main() -> None:
    """Entry point for ``python -m ainimator.cli.train_text_encoder``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    parser = _buildParser()
    args = parser.parse_args()

    if args.subcommand == "save":
        sys.exit(_runSave(args))
    elif args.subcommand == "info":
        sys.exit(_runInfo(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
