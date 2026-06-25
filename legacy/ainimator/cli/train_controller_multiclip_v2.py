"""ARCHIVED (A7, 2026-06-25) — superseded by train_controller_v2 --profile full.

This CLI is kept in ``legacy/`` for historical reference only.  Use
``python -m ainimator.cli.train_controller_v2 --profile full`` instead.
Nothing in ``ainimator.*`` may import from ``legacy.*``
(lint-imports ``no_legacy_imports``).

Original module docstring follows.
----------------------------------------------------------------------
CLI — Goal A multi-clip controller training (A2/A4 validation).

Zero logic: select N clips, load them, delegate to
:func:`ainimator.training.controller_multiclip_v2.runControllerMultiClip`.

Clip selection
--------------
* ``--sample-indices 3,9,18`` — explicit link indices, or
* ``--num-clips N --min-frames F`` — the N longest clips with at least F
  frames (read from ``link_index.json``), which gives control variety
  without hand-picking.

Example
-------
``python -m ainimator.cli.train_controller_multiclip_v2 \\
    --num-clips 16 --min-frames 200 --phase explicit \\
    --output-dir output/controller_multi``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ainimator.core.config_loader import defaultPreprocessedDatasetRoot
from ainimator.core.constants.controller import PhaseMode
from ainimator.model.losses_controller_v2 import ControllerLossWeights
from ainimator.training.controller_multiclip_v2 import runControllerMultiClip
from ainimator.training.controller_training_v2 import ControllerTrainingConfig
from ainimator.training.training_v2 import loadDatasetSample


def _parseArgs() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Goal A multi-clip controller training."
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated link indices, e.g. '3,9,18'.",
    )
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--min-frames", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    # 1e-3 matches the single-clip default; 5e-4 under-optimised the
    # multi-clip run (debug 2026-06-23: 4 clips lr5e-4/400ep=0.55 vs
    # lr1e-3/1000ep=0.17).
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--phase", type=str, default="none")
    parser.add_argument("--aim-direction", action="store_true")
    parser.add_argument("--foot-contact-weight", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling", type=float, default=0.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _resolveDatasetRoot(explicit: Path | None) -> Path:
    """Return the explicit dataset root or the preprocess.yaml default."""
    if explicit is not None:
        return explicit
    fallback = defaultPreprocessedDatasetRoot()
    if fallback is None:
        raise SystemExit(
            "--dataset-root not given and no output-root in "
            "preprocess_dataset.yaml."
        )
    logging.info("Using default dataset root: %s", fallback)
    return fallback


def _selectIndices(args: argparse.Namespace, datasetRoot: Path) -> list[int]:
    """Resolve the clip link indices from the CLI selection options."""
    if args.sample_indices:
        return [int(token) for token in args.sample_indices.split(",")]
    linkIndexPath = datasetRoot / "link_index.json"
    links = json.loads(linkIndexPath.read_text(encoding="utf-8"))
    eligible = [link for link in links if link["frames"] >= args.min_frames]
    eligible.sort(key=lambda link: link["frames"], reverse=True)
    chosen = [int(link["linkId"]) for link in eligible[: args.num_clips]]
    if len(chosen) < 2:
        raise SystemExit(
            f"Only {len(chosen)} clips with >= {args.min_frames} frames; "
            "multi-clip needs at least 2."
        )
    return chosen


def main() -> None:
    """Train the controller on several dataset clips."""
    logging.basicConfig(level=logging.INFO)
    args = _parseArgs()
    datasetRoot = _resolveDatasetRoot(args.dataset_root)
    indices = _selectIndices(args, datasetRoot)
    logging.info("Selected %d clips: %s", len(indices), indices)
    clips = []
    for index in indices:
        sample = loadDatasetSample(datasetRoot, index)
        clips.append((sample.rotation6d, sample.rootTranslation))

    config = ControllerTrainingConfig(
        outputDir=args.output_dir,
        epochs=args.epochs,
        learningRate=args.learning_rate,
        embedDim=args.embed_dim,
        numHeads=args.num_heads,
        numLayers=args.num_layers,
        contextFrames=args.context_frames,
        phaseMode=PhaseMode(args.phase),
        useAimDirection=args.aim_direction,
        seed=args.seed,
        device=args.device,
        lossWeights=ControllerLossWeights(
            velocity=1.0,
            geodesic=1.0,
            footContact=args.foot_contact_weight,
        ),
        scheduledSampling=args.scheduled_sampling,
        resumeCheckpoint=args.resume,
    )
    result = runControllerMultiClip(clips, config)
    logging.info("final loss: %.6f", result.finalLoss)
    for name, verdict in result.verdicts.items():
        logging.info("contract %s: %s", name, verdict.value)


if __name__ == "__main__":
    main()
