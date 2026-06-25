"""ARCHIVED (A7, 2026-06-25) — superseded by train_controller_v2 --profile full.

Use ``python -m ainimator.cli.train_controller_v2 --profile full`` instead.
This CLI is kept in ``legacy/`` for historical reference only.

Original module docstring follows.
----------------------------------------------------------------------
CLI — Goal A controller **generalization** training (phase A6).

Zero logic: pick a working set of clips, split it into a disjoint
train / held-out partition (deterministic, seeded), load the tensors and
delegate to
:func:`ainimator.training.controller_generalization_v2.runControllerGeneralization`.

Unlike the overfit / multi-clip CLIs, the metrics that matter are reported
on the **held-out** clips — the controller never trained on them, so they
are the honest generalization judge (ROADMAP_DETERMINIST A6).

Clip selection
--------------
``--num-clips N`` takes the ``N`` longest clips with at least
``--min-frames`` frames, then reserves ``--held-out-clips K`` of them
(seeded shuffle) as the held-out partition; the remaining ``N - K`` are the
train set.

Example
-------
``python -m ainimator.cli.train_controller_generalization_v2 \\
    --dataset-root /Users/pazimor/dataset_preprocessed_canon \\
    --num-clips 1000 --held-out-clips 64 --min-frames 200 \\
    --phase explicit --foot-contact-weight 1.0 \\
    --embed-dim 384 --num-heads 6 --num-layers 6 \\
    --epochs 200 --clip-batch-size 8 \\
    --output-dir output/controller_gen_n1000``
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from ainimator.core.config_loader import defaultPreprocessedDatasetRoot
from ainimator.core.constants.controller import PhaseMode
from ainimator.model.losses_controller_v2 import ControllerLossWeights
from ainimator.training.controller_generalization_v2 import (
    runControllerGeneralization,
)
from ainimator.training.controller_training_v2 import ControllerTrainingConfig
from ainimator.training.training_v2 import loadDatasetSample


def _parseArgs() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Goal A controller generalization training (A6)."
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--num-clips", type=int, default=256)
    parser.add_argument("--held-out-clips", type=int, default=32)
    parser.add_argument("--min-frames", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--eval-sample-clips", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--phase", type=str, default="none")
    parser.add_argument("--aim-direction", action="store_true")
    parser.add_argument("--foot-contact-weight", type=float, default=0.0)
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


def _splitIndices(
    args: argparse.Namespace, datasetRoot: Path
) -> tuple[list[int], list[int]]:
    """Resolve a working set, then split it into train / held-out.

    The eligible clips (>= ``min_frames``) are taken longest-first up to
    ``num_clips``, then shuffled with a seeded RNG so the held-out reserve
    is not biased toward the shortest clips.
    """
    linkIndexPath = datasetRoot / "link_index.json"
    links = json.loads(linkIndexPath.read_text(encoding="utf-8"))
    eligible = [link for link in links if link["frames"] >= args.min_frames]
    eligible.sort(key=lambda link: link["frames"], reverse=True)
    chosen = [int(link["linkId"]) for link in eligible[: args.num_clips]]
    if len(chosen) < args.held_out_clips + 2:
        raise SystemExit(
            f"Only {len(chosen)} clips with >= {args.min_frames} frames; "
            f"need >= held-out ({args.held_out_clips}) + 2 train."
        )
    random.Random(args.seed).shuffle(chosen)
    heldOut = chosen[: args.held_out_clips]
    train = chosen[args.held_out_clips :]
    return train, heldOut


def _loadClips(
    datasetRoot: Path, indices: list[int]
) -> list[tuple[object, object]]:
    """Load (rotation6d, rootTranslation) for each link index."""
    clips = []
    for index in indices:
        sample = loadDatasetSample(datasetRoot, index)
        clips.append((sample.rotation6d, sample.rootTranslation))
    return clips


def _buildConfig(args: argparse.Namespace) -> ControllerTrainingConfig:
    """Map CLI arguments onto the controller training config."""
    return ControllerTrainingConfig(
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
    )


def main() -> None:
    """Train the controller for generalization and report held-out metrics."""
    logging.basicConfig(level=logging.INFO)
    args = _parseArgs()
    datasetRoot = _resolveDatasetRoot(args.dataset_root)
    trainIndices, heldOutIndices = _splitIndices(args, datasetRoot)
    logging.info(
        "Split: %d train / %d held-out clips", len(trainIndices),
        len(heldOutIndices),
    )
    trainClips = _loadClips(datasetRoot, trainIndices)
    heldOutClips = _loadClips(datasetRoot, heldOutIndices)

    result = runControllerGeneralization(
        trainClips, heldOutClips, _buildConfig(args),
        clipBatchSize=args.clip_batch_size,
        evalSampleClips=args.eval_sample_clips,
    )
    logging.info("final train loss: %.6f", result.finalLoss)
    logging.info("train  metrics: %s", result.trainMetrics)
    logging.info("HELD-OUT metrics: %s", result.heldOutMetrics)
    logging.info("held-out drift curve: %s", result.driftCurve)
    for name, verdict in result.verdicts.items():
        logging.info("contract %s: %s", name, verdict.value)


if __name__ == "__main__":
    main()
