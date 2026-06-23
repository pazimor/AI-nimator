"""CLI entry point — Goal C controller overfit / training (phase C1).

Zero logic: parse args, load one clip, delegate to
:func:`ainimator.training.controller_training_v2.runControllerOverfit`.

Example
-------
``python -m ainimator.cli.train_controller_v2 \\
    --dataset-root <preprocessed> --sample-index 0 \\
    --output-dir output/controller_overfit``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ainimator.core.config_loader import defaultPreprocessedDatasetRoot
from ainimator.core.constants.controller import PhaseMode
from ainimator.model.losses_controller_v2 import ControllerLossWeights
from ainimator.training.controller_training_v2 import (
    ControllerTrainingConfig,
    runControllerOverfit,
)
from ainimator.training.training_v2 import loadDatasetSample


def _parseArgs() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Goal C controller train.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Preprocessed dataset root. Defaults to output-root from "
        "src/configs/preprocess_dataset.yaml when omitted.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--context-frames", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--phase", type=str, default="none")
    parser.add_argument("--aim-direction", action="store_true")
    parser.add_argument("--foot-contact-weight", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Warm-start from a controller checkpoint (.pt). Architecture "
        "and normalization come from it; arch flags are ignored. Use this "
        "to fine-tune with --scheduled-sampling instead of from scratch.",
    )
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
            "--dataset-root not given and no output-root found in "
            "src/configs/preprocess_dataset.yaml."
        )
    logging.info("Using default dataset root: %s", fallback)
    return fallback


def main() -> None:
    """Run the controller overfit on one dataset sample."""
    logging.basicConfig(level=logging.INFO)
    args = _parseArgs()
    datasetRoot = _resolveDatasetRoot(args.dataset_root)
    sample = loadDatasetSample(datasetRoot, args.sample_index)
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
    result = runControllerOverfit(
        sample.rotation6d, sample.rootTranslation, config
    )
    logging.info("final loss: %.6f", result.finalLoss)
    for name, verdict in result.verdicts.items():
        logging.info("contract %s: %s", name, verdict.value)


if __name__ == "__main__":
    main()
