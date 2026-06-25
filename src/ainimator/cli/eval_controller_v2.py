"""CLI — held-out evaluation of a trained controller (A7).

Replays the deterministic train/held-out split (same seed as the
training run) and computes held-out metrics WITHOUT retraining.
Decouples evaluation from training (ROADMAP_DETERMINIST §3.2 / A7).

Usage
-----
``python -m ainimator.cli.eval_controller_v2 \\
    --checkpoint output/controller/controller_overfit_checkpoint.pt \\
    --dataset-root /path/to/preprocessed \\
    --profile full \\
    --output-dir output/eval_results``

The ``--profile`` flag is used to reproduce the same clip-selection
parameters (numClips, heldOutClips, minFrames, seed) that were used
during training.  The checkpoint carries the model + normalizers;
evaluation uses only the ``data`` and ``training.seed`` fields of the
profile.

Non-config per-invocation args: ``--checkpoint``, ``--dataset-root``,
``--output-dir``.  Zero hyperparameter flags (G-PROFILES).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import argparse

from ainimator.core.config_loader import (
    defaultPreprocessedDatasetRoot,
    loadControllerProfile,
)
from ainimator.core.resolved_config import writeResolvedConfig

LOGGER = logging.getLogger("ainimator.cli.eval_controller_v2")


def _parseArgs() -> argparse.Namespace:
    """Parse the minimal CLI surface for evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Held-out evaluation of a trained controller — "
            "replays the deterministic split without retraining (A7)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="FILE",
        help="Trained controller checkpoint (.pt).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Preprocessed dataset root.  Defaults to output-root from "
            "src/configs/preprocess_dataset.yaml when omitted."
        ),
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["overfit", "full", "debug"],
        metavar="PROFILE",
        help=(
            "Controller profile whose data/seed settings are used to "
            "replay the clip split.  Default: full."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Where to write eval_results.json and resolved_config.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="Override the default network.yaml path.",
    )
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
    LOGGER.info("Using default dataset root: %s", fallback)
    return fallback


def _splitIndices(
    datasetRoot: Path,
    numClips: int,
    heldOutClips: int,
    minFrames: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Deterministic split — identical to the training split (same seed).

    Parameters
    ----------
    datasetRoot : Path
        Preprocessed dataset root (must contain link_index.json).
    numClips : int
        Total clips from which to draw.
    heldOutClips : int
        How many to reserve as held-out.
    minFrames : int
        Minimum frame count filter.
    seed : int
        RNG seed — must match the training run's seed for valid replay.

    Returns
    -------
    tuple[list[int], list[int]]
        (trainIndices, heldOutIndices)
    """
    linkIndexPath = datasetRoot / "link_index.json"
    links = json.loads(linkIndexPath.read_text(encoding="utf-8"))
    eligible = [link for link in links if link["frames"] >= minFrames]
    eligible.sort(key=lambda link: link["frames"], reverse=True)
    chosen = [int(link["linkId"]) for link in eligible[:numClips]]
    if len(chosen) < heldOutClips + 2:
        raise SystemExit(
            f"Only {len(chosen)} clips with >= {minFrames} frames; "
            f"need >= held-out ({heldOutClips}) + 2 train clips."
        )
    random.Random(seed).shuffle(chosen)
    return chosen[heldOutClips:], chosen[:heldOutClips]


def _loadClipsList(
    datasetRoot: Path, indices: list[int]
) -> list[tuple[object, object]]:
    """Load (rotation6d, rootTranslation) for each link index."""
    from ainimator.training.training_v2 import loadDatasetSample

    clips = []
    for index in indices:
        sample = loadDatasetSample(datasetRoot, index)
        clips.append((sample.rotation6d, sample.rootTranslation))
    return clips


def _evalOverfitProfile(
    args: argparse.Namespace,
    datasetRoot: Path,
    profile: object,
) -> None:
    """Evaluate overfit / debug profile — single clip held-out replay."""
    from ainimator.training.controller_generalization_v2 import (
        _buildClipBatches,
        _concatClipBatches,
        _evaluateClips,
        _fitControlStats,
        _fitStateDeltaNormalizers,
        _sequenceConfig,
    )
    from ainimator.training.controller_training_v2 import (
        loadControllerCheckpoint,
        resolveControllerDevice,
    )

    profileData = profile.data  # type: ignore[union-attr]
    profileTraining = profile.training  # type: ignore[union-attr]

    model, stateNorm, deltaNorm, _mean, _std = loadControllerCheckpoint(
        args.checkpoint
    )
    device = resolveControllerDevice(profileTraining.device)
    model = model.to(device)
    stateNorm = stateNorm.to(device)
    deltaNorm = deltaNorm.to(device)

    # Single-clip: load the overfit clip as both "train" and "held-out"
    # for the purpose of metric computation.
    rotation6d, rootTranslation = _loadClipsList(
        datasetRoot, [profileData.sampleIndex]
    )[0]
    clips = [(rotation6d, rootTranslation)]
    seqCfg = _sequenceConfig(
        # Minimal dummy config to access lossWeights.footContact
        _buildDummyConfig(args.output_dir),
        model.config.phaseMode,
        model.config.useAimDirection,
        model.config.contextFrames,
    )
    controlStats = _fitControlStats(clips, seqCfg, device)
    norms = (stateNorm, deltaNorm)
    metrics, _ = _evaluateClips(
        model, clips, seqCfg, norms, controlStats, device
    )
    return metrics


def _buildDummyConfig(outputDir: Path) -> object:
    """Minimal ControllerTrainingConfig with zero loss weights."""
    from ainimator.training.controller_training_v2 import (
        ControllerTrainingConfig,
    )
    from ainimator.model.losses_controller_v2 import ControllerLossWeights

    return ControllerTrainingConfig(
        outputDir=outputDir,
        lossWeights=ControllerLossWeights(
            velocity=1.0, geodesic=1.0, footContact=0.0
        ),
    )


def main() -> None:
    """Entry point for controller held-out evaluation."""
    logging.basicConfig(level=logging.INFO)
    args = _parseArgs()
    datasetRoot = _resolveDatasetRoot(args.dataset_root)
    profile = loadControllerProfile(args.profile, args.config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.profile in ("overfit", "debug"):
        metrics = _evalOverfitProfile(args, datasetRoot, profile)
    else:
        metrics = _evalFullProfile(args, datasetRoot, profile)

    # Persist results.
    resultsPath = args.output_dir / "eval_results.json"
    resultsPath.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    LOGGER.info("Eval results written to %s", resultsPath)
    for key, value in metrics.items():
        LOGGER.info("  %-30s %s", key, value)

    writeResolvedConfig(
        _EvalRunConfig(
            checkpoint=str(args.checkpoint),
            profile=args.profile,
            datasetRoot=str(datasetRoot),
            outputDir=str(args.output_dir),
        ),
        args.output_dir,
    )


def _evalFullProfile(
    args: argparse.Namespace,
    datasetRoot: Path,
    profile: object,
) -> dict[str, float]:
    """Replay the deterministic split and evaluate on held-out clips."""
    from ainimator.training.controller_generalization_v2 import (
        _evaluateClips,
        _fitControlStats,
        _fitStateDeltaNormalizers,
        _sequenceConfig,
    )
    from ainimator.training.controller_training_v2 import (
        loadControllerCheckpoint,
        resolveControllerDevice,
    )

    profileData = profile.data  # type: ignore[union-attr]
    profileTraining = profile.training  # type: ignore[union-attr]

    trainIndices, heldOutIndices = _splitIndices(
        datasetRoot,
        profileData.numClips,
        profileData.heldOutClips,
        profileData.minFrames,
        profileTraining.seed,
    )
    LOGGER.info(
        "Replayed split: %d train / %d held-out clips",
        len(trainIndices),
        len(heldOutIndices),
    )
    trainClips = _loadClipsList(datasetRoot, trainIndices)
    heldOutClips = _loadClipsList(datasetRoot, heldOutIndices)

    model, stateNorm, deltaNorm, _mean, _std = loadControllerCheckpoint(
        args.checkpoint
    )
    device = resolveControllerDevice(profileTraining.device)
    model = model.to(device)
    stateNorm = stateNorm.to(device)
    deltaNorm = deltaNorm.to(device)

    seqCfg = _sequenceConfig(
        _buildDummyConfig(args.output_dir),
        model.config.phaseMode,
        model.config.useAimDirection,
        model.config.contextFrames,
    )
    numBones = int(trainClips[0][0].shape[-2])  # type: ignore[union-attr]
    stateNorm2, deltaNorm2 = _fitStateDeltaNormalizers(trainClips, numBones)
    norms = (stateNorm2.to(device), deltaNorm2.to(device))
    controlStats = _fitControlStats(trainClips, seqCfg, device)

    evalClips = heldOutClips[: max(2, profileData.evalSampleClips)]
    metrics, _ = _evaluateClips(
        model, evalClips, seqCfg, norms, controlStats, device
    )
    return metrics


# Simple dataclass for resolved_config serialization.
from dataclasses import dataclass


@dataclass(frozen=True)
class _EvalRunConfig:
    """Minimal config for the eval resolved_config.yaml.

    Attributes
    ----------
    checkpoint : str
        Path to the evaluated checkpoint.
    profile : str
        Profile name used to replay the split.
    datasetRoot : str
        Dataset root path.
    outputDir : str
        Output directory.
    """

    checkpoint: str
    profile: str
    datasetRoot: str
    outputDir: str


if __name__ == "__main__":
    main()
