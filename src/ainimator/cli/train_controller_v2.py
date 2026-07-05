"""CLI — Goal A controller training (A1/A6/A7).

Config-first entry point (G-PROFILES): a run is driven entirely by a
named *profile* loaded from ``src/configs/network.yaml``.  No
hyperparameter flags are exposed — only the three non-config
per-invocation args (I/O paths + resume) and ``--profile``.

Profiles
--------
* ``overfit``  — single-clip overfit, smoke-test canonical (A1, < 2 min).
* ``full``     — train/held-out split, generalization run (A6/A7).
* ``debug``    — same arch as overfit, 10 epochs; CI/pre-commit gate.

Dispatches to:

* :func:`ainimator.training.controller_training_v2.runControllerOverfit`
  for ``overfit`` / ``debug``.
* :func:`ainimator.training.controller_generalization_v2\
.runControllerGeneralization` for ``full``.

Zero logic lives here (G-PROFILES, G-COMMITS).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

from ainimator.core.config_loader import (
    defaultPreprocessedDatasetRoot,
    loadControllerProfile,
)
from ainimator.core.constants.controller import PhaseMode
from ainimator.model.losses_controller_v2 import ControllerLossWeights
from ainimator.training.controller_training_v2 import (
    ControllerTrainingConfig,
    runControllerOverfit,
)

LOGGER = logging.getLogger("ainimator.cli.train_controller_v2")

_OVERFIT_PROFILES = ("overfit", "debug", "controller_text")
#: Any profile not in :data:`_OVERFIT_PROFILES` is dispatched to the
#: generalization (train/held-out) loop and trained *as named* (``full``,
#: ``full-long``, …) -- see :func:`_runFull`.


def _parseArgs() -> argparse.Namespace:
    """Parse the minimal CLI surface (profile + I/O paths)."""
    parser = argparse.ArgumentParser(
        description=(
            "Goal A controller training — config-first "
            "(G-PROFILES, ROADMAP_DETERMINIST §3.2)."
        )
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="overfit",
        help=(
            "Named training profile from controller_profiles in "
            "src/configs/network.yaml.  Default: overfit."
        ),
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
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Where to write checkpoints and resolved_config.yaml.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Warm-start from a controller checkpoint (.pt).  "
            "Architecture and normalization come from it."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Override the default network.yaml path.  "
            "Useful for testing alternate configs."
        ),
    )
    parser.add_argument(
        "--encoder-artifact",
        dest="encoderArtifact",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Path to a frozen text encoder artifact directory.  "
            "Required when the profile has prompt-emb-channels > 0."
        ),
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


def _loadDatasetSample(
    datasetRoot: Path, sampleIndex: int
) -> tuple[object, object, str]:
    """Load a single clip from the dataset."""
    from ainimator.training.training_v2 import loadDatasetSample

    sample = loadDatasetSample(datasetRoot, sampleIndex)
    return sample.rotation6d, sample.rootTranslation, sample.rawText or ""


def _loadCaptionsByTextId(
    datasetRoot: Path, textIds: set[int]
) -> dict[int, str]:
    """Return ``{textId: raw_text}`` for the requested text ids only.

    Reads ``manifest.json`` + ``text_index.json`` to resolve each
    requested ``textId`` to its shard/offset, then loads every needed
    text shard exactly once (grouped by shard index) rather than the
    whole ``text_shards/`` directory.
    """
    import torch

    manifest = json.loads(
        (datasetRoot / "manifest.json").read_text(encoding="utf-8")
    )
    textIndexPath = datasetRoot / manifest.get(
        "textIndexPath", "text_index.json"
    )
    textEntries = json.loads(textIndexPath.read_text(encoding="utf-8"))

    neededByShard: dict[int, list[tuple[int, int]]] = {}
    for position, entry in enumerate(textEntries):
        textId = int(entry.get("textId", position))
        if textId not in textIds:
            continue
        shardIndex = int(entry["shardIndex"])
        shardOffset = int(entry["shardOffset"])
        neededByShard.setdefault(shardIndex, []).append(
            (textId, shardOffset)
        )

    captions: dict[int, str] = {}
    for shardIndex, entries in neededByShard.items():
        shardPath = datasetRoot / manifest["textShards"][shardIndex]["path"]
        shard = torch.load(shardPath, map_location="cpu", weights_only=False)
        for textId, shardOffset in entries:
            captions[textId] = str(shard[shardOffset].get("raw_text", ""))
    return captions


def _filterLinksByCaption(
    datasetRoot: Path,
    links: list[dict[str, object]],
    captionFilter: str,
) -> list[dict[str, object]]:
    """Keep only links whose caption matches ``captionFilter`` (regex).

    The regex is matched case-insensitively against each link's
    ``raw_text`` (resolved once via :func:`_loadCaptionsByTextId`).
    """
    pattern = re.compile(captionFilter, re.IGNORECASE)
    textIds = {int(link["textId"]) for link in links}
    captions = _loadCaptionsByTextId(datasetRoot, textIds)
    return [
        link
        for link in links
        if pattern.search(captions.get(int(link["textId"]), ""))
    ]


def _selectFullIndices(
    datasetRoot: Path,
    numClips: int,
    heldOutClips: int,
    minFrames: int,
    seed: int,
    captionFilter: str | None = None,
) -> tuple[list[int], list[int]]:
    """Deterministic train / held-out split for the ``full`` profile.

    Parameters
    ----------
    captionFilter : str or None
        Optional regex (case-insensitive, ``re.search``) applied to each
        candidate clip's caption (``raw_text``).  Clips whose caption
        does not match are excluded before the frame-count sort and
        ``numClips`` truncation.  ``None`` disables filtering (default,
        unchanged behaviour for the ``full`` / ``full-long`` profiles).
    """
    linkIndexPath = datasetRoot / "link_index.json"
    links = json.loads(linkIndexPath.read_text(encoding="utf-8"))
    eligible = [
        link for link in links if link["frames"] >= minFrames
    ]
    if captionFilter:
        eligible = _filterLinksByCaption(datasetRoot, eligible, captionFilter)
    eligible.sort(key=lambda link: link["frames"], reverse=True)
    chosen = [int(link["linkId"]) for link in eligible[:numClips]]
    if len(chosen) < heldOutClips + 2:
        raise SystemExit(
            f"Only {len(chosen)} clips with >= {minFrames} frames"
            f"{' matching caption-filter' if captionFilter else ''}; "
            f"need >= held-out ({heldOutClips}) + 2 train clips."
        )
    random.Random(seed).shuffle(chosen)
    return chosen[heldOutClips:], chosen[:heldOutClips]


def _loadClipsList(
    datasetRoot: Path, indices: list[int]
) -> tuple[list[tuple[object, object]], list[str]]:
    """Load (rotation6d, rootTranslation) and rawText for each link index."""
    from ainimator.training.training_v2 import loadDatasetSample

    clips = []
    texts = []
    for index in indices:
        sample = loadDatasetSample(datasetRoot, index)
        clips.append((sample.rotation6d, sample.rootTranslation))
        texts.append(sample.rawText or "")
    return clips, texts


def _runOverfit(
    profileName: str,
    args: argparse.Namespace,
    datasetRoot: Path,
) -> None:
    """Dispatch to the single-clip overfit loop."""
    profile = loadControllerProfile(profileName, args.config)
    arch = profile.arch
    training = profile.training
    data = profile.data
    text = profile.text

    rotation6d, rootTranslation, rawText = _loadDatasetSample(
        datasetRoot, data.sampleIndex
    )
    encoderArtifactPath = (
        args.encoderArtifact
        or (Path(text.encoderArtifactPath) if text.encoderArtifactPath else None)
    )
    config = ControllerTrainingConfig(
        outputDir=args.output_dir,
        epochs=training.epochs,
        learningRate=training.learningRate,
        weightDecay=training.weightDecay,
        embedDim=arch.embedDim,
        numHeads=arch.numHeads,
        numLayers=arch.numLayers,
        contextFrames=arch.contextFrames,
        phaseMode=PhaseMode(arch.phase),
        useAimDirection=arch.aimDirection,
        seed=training.seed,
        device=training.device,
        logEvery=training.logEvery,
        lossWeights=ControllerLossWeights(
            velocity=1.0,
            geodesic=1.0,
            footContact=training.footContactWeight,
        ),
        scheduledSampling=training.scheduledSampling,
        rolloutLossHorizon=training.rolloutLossHorizon,
        rolloutLossWeight=training.rolloutLossWeight,
        resumeCheckpoint=args.resume,
        promptEmbChannels=text.promptEmbChannels,
        condDropoutProb=text.condDropoutProb,
        encoderArtifactPath=encoderArtifactPath,
    )
    result = runControllerOverfit(
        rotation6d, rootTranslation, config, clipRawText=rawText
    )
    LOGGER.info("final loss: %.6f", result.finalLoss)
    for name, verdict in result.verdicts.items():
        LOGGER.info("contract %s: %s", name, verdict.value)


def _runFull(args: argparse.Namespace, datasetRoot: Path) -> None:
    """Dispatch to the generalization (train/held-out split) loop."""
    from ainimator.training.controller_generalization_v2 import (
        runControllerGeneralization,
    )

    # Honour the requested profile (e.g. ``full-long``).  Previously this
    # loaded a hard-coded ``"full"`` unconditionally, so ``--profile
    # full-long`` silently trained ``full`` (smaller arch, and — before the
    # rollout-loss field existed — no exposure-bias correction at all).
    profile = loadControllerProfile(args.profile, args.config)
    arch = profile.arch
    training = profile.training
    data = profile.data
    text = profile.text

    trainIndices, heldOutIndices = _selectFullIndices(
        datasetRoot,
        data.numClips,
        data.heldOutClips,
        data.minFrames,
        training.seed,
        captionFilter=data.captionFilter,
    )
    LOGGER.info(
        "Split: %d train / %d held-out clips",
        len(trainIndices),
        len(heldOutIndices),
    )
    trainClips, trainTexts = _loadClipsList(datasetRoot, trainIndices)
    heldOutClips, heldOutTexts = _loadClipsList(datasetRoot, heldOutIndices)

    encoderArtifactPath = (
        args.encoderArtifact
        or (Path(text.encoderArtifactPath) if text.encoderArtifactPath else None)
    )
    config = ControllerTrainingConfig(
        outputDir=args.output_dir,
        epochs=training.epochs,
        learningRate=training.learningRate,
        weightDecay=training.weightDecay,
        embedDim=arch.embedDim,
        numHeads=arch.numHeads,
        numLayers=arch.numLayers,
        contextFrames=arch.contextFrames,
        phaseMode=PhaseMode(arch.phase),
        useAimDirection=arch.aimDirection,
        seed=training.seed,
        device=training.device,
        logEvery=training.logEvery,
        lossWeights=ControllerLossWeights(
            velocity=1.0,
            geodesic=1.0,
            footContact=training.footContactWeight,
        ),
        scheduledSampling=training.scheduledSampling,
        rolloutLossHorizon=training.rolloutLossHorizon,
        rolloutLossWeight=training.rolloutLossWeight,
        resumeCheckpoint=args.resume,
        promptEmbChannels=text.promptEmbChannels,
        condDropoutProb=text.condDropoutProb,
        encoderArtifactPath=encoderArtifactPath,
    )
    result = runControllerGeneralization(
        trainClips,
        heldOutClips,
        config,
        clipBatchSize=data.clipBatchSize,
        evalSampleClips=data.evalSampleClips,
        trainClipTexts=trainTexts if text.promptEmbChannels > 0 else None,
        heldOutClipTexts=(
            heldOutTexts if text.promptEmbChannels > 0 else None
        ),
    )
    LOGGER.info("final train loss: %.6f", result.finalLoss)
    LOGGER.info("train  metrics: %s", result.trainMetrics)
    LOGGER.info("HELD-OUT metrics: %s", result.heldOutMetrics)
    for name, verdict in result.verdicts.items():
        LOGGER.info("contract %s: %s", name, verdict.value)


def main() -> None:
    """Entry point for controller training (config-first, G-PROFILES)."""
    logging.basicConfig(level=logging.INFO)
    args = _parseArgs()
    datasetRoot = _resolveDatasetRoot(args.dataset_root)

    if args.profile in _OVERFIT_PROFILES:
        _runOverfit(args.profile, args, datasetRoot)
    else:
        _runFull(args, datasetRoot)


if __name__ == "__main__":
    main()
