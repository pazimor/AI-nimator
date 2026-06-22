"""CLI entry point — Goal C controller rollout / generation (phase C1).

Zero logic: load a trained controller checkpoint, derive the
ground-truth control for a dataset sample, roll the controller forward
under that control and save the trajectory.

Example
-------
``python -m ainimator.cli.generate_controller_v2 \\
    --checkpoint output/controller_overfit/checkpoints/...pt \\
    --dataset-root <preprocessed> --sample-index 0 \\
    --output output/controller_rollout.pt``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from ainimator.core.config_loader import defaultPreprocessedDatasetRoot
from ainimator.core.constants.controller import PhaseMode
from ainimator.data.controller_sequences import (
    ControllerSequenceConfig,
    buildControllerSequences,
)
from ainimator.model.controller_rollout import rolloutController
from ainimator.training.controller_training_v2 import (
    loadControllerCheckpoint,
    resolveControllerDevice,
)
from ainimator.training.training_v2 import loadDatasetSample


def _parseArgs() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Goal C controller roll.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Preprocessed dataset root. Defaults to output-root from "
        "src/configs/preprocess_dataset.yaml when omitted.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dae",
        type=Path,
        default=None,
        help="Optional .dae (Collada) path. When set, the rollout is "
        "also exported as an animation viewable in Blender.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Loop the GT-derived control (and phase) this many times to "
        "drive a longer rollout — tests long-horizon stability.",
    )
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
    """Roll the controller forward under a sample's GT control."""
    logging.basicConfig(level=logging.INFO)
    args = _parseArgs()
    device = resolveControllerDevice(args.device)
    model, stateNorm, deltaNorm, controlMean, controlStd = (
        loadControllerCheckpoint(args.checkpoint, device)
    )
    model = model.to(device).eval()
    stateNorm = stateNorm.to(device)
    deltaNorm = deltaNorm.to(device)

    datasetRoot = _resolveDatasetRoot(args.dataset_root)
    sample = loadDatasetSample(datasetRoot, args.sample_index)
    needPhase = model.config.phaseMode is not PhaseMode.NONE
    sequenceConfig = ControllerSequenceConfig(
        contextFrames=model.config.contextFrames,
        useAimDirection=model.config.useAimDirection,
        emitPhase=needPhase,
    )
    batch = buildControllerSequences(
        sample.rotation6d.to(device),
        sample.rootTranslation.to(device),
        sequenceConfig,
    )
    controlNorm = (batch.control - controlMean.to(device)) / controlStd.to(
        device
    )
    repeat = max(1, int(args.repeat))
    controlSequence = controlNorm.repeat(repeat, 1).unsqueeze(0)
    phaseSequence = (
        None
        if batch.phase is None
        else batch.phase.repeat(repeat, 1).unsqueeze(0)
    )
    rollout = rolloutController(
        model,
        stateNorm,
        deltaNorm,
        batch.boneWindow[:1],
        batch.globalWindow[:1],
        controlSequence,
        phaseSequence=phaseSequence,
    )
    torch.save(
        {
            "rotation6d": rollout.rotation6d.detach().cpu(),
            "rootTranslation": rollout.rootTranslation.detach().cpu(),
        },
        args.output,
    )
    logging.info("rollout saved to %s", args.output)
    if args.dae is not None:
        _exportRolloutDae(rollout, args.dae, args.fps)


def _exportRolloutDae(rollout: object, daePath: Path, fps: int) -> None:
    """Export a single-clip rollout to a Blender-viewable .dae file.

    Reuses the v2 rotation6d → Collada converter so the deterministic
    rollout renders with the exact same skeleton/axis conventions as the
    diffusion exporter.
    """
    from ainimator.cli.generate_animation_v2 import (
        _buildAnimationSampleV2,
        _buildRebuilder,
    )

    rotation6d = rollout.rotation6d[0].detach().cpu()  # (F, 22, 6)
    rootTranslation = rollout.rootTranslation[0].detach().cpu()  # (F, 3)
    sample = _buildAnimationSampleV2(
        boneRotation6d=rotation6d,
        rootTranslation=rootTranslation,
        fps=fps,
        outputPath=daePath,
        extras={"engine": "controller_v2"},
    )
    daePath.parent.mkdir(parents=True, exist_ok=True)
    _buildRebuilder(daePath).exportCollada(sample, daePath)
    logging.info("rollout .dae exported to %s", daePath)


if __name__ == "__main__":
    main()
