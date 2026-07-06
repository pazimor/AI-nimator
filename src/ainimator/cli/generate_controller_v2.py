"""CLI entry point — Goal A controller rollout / generation (phase A1).

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
from ainimator.model.controller_rollout import (
    RolloutResult,
    rolloutController,
    rolloutControllerClosedLoop,
)
from ainimator.training.controller_training_v2 import (
    _groundTruthTrajectory,
    encodeTextToPooled,
    loadControllerCheckpoint,
    loadFrozenTextEncoder,
    resolveControllerDevice,
)
from ainimator.training.training_v2 import loadDatasetSample


def _parseArgs() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Goal A controller roll.")
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
        "drive a longer rollout.",
    )
    parser.add_argument(
        "--reinject",
        type=int,
        default=0,
        help="Closed-loop: re-inject ground truth every N frames (models "
        "engine re-grounding). 0 = pure open-loop. Use a small N (e.g. 32) "
        "to keep a long --repeat animation clean.",
    )
    parser.add_argument(
        "--dae-gt",
        type=Path,
        default=None,
        help="Also export the ground-truth base animation of the sample.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Temporal Gaussian smoothing sigma (frames) applied to the "
        "rollout before export. Softens re-injection snaps / single-frame "
        "outliers. 0 disables. Try 2-3.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional text prompt (e.g. 'a person dances').  When "
        "provided and --encoder-artifact is set, the frozen text encoder "
        "encodes the prompt ONCE upstream of the rollout loop.",
    )
    parser.add_argument(
        "--encoder-artifact",
        dest="encoderArtifact",
        type=Path,
        default=None,
        help="Path to a frozen encoder artifact directory produced by "
        "train_text_encoder.  Required when --prompt is used.",
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
    logging.info("Using default dataset root: %s", fallback)
    return fallback


def _resolvePromptEmb(
    model: object,
    prompt: str | None,
    encoderArtifact: Path | None,
    device: torch.device,
) -> "torch.Tensor | None":
    """Encode prompt text to ``(1, D)`` pooled embedding or return None.

    Returns None only when no prompt is requested (a legitimate
    unconditioned rollout). When a prompt IS given, any condition that
    would silently drop it — a checkpoint trained without text
    conditioning (``promptEmbChannels == 0``) or a missing encoder
    artifact — raises an explicit error instead of falling back to the
    null embedding (A6-bis: no silent prompt ignore).

    The encoder is loaded, runs one forward pass, then discarded — only
    the embedding tensor is kept for the rollout loop.
    """
    if prompt is None or not prompt.strip():
        return None
    promptEmbChannels = getattr(
        getattr(model, "config", None), "promptEmbChannels", 0
    )
    if promptEmbChannels == 0:
        raise SystemExit(
            f"--prompt {prompt!r} given but the checkpoint was trained "
            "without text conditioning (promptEmbChannels=0). Re-train "
            "with a text-enabled profile or drop --prompt."
        )
    if encoderArtifact is None:
        raise SystemExit(
            f"--prompt {prompt!r} given but --encoder-artifact is "
            "missing; it is required to encode the prompt."
        )
    encoder, tokenizer = loadFrozenTextEncoder(encoderArtifact, device)
    pooled = encodeTextToPooled([prompt], encoder, tokenizer, device)
    logging.info("prompt encoded: %r  dim=%d", prompt, pooled.shape[-1])
    return pooled  # (1, D)


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

    # Encode the prompt ONCE upstream of the autoregressive loop.
    promptEmb = _resolvePromptEmb(
        model, args.prompt, args.encoderArtifact, device
    )

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
    rootTranslation = sample.rootTranslation.to(device)
    gtBone, gtRoot = _groundTruthTrajectory(batch, rootTranslation)
    rollout = _generate(
        model, stateNorm, deltaNorm, batch, rootTranslation,
        controlSequence, phaseSequence,
        gtBone, gtRoot, model.config.contextFrames, repeat, int(args.reinject),
        promptEmb=promptEmb,
    )
    rollout = _smoothRollout(rollout, float(args.smooth))
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
    if args.dae_gt is not None:
        gtLocalMotion = torch.zeros(
            gtRoot.shape[0], gtRoot.shape[1], 4, device=gtRoot.device
        )
        _exportRolloutDae(
            RolloutResult(
                rotation6d=gtBone,
                rootTranslation=gtRoot,
                rootLocalMotion=gtLocalMotion,
            ),
            args.dae_gt, args.fps,
        )
        logging.info("ground-truth base animation -> %s", args.dae_gt)


def _smoothRollout(rollout: RolloutResult, sigma: float) -> RolloutResult:
    """Temporally Gaussian-smooth the rollout rotations (no-op if sigma<=0)."""
    if sigma <= 0.0:
        return rollout
    from ainimator.model.postprocess_v2 import smoothRotation6dTemporal

    smoothed = smoothRotation6dTemporal(rollout.rotation6d[0], sigma)
    return RolloutResult(
        rotation6d=smoothed.unsqueeze(0),
        rootTranslation=rollout.rootTranslation,
        rootLocalMotion=rollout.rootLocalMotion,
    )


def _generate(
    model: object,
    stateNorm: object,
    deltaNorm: object,
    batch: object,
    clipRootTranslation: torch.Tensor,
    controlSequence: torch.Tensor,
    phaseSequence: torch.Tensor | None,
    gtBone: torch.Tensor,
    gtRoot: torch.Tensor,
    contextFrames: int,
    repeat: int,
    reinject: int,
    promptEmb: torch.Tensor | None = None,
) -> RolloutResult:
    """Open-loop, or closed-loop (GT re-injection) when ``reinject > 0``."""
    if reinject <= 0:
        seedRootTranslation = clipRootTranslation[:contextFrames].unsqueeze(0)
        return rolloutController(
            model, stateNorm, deltaNorm,
            batch.boneWindow[:1],  # type: ignore[attr-defined]
            seedRootTranslation,
            controlSequence,
            phaseSequence=phaseSequence,
            seedRootLocalMotion=batch.globalWindow[:1],  # type: ignore[attr-defined]
            promptEmb=promptEmb,
        )
    tiledBone, tiledRoot = _tileGroundTruth(
        gtBone, gtRoot, contextFrames, repeat
    )
    return rolloutControllerClosedLoop(
        model, stateNorm, deltaNorm, tiledBone, tiledRoot, controlSequence,
        reinject, phaseSequence=phaseSequence, promptEmb=promptEmb,
    )


def _tileGroundTruth(
    gtBone: torch.Tensor,
    gtRoot: torch.Tensor,
    contextFrames: int,
    repeat: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat the GT targets ``repeat`` times after the seed window."""
    seedBone, targetBone = gtBone[:, :contextFrames], gtBone[:, contextFrames:]
    seedRoot, targetRoot = gtRoot[:, :contextFrames], gtRoot[:, contextFrames:]
    bone = torch.cat(
        [seedBone, targetBone.repeat(1, repeat, 1, 1)], dim=1
    )
    root = torch.cat([seedRoot, targetRoot.repeat(1, repeat, 1)], dim=1)
    return bone, root


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
