"""CLI entry-point for generating motion from a v2 checkpoint.

The script loads a checkpoint produced by
``src.cli.train_generation_v2`` and runs DDIM sampling to produce a
batch of motion tensors which are then exported as a Collada (.dae)
animation, matching the format used by the v1 generation pipeline.

The v2 stack predicts ``rotation6d`` for the 22 SMPL bones plus a
3-channel ``root_translation``.  This module:

* Converts the rotation6d output to axis-angle via the canonical
  :class:`Rotation` utility (same path used by the v1 pipeline so the
  numerical behaviour is identical).
* Pads the 22-bone SMPL skeleton up to 24 bones (the SMPL-24 layout
  expected by :class:`AnimationRebuilder`) with zero rotations on the
  hand-tip joints.
* Reuses :meth:`AnimationRebuilder.exportCollada` for the actual XML
  serialisation, which keeps the bone hierarchy, default offsets and
  matrix conventions in sync with the v1 exporter.

Usage example
-------------
.. code-block:: bash

    poetry run python -m src.cli.generate_animation_v2 \\
        --checkpoint output/generation_v2/v2_overfit_checkpoint.pt \\
        --prompt "a person walks forward." \\
        --frames 120 \\
        --num-steps 100 \\
        --cfg-scale 3.5 \\
        --output output/generation_v2/sample.dae \\
        --fps 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.features.dataset_builder.animation_rebuilder import (
    AnimationRebuilder,
)
from src.features.generation.training_v2 import (
    EMPTY_PROMPT,
    loadCheckpointV2,
    resolveDevice,
)
from src.shared.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL24_BONE_ORDER,
)
from src.shared.model.generation.postprocess_v2 import (
    dampFootMotionDuringContact,
    smoothRootYaw,
    smoothRotation6dTemporal,
)
from src.shared.model.generation.sampler_v2 import DDIMSamplerV2
from src.shared.quaternion import Rotation
from src.shared.types import (
    AnimationSample,
    DatasetBuilderConfig,
    DatasetBuilderPaths,
    DatasetBuilderProcessing,
)

LOGGER = logging.getLogger(__name__)

AXIS_ANGLE_CHANNELS = 3
SMPL24_BONE_COUNT = len(SMPL24_BONE_ORDER)


# ---------------------------------------------------------------------
# Skeleton mapping
# ---------------------------------------------------------------------
def _mapAxisAnglesToSmpl24(
    axisAngles: np.ndarray,
    boneOrder: Sequence[str],
) -> np.ndarray:
    """Pad a 22-bone axis-angle tensor to the SMPL-24 layout.

    Bones not present in ``boneOrder`` (the two SMPL hand-tip joints)
    receive identity rotations.  This mirrors
    :func:`src.features.generation.generate_animation.mapAxisAnglesToSmpl24`
    so the v2 export stays binary-compatible with the v1 pipeline.
    """
    boneCount = axisAngles.shape[1]
    if boneCount != len(boneOrder):
        raise ValueError(
            f"Bone count mismatch: {boneCount} != {len(boneOrder)}."
        )
    frameCount = axisAngles.shape[0]
    target = np.zeros(
        (frameCount, SMPL24_BONE_COUNT, AXIS_ANGLE_CHANNELS),
        dtype=np.float32,
    )
    smpl24Index = {
        boneName: index for index, boneName in enumerate(SMPL24_BONE_ORDER)
    }
    for sourceIndex, boneName in enumerate(boneOrder):
        targetIndex = smpl24Index.get(boneName)
        if targetIndex is None:
            raise KeyError(
                f"Bone {boneName!r} from the v2 output is not part of the "
                f"SMPL-24 ordering."
            )
        target[:, targetIndex, :] = axisAngles[:, sourceIndex, :]
    return target


# ---------------------------------------------------------------------
# Tensor → AnimationSample
# ---------------------------------------------------------------------
def _buildAnimationSampleV2(
    boneRotation6d: torch.Tensor,
    rootTranslation: torch.Tensor | None,
    fps: int,
    outputPath: Path,
    extras: dict[str, object],
) -> AnimationSample:
    """Convert v2 sampler output to an :class:`AnimationSample`.

    Parameters
    ----------
    boneRotation6d : torch.Tensor
        Rotation-6D tensor from the sampler shaped
        ``(frames, numBones=22, 6)``.
    rootTranslation : torch.Tensor or None
        Per-frame root translation of shape ``(frames, 3)``.  When
        provided it is stored under ``extras["trans"]`` so
        :meth:`AnimationRebuilder.exportCollada` can apply it to the
        pelvis.
    fps : int
        Frame rate written to the Collada asset block.
    outputPath : Path
        Path of the eventual ``.dae`` file (used as the sample source
        identifier; not written to here).
    extras : dict
        Extra metadata to attach (prompt, checkpoint, sampler params,
        etc.).
    """
    if boneRotation6d.ndim != 3:
        raise ValueError(
            "boneRotation6d must be 3-D (frames, bones, 6); got shape "
            f"{tuple(boneRotation6d.shape)}."
        )
    if boneRotation6d.shape[-1] != 6:
        raise ValueError(
            "Rotation-6D tensor must end with a 6-channel axis; got "
            f"shape {tuple(boneRotation6d.shape)}."
        )
    if boneRotation6d.shape[1] != len(SMPL22_BONE_ORDER):
        raise ValueError(
            f"Expected {len(SMPL22_BONE_ORDER)} bones, got "
            f"{boneRotation6d.shape[1]}."
        )

    rotation = boneRotation6d.detach().to(torch.float32).cpu()
    # rot6d (F, 22, 6) → quat (F, 22, 4) → axis-angle (F, 22, 3).
    axisAngles = (
        Rotation(rotation, kind="rot6d").axis_angle.numpy().astype(np.float32)
    )
    axisAnglesFull = _mapAxisAnglesToSmpl24(axisAngles, SMPL22_BONE_ORDER)
    flatAngles = axisAnglesFull.reshape(axisAnglesFull.shape[0], -1)

    extrasOut = dict(extras)
    if rootTranslation is not None:
        if rootTranslation.shape != (boneRotation6d.shape[0], 3):
            raise ValueError(
                "rootTranslation shape "
                f"{tuple(rootTranslation.shape)} does not match "
                f"({boneRotation6d.shape[0]}, 3)."
            )
        extrasOut["trans"] = (
            rootTranslation.detach().to(torch.float32).cpu().tolist()
        )

    return AnimationSample(
        relativePath=outputPath,
        resolvedPath=outputPath.resolve(),
        axisAngles=flatAngles,
        fps=int(fps),
        extras=extrasOut,
    )


def _buildRebuilder(outputPath: Path) -> AnimationRebuilder:
    """Build the minimal :class:`AnimationRebuilder` we need for export.

    The exporter only consumes the rotation/translation tensors we hand
    it via :class:`AnimationSample`; the dataset paths are unused at
    export time but the dataclass requires them, so we point everything
    to ``outputPath.parent`` to keep the contract satisfied.
    """
    parentDir = outputPath.parent.resolve()
    paths = DatasetBuilderPaths(
        animationRoot=parentDir,
        promptRoot=parentDir,
        promptSources=[parentDir],
        indexCsv=parentDir / "missing-index.csv",
        outputRoot=parentDir,
    )
    config = DatasetBuilderConfig(
        paths=paths,
        processing=DatasetBuilderProcessing(),
    )
    return AnimationRebuilder(config)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def buildArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate motion from a v2 checkpoint and export to .dae."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the v2 checkpoint .pt file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt used to condition the generation.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of motion frames to generate. (default: %(default)s)",
    )
    parser.add_argument(
        "--num-steps",
        dest="numSteps",
        type=int,
        default=100,
        help=(
            "Number of DDIM sampling steps (50–200 is typical). "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--cfg-scale",
        dest="cfgScale",
        type=float,
        default=3.5,
        help=(
            "Classifier-free guidance scale; 1.0 disables CFG. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity (0=deterministic). (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for the initial noise. (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Torch device — 'auto' (mps→cuda→cpu), 'cpu', 'mps', 'cuda'. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output .dae file (Collada animation).  A sibling .json "
            "metadata file is written next to it."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help=(
            "Frame rate written into the .dae asset block. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--colladaInterpolation",
        dest="colladaInterpolation",
        type=str,
        default="linear",
        choices=["linear", "step"],
        help="Collada keyframe interpolation. (default: %(default)s)",
    )
    parser.add_argument(
        "--zeroRootTranslation",
        dest="zeroRootTranslation",
        action="store_true",
        help=(
            "Force pelvis translation to zero (useful for in-place "
            "inspection)."
        ),
    )
    parser.add_argument(
        "--anchorRootTranslation",
        dest="anchorRootTranslation",
        action="store_true",
        help=(
            "Subtract the first-frame translation from the trajectory "
            "so the animation starts at the origin."
        ),
    )
    parser.add_argument(
        "--smooth-sigma",
        dest="smoothSigma",
        type=float,
        default=0.0,
        help=(
            "2026-06-05 — Gaussian sigma (in frames) applied to ALL "
            "rotation6d channels before export.  Removes the high-"
            "frequency frame-to-frame jitter ('trembling': generated "
            "Δ²/frame is ~30× the AMASS reference).  0.0 disables; "
            "~2.0 matches real-motion smoothness while preserving the "
            "motion. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--yawSmoothSigma",
        dest="yawSmoothSigma",
        type=float,
        default=0.0,
        help=(
            "Phase 4.2 — Gaussian sigma (in frames) applied to the "
            "pelvis yaw signal before export.  0.0 disables.  1.5–3.0 "
            "kills high-frequency spinning without flattening real "
            "turns. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--footDampThreshold",
        dest="footDampThreshold",
        type=float,
        default=0.0,
        help=(
            "Phase 4.1 — foot speed (m/frame) below which the foot is "
            "treated as in contact.  Triggers a root-translation "
            "counter-correction to suppress foot-skating.  0.0 "
            "disables.  0.03–0.05 is the typical range. (default: "
            "%(default)s)"
        ),
    )
    parser.add_argument(
        "--footDampBlend",
        dest="footDampBlend",
        type=float,
        default=0.6,
        help=(
            "Phase 4.1 — blend ratio for the foot-contact root "
            "correction (0 = off, 1 = full cancellation). Only used "
            "when --footDampThreshold > 0. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = buildArgumentParser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.logLevel),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not arguments.checkpoint.exists():
        LOGGER.error("Checkpoint file not found: %s", arguments.checkpoint)
        return 1
    outputPath: Path = arguments.output
    if outputPath.suffix.lower() != ".dae":
        LOGGER.warning(
            "Output suffix is %s; the v2 exporter writes Collada — "
            "consider using .dae for clarity.",
            outputPath.suffix,
        )

    device = resolveDevice(arguments.device)
    LOGGER.info("Loading checkpoint %s on %s.", arguments.checkpoint, device)
    (
        tokenizer,
        encoder,
        denoiser,
        schedule,
        normalizer,
        payload,
    ) = loadCheckpointV2(arguments.checkpoint, device=device)
    encoder.eval()
    denoiser.eval()

    sampler = DDIMSamplerV2(
        schedule,
        predictionMode=str(payload["training_config"]["predictionMode"]),
    )

    encodedCond = tokenizer.encode(arguments.prompt)
    inputIdsCond = encodedCond.inputIds.to(device)
    attentionMaskCond = encodedCond.attentionMask.to(device)
    condOutput = encoder(inputIdsCond, attentionMaskCond)

    uncondOutput = None
    if arguments.cfgScale != 1.0:
        if encoder.nullEmbedding is not None:
            uncondOutput = encoder.forwardNull(batchSize=1, device=device)
            LOGGER.info("Using learnable null embedding for CFG uncond branch.")
        else:
            encodedUncond = tokenizer.encode(EMPTY_PROMPT)
            inputIdsUncond = encodedUncond.inputIds.to(device)
            attentionMaskUncond = encodedUncond.attentionMask.to(device)
            uncondOutput = encoder(inputIdsUncond, attentionMaskUncond)

    LOGGER.info(
        "Sampling %d frame(s) over %d DDIM steps (cfg=%.2f, eta=%.2f).",
        arguments.frames,
        arguments.numSteps,
        arguments.cfgScale,
        arguments.eta,
    )
    output = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=condOutput.hiddenStates,
        textKeyPaddingMask=condOutput.keyPaddingMask,
        unconditionalTextHiddenStates=(
            uncondOutput.hiddenStates if uncondOutput is not None else None
        ),
        unconditionalTextKeyPaddingMask=(
            uncondOutput.keyPaddingMask if uncondOutput is not None else None
        ),
        frames=int(arguments.frames),
        numSteps=int(arguments.numSteps),
        cfgScale=float(arguments.cfgScale),
        eta=float(arguments.eta),
        device=device,
        seed=arguments.seed,
        normalizer=normalizer,
    )

    extras: dict[str, object] = {
        "prompt": arguments.prompt,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "ddim_steps": int(arguments.numSteps),
        "cfg_scale": float(arguments.cfgScale),
        "eta": float(arguments.eta),
        "seed": arguments.seed,
        "fps": int(arguments.fps),
        "trainingPrompt": payload.get("training_sample", {}).get("rawText"),
    }

    # Phase 4 — post-processing.  Both filters short-circuit when their
    # threshold is 0 so this block is a no-op by default.
    boneRotation6dOut = output.boneMotion[0]
    rootTranslationOut = (
        output.globalMotion[0] if output.globalMotion is not None else None
    )

    if float(arguments.smoothSigma) > 0.0:
        boneRotation6dOut = smoothRotation6dTemporal(
            boneRotation6dOut, sigma=float(arguments.smoothSigma)
        )
        LOGGER.info(
            "Post-proc: temporal rotation6d smoothing (sigma=%.2f "
            "frames) — removes frame-to-frame jitter.",
            float(arguments.smoothSigma),
        )

    if float(arguments.yawSmoothSigma) > 0.0:
        boneRotation6dOut = smoothRootYaw(
            boneRotation6dOut, sigma=float(arguments.yawSmoothSigma)
        )
        LOGGER.info(
            "Post-proc: smoothed pelvis yaw (sigma=%.2f frames).",
            float(arguments.yawSmoothSigma),
        )

    if (
        float(arguments.footDampThreshold) > 0.0
        and rootTranslationOut is not None
    ):
        rootTranslationOut, footReport = dampFootMotionDuringContact(
            rotation6d=boneRotation6dOut,
            rootTranslation=rootTranslationOut,
            velocityThreshold=float(arguments.footDampThreshold),
            blendAlpha=float(arguments.footDampBlend),
        )
        LOGGER.info(
            "Post-proc: foot-damp contacts per foot=%s, total root "
            "correction L1=%.3f m.",
            footReport.contactFramesPerFoot,
            footReport.rootCorrectionNorm,
        )
        extras["postproc_foot_contacts"] = list(
            footReport.contactFramesPerFoot
        )
        extras["postproc_root_correction"] = float(
            footReport.rootCorrectionNorm
        )

    sample = _buildAnimationSampleV2(
        boneRotation6d=boneRotation6dOut,
        rootTranslation=rootTranslationOut,
        fps=arguments.fps,
        outputPath=outputPath,
        extras=extras,
    )

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    rebuilder = _buildRebuilder(outputPath)
    rebuilder.exportCollada(
        sample,
        outputPath,
        interpolation=arguments.colladaInterpolation,
        zeroRootTranslation=bool(arguments.zeroRootTranslation),
        anchorRootTranslation=bool(arguments.anchorRootTranslation),
    )
    LOGGER.info("Animation saved to %s.", outputPath)

    metadataPath = outputPath.with_suffix(".json")
    metadataPath.write_text(
        json.dumps(extras, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info("Metadata saved to %s.", metadataPath)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
