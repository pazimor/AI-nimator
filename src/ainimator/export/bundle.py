"""Controller bundle — ONNX + norm_stats + manifest + presets (B0/A7).

A **controller bundle** is the minimal artefact set that a plugin
(Unity Sentis / Unreal NNE) needs to run the controller in production
without any Python dependency:

* ``controller.onnx``       — one-frame forward graph (§2.10 ONNX rules).
* ``norm_stats.json``       — state / delta / control z-norm statistics.
* ``manifest.json``         — I/O contract serialization (§2.2 layout).
* ``resolved_config.yaml``  — provenance: full config + git SHA + date.
* ``presets/``              — optional YAML preset files (one per preset).

The :func:`exportControllerBundle` function is the single entry point.
It delegates the ONNX export to :func:`ainimator.export.onnx.exportController`
and assembles the companion JSON files from the loaded checkpoint data.

Design rules (ROADMAP_DETERMINIST §2.2 / §2.10 / B0)
-----------------------------------------------------
* ``manifest.json`` serialises the **frozen** I/O contract defined in
  §2.2.  It is NOT a second definition — it is a serialization.
* ``norm_stats.json`` carries the z-norm statistics the engine needs to
  normalize inputs and denormalize outputs. ``aim_x/aim_z`` are NOT
  included (they are unit-norm by construction, excluded from z-norm).
* The bundle version is ``"A7.0"`` until the B0 plugin integration
  bump.  A semver-like string keeps the manifest forward-compatible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ainimator.core.constants.controller import (
    LEAN_STATE_CHANNELS,
    NUM_SMPL22_BONES,
    ROOT_LOCAL_MOTION_CHANNELS,
    ROTATION6D_CHANNELS,
    CONTROL_PLANAR_VELOCITY_CHANNELS,
    CONTROL_AIM_DIRECTION_CHANNELS,
    controlSignalChannels,
    phaseConditioningChannels,
)
from ainimator.export.onnx import exportController
from ainimator.model.controller_v2 import MotionController
from ainimator.model.motion_normalizer import MotionNormalizer

LOGGER = logging.getLogger(__name__)

# Bundle format version — bump on breaking manifest changes.
BUNDLE_VERSION = "A7.0"

# Canonical filenames inside the bundle directory.
ONNX_FILENAME = "controller.onnx"
NORM_STATS_FILENAME = "norm_stats.json"
MANIFEST_FILENAME = "manifest.json"
PRESETS_SUBDIR = "presets"


@dataclass(frozen=True)
class BundleManifest:
    """Serialisation of the deterministic I/O contract (§2.2).

    This dataclass is the Python representation of ``manifest.json``.
    It contains ONLY the frozen layout constants; it does NOT duplicate
    the ROADMAP — it serialises it for the engine.

    Attributes
    ----------
    bundleVersion : str
        Bundle format version string (``"A7.0"``).
    stateChannels : int
        Total regressed state channels (136 = 132 rotation6d + 4 root).
    numBones : int
        Number of SMPL bones (22).
    rotationChannelsPerBone : int
        6D rotation channels per bone (6).
    rootLocalMotionChannels : int
        Root-local motion channels (4 = Δfwd/Δlat/Δheight/Δyaw).
    controlChannels : int
        Control signal width (2 or 4 depending on aim-direction flag).
    controlLayout : list[str]
        Ordered control channel names.
    phaseChannels : int
        Phase conditioning channels (0 or 2).
    promptEmbChannels : int
        Prompt embedding channels (0 when text encoder is absent).
    contextFrames : int
        Autoregressive context window length.
    outputLayout : str
        Layout of the ``Δstate`` output (``"bone_delta|global_delta"``).
    coordSystem : str
        Coordinate system convention (``"Y-up right-handed"``).
    normalizationNote : str
        Human-readable note about what is z-normalized.
    reservedInputGroups : list[str]
        Bus groups reserved for future extension (§2.2.d).
    """

    bundleVersion: str
    stateChannels: int
    numBones: int
    rotationChannelsPerBone: int
    rootLocalMotionChannels: int
    controlChannels: int
    controlLayout: list[str]
    phaseChannels: int
    promptEmbChannels: int
    contextFrames: int
    outputLayout: str
    coordSystem: str
    normalizationNote: str
    reservedInputGroups: list[str] = field(default_factory=list)

    def toDict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary.

        Returns
        -------
        dict[str, Any]
            Plain dict with all fields (lists preserved as lists).
        """
        return {
            "bundle_version": self.bundleVersion,
            "state_channels": self.stateChannels,
            "num_bones": self.numBones,
            "rotation_channels_per_bone": self.rotationChannelsPerBone,
            "root_local_motion_channels": self.rootLocalMotionChannels,
            "control_channels": self.controlChannels,
            "control_layout": self.controlLayout,
            "phase_channels": self.phaseChannels,
            "prompt_emb_channels": self.promptEmbChannels,
            "context_frames": self.contextFrames,
            "output_layout": self.outputLayout,
            "coord_system": self.coordSystem,
            "normalization_note": self.normalizationNote,
            "reserved_input_groups": self.reservedInputGroups,
        }


def _buildManifest(controller: MotionController) -> BundleManifest:
    """Build the manifest from the controller's frozen config.

    Parameters
    ----------
    controller : MotionController
        Trained controller (config is authoritative for the manifest).

    Returns
    -------
    BundleManifest
        Populated manifest ready for JSON serialization.
    """
    cfg = controller.config
    useAim = cfg.useAimDirection
    controlLayout = ["vx", "vz"]
    if useAim:
        controlLayout += ["aim_x", "aim_z"]
    return BundleManifest(
        bundleVersion=BUNDLE_VERSION,
        stateChannels=LEAN_STATE_CHANNELS,
        numBones=NUM_SMPL22_BONES,
        rotationChannelsPerBone=ROTATION6D_CHANNELS,
        rootLocalMotionChannels=ROOT_LOCAL_MOTION_CHANNELS,
        controlChannels=controlSignalChannels(useAim),
        controlLayout=controlLayout,
        phaseChannels=phaseConditioningChannels(cfg.phaseMode),
        promptEmbChannels=cfg.promptEmbChannels,
        contextFrames=cfg.contextFrames,
        outputLayout="bone_delta|global_delta",
        coordSystem="Y-up right-handed",
        normalizationNote=(
            "vx,vz are z-normalized (mean/std in norm_stats.json). "
            "aim_x,aim_z are unit-norm by construction (not z-normalized)."
        ),
        reservedInputGroups=[
            "interaction",
            "perception",
            "reaction",
            "morphology",
        ],
    )


def _normStatsToDict(
    stateNorm: MotionNormalizer,
    deltaNorm: MotionNormalizer,
    controlMean: torch.Tensor,
    controlStd: torch.Tensor,
) -> dict[str, Any]:
    """Serialize z-norm statistics as a plain dict.

    Parameters
    ----------
    stateNorm : MotionNormalizer
        State normalizer (bone + global channels).
    deltaNorm : MotionNormalizer
        Delta normalizer (bone + global channels).
    controlMean : torch.Tensor
        Control signal mean (vx, vz only; aim is excluded).
    controlStd : torch.Tensor
        Control signal std (vx, vz only; aim is excluded).

    Returns
    -------
    dict[str, Any]
        JSON-serialisable stats with ``state``, ``delta``, ``control``
        sections.
    """
    def _tensorsToList(
        tensor: torch.Tensor | None,
    ) -> list[float] | None:
        if tensor is None:
            return None
        return tensor.detach().cpu().float().tolist()

    return {
        "state": {
            "bone_mean": _tensorsToList(stateNorm.boneMean),
            "bone_std": _tensorsToList(stateNorm.boneStd),
            "global_mean": _tensorsToList(stateNorm.globalMean),
            "global_std": _tensorsToList(stateNorm.globalStd),
        },
        "delta": {
            "bone_mean": _tensorsToList(deltaNorm.boneMean),
            "bone_std": _tensorsToList(deltaNorm.boneStd),
            "global_mean": _tensorsToList(deltaNorm.globalMean),
            "global_std": _tensorsToList(deltaNorm.globalStd),
        },
        "control": {
            "mean": _tensorsToList(controlMean),
            "std": _tensorsToList(controlStd),
            "channels": ["vx", "vz"],
            "note": (
                "aim_x, aim_z are unit-norm by construction; "
                "no z-norm stats stored for them."
            ),
        },
    }


def exportControllerBundle(
    controller: MotionController,
    stateNorm: MotionNormalizer,
    deltaNorm: MotionNormalizer,
    controlMean: torch.Tensor,
    controlStd: torch.Tensor,
    outputDir: Path,
    resolvedConfigPath: Path | None = None,
    presets: dict[str, dict[str, Any]] | None = None,
    batchSize: int = 1,
) -> Path:
    """Assemble a complete controller bundle in ``outputDir``.

    Produces:

    * ``controller.onnx``       — one-frame forward ONNX graph.
    * ``norm_stats.json``       — z-norm statistics for the engine.
    * ``manifest.json``         — frozen I/O contract (§2.2 layout).
    * ``presets/<name>.json``   — optional preset files.

    If ``resolvedConfigPath`` is provided it is *copied* into
    ``outputDir/resolved_config.yaml`` so the bundle is self-contained.

    Parameters
    ----------
    controller : MotionController
        Trained controller in eval-ready state.
    stateNorm : MotionNormalizer
        State z-normalizer fitted on the training data.
    deltaNorm : MotionNormalizer
        Delta z-normalizer fitted on the training data.
    controlMean : torch.Tensor
        Mean of the control channels (vx, vz) across training data.
    controlStd : torch.Tensor
        Std of the control channels (vx, vz) across training data.
    outputDir : Path
        Destination directory (created if absent).
    resolvedConfigPath : Path | None
        Path to an existing ``resolved_config.yaml`` to copy in.
        When ``None`` the file is not included in the bundle.
    presets : dict[str, dict[str, Any]] | None
        Mapping ``presetName → preset_dict`` to write as JSON files
        under ``presets/``.  Pass ``None`` or ``{}`` for no presets.
    batchSize : int
        Concrete batch size for the ONNX trace example.

    Returns
    -------
    Path
        The bundle directory (same as ``outputDir``).
    """
    outputDir.mkdir(parents=True, exist_ok=True)

    # 1. ONNX graph.
    onnxPath = outputDir / ONNX_FILENAME
    controller.eval()
    exportController(
        controller=controller,
        outputPath=onnxPath,
        batchSize=batchSize,
    )
    LOGGER.info("ONNX graph written: %s", onnxPath)

    # 2. Norm stats.
    normPath = outputDir / NORM_STATS_FILENAME
    normStats = _normStatsToDict(
        stateNorm, deltaNorm, controlMean, controlStd
    )
    normPath.write_text(
        json.dumps(normStats, indent=2), encoding="utf-8"
    )
    LOGGER.info("Norm stats written: %s", normPath)

    # 3. Manifest.
    manifestPath = outputDir / MANIFEST_FILENAME
    manifest = _buildManifest(controller)
    manifestPath.write_text(
        json.dumps(manifest.toDict(), indent=2), encoding="utf-8"
    )
    LOGGER.info("Manifest written: %s", manifestPath)

    # 4. Resolved config (optional copy).
    if resolvedConfigPath is not None and resolvedConfigPath.exists():
        import shutil

        destConfig = outputDir / "resolved_config.yaml"
        shutil.copy2(resolvedConfigPath, destConfig)
        LOGGER.info("Resolved config copied: %s", destConfig)

    # 5. Presets.
    if presets:
        presetsDir = outputDir / PRESETS_SUBDIR
        presetsDir.mkdir(exist_ok=True)
        for name, presetData in presets.items():
            presetPath = presetsDir / f"{name}.json"
            presetPath.write_text(
                json.dumps(presetData, indent=2), encoding="utf-8"
            )
            LOGGER.info("Preset written: %s", presetPath)

    LOGGER.info("Controller bundle complete at: %s", outputDir)
    return outputDir
