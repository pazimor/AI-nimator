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
    controlSignalChannels,
    phaseConditioningChannels,
)
from ainimator.export.onnx import exportController
from ainimator.model.controller_v2 import MotionController
from ainimator.model.motion_normalizer import MotionNormalizer

LOGGER = logging.getLogger(__name__)

# Bundle format version — bump on breaking manifest changes.
# A7.1 (B7, 2026-07-04): optional text_encoder.onnx + tokenizer/ for
# in-engine prompt encoding — minor bump, engine gates filter on the
# major only (apps/spec/text_encoding.md §1).
BUNDLE_VERSION = "A7.1"

# Canonical filenames inside the bundle directory.
ONNX_FILENAME = "controller.onnx"
NORM_STATS_FILENAME = "norm_stats.json"
MANIFEST_FILENAME = "manifest.json"
PRESETS_SUBDIR = "presets"
TEXT_ENCODER_FILENAME = "text_encoder.onnx"
TOKENIZER_SUBDIR = "tokenizer"

# Special-token strings of the CLIP vocabulary — resolved to ids from
# the exported vocab.json (never hardcoded ids).
_BOS_TOKEN = "<|startoftext|>"
_EOS_TOKEN = "<|endoftext|>"

# Default preset walking speed, meters per frame in the root-local
# ground frame (~1 m/s at the 30 fps dataset rate).  Presets are
# engine-editable examples, not tuned hyperparameters.
DEFAULT_PRESET_SPEED = 0.033


def defaultControlPresets(
    useAimDirection: bool,
) -> dict[str, dict[str, Any]]:
    """Build the default ControlPreset set shipped in every bundle.

    Five locomotion presets covering the raw control space
    (``vx, vz`` in meters/frame, root-local ground frame — the engine
    z-normalizes them with ``norm_stats.json``).  ``aim_x, aim_z``
    (unit-norm, facing forward) are included only when the checkpoint
    was trained with aim conditioning.

    Parameters
    ----------
    useAimDirection : bool
        Whether the controller consumes the 2-channel aim direction.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping ``presetName -> preset dict`` matching
        ``apps/spec/control_preset.schema.json``.
    """
    speed = DEFAULT_PRESET_SPEED
    velocities = {
        "idle": (0.0, 0.0),
        "forward": (0.0, speed),
        "backward": (0.0, -speed),
        "strafe_left": (-speed, 0.0),
        "strafe_right": (speed, 0.0),
    }
    presets: dict[str, dict[str, Any]] = {}
    for name, (vx, vz) in velocities.items():
        control: dict[str, float] = {"vx": vx, "vz": vz}
        if useAimDirection:
            control["aim_x"] = 0.0
            control["aim_z"] = 1.0
        presets[name] = {"name": name, "control": control}
    return presets


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
    textEncoder : dict[str, Any] | None
        Optional in-engine text-encoder section (B7, bundle A7.1) —
        ``None`` for embedding-only bundles (A7.0 behaviour).
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
    textEncoder: dict[str, Any] | None = None

    def toDict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary.

        Returns
        -------
        dict[str, Any]
            Plain dict with all fields (lists preserved as lists).
        """
        payload: dict[str, Any] = {
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
        if self.textEncoder is not None:
            payload["text_encoder"] = self.textEncoder
        return payload


def _buildManifest(
    controller: MotionController,
    textEncoderSection: dict[str, Any] | None = None,
) -> BundleManifest:
    """Build the manifest from the controller's frozen config.

    Parameters
    ----------
    controller : MotionController
        Trained controller (config is authoritative for the manifest).
    textEncoderSection : dict[str, Any] | None
        Optional ``text_encoder`` manifest section (B7 bundles).

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
        textEncoder=textEncoderSection,
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
    encoderArtifactPath: Path | None = None,
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
        under ``presets/``.  ``None`` (default) writes the standard
        locomotion set from :func:`defaultControlPresets`; pass ``{}``
        for an explicitly preset-free bundle.
    batchSize : int
        Concrete batch size for the ONNX trace example.
    encoderArtifactPath : Path | None
        Frozen text-encoder artifact directory.  When provided the
        bundle additionally ships ``text_encoder.onnx`` +
        ``tokenizer/`` for in-engine prompt encoding (B7 / A7.1 —
        ``apps/spec/text_encoding.md``).  Requires a text-conditioned
        controller (``promptEmbChannels > 0``).

    Returns
    -------
    Path
        The bundle directory (same as ``outputDir``).
    """
    outputDir.mkdir(parents=True, exist_ok=True)
    if presets is None:
        presets = defaultControlPresets(controller.config.useAimDirection)
    _writeOnnxGraph(controller, outputDir, batchSize)
    _writeNormStats(
        stateNorm, deltaNorm, controlMean, controlStd, outputDir,
        controller=controller,
    )
    textEncoderSection = _writeTextEncoder(
        encoderArtifactPath, controller, outputDir, batchSize
    )
    _writeManifest(controller, outputDir, textEncoderSection)
    _copyResolvedConfig(resolvedConfigPath, outputDir)
    _writePresets(presets, outputDir)
    LOGGER.info("Controller bundle complete at: %s", outputDir)
    return outputDir


def _writeOnnxGraph(
    controller: MotionController,
    outputDir: Path,
    batchSize: int,
) -> None:
    """Export the controller ONNX graph into ``outputDir``."""
    onnxPath = outputDir / ONNX_FILENAME
    controller.eval()
    exportController(controller=controller, outputPath=onnxPath,
                     batchSize=batchSize)
    LOGGER.info("ONNX graph written: %s", onnxPath)


def _writeNormStats(
    stateNorm: MotionNormalizer,
    deltaNorm: MotionNormalizer,
    controlMean: torch.Tensor,
    controlStd: torch.Tensor,
    outputDir: Path,
    controller: MotionController | None = None,
) -> None:
    """Serialise z-norm statistics to ``norm_stats.json``.

    When the controller carries a learned null prompt embedding it is
    serialised too: the ONNX ``promptEmb`` input is REQUIRED whenever
    ``prompt_emb_channels > 0``, so a promptless engine must feed this
    exact vector (zeros are NOT a valid substitute).
    """
    normPath = outputDir / NORM_STATS_FILENAME
    normStats = _normStatsToDict(stateNorm, deltaNorm, controlMean, controlStd)
    if controller is not None and controller.nullPromptEmb is not None:
        normStats["prompt"] = {
            "null_emb": (
                controller.nullPromptEmb.detach().cpu().float().tolist()
            ),
            "channels": int(controller.config.promptEmbChannels),
            "note": (
                "Feed null_emb as promptEmb when no prompt is active; "
                "it is a trained parameter, not zeros."
            ),
        }
    normPath.write_text(json.dumps(normStats, indent=2), encoding="utf-8")
    LOGGER.info("Norm stats written: %s", normPath)


def _writeManifest(
    controller: MotionController,
    outputDir: Path,
    textEncoderSection: dict[str, Any] | None = None,
) -> None:
    """Write the I/O contract manifest to ``manifest.json``."""
    manifestPath = outputDir / MANIFEST_FILENAME
    manifest = _buildManifest(controller, textEncoderSection)
    manifestPath.write_text(
        json.dumps(manifest.toDict(), indent=2), encoding="utf-8"
    )
    LOGGER.info("Manifest written: %s", manifestPath)


def _writeTextEncoder(
    encoderArtifactPath: Path | None,
    controller: MotionController,
    outputDir: Path,
    batchSize: int,
) -> dict[str, Any] | None:
    """Ship the in-engine text encoder in the bundle (B7 / A7.1).

    Exports the pooled encoder graph to ``text_encoder.onnx`` and the
    verbatim HuggingFace ``vocab.json`` / ``merges.txt`` under
    ``tokenizer/``, then returns the manifest ``text_encoder`` section
    (``apps/spec/text_encoding.md`` §1).  Returns ``None`` when no
    artifact is supplied (A7.0-style bundle, embeddings only).

    Raises
    ------
    ValueError
        If the controller has no prompt channel, the artifact is not a
        CLIP-type artifact, or the encoder width does not match the
        controller's ``promptEmbChannels`` (fail-fast, never a silent
        mismatch).
    """
    if encoderArtifactPath is None:
        return None
    if controller.config.promptEmbChannels <= 0:
        raise ValueError(
            "encoderArtifactPath given but the controller has "
            "promptEmbChannels == 0 — a text encoder cannot condition "
            "a promptless controller."
        )
    from ainimator.export.onnx import exportPooledTextEncoder
    from ainimator.text.artifact import loadEncoderArtifact

    encoder, tokenizer = loadEncoderArtifact(
        encoderArtifactPath, device=torch.device("cpu")
    )
    if not hasattr(tokenizer, "saveVocabulary"):
        raise ValueError(
            "In-engine text encoding (B7) requires a CLIP-type "
            f"artifact; got tokenizer {type(tokenizer).__name__} "
            "without a serialisable vocabulary."
        )
    if encoder.outputDim != controller.config.promptEmbChannels:
        raise ValueError(
            f"Encoder outputDim ({encoder.outputDim}) != controller "
            f"promptEmbChannels ({controller.config.promptEmbChannels})."
        )
    encoder.eval()
    maxLength = tokenizer.config.maxLength
    exportPooledTextEncoder(
        encoder=encoder,
        outputPath=outputDir / TEXT_ENCODER_FILENAME,
        maxLength=maxLength,
        batchSize=batchSize,
    )
    vocabPath, mergesPath = tokenizer.saveVocabulary(
        outputDir / TOKENIZER_SUBDIR
    )
    vocab = json.loads(vocabPath.read_text(encoding="utf-8"))
    LOGGER.info(
        "Text encoder shipped: %s + %s",
        TEXT_ENCODER_FILENAME,
        TOKENIZER_SUBDIR,
    )
    return {
        "file": TEXT_ENCODER_FILENAME,
        "tokenizer": {
            "type": "clip-bpe",
            "vocab": f"{TOKENIZER_SUBDIR}/{vocabPath.name}",
            "merges": f"{TOKENIZER_SUBDIR}/{mergesPath.name}",
            "max_length": maxLength,
            "bos_id": int(vocab[_BOS_TOKEN]),
            "eos_id": int(vocab[_EOS_TOKEN]),
            "pad_id": int(vocab[_EOS_TOKEN]),
        },
        "pooling": "masked_mean",
        "embedding_channels": int(encoder.outputDim),
    }


def _copyResolvedConfig(
    resolvedConfigPath: Path | None, outputDir: Path
) -> None:
    """Copy resolved_config.yaml into the bundle if provided."""
    if resolvedConfigPath is not None and resolvedConfigPath.exists():
        import shutil
        destConfig = outputDir / "resolved_config.yaml"
        shutil.copy2(resolvedConfigPath, destConfig)
        LOGGER.info("Resolved config copied: %s", destConfig)


def _writePresets(
    presets: dict[str, dict[str, Any]] | None, outputDir: Path
) -> None:
    """Write preset JSON files under ``presets/`` sub-directory."""
    if not presets:
        return
    presetsDir = outputDir / PRESETS_SUBDIR
    presetsDir.mkdir(exist_ok=True)
    for name, presetData in presets.items():
        presetPath = presetsDir / f"{name}.json"
        presetPath.write_text(
            json.dumps(presetData, indent=2), encoding="utf-8"
        )
        LOGGER.info("Preset written: %s", presetPath)
