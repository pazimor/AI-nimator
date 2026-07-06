"""Tests for the controller bundle assembly (A7/B0).

Validates that :func:`exportControllerBundle` produces all required
artefacts with the correct structure and content.  Uses a small
synthetic controller (no dataset on disk required).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ainimator.core.constants.controller import (
    LEAN_STATE_CHANNELS,
    NUM_SMPL22_BONES,
    PhaseMode,
    ROOT_LOCAL_MOTION_CHANNELS,
)
from ainimator.core.types.controller import ControllerV2Config
from ainimator.export.bundle import (
    MANIFEST_FILENAME,
    NORM_STATS_FILENAME,
    ONNX_FILENAME,
    PRESETS_SUBDIR,
    BundleManifest,
    exportControllerBundle,
)
from ainimator.model.controller_v2 import MotionController
from ainimator.model.motion_normalizer import MotionNormalizer


def _tinyController(phaseMode: PhaseMode = PhaseMode.NONE) -> MotionController:
    """Build a minimal controller for export tests.

    Parameters
    ----------
    phaseMode : PhaseMode
        Locomotor phase mode.  Default ``NONE`` for simple export tests.
    """
    cfg = ControllerV2Config(
        embedDim=64,
        numHeads=4,
        numLayers=2,
        numBones=NUM_SMPL22_BONES,
        contextFrames=1,
        phaseMode=phaseMode,
    )
    return MotionController(cfg)


def _tinyNorms(numBones: int) -> tuple[MotionNormalizer, MotionNormalizer]:
    """Build unit-fitted normalizers (mean=0, std=1)."""
    state = MotionNormalizer(numBones, 6, ROOT_LOCAL_MOTION_CHANNELS)
    delta = MotionNormalizer(numBones, 6, ROOT_LOCAL_MOTION_CHANNELS)
    frames = 32
    bone = torch.randn(frames, numBones, 6)
    root = torch.randn(frames, ROOT_LOCAL_MOTION_CHANNELS)
    state.fitFromTensors([bone], [root])
    delta.fitFromTensors([bone[1:] - bone[:-1]], [root[1:] - root[:-1]])
    return state, delta


def test_bundle_creates_required_files(tmp_path: Path) -> None:
    """exportControllerBundle writes onnx, norm_stats, manifest."""
    controller = _tinyController()
    state, delta = _tinyNorms(NUM_SMPL22_BONES)
    controlMean = torch.zeros(2)
    controlStd = torch.ones(2)

    bundleDir = exportControllerBundle(
        controller=controller,
        stateNorm=state,
        deltaNorm=delta,
        controlMean=controlMean,
        controlStd=controlStd,
        outputDir=tmp_path / "bundle",
    )
    assert (bundleDir / ONNX_FILENAME).exists()
    assert (bundleDir / NORM_STATS_FILENAME).exists()
    assert (bundleDir / MANIFEST_FILENAME).exists()


def test_manifest_contract_fields(tmp_path: Path) -> None:
    """manifest.json carries all frozen I/O contract fields (§2.2)."""
    controller = _tinyController()
    state, delta = _tinyNorms(NUM_SMPL22_BONES)
    controlMean = torch.zeros(2)
    controlStd = torch.ones(2)

    bundleDir = exportControllerBundle(
        controller=controller,
        stateNorm=state,
        deltaNorm=delta,
        controlMean=controlMean,
        controlStd=controlStd,
        outputDir=tmp_path / "bundle",
    )
    manifest = json.loads((bundleDir / MANIFEST_FILENAME).read_text())
    assert manifest["state_channels"] == LEAN_STATE_CHANNELS
    assert manifest["num_bones"] == NUM_SMPL22_BONES
    assert manifest["root_local_motion_channels"] == ROOT_LOCAL_MOTION_CHANNELS
    assert manifest["control_channels"] == 2  # vx, vz (no aim by default)
    assert manifest["control_layout"] == ["vx", "vz"]
    assert manifest["phase_channels"] == 0
    assert manifest["output_layout"] == "bone_delta|global_delta"
    assert "reserved_input_groups" in manifest
    assert len(manifest["reserved_input_groups"]) == 4


def test_norm_stats_structure(tmp_path: Path) -> None:
    """norm_stats.json carries state/delta/control sections."""
    controller = _tinyController()
    state, delta = _tinyNorms(NUM_SMPL22_BONES)
    controlMean = torch.zeros(2)
    controlStd = torch.ones(2) * 0.5

    bundleDir = exportControllerBundle(
        controller=controller,
        stateNorm=state,
        deltaNorm=delta,
        controlMean=controlMean,
        controlStd=controlStd,
        outputDir=tmp_path / "bundle",
    )
    stats = json.loads((bundleDir / NORM_STATS_FILENAME).read_text())
    assert "state" in stats
    assert "delta" in stats
    assert "control" in stats
    assert stats["control"]["channels"] == ["vx", "vz"]
    # Std values round-trip correctly.
    controlStdList = stats["control"]["std"]
    assert all(math.isclose(v, 0.5, abs_tol=1e-5) for v in controlStdList)


def test_bundle_with_presets(tmp_path: Path) -> None:
    """Preset files are written under presets/ when provided."""
    controller = _tinyController()
    state, delta = _tinyNorms(NUM_SMPL22_BONES)
    controlMean = torch.zeros(2)
    controlStd = torch.ones(2)

    presets = {
        "walk": {"vx": 0.05, "vz": 0.0, "description": "default walk"},
        "run": {"vx": 0.15, "vz": 0.0, "description": "fast run"},
    }
    bundleDir = exportControllerBundle(
        controller=controller,
        stateNorm=state,
        deltaNorm=delta,
        controlMean=controlMean,
        controlStd=controlStd,
        outputDir=tmp_path / "bundle",
        presets=presets,
    )
    presetsDir = bundleDir / PRESETS_SUBDIR
    assert presetsDir.exists()
    walkPreset = json.loads((presetsDir / "walk.json").read_text())
    assert walkPreset["vx"] == pytest.approx(0.05)
    assert walkPreset["description"] == "default walk"


import pytest  # noqa: E402 (placed here to keep imports at module top)


# ---------------------------------------------------------------------
# B0 acceptance: schemas + default presets + engine-style parity
# ---------------------------------------------------------------------
_SPEC_DIR = Path(__file__).parents[3] / "apps" / "spec"


def test_default_presets_written_and_valid(tmp_path: Path) -> None:
    """No presets arg -> the standard locomotion set, schema-valid."""
    import jsonschema

    controller = _tinyController()
    state, delta = _tinyNorms(NUM_SMPL22_BONES)
    exportControllerBundle(
        controller, state, delta,
        torch.zeros(2), torch.ones(2), tmp_path,
    )
    schema = json.loads(
        (_SPEC_DIR / "control_preset.schema.json").read_text()
    )
    presetDir = tmp_path / PRESETS_SUBDIR
    names = sorted(p.stem for p in presetDir.glob("*.json"))
    assert names == [
        "backward", "forward", "idle", "strafe_left", "strafe_right",
    ]
    for path in presetDir.glob("*.json"):
        jsonschema.validate(json.loads(path.read_text()), schema)


def test_manifest_validates_schema(tmp_path: Path) -> None:
    """The generated manifest must satisfy apps/spec/manifest.schema.json."""
    import jsonschema

    controller = _tinyController()
    state, delta = _tinyNorms(NUM_SMPL22_BONES)
    exportControllerBundle(
        controller, state, delta,
        torch.zeros(2), torch.ones(2), tmp_path,
    )
    schema = json.loads((_SPEC_DIR / "manifest.schema.json").read_text())
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    jsonschema.validate(manifest, schema)


def test_null_prompt_emb_serialized_for_text_models(tmp_path: Path) -> None:
    """Text-conditioned checkpoints ship the learned null embedding."""
    cfg = ControllerV2Config(
        embedDim=64, numHeads=4, numLayers=2,
        numBones=NUM_SMPL22_BONES, contextFrames=1,
        phaseMode=PhaseMode.NONE, promptEmbChannels=16,
    )
    controller = MotionController(cfg)
    state, delta = _tinyNorms(NUM_SMPL22_BONES)
    exportControllerBundle(
        controller, state, delta,
        torch.zeros(2), torch.ones(2), tmp_path,
    )
    stats = json.loads((tmp_path / NORM_STATS_FILENAME).read_text())
    assert stats["prompt"]["channels"] == 16
    assert len(stats["prompt"]["null_emb"]) == 16


def test_bundle_engine_parity_with_normalization(tmp_path: Path) -> None:
    """Engine-style step from bundle files only == torch path (1e-3).

    Reads norm_stats.json as plain JSON (no MotionNormalizer), z-norms
    a raw state window, runs controller.onnx via ONNXRuntime, and
    denormalizes the delta — exactly what a plugin does per frame.
    """
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(0)
    controller = _tinyController().eval()
    state, delta = _tinyNorms(NUM_SMPL22_BONES)
    controlMean, controlStd = torch.zeros(2), torch.ones(2) * 2.0
    exportControllerBundle(
        controller, state, delta, controlMean, controlStd, tmp_path,
    )
    stats = json.loads((tmp_path / NORM_STATS_FILENAME).read_text())

    rawBone = torch.randn(1, 1, NUM_SMPL22_BONES, 6)
    rawGlobal = torch.randn(1, 1, ROOT_LOCAL_MOTION_CHANNELS)
    rawControl = torch.tensor([[0.5, 1.0]])

    # --- engine side: JSON stats only -----------------------------
    bMean = np.array(stats["state"]["bone_mean"], dtype=np.float32)
    bStd = np.array(stats["state"]["bone_std"], dtype=np.float32)
    gMean = np.array(stats["state"]["global_mean"], dtype=np.float32)
    gStd = np.array(stats["state"]["global_std"], dtype=np.float32)
    cMean = np.array(stats["control"]["mean"], dtype=np.float32)
    cStd = np.array(stats["control"]["std"], dtype=np.float32)
    normBoneNp = (rawBone.numpy() - bMean) / bStd
    normGlobalNp = (rawGlobal.numpy() - gMean) / gStd
    normControlNp = (rawControl.numpy() - cMean) / cStd

    session = ort.InferenceSession(
        str(tmp_path / ONNX_FILENAME), providers=["CPUExecutionProvider"]
    )
    inputNames = [i.name for i in session.get_inputs()]
    feeds = dict(zip(inputNames, [
        normBoneNp.astype(np.float32),
        normControlNp.astype(np.float32),
        normGlobalNp.astype(np.float32),
    ]))
    boneDeltaOrt, globalDeltaOrt = session.run(None, feeds)
    dbMean = np.array(stats["delta"]["bone_mean"], dtype=np.float32)
    dbStd = np.array(stats["delta"]["bone_std"], dtype=np.float32)
    engineBoneDelta = boneDeltaOrt * dbStd[0] + dbMean[0]

    # --- torch reference side --------------------------------------
    with torch.no_grad():
        normBone = (rawBone - state.boneMean) / state.boneStd
        normGlobal = (rawGlobal - state.globalMean) / state.globalStd
        normControl = (rawControl - controlMean) / controlStd
        out = controller(normBone, normControl, globalWindow=normGlobal)
        torchBoneDelta = out.boneDelta * delta.boneStd + delta.boneMean

    assert np.allclose(
        engineBoneDelta, torchBoneDelta.numpy(), atol=1e-3
    ), "engine-style bundle step diverges from the torch path"
