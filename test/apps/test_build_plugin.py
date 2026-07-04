"""Tests for the B5 build orchestrator (export → validate → deliver).

The engine step is not testable here (no Unity/Unreal install); the
pipeline is exercised end-to-end up to delivery with a real tiny
checkpoint, plus the fail-fast paths (missing checkpoint, incompatible
contract version, missing engine env var).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from apps.build import build_plugin
from apps.build.build_plugin import (
    BuildError,
    TargetConfig,
    _contractKey,
    validateBundle,
)


@pytest.fixture(scope="module")
def tinyCheckpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a real minimal controller checkpoint on disk."""
    from ainimator.core.constants.controller import (
        NUM_SMPL22_BONES,
        PhaseMode,
        ROOT_LOCAL_MOTION_CHANNELS,
    )
    from ainimator.core.types.controller import ControllerV2Config
    from ainimator.model.controller_v2 import MotionController
    from ainimator.model.motion_normalizer import MotionNormalizer
    from ainimator.training.controller_training_v2 import (
        saveControllerCheckpoint,
    )

    outDir = tmp_path_factory.mktemp("ckpt")
    cfg = ControllerV2Config(
        embedDim=64, numHeads=4, numLayers=2,
        numBones=NUM_SMPL22_BONES, contextFrames=1,
        phaseMode=PhaseMode.NONE,
    )
    model = MotionController(cfg)
    state = MotionNormalizer(
        NUM_SMPL22_BONES, 6, ROOT_LOCAL_MOTION_CHANNELS
    )
    delta = MotionNormalizer(
        NUM_SMPL22_BONES, 6, ROOT_LOCAL_MOTION_CHANNELS
    )
    bone = torch.randn(16, NUM_SMPL22_BONES, 6)
    root = torch.randn(16, ROOT_LOCAL_MOTION_CHANNELS)
    state.fitFromTensors([bone], [root])
    delta.fitFromTensors([bone[1:] - bone[:-1]], [root[1:] - root[:-1]])
    return saveControllerCheckpoint(
        model, state, delta, torch.zeros(2), torch.ones(2), outDir
    )


def _tmpTarget(tmp_path: Path, name: str = "unity") -> TargetConfig:
    """Target config delivering into a temporary directory."""
    return TargetConfig(
        name=name,
        deliveryDir=tmp_path / "delivery" / "AInimatorBundle",
        engineEnvVar="UNITY_PATH",
    )


def test_pipeline_export_validate_deliver(
    tinyCheckpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full pipeline (minus engine) delivers a fresh, valid bundle."""
    target = _tmpTarget(tmp_path)
    monkeypatch.setitem(build_plugin.TARGETS, "unity", target)
    build_plugin.main([
        "--target", "unity",
        "--checkpoint", str(tinyCheckpoint),
        "--skip-engine-build",
    ])
    delivered = target.deliveryDir
    assert (delivered / "controller.onnx").exists()
    assert (delivered / "manifest.json").exists()
    assert (delivered / "norm_stats.json").exists()
    assert sorted(p.stem for p in (delivered / "presets").glob("*.json")) == [
        "backward", "forward", "idle", "strafe_left", "strafe_right",
    ]


def test_missing_checkpoint_fails_clearly(tmp_path: Path) -> None:
    """A nonexistent checkpoint aborts with an explicit message."""
    with pytest.raises(SystemExit, match="checkpoint not found"):
        build_plugin.exportBundle(
            tmp_path / "nope.pt", tmp_path / "bundle"
        )


def test_incompatible_contract_version_stops(tmp_path: Path) -> None:
    """A bundle from a future contract major is refused (STOP)."""
    bundleDir = tmp_path / "bundle"
    (bundleDir / "presets").mkdir(parents=True)
    manifest = {
        "bundle_version": "B9.0",
        "state_channels": 136,
        "num_bones": 22,
        "rotation_channels_per_bone": 6,
        "root_local_motion_channels": 4,
        "control_channels": 2,
        "control_layout": ["vx", "vz"],
        "phase_channels": 0,
        "prompt_emb_channels": 0,
        "context_frames": 1,
        "output_layout": "bone_delta|global_delta",
        "coord_system": "Y-up right-handed",
        "normalization_note": "n/a",
        "reserved_input_groups": [],
    }
    (bundleDir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="not accepted"):
        validateBundle(bundleDir, _tmpTarget(tmp_path))


def test_engine_step_requires_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the engine env var, the build step fails explicitly."""
    monkeypatch.delenv("UNITY_PATH", raising=False)
    with pytest.raises(SystemExit, match="UNITY_PATH"):
        build_plugin.buildEngine(_tmpTarget(tmp_path))


def test_contract_key_parsing() -> None:
    """bundle_version → prefix+major compatibility key."""
    assert _contractKey("A7.0") == "A7"
    assert _contractKey("12.3") == "12"
    with pytest.raises(SystemExit, match="pattern"):
        _contractKey("garbage")
