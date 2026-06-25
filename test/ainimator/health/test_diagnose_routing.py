"""LOT-4 tests — health diagnose routing and controller diagnose path.

Tests cover:
1. _detectModelType: controller checkpoint detected via payload keys.
2. _detectModelType: diffusion checkpoint detected via payload keys.
3. _detectModelType: resolved_config.yaml takes precedence.
4. _detectModelType: defaults to "diffusion" when nothing matches.
5. _loadDiagnoseProfile: loads known profile from network.yaml.
6. _loadDiagnoseProfile: returns empty dict for unknown profile.
7. HealthHub.diagnoseController: returns expected keys and contract values.
8. _buildControlVector: forward spec builds expected tensor.
9. _buildControlVector: random spec is normalized and scaled.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from ainimator.core.constants.controller import PhaseMode
from ainimator.core.types.controller import ControllerV2Config
from ainimator.model.controller_v2 import MotionController
from ainimator.model.motion_normalizer import MotionNormalizer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _makeControllerCheckpoint(tmpDir: Path) -> Path:
    """Write a minimal controller checkpoint to disk."""
    path = tmpDir / "checkpoints" / "controller_overfit_checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": {
            "embedDim": 64,
            "numHeads": 4,
            "numLayers": 2,
            "numBones": 22,
            "motionChannels": 6,
            "globalChannels": 4,
            "contextFrames": 1,
            "phaseMode": "none",
            "useAimDirection": False,
            "useFilmConditioning": False,
            "usePerBlockFilm": False,
            "filmInitStd": 0.02,
            "dropout": 0.0,
            "maxFrames": 256,
            "styleLatentEnabled": False,
            "styleLatentDim": 16,
            "promptEmbChannels": 0,
        },
        "model_state": {},
        "state_normalizer_config": {
            "boneMean": [0.0] * 132,
            "boneStd": [1.0] * 132,
            "globalMean": [0.0] * 4,
            "globalStd": [1.0] * 4,
        },
        "state_normalizer_state": {},
        "delta_normalizer_config": {
            "boneMean": [0.0] * 132,
            "boneStd": [1.0] * 132,
            "globalMean": [0.0] * 4,
            "globalStd": [1.0] * 4,
        },
        "delta_normalizer_state": {},
        "control_mean": torch.zeros(4),
        "control_std": torch.ones(4),
    }
    torch.save(payload, path)
    return path


def _makeDiffusionCheckpoint(tmpDir: Path) -> Path:
    """Write a minimal diffusion-style checkpoint to disk."""
    path = tmpDir / "checkpoints" / "diffusion_checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_config": {"predictionMode": "v_prediction"},
        "encoder_state": {},
        "denoiser_state": {},
    }
    torch.save(payload, path)
    return path


def _makeResolvedConfig(dirPath: Path, modelType: str) -> Path:
    """Write a minimal resolved_config.yaml with the given model-type."""
    path = dirPath / "resolved_config.yaml"
    data: dict[str, Any] = {
        "git_sha": "abc1234",
        "timestamp": "2026-06-25T00:00:00+00:00",
        "config": {"modelType": modelType},
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    return path


def _makeMinimalControllerModel() -> MotionController:
    """Build a minimal MotionController for diagnose tests."""
    config = ControllerV2Config(
        embedDim=64,
        numHeads=4,
        numLayers=2,
        contextFrames=1,
        phaseMode=PhaseMode.NONE,
        useFilmConditioning=False,
        usePerBlockFilm=False,
    )
    return MotionController(config).eval()


def _makeIdentityNormalizer(
    numBones: int = 22,
    motionChannels: int = 6,
) -> MotionNormalizer:
    """Build an identity MotionNormalizer (no-op normalization)."""
    norm = MotionNormalizer(
        numBones=numBones,
        motionChannels=motionChannels,
        globalChannels=4,
    )
    norm.boneMean.zero_()
    norm.boneStd.fill_(1.0)
    norm.globalMean.zero_()
    norm.globalStd.fill_(1.0)
    return norm


# ------------------------------------------------------------------
# _detectModelType tests
# ------------------------------------------------------------------
class TestDetectModelType:
    """Tests for the model-type detection heuristic."""

    def test_controller_detected_from_payload_keys(self) -> None:
        from ainimator.cli.health import _detectModelType

        with tempfile.TemporaryDirectory() as tmp:
            ckptPath = _makeControllerCheckpoint(Path(tmp))
            result = _detectModelType(ckptPath)
        assert result == "controller"

    def test_diffusion_detected_from_payload_keys(self) -> None:
        from ainimator.cli.health import _detectModelType

        with tempfile.TemporaryDirectory() as tmp:
            ckptPath = _makeDiffusionCheckpoint(Path(tmp))
            result = _detectModelType(ckptPath)
        assert result == "diffusion"

    def test_resolved_config_takes_precedence_over_payload(self) -> None:
        """A resolved_config.yaml with model-type=controller overrides
        even a payload that would otherwise suggest diffusion."""
        from ainimator.cli.health import _detectModelType

        with tempfile.TemporaryDirectory() as tmp:
            tmpPath = Path(tmp)
            # Diffusion payload but controller resolved_config.
            ckptPath = _makeDiffusionCheckpoint(tmpPath)
            _makeResolvedConfig(ckptPath.parent.parent, "controller")
            result = _detectModelType(ckptPath)
        assert result == "controller"

    def test_defaults_to_diffusion_for_empty_checkpoint(self) -> None:
        """An unknown checkpoint should default to diffusion."""
        from ainimator.cli.health import _detectModelType

        with tempfile.TemporaryDirectory() as tmp:
            ckptPath = Path(tmp) / "mystery.pt"
            torch.save({"unknown_key": 42}, ckptPath)
            result = _detectModelType(ckptPath)
        assert result == "diffusion"

    def test_controller_heuristic_from_resolved_config_fields(self) -> None:
        """A resolved_config.yaml with phaseMode field → controller."""
        from ainimator.cli.health import _detectModelType

        with tempfile.TemporaryDirectory() as tmp:
            tmpPath = Path(tmp)
            ckptPath = tmpPath / "mystery.pt"
            torch.save({}, ckptPath)
            # Write resolved_config.yaml with controller fields but no
            # explicit modelType key.
            data: dict[str, Any] = {
                "config": {"phaseMode": "none", "contextFrames": 1}
            }
            rcPath = tmpPath / "resolved_config.yaml"
            with rcPath.open("w") as fh:
                yaml.dump(data, fh)
            result = _detectModelType(ckptPath)
        assert result == "controller"


# ------------------------------------------------------------------
# _loadDiagnoseProfile tests
# ------------------------------------------------------------------
class TestLoadDiagnoseProfile:
    """Tests for diagnose profile loading from network.yaml."""

    def test_loads_known_controller_profile(self) -> None:
        from ainimator.cli.health import _loadDiagnoseProfile

        networkYaml = Path("src/configs/network.yaml")
        profile = _loadDiagnoseProfile(
            "controller_default", networkYaml=networkYaml
        )
        assert "diagnose" in profile
        diag = profile["diagnose"]
        assert "control" in diag
        assert "rollout-frames" in diag

    def test_loads_controller_short_profile(self) -> None:
        from ainimator.cli.health import _loadDiagnoseProfile

        networkYaml = Path("src/configs/network.yaml")
        profile = _loadDiagnoseProfile(
            "controller_short", networkYaml=networkYaml
        )
        diag = profile.get("diagnose", {})
        rolloutFrames = diag.get("rollout-frames", 120)
        assert rolloutFrames < 120, "controller_short should have short rollout"

    def test_returns_empty_dict_for_unknown_profile(self) -> None:
        from ainimator.cli.health import _loadDiagnoseProfile

        networkYaml = Path("src/configs/network.yaml")
        profile = _loadDiagnoseProfile(
            "nonexistent_profile_xyz", networkYaml=networkYaml
        )
        assert profile == {}

    def test_returns_empty_dict_for_missing_yaml(self) -> None:
        from ainimator.cli.health import _loadDiagnoseProfile

        profile = _loadDiagnoseProfile(
            "controller_default",
            networkYaml=Path("/nonexistent/network.yaml"),
        )
        assert profile == {}


# ------------------------------------------------------------------
# _buildControlVector tests
# ------------------------------------------------------------------
class TestBuildControlVector:
    """Tests for the control vector builder."""

    def test_forward_spec_sets_first_channel(self) -> None:
        from ainimator.health.hub import _buildControlVector

        ctrl = _buildControlVector("forward:1.0", controlChannels=4)
        assert ctrl.shape == (4,)
        assert float(ctrl[0].item()) == pytest.approx(1.0)
        assert float(ctrl[1].item()) == pytest.approx(0.0)

    def test_forward_spec_respects_scale(self) -> None:
        from ainimator.health.hub import _buildControlVector

        ctrl = _buildControlVector("forward:2.5", controlChannels=4)
        assert float(ctrl[0].item()) == pytest.approx(2.5)

    def test_random_spec_produces_correct_shape(self) -> None:
        from ainimator.health.hub import _buildControlVector

        ctrl = _buildControlVector("random:1.0", controlChannels=8, seed=0)
        assert ctrl.shape == (8,)

    def test_random_spec_is_scaled_by_scale(self) -> None:
        from ainimator.health.hub import _buildControlVector

        ctrl = _buildControlVector("random:0.5", controlChannels=4, seed=0)
        assert float(ctrl.norm().item()) == pytest.approx(0.5, abs=1e-5)

    def test_unknown_keyword_falls_back_to_random(self) -> None:
        from ainimator.health.hub import _buildControlVector

        ctrl = _buildControlVector("sideways:1.0", controlChannels=4, seed=0)
        assert ctrl.shape == (4,)


# ------------------------------------------------------------------
# HealthHub.diagnoseController tests
# ------------------------------------------------------------------
class TestDiagnoseController:
    """Tests for HealthHub.diagnoseController."""

    def test_returns_expected_metric_keys(self) -> None:
        from ainimator.health.hub import buildHealthHub

        with tempfile.TemporaryDirectory() as tmp:
            outputDir = Path(tmp)
            hub = buildHealthHub(outputDir)

            model = _makeMinimalControllerModel()
            stateNorm = _makeIdentityNormalizer()
            deltaNorm = _makeIdentityNormalizer()

            results = hub.diagnoseController(
                model=model,
                stateNormalizer=stateNorm,
                deltaNormalizer=deltaNorm,
                device=torch.device("cpu"),
                controlSpec="forward:1.0",
                rolloutFrames=10,
                shuffleControl=False,
                seeds=(0, 1),
            )
            hub.close()

        assert results["model_type"] == "controller"
        assert "mean_collapse_rank" in results
        assert "mean_collapse_sim" in results
        assert "post_norm_stats" in results
        assert "rollout_drift" in results

    def test_control_sensitivity_populated_when_shuffle_enabled(self) -> None:
        from ainimator.health.hub import buildHealthHub

        with tempfile.TemporaryDirectory() as tmp:
            outputDir = Path(tmp)
            hub = buildHealthHub(outputDir)

            model = _makeMinimalControllerModel()
            stateNorm = _makeIdentityNormalizer()
            deltaNorm = _makeIdentityNormalizer()

            results = hub.diagnoseController(
                model=model,
                stateNormalizer=stateNorm,
                deltaNormalizer=deltaNorm,
                device=torch.device("cpu"),
                controlSpec="forward:1.0",
                rolloutFrames=5,
                shuffleControl=True,
                seeds=(0, 1, 2),
            )
            hub.close()

        assert results["control_sensitivity"] is not None
        assert isinstance(results["control_sensitivity"], float)

    def test_control_sensitivity_none_when_shuffle_disabled(self) -> None:
        from ainimator.health.hub import buildHealthHub

        with tempfile.TemporaryDirectory() as tmp:
            outputDir = Path(tmp)
            hub = buildHealthHub(outputDir)

            model = _makeMinimalControllerModel()
            stateNorm = _makeIdentityNormalizer()
            deltaNorm = _makeIdentityNormalizer()

            results = hub.diagnoseController(
                model=model,
                stateNormalizer=stateNorm,
                deltaNormalizer=deltaNorm,
                device=torch.device("cpu"),
                rolloutFrames=5,
                shuffleControl=False,
                seeds=(0, 1),
            )
            hub.close()

        assert results["control_sensitivity"] is None

    def test_mean_collapse_rank_in_valid_range(self) -> None:
        from ainimator.health.hub import buildHealthHub

        with tempfile.TemporaryDirectory() as tmp:
            hub = buildHealthHub(Path(tmp))
            model = _makeMinimalControllerModel()
            stateNorm = _makeIdentityNormalizer()
            deltaNorm = _makeIdentityNormalizer()

            results = hub.diagnoseController(
                model=model,
                stateNormalizer=stateNorm,
                deltaNormalizer=deltaNorm,
                device=torch.device("cpu"),
                rolloutFrames=5,
                shuffleControl=False,
                seeds=(0, 1, 2),
            )
            hub.close()

        rank = results["mean_collapse_rank"]
        assert 0.0 <= rank <= 1.0

    def test_output_file_written(self) -> None:
        from ainimator.health.hub import buildHealthHub

        with tempfile.TemporaryDirectory() as tmp:
            outputDir = Path(tmp)
            hub = buildHealthHub(outputDir)
            model = _makeMinimalControllerModel()
            stateNorm = _makeIdentityNormalizer()
            deltaNorm = _makeIdentityNormalizer()

            hub.diagnoseController(
                model=model,
                stateNormalizer=stateNorm,
                deltaNormalizer=deltaNorm,
                device=torch.device("cpu"),
                rolloutFrames=5,
                shuffleControl=False,
                seeds=(0, 1),
            )
            hub.close()

            diagFile = outputDir / "health" / "diagnose_controller.json"
            assert diagFile.exists(), (
                "diagnose_controller.json should be written"
            )


# ------------------------------------------------------------------
# Resolve helpers tests
# ------------------------------------------------------------------
class TestResolveHelpers:
    """Tests for the _resolve / _resolveBool helpers."""

    def test_resolve_cli_wins_over_profile(self) -> None:
        from ainimator.cli.health import _resolve

        assert _resolve("cli_val", "profile_val", "default") == "cli_val"

    def test_resolve_profile_wins_over_default(self) -> None:
        from ainimator.cli.health import _resolve

        assert _resolve(None, "profile_val", "default") == "profile_val"

    def test_resolve_returns_default_when_both_none(self) -> None:
        from ainimator.cli.health import _resolve

        assert _resolve(None, None, "default") == "default"

    def test_resolve_bool_handles_string_true(self) -> None:
        from ainimator.cli.health import _resolveBool

        assert _resolveBool(None, "true", False) is True

    def test_resolve_bool_handles_string_false(self) -> None:
        from ainimator.cli.health import _resolveBool

        assert _resolveBool(None, "false", True) is False
