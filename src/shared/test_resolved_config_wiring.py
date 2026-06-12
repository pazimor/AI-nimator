"""Integration tests verifying resolved_config.yaml wiring.

Tests confirm that both training entry-points write
``resolved_config.yaml`` into ``outputDir`` immediately after the
directory is created.

These tests mock the heavy dataset/model dependencies so they run in
under one second without GPU access.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shared.resolved_config import (
    RESOLVED_CONFIG_FILENAME,
    writeResolvedConfig,
    _gitSha,
    _toSerializable,
    _configToDict,
)


# =====================================================================
# Unit tests for helper functions
# =====================================================================

def test_git_sha_returns_string() -> None:
    """_gitSha must always return a non-empty string."""
    sha = _gitSha()
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_git_sha_returns_untracked_on_bad_git(tmp_path: Path) -> None:
    """_gitSha must return 'untracked' when git is unavailable."""
    with patch("subprocess.run", side_effect=OSError("no git")):
        sha = _gitSha()
    assert sha == "untracked"


def test_to_serializable_converts_path() -> None:
    """Path values must be converted to strings."""
    result = _toSerializable(Path("/tmp/foo"))
    assert result == "/tmp/foo"
    assert isinstance(result, str)


def test_to_serializable_converts_nested_dict_paths() -> None:
    """Nested Path values inside dicts must be stringified."""
    raw = {"a": {"b": Path("/x/y")}}
    result = _toSerializable(raw)
    assert result == {"a": {"b": "/x/y"}}


def test_to_serializable_converts_tuple_to_list() -> None:
    """Tuple values must be converted to lists."""
    result = _toSerializable((1, 2, 3))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_to_serializable_passes_none_through() -> None:
    """None must be returned unchanged."""
    assert _toSerializable(None) is None


def test_config_to_dict_handles_pydantic_model() -> None:
    """Pydantic BaseModel must be converted via model_dump."""
    from pydantic import BaseModel

    class SampleModel(BaseModel, extra="forbid"):
        """Test model."""

        value: int = 5
        label: str = "ok"

    result = _configToDict(SampleModel())
    assert result == {"value": 5, "label": "ok"}


def test_config_to_dict_handles_dataclass() -> None:
    """Plain dataclass must be converted via dataclasses.asdict."""

    @dataclasses.dataclass(frozen=True)
    class SampleDC:
        """Test dataclass."""

        count: int = 3
        name: str = "test"

    result = _configToDict(SampleDC())
    assert result == {"count": 3, "name": "test"}


def test_config_to_dict_raises_on_arbitrary_object() -> None:
    """Non-model, non-dataclass objects must raise TypeError."""
    with pytest.raises(TypeError):
        _configToDict(object())


# =====================================================================
# writeResolvedConfig integration tests
# =====================================================================

def test_write_resolved_config_overwrites_existing(
    tmp_path: Path,
) -> None:
    """A second call must overwrite the first resolved_config.yaml."""
    import yaml
    from src.shared.config_schema import V2TrainingConfigSchema

    schema1 = V2TrainingConfigSchema.model_validate({
        "datasetRoot": str(tmp_path / "d"),
        "tokenizerDir": str(tmp_path / "t"),
        "outputDir": str(tmp_path / "o"),
        "epochs": 10,
    })
    schema2 = V2TrainingConfigSchema.model_validate({
        "datasetRoot": str(tmp_path / "d"),
        "tokenizerDir": str(tmp_path / "t"),
        "outputDir": str(tmp_path / "o"),
        "epochs": 99,
    })
    writeResolvedConfig(schema1, tmp_path)
    writeResolvedConfig(schema2, tmp_path)
    payload = yaml.safe_load(
        (tmp_path / RESOLVED_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["config"]["epochs"] == 99


def test_write_resolved_config_paths_are_strings(
    tmp_path: Path,
) -> None:
    """Path fields in the config section must be serialised as strings."""
    import yaml
    from src.shared.config_schema import V2TrainingConfigSchema

    schema = V2TrainingConfigSchema.model_validate({
        "datasetRoot": str(tmp_path / "dataset"),
        "tokenizerDir": str(tmp_path / "tokenizer"),
        "outputDir": str(tmp_path / "output"),
    })
    writeResolvedConfig(schema, tmp_path)
    payload = yaml.safe_load(
        (tmp_path / RESOLVED_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    # Path fields should be strings in the output, not Path objects
    assert isinstance(payload["config"]["datasetRoot"], str)
    assert isinstance(payload["config"]["outputDir"], str)


# =====================================================================
# Wiring: overfit training loop
# =====================================================================

def test_run_overfit_writes_resolved_config(tmp_path: Path) -> None:
    """runOverfit must write resolved_config.yaml before loading data."""
    from src.features.generation.training_v2 import (
        V2TrainingConfig,
        runOverfit,
    )

    config = V2TrainingConfig(
        datasetRoot=tmp_path / "dataset",
        tokenizerDir=tmp_path / "tokenizer",
        outputDir=tmp_path / "out",
        epochs=1,
    )

    with (
        patch(
            "src.features.generation.training_v2.loadDatasetSample",
        ) as mockLoad,
        patch(
            "src.features.generation.training_v2.buildTrainingComponents",
        ) as mockBuild,
        patch(
            "src.features.generation.training_v2.trainStep",
        ) as mockStep,
        patch(
            "src.features.generation.training_v2.saveCheckpointV2",
        ),
    ):
        # Minimal mocks so the loop can run one iteration
        _fakeSample = MagicMock()
        _fakeSample.rotation6d = MagicMock()
        _fakeSample.rootTranslation = MagicMock()
        _fakeSample.sampleId = 0
        _fakeSample.textId = 0
        _fakeSample.numFrames = 10
        _fakeSample.rawText = "walk"
        mockLoad.return_value = _fakeSample

        _fakeComponents = MagicMock()
        mockBuild.return_value = _fakeComponents

        mockStep.return_value = {
            "loss_total": 0.5,
            "loss_bone": 0.25,
            "loss_global": 0.25,
            "loss_vel_xyz": 0.0,
            "timestep": 42,
        }

        config.outputDir.mkdir(parents=True, exist_ok=True)
        runOverfit(config)

    assert (config.outputDir / RESOLVED_CONFIG_FILENAME).exists()


# =====================================================================
# Wiring: full training loop
# =====================================================================

def test_run_full_training_writes_resolved_config(
    tmp_path: Path,
) -> None:
    """runFullTraining must write resolved_config.yaml before loading data."""
    from src.features.generation.full_training_v2 import (
        V2FullTrainingConfig,
        runFullTraining,
    )

    config = V2FullTrainingConfig(
        datasetRoot=tmp_path / "dataset",
        tokenizerDir=tmp_path / "tokenizer",
        outputDir=tmp_path / "out",
        epochs=1,
        maxSamplesPerEpoch=1,
    )

    _fakeEntry = MagicMock()
    _fakeEntry.datasetFolder = "ACCAD"
    _fakeDataset = MagicMock()
    _fakeDataset.linkEntries = [_fakeEntry]
    _fakeDataset.manifest = MagicMock()
    _fakeDataset.manifest.modelName = "test"

    with (
        patch(
            "src.features.generation.full_training_v2"
            ".PreprocessedLinkDataset",
            return_value=_fakeDataset,
        ),
        patch(
            "src.features.generation.full_training_v2"
            ".fitNormalizerFromDataset",
        ),
        patch(
            "src.features.generation.full_training_v2"
            ".buildFullTrainingComponents",
        ) as mockBuild,
        patch(
            "src.features.generation.full_training_v2.iterBatches",
            return_value=iter([]),
        ),
        patch(
            "src.features.generation.full_training_v2.validateEpoch",
            return_value={
                "loss_total": 0.5,
                "loss_bone": 0.25,
                "loss_global": 0.25,
                "loss_vel_xyz": 0.0,
                "loss_joint_xyz": 0.0,
                "loss_foot_contact": 0.0,
                "loss_clip_guidance": 0.0,
                "loss_clip_aux_pool": 0.0,
                "loss_x0_contrastive": 0.0,
            },
        ),
        # Patch the checkpoint saver to avoid MagicMock pickling errors
        patch(
            "src.features.generation.full_training_v2"
            ".saveTorchObjectAtomically",
        ),
    ):
        _fakeComponents = MagicMock()
        _fakeComponents.ema = None
        mockBuild.return_value = _fakeComponents

        config.outputDir.mkdir(parents=True, exist_ok=True)
        runFullTraining(config)

    assert (config.outputDir / RESOLVED_CONFIG_FILENAME).exists()
