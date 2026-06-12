"""Isolated unit tests for the Pydantic config schema.

Coverage:
* Unknown keys raise ``ValidationError`` naming the offending field.
* Wrong types raise ``ValidationError`` naming the offending field.
* Valid minimal configs (required fields only) parse without error.
* All defaults match the runtime dataclass defaults.
* Sub-schema field-level validators (e.g. ``clipMaxLength``) fire.
* Cross-field validators (e.g. ``validationFraction`` in (0, 1)) fire.
* :func:`writeResolvedConfig` produces a readable YAML file with the
  expected top-level keys (``git_sha``, ``timestamp``, ``config``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.shared.config_schema import (
    DiffusionConfigSchema,
    LossesConfigSchema,
    MotionDenoiserV2ConfigSchema,
    RegularisationConfigSchema,
    TextEncoderConfigSchema,
    V2FullTrainingConfigSchema,
    V2TrainingConfigSchema,
    ValidationConfigSchema,
)
from src.shared.resolved_config import writeResolvedConfig


# =====================================================================
# Helpers
# =====================================================================

def _makeMinimalFullConfig(tmp_path: Path) -> dict:
    """Return the minimum dict needed by V2FullTrainingConfigSchema."""
    return {
        "datasetRoot": str(tmp_path / "dataset"),
        "tokenizerDir": str(tmp_path / "tokenizer"),
        "outputDir": str(tmp_path / "output"),
    }


def _makeMinimalOverfitConfig(tmp_path: Path) -> dict:
    """Return the minimum dict needed by V2TrainingConfigSchema."""
    return {
        "datasetRoot": str(tmp_path / "dataset"),
        "tokenizerDir": str(tmp_path / "tokenizer"),
        "outputDir": str(tmp_path / "output"),
    }


# =====================================================================
# Unknown-key rejection tests
# =====================================================================

def test_full_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    """Unknown top-level key must name the offending field in the error."""
    raw = _makeMinimalFullConfig(tmp_path)
    raw["unknownField"] = "oops"
    with pytest.raises(ValidationError) as excInfo:
        V2FullTrainingConfigSchema.model_validate(raw)
    assert "unknownField" in str(excInfo.value)


def test_overfit_config_rejects_unknown_key(tmp_path: Path) -> None:
    """Unknown key in V2TrainingConfigSchema raises ValidationError."""
    raw = _makeMinimalOverfitConfig(tmp_path)
    raw["notAField"] = 99
    with pytest.raises(ValidationError) as excInfo:
        V2TrainingConfigSchema.model_validate(raw)
    assert "notAField" in str(excInfo.value)


def test_diffusion_config_rejects_unknown_key() -> None:
    """Unknown key in DiffusionConfigSchema raises ValidationError."""
    with pytest.raises(ValidationError) as excInfo:
        DiffusionConfigSchema.model_validate({"bogusKey": 1})
    assert "bogusKey" in str(excInfo.value)


def test_losses_config_rejects_unknown_key() -> None:
    """Unknown key in LossesConfigSchema raises ValidationError."""
    with pytest.raises(ValidationError) as excInfo:
        LossesConfigSchema.model_validate({"mystery": 0.5})
    assert "mystery" in str(excInfo.value)


def test_denoiser_config_rejects_unknown_key() -> None:
    """Unknown key in MotionDenoiserV2ConfigSchema raises ValidationError."""
    with pytest.raises(ValidationError) as excInfo:
        MotionDenoiserV2ConfigSchema.model_validate({"badField": True})
    assert "badField" in str(excInfo.value)


def test_text_encoder_config_rejects_unknown_key() -> None:
    """Unknown key in TextEncoderConfigSchema raises ValidationError."""
    with pytest.raises(ValidationError) as excInfo:
        TextEncoderConfigSchema.model_validate({"wrongKey": 42})
    assert "wrongKey" in str(excInfo.value)


# =====================================================================
# Wrong-type rejection tests
# =====================================================================

def test_full_config_rejects_wrong_type_for_epochs(
    tmp_path: Path,
) -> None:
    """String value for ``epochs`` must raise ValidationError."""
    raw = _makeMinimalFullConfig(tmp_path)
    raw["epochs"] = "notAnInt"
    with pytest.raises(ValidationError) as excInfo:
        V2FullTrainingConfigSchema.model_validate(raw)
    assert "epochs" in str(excInfo.value)


def test_full_config_rejects_wrong_type_for_learning_rate(
    tmp_path: Path,
) -> None:
    """List value for ``learningRate`` must raise ValidationError."""
    raw = _makeMinimalFullConfig(tmp_path)
    raw["learningRate"] = [0.1, 0.2]
    with pytest.raises(ValidationError) as excInfo:
        V2FullTrainingConfigSchema.model_validate(raw)
    assert "learningRate" in str(excInfo.value)


def test_diffusion_config_rejects_invalid_schedule_type() -> None:
    """Invalid literal value for ``scheduleType`` raises ValidationError."""
    with pytest.raises(ValidationError) as excInfo:
        DiffusionConfigSchema.model_validate({"scheduleType": "quadratic"})
    assert "scheduleType" in str(excInfo.value)


def test_diffusion_config_rejects_invalid_prediction_mode() -> None:
    """Invalid literal value for ``predictionMode`` raises ValidationError."""
    with pytest.raises(ValidationError) as excInfo:
        DiffusionConfigSchema.model_validate({"predictionMode": "flux"})
    assert "predictionMode" in str(excInfo.value)


# =====================================================================
# Field-level validator tests
# =====================================================================

def test_encoder_config_rejects_clip_max_length_zero() -> None:
    """clipMaxLength=0 must raise (must be in [1, 77])."""
    with pytest.raises(ValidationError) as excInfo:
        TextEncoderConfigSchema.model_validate({"clipMaxLength": 0})
    assert "clipMaxLength" in str(excInfo.value)


def test_encoder_config_rejects_clip_max_length_too_large() -> None:
    """clipMaxLength=78 must raise (must be in [1, 77])."""
    with pytest.raises(ValidationError) as excInfo:
        TextEncoderConfigSchema.model_validate({"clipMaxLength": 78})
    assert "clipMaxLength" in str(excInfo.value)


def test_encoder_config_accepts_clip_max_length_boundary() -> None:
    """clipMaxLength=77 must be accepted."""
    schema = TextEncoderConfigSchema.model_validate({"clipMaxLength": 77})
    assert schema.clipMaxLength == 77


# =====================================================================
# Cross-field validator tests
# =====================================================================

def test_full_config_rejects_validation_fraction_zero(
    tmp_path: Path,
) -> None:
    """validationFraction=0.0 must raise (must be in (0, 1))."""
    raw = _makeMinimalFullConfig(tmp_path)
    raw["validation"] = {"validationFraction": 0.0}
    with pytest.raises(ValidationError):
        V2FullTrainingConfigSchema.model_validate(raw)


def test_full_config_rejects_validation_fraction_one(
    tmp_path: Path,
) -> None:
    """validationFraction=1.0 must raise (must be in (0, 1))."""
    raw = _makeMinimalFullConfig(tmp_path)
    raw["validation"] = {"validationFraction": 1.0}
    with pytest.raises(ValidationError):
        V2FullTrainingConfigSchema.model_validate(raw)


def test_full_config_rejects_guidance_weight_start_too_small(
    tmp_path: Path,
) -> None:
    """clipGuidanceWeightStart < clipGuidanceWeight must raise."""
    raw = _makeMinimalFullConfig(tmp_path)
    raw["losses"] = {
        "clipGuidanceWeight": 0.5,
        "clipGuidanceWeightStart": 0.3,  # < clipGuidanceWeight
    }
    with pytest.raises(ValidationError):
        V2FullTrainingConfigSchema.model_validate(raw)


def test_full_config_rejects_negative_epochs(tmp_path: Path) -> None:
    """epochs=0 must raise."""
    raw = _makeMinimalFullConfig(tmp_path)
    raw["epochs"] = 0
    with pytest.raises(ValidationError):
        V2FullTrainingConfigSchema.model_validate(raw)


# =====================================================================
# Valid config tests
# =====================================================================

def test_full_config_minimal_parses(tmp_path: Path) -> None:
    """Minimal config (required fields only) must parse without error."""
    schema = V2FullTrainingConfigSchema.model_validate(
        _makeMinimalFullConfig(tmp_path)
    )
    assert schema.epochs == 100
    assert schema.batchSize == 8
    assert schema.diffusion.scheduleType == "cosine"
    assert schema.diffusion.predictionMode == "v"
    assert schema.diffusion.minSnrGamma == 5.0
    assert schema.encoder.encoderHiddenDim == 256
    assert schema.denoiser.denoiserEmbedDim == 384
    assert schema.regularisation.condMaskProb == 0.20


def test_overfit_config_minimal_parses(tmp_path: Path) -> None:
    """Minimal overfit config must parse and expose correct defaults."""
    schema = V2TrainingConfigSchema.model_validate(
        _makeMinimalOverfitConfig(tmp_path)
    )
    assert schema.sampleLinkIndex == 0
    assert schema.epochs == 200
    assert schema.condMaskProb == 0.0
    assert schema.dropout == 0.0
    assert schema.diffusion.predictionMode == "v"
    assert schema.denoiser.useFilmConditioning is True
    assert schema.denoiser.usePerBlockFilm is True


def test_full_config_denoiser_dropout_default_is_0_1(
    tmp_path: Path,
) -> None:
    """Full config with dropout omitted must resolve denoiser dropout=0.1.

    Matches V2FullTrainingConfig.dropout runtime default (0.1).
    """
    schema = V2FullTrainingConfigSchema.model_validate(
        _makeMinimalFullConfig(tmp_path)
    )
    assert schema.denoiser.dropout == 0.1


def test_overfit_config_denoiser_dropout_default_is_0_0(
    tmp_path: Path,
) -> None:
    """Overfit config with dropout omitted must resolve denoiser dropout=0.0.

    Matches V2TrainingConfig.dropout runtime default (0.0).
    """
    schema = V2TrainingConfigSchema.model_validate(
        _makeMinimalOverfitConfig(tmp_path)
    )
    assert schema.denoiser.dropout == 0.0


def test_full_config_nested_override(tmp_path: Path) -> None:
    """Nested sub-schema overrides must be accepted and applied."""
    raw = _makeMinimalFullConfig(tmp_path)
    raw["diffusion"] = {"scheduleType": "linear", "minSnrGamma": 0.0}
    schema = V2FullTrainingConfigSchema.model_validate(raw)
    assert schema.diffusion.scheduleType == "linear"
    assert schema.diffusion.minSnrGamma == 0.0


# =====================================================================
# writeResolvedConfig tests
# =====================================================================

def test_write_resolved_config_creates_file(tmp_path: Path) -> None:
    """writeResolvedConfig must create resolved_config.yaml."""
    schema = V2TrainingConfigSchema.model_validate(
        _makeMinimalOverfitConfig(tmp_path)
    )
    dest = writeResolvedConfig(schema, tmp_path)
    assert dest.exists()
    assert dest.name == "resolved_config.yaml"


def test_write_resolved_config_contains_required_keys(
    tmp_path: Path,
) -> None:
    """The written YAML must contain git_sha, timestamp, config."""
    schema = V2TrainingConfigSchema.model_validate(
        _makeMinimalOverfitConfig(tmp_path)
    )
    writeResolvedConfig(schema, tmp_path)
    payload = yaml.safe_load(
        (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert "git_sha" in payload
    assert "timestamp" in payload
    assert "config" in payload


def test_write_resolved_config_embeds_field_values(
    tmp_path: Path,
) -> None:
    """The config section must contain the actual field values."""
    schema = V2FullTrainingConfigSchema.model_validate(
        {**_makeMinimalFullConfig(tmp_path), "epochs": 42}
    )
    writeResolvedConfig(schema, tmp_path)
    payload = yaml.safe_load(
        (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert payload["config"]["epochs"] == 42


def test_write_resolved_config_accepts_dataclass(
    tmp_path: Path,
) -> None:
    """writeResolvedConfig must accept plain dataclasses too."""

    @dataclasses.dataclass(frozen=True)
    class SimpleConfig:
        """A minimal dataclass config for testing."""

        value: int = 7
        label: str = "test"

    writeResolvedConfig(SimpleConfig(), tmp_path)
    payload = yaml.safe_load(
        (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert payload["config"]["value"] == 7
    assert payload["config"]["label"] == "test"


def test_write_resolved_config_raises_on_missing_dir(
    tmp_path: Path,
) -> None:
    """Passing a non-existent outputDir must raise FileNotFoundError."""
    schema = V2TrainingConfigSchema.model_validate(
        _makeMinimalOverfitConfig(tmp_path)
    )
    missingDir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        writeResolvedConfig(schema, missingDir)
