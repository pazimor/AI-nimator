"""Phase C0 tests — deterministic controller state, flags and schema.

Covers ROADMAP_DETERMINIST C0 acceptance:
* a controller YAML block with an unknown key raises a validation error
  that *names* the offending field;
* the state dataclasses and architecture config behave as frozen;
* the network.yaml selector loads with the canonical default
  (``model-type: diffusion``) and never breaks the diffusion path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from ainimator.core.config_loader import (
    buildControllerV2Config,
    loadGenerationModelSelector,
)
from ainimator.core.config_schema import (
    ControllerConfigSchema,
    GenerationModelSelectorSchema,
)
from ainimator.core.constants.controller import (
    CONTROL_PLANAR_VELOCITY_CHANNELS,
    LEAN_STATE_CHANNELS,
    PHASE_CHANNELS,
    GenerationModelType,
    PhaseMode,
    controlSignalChannels,
)
from ainimator.core.types import (
    ControlSignal,
    ControllerState,
    ControllerV2Config,
    StylePreset,
)

_NETWORK_YAML = Path("src/configs/network.yaml")


# ---------------------------------------------------------------------
# Constants / channel layout
# ---------------------------------------------------------------------
def test_lean_state_channels_match_smpl22() -> None:
    assert LEAN_STATE_CHANNELS == 22 * 6 + 3


def test_control_signal_channels_toggle_aim() -> None:
    assert controlSignalChannels(False) == CONTROL_PLANAR_VELOCITY_CHANNELS
    assert controlSignalChannels(True) == (
        CONTROL_PLANAR_VELOCITY_CHANNELS + 2
    )


# ---------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------
def test_controller_state_is_frozen_and_reports_bones() -> None:
    state = ControllerState(
        rotation6d=torch.zeros(4, 22, 6),
        rootTranslation=torch.zeros(4, 3),
    )
    assert state.numBones == 22
    with pytest.raises(AttributeError):
        state.rotation6d = torch.zeros(1)  # type: ignore[misc]


def test_control_signal_aim_optional() -> None:
    signal = ControlSignal(desiredPlanarVelocity=torch.zeros(4, 2))
    assert signal.aimDirection is None


def test_style_preset_locked_default() -> None:
    preset = StylePreset(name="ninja", vector=torch.zeros(64))
    assert preset.locked is True


# ---------------------------------------------------------------------
# Architecture config
# ---------------------------------------------------------------------
def test_controller_config_conditioning_width_explicit_no_aim() -> None:
    config = ControllerV2Config(
        phaseMode=PhaseMode.EXPLICIT, useAimDirection=False
    )
    assert config.controlChannels == 2
    assert config.phaseChannels == PHASE_CHANNELS
    assert config.styleChannels == 0
    assert config.conditioningChannels == 2 + PHASE_CHANNELS


def test_controller_config_conditioning_width_rich_control() -> None:
    config = ControllerV2Config(
        phaseMode=PhaseMode.EXPLICIT, useAimDirection=True
    )
    assert config.controlChannels == 4
    assert config.conditioningChannels == 4 + PHASE_CHANNELS


def test_controller_config_phase_none_drops_phase_channels() -> None:
    config = ControllerV2Config(phaseMode=PhaseMode.NONE)
    assert config.phaseChannels == 0


def test_controller_config_output_dims() -> None:
    config = ControllerV2Config(numBones=22, motionChannels=6, globalChannels=3)
    assert config.boneOutputDim == 132
    assert config.totalOutputDim == 135


def test_controller_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ControllerV2Config(embedDim=250, numHeads=8)


# ---------------------------------------------------------------------
# Pydantic schema — unknown-key guard (C0 acceptance)
# ---------------------------------------------------------------------
def test_controller_schema_unknown_key_names_field() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ControllerConfigSchema.model_validate({"phaze": "explicit"})
    assert "phaze" in str(excinfo.value)


def test_controller_schema_unknown_nested_key_names_field() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ControllerConfigSchema.model_validate(
            {"losses": {"velocity-loss": True, "bogus-loss": True}}
        )
    assert "bogus-loss" in str(excinfo.value)


def test_controller_schema_kebab_aliases_parse() -> None:
    schema = ControllerConfigSchema.model_validate(
        {"phase": "explicit", "context-frames": 3, "style-latent": False}
    )
    assert schema.contextFrames == 3
    assert schema.styleLatent is False


def test_controller_schema_style_latent_true_is_blocked() -> None:
    with pytest.raises(ValidationError, match="style-latent"):
        ControllerConfigSchema.model_validate({"style-latent": True})


def test_controller_schema_bad_phase_rejected() -> None:
    with pytest.raises(ValidationError):
        ControllerConfigSchema.model_validate({"phase": "magic"})


def test_selector_requires_controller_block_when_controller() -> None:
    with pytest.raises(ValidationError, match="controller"):
        GenerationModelSelectorSchema.model_validate(
            {"model-type": "controller"}
        )


# ---------------------------------------------------------------------
# Loader — canonical network.yaml stays diffusion
# ---------------------------------------------------------------------
def test_network_yaml_default_is_diffusion() -> None:
    selector = loadGenerationModelSelector(_NETWORK_YAML, profile="v2")
    assert selector.modelType == "diffusion"
    assert GenerationModelType(selector.modelType) is (
        GenerationModelType.DIFFUSION
    )


def test_network_yaml_controller_block_parses() -> None:
    selector = loadGenerationModelSelector(_NETWORK_YAML, profile="v2")
    assert selector.controller is not None
    assert selector.controller.phase == "explicit"
    assert selector.controller.contextFrames == 1
    assert selector.controller.styleLatent is False


def test_build_controller_v2_config_from_selector() -> None:
    selector = loadGenerationModelSelector(_NETWORK_YAML, profile="v2")
    config = buildControllerV2Config(
        selector, embedDim=256, numHeads=8, numLayers=4
    )
    assert config.phaseMode is PhaseMode.EXPLICIT
    assert config.contextFrames == 1
    assert config.styleLatentEnabled is False
