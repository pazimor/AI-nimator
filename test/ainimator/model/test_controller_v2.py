"""Phase C1 tests — MotionController forward, shapes, ONNX-friendliness."""

from __future__ import annotations

import pytest
import torch

from ainimator.core.constants.controller import PhaseMode
from ainimator.core.types.controller import ControllerV2Config
from ainimator.model.controller_v2 import (
    ControllerForwardWrapper,
    ControllerOutput,
    MotionController,
)


def _config(**overrides: object) -> ControllerV2Config:
    base = dict(embedDim=64, numHeads=4, numLayers=2, contextFrames=1)
    base.update(overrides)
    return ControllerV2Config(**base)  # type: ignore[arg-type]


def test_forward_output_shapes_no_phase() -> None:
    config = _config(phaseMode=PhaseMode.NONE)
    model = MotionController(config)
    batch, window = 5, config.contextFrames
    out = model(
        torch.randn(batch, window, 22, 6),
        torch.randn(batch, config.controlChannels),
        globalWindow=torch.randn(batch, window, 3),
    )
    assert isinstance(out, ControllerOutput)
    assert out.boneDelta.shape == (batch, 22, 6)
    assert out.globalDelta is not None
    assert out.globalDelta.shape == (batch, 3)


def test_forward_with_explicit_phase() -> None:
    config = _config(phaseMode=PhaseMode.EXPLICIT)
    model = MotionController(config)
    out = model(
        torch.randn(3, 1, 22, 6),
        torch.randn(3, config.controlChannels),
        globalWindow=torch.randn(3, 1, 3),
        phase=torch.randn(3, config.phaseChannels),
    )
    assert out.boneDelta.shape == (3, 22, 6)


def test_missing_phase_raises_when_required() -> None:
    config = _config(phaseMode=PhaseMode.EXPLICIT)
    model = MotionController(config)
    with pytest.raises(ValueError, match="phase is required"):
        model(
            torch.randn(2, 1, 22, 6),
            torch.randn(2, config.controlChannels),
            globalWindow=torch.randn(2, 1, 3),
        )


def test_wrong_window_length_raises() -> None:
    config = _config(contextFrames=2, phaseMode=PhaseMode.NONE)
    model = MotionController(config)
    with pytest.raises(ValueError, match="contextFrames"):
        model(
            torch.randn(2, 1, 22, 6),
            torch.randn(2, config.controlChannels),
            globalWindow=torch.randn(2, 1, 3),
        )


def test_context_window_greater_than_one() -> None:
    config = _config(contextFrames=4, phaseMode=PhaseMode.NONE)
    model = MotionController(config)
    out = model(
        torch.randn(2, 4, 22, 6),
        torch.randn(2, config.controlChannels),
        globalWindow=torch.randn(2, 4, 3),
    )
    assert out.boneDelta.shape == (2, 22, 6)


def test_forward_is_deterministic_in_eval() -> None:
    config = _config(phaseMode=PhaseMode.NONE)
    model = MotionController(config).eval()
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 3)
    first = model(bone, ctrl, globalWindow=glob).boneDelta
    second = model(bone, ctrl, globalWindow=glob).boneDelta
    assert torch.allclose(first, second)


def test_single_forward_is_onnx_exportable(tmp_path) -> None:
    """The single forward must export to ONNX (the §2.10 acquis)."""
    config = _config(phaseMode=PhaseMode.NONE)
    wrapper = ControllerForwardWrapper(MotionController(config)).eval()
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 3)
    destination = tmp_path / "controller.onnx"
    torch.onnx.export(
        wrapper,
        (bone, ctrl, glob),
        str(destination),
        input_names=["bone_window", "control", "global_window"],
        output_names=["bone_delta", "global_delta"],
        dynamic_axes={
            "bone_window": {0: "batch"},
            "control": {0: "batch"},
            "global_window": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    assert destination.exists() and destination.stat().st_size > 0


def test_control_modulates_output() -> None:
    """Different control must change the output (anti-collapse init)."""
    config = _config(phaseMode=PhaseMode.NONE)
    model = MotionController(config).eval()
    bone = torch.randn(4, 1, 22, 6)
    glob = torch.randn(4, 1, 3)
    out_a = model(bone, torch.zeros(4, config.controlChannels),
                  globalWindow=glob).boneDelta
    out_b = model(bone, torch.ones(4, config.controlChannels) * 5.0,
                  globalWindow=glob).boneDelta
    assert not torch.allclose(out_a, out_b)
