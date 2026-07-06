"""Phase A1 tests — MotionController forward, shapes, ONNX-friendliness."""

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
        globalWindow=torch.randn(batch, window, 4),
    )
    assert isinstance(out, ControllerOutput)
    assert out.boneDelta.shape == (batch, 22, 6)
    assert out.globalDelta is not None
    assert out.globalDelta.shape == (batch, 4)


def test_forward_with_explicit_phase() -> None:
    config = _config(phaseMode=PhaseMode.EXPLICIT)
    model = MotionController(config)
    out = model(
        torch.randn(3, 1, 22, 6),
        torch.randn(3, config.controlChannels),
        globalWindow=torch.randn(3, 1, 4),
        phase=torch.randn(3, config.phaseChannels),
    )
    assert out.boneDelta.shape == (3, 22, 6)


def test_missing_phase_raises_when_required() -> None:
    """Shape validation via validateInputs() raises for missing phase.

    Validation is intentionally absent from forward() to keep the ONNX
    graph free of data-dependent control flow (G-ONNX).  Callers invoke
    validateInputs() before the first step.
    """
    config = _config(phaseMode=PhaseMode.EXPLICIT)
    model = MotionController(config)
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    with pytest.raises(ValueError, match="phase is required"):
        model.validateInputs(bone, ctrl, glob, None, None, None)


def test_wrong_window_length_raises() -> None:
    """Shape validation via validateInputs() raises for wrong window length."""
    config = _config(contextFrames=2, phaseMode=PhaseMode.NONE)
    model = MotionController(config)
    bone = torch.randn(2, 1, 22, 6)  # window=1 but config wants 2
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    with pytest.raises(ValueError, match="contextFrames"):
        model.validateInputs(bone, ctrl, glob, None, None, None)


def test_context_window_greater_than_one() -> None:
    config = _config(contextFrames=4, phaseMode=PhaseMode.NONE)
    model = MotionController(config)
    out = model(
        torch.randn(2, 4, 22, 6),
        torch.randn(2, config.controlChannels),
        globalWindow=torch.randn(2, 4, 4),
    )
    assert out.boneDelta.shape == (2, 22, 6)


def test_forward_is_deterministic_in_eval() -> None:
    config = _config(phaseMode=PhaseMode.NONE)
    model = MotionController(config).eval()
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    first = model(bone, ctrl, globalWindow=glob).boneDelta
    second = model(bone, ctrl, globalWindow=glob).boneDelta
    assert torch.allclose(first, second)


def test_single_forward_is_onnx_exportable(tmp_path) -> None:
    """The single forward must export to ONNX (the §2.10 acquis)."""
    config = _config(phaseMode=PhaseMode.NONE)
    wrapper = ControllerForwardWrapper(MotionController(config)).eval()
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
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
    glob = torch.randn(4, 1, 4)
    out_a = model(bone, torch.zeros(4, config.controlChannels),
                  globalWindow=glob).boneDelta
    out_b = model(bone, torch.ones(4, config.controlChannels) * 5.0,
                  globalWindow=glob).boneDelta
    assert not torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# LOT-2: integrated text encoder (promptEmb conditioning)
# ---------------------------------------------------------------------------

def test_prompt_emb_channels_zero_backward_compatible() -> None:
    """promptEmbChannels=0 (default) — forward with no promptEmb unchanged."""
    config = _config(phaseMode=PhaseMode.NONE, promptEmbChannels=0)
    model = MotionController(config).eval()
    assert model.nullPromptEmb is None
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    out = model(bone, ctrl, globalWindow=glob)
    assert out.boneDelta.shape == (2, 22, 6)


def test_prompt_emb_with_explicit_emb() -> None:
    """Forward with promptEmbChannels > 0 and an explicit promptEmb tensor."""
    config = _config(phaseMode=PhaseMode.NONE, promptEmbChannels=32)
    model = MotionController(config).eval()
    assert model.nullPromptEmb is not None
    assert model.nullPromptEmb.shape == (32,)
    bone = torch.randn(3, 1, 22, 6)
    ctrl = torch.randn(3, config.controlChannels)
    glob = torch.randn(3, 1, 4)
    promptEmb = torch.randn(3, 32)
    out = model(bone, ctrl, globalWindow=glob, promptEmb=promptEmb)
    assert out.boneDelta.shape == (3, 22, 6)


def test_prompt_emb_none_uses_null_embedding() -> None:
    """Forward with promptEmb=None uses the learnable null embedding."""
    config = _config(phaseMode=PhaseMode.NONE, promptEmbChannels=32)
    model = MotionController(config).eval()
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    # Should run without error using the null embedding path.
    out = model(bone, ctrl, globalWindow=glob, promptEmb=None)
    assert out.boneDelta.shape == (2, 22, 6)


def test_null_emb_vs_explicit_emb_differ() -> None:
    """Null embedding and explicit zero-vector embedding yield same result.

    When the null embedding is zero-initialized (fresh model) and we pass
    an explicit all-zeros tensor of the same shape, outputs must be equal.
    This validates that the null embedding is correctly substituted.
    """
    config = _config(phaseMode=PhaseMode.NONE, promptEmbChannels=16)
    model = MotionController(config).eval()
    # Force null embedding to zeros for determinism.
    assert model.nullPromptEmb is not None
    model.nullPromptEmb.data.zero_()
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    zero_emb = torch.zeros(2, 16)
    with torch.no_grad():
        out_null = model(bone, ctrl, globalWindow=glob, promptEmb=None)
        out_zero = model(bone, ctrl, globalWindow=glob, promptEmb=zero_emb)
    assert torch.allclose(out_null.boneDelta, out_zero.boneDelta), (
        "null embedding path and explicit zero-emb path must be identical "
        "when nullPromptEmb is all zeros."
    )


def test_prompt_emb_wrong_channels_raises() -> None:
    """validateInputs() raises ValueError when promptEmb channels mismatch."""
    config = _config(phaseMode=PhaseMode.NONE, promptEmbChannels=32)
    model = MotionController(config).eval()
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    bad_emb = torch.randn(2, 16)  # 16 != 32
    with pytest.raises(ValueError, match="promptEmb has"):
        model.validateInputs(bone, ctrl, glob, bad_emb, None, None)


def test_prompt_emb_no_grad_to_frozen_encoder() -> None:
    """nullPromptEmb receives gradients; a frozen upstream tensor does not."""
    config = _config(phaseMode=PhaseMode.NONE, promptEmbChannels=16)
    model = MotionController(config).train()
    assert model.nullPromptEmb is not None
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    # Simulate a frozen encoder: detach the embedding before passing it.
    frozen_emb = torch.randn(2, 16).requires_grad_(False)
    out = model(bone, ctrl, globalWindow=glob, promptEmb=frozen_emb)
    out.boneDelta.sum().backward()
    # nullPromptEmb has grad only when used — here promptEmb is provided
    # explicitly so the null path is NOT taken; null grad must be None.
    assert model.nullPromptEmb.grad is None


def test_null_emb_receives_grad_when_used() -> None:
    """nullPromptEmb must have a gradient when the null path is taken."""
    config = _config(phaseMode=PhaseMode.NONE, promptEmbChannels=16)
    model = MotionController(config).train()
    assert model.nullPromptEmb is not None
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    out = model(bone, ctrl, globalWindow=glob, promptEmb=None)
    out.boneDelta.sum().backward()
    assert model.nullPromptEmb.grad is not None


def test_conditioning_channels_property_includes_prompt() -> None:
    """conditioningChannels must include promptEmbChannels."""
    config = _config(
        phaseMode=PhaseMode.NONE,
        promptEmbChannels=32,
        useAimDirection=False,
    )
    # control=2, phase=0, style=0, prompt=32 → 34
    assert config.conditioningChannels == config.controlChannels + 32


def test_prompt_emb_onnx_exportable(tmp_path) -> None:
    """A controller with promptEmbChannels > 0 must export to ONNX."""
    config = _config(phaseMode=PhaseMode.NONE, promptEmbChannels=32)
    wrapper = ControllerForwardWrapper(MotionController(config)).eval()
    bone = torch.randn(2, 1, 22, 6)
    ctrl = torch.randn(2, config.controlChannels)
    glob = torch.randn(2, 1, 4)
    prompt = torch.randn(2, 32)
    destination = tmp_path / "controller_prompt.onnx"
    torch.onnx.export(
        wrapper,
        (bone, ctrl, glob, prompt),
        str(destination),
        input_names=["bone_window", "control", "global_window", "prompt_emb"],
        output_names=["bone_delta", "global_delta"],
        dynamic_axes={
            "bone_window": {0: "batch"},
            "control": {0: "batch"},
            "global_window": {0: "batch"},
            "prompt_emb": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    assert destination.exists() and destination.stat().st_size > 0
