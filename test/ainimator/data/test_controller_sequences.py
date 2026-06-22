"""Phase C1 tests — autoregressive sequence builder."""

from __future__ import annotations

import pytest
import torch

from ainimator.data.controller_sequences import (
    ControllerSequenceConfig,
    buildControllerSequences,
)


def _clip(frames: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    rotation6d = torch.randn(frames, 22, 6)
    rootTranslation = torch.randn(frames, 3)
    return rotation6d, rootTranslation


def test_transition_count_context_one() -> None:
    rot, root = _clip(10)
    batch = buildControllerSequences(
        rot, root, ControllerSequenceConfig(contextFrames=1)
    )
    assert batch.numTransitions == 9
    assert batch.boneWindow.shape == (9, 1, 22, 6)
    assert batch.control.shape == (9, 2)


def test_transition_count_context_three() -> None:
    rot, root = _clip(10)
    batch = buildControllerSequences(
        rot, root, ControllerSequenceConfig(contextFrames=3)
    )
    assert batch.numTransitions == 7
    assert batch.boneWindow.shape == (7, 3, 22, 6)


def test_delta_targets_are_consistent() -> None:
    rot, root = _clip(8)
    batch = buildControllerSequences(
        rot, root, ControllerSequenceConfig(contextFrames=1)
    )
    # target delta == next - last-of-window
    lastBone = batch.boneWindow[:, -1, :, :]
    assert torch.allclose(
        batch.targetBoneDelta, batch.targetBoneNext - lastBone, atol=1e-5
    )


def test_control_is_planar_velocity() -> None:
    rot, root = _clip(6)
    batch = buildControllerSequences(
        rot, root, ControllerSequenceConfig(contextFrames=1)
    )
    expected = batch.targetGlobalDelta[:, [0, 2]]
    assert torch.allclose(batch.control, expected, atol=1e-5)


def test_aim_direction_appends_two_channels() -> None:
    rot, root = _clip(6)
    batch = buildControllerSequences(
        rot,
        root,
        ControllerSequenceConfig(contextFrames=1, useAimDirection=True),
    )
    assert batch.control.shape[-1] == 4
    aim = batch.control[:, 2:]
    norms = aim.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_clip_too_short_raises() -> None:
    rot, root = _clip(2)
    with pytest.raises(ValueError, match="too short"):
        buildControllerSequences(
            rot, root, ControllerSequenceConfig(contextFrames=4)
        )
