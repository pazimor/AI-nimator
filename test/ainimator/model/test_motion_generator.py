"""Tests for motion generator construction."""

from __future__ import annotations

import torch
import torch.nn as nn

from ainimator.model.motion_generator import MotionGenerator


def test_motion_generator_passes_clip_motion_architecture(
    monkeypatch,
) -> None:
    """The frozen CLIP encoder must use the configured motion architecture."""
    captured: dict[str, object] = {}

    class FakeClipModel(nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            captured.update(kwargs)
            self.weight = nn.Parameter(torch.ones(1))

    monkeypatch.setattr(
        "ainimator.model.motion_generator.ClipModel",
        FakeClipModel,
    )

    model = MotionGenerator(
        embedDim=768,
        numHeads=1,
        numLayers=1,
        numBones=22,
        clipMotionNumHeads=8,
        clipMotionNumLayers=4,
    )

    assert captured["motionNumHeads"] == 8
    assert captured["motionNumLayers"] == 4
    assert captured["embedDim"] == 768
    assert all(not param.requires_grad for param in model.clip.parameters())
