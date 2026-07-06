"""Tests for the temporal motion encoder."""

from __future__ import annotations

import torch

from ainimator.model.layers.temporal_unet import TemporalUNet


def test_temporal_unet_disables_nested_tensor_fast_path() -> None:
    """MPS validation with padding masks requires nested tensor to stay off."""
    model = TemporalUNet(embedDim=32, numHeads=4, numLayers=2)

    assert model.transformer.enable_nested_tensor is False


def test_temporal_unet_supports_late_fusion_of_bone_and_global_branches() -> None:
    """The motion encoder must accept separate bone/global branches."""
    model = TemporalUNet(
        embedDim=32,
        numHeads=4,
        numLayers=2,
        numBones=3,
        numChannels=6,
        globalChannels=2,
    )
    boneInput = torch.randn(2, 5, 3, 6)
    globalInput = torch.randn(2, 5, 2)

    output = model(
        boneInput=boneInput,
        globalInput=globalInput,
    )

    assert output.shape == (2, 32)
    assert model.globalTransformer is not None
    assert model.globalTransformer.enable_nested_tensor is False
