"""Unit tests for :mod:`motion_alignment_encoder`."""

from __future__ import annotations

import pytest
import torch

from src.shared.model.generation.motion_alignment_encoder import (
    MotionAlignmentEncoder,
    MotionAlignmentEncoderConfig,
)

_BATCH = 3
_FRAMES = 12
_BONES = 22
_CHANNELS = 6
_TEXT_DIM = 384
_ALIGN_DIM = 128


def _buildEncoder() -> MotionAlignmentEncoder:
    """Construct a small encoder with the test geometry."""
    return MotionAlignmentEncoder(
        MotionAlignmentEncoderConfig(
            numBones=_BONES,
            motionChannels=_CHANNELS,
            embedDim=64,
            numLayers=2,
            numHeads=4,
            alignmentDim=_ALIGN_DIM,
            textDim=_TEXT_DIM,
            maxFrames=64,
            dropout=0.0,
        )
    )


def test_config_rejects_indivisible_heads() -> None:
    """embedDim must be divisible by numHeads."""
    with pytest.raises(ValueError):
        MotionAlignmentEncoderConfig(embedDim=65, numHeads=4)


def test_encode_motion_shape_and_unit_norm() -> None:
    """encodeMotion returns ``(B, alignmentDim)`` L2-normalised rows."""
    encoder = _buildEncoder().eval()
    motion = torch.randn(_BATCH, _FRAMES, _BONES, _CHANNELS)
    embedding = encoder.encodeMotion(motion)
    assert embedding.shape == (_BATCH, _ALIGN_DIM)
    norms = embedding.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_project_text_shape_and_unit_norm() -> None:
    """projectText maps ``(B, textDim)`` to a unit-norm alignment vector."""
    encoder = _buildEncoder().eval()
    pooled = torch.randn(_BATCH, _TEXT_DIM)
    projected = encoder.projectText(pooled)
    assert projected.shape == (_BATCH, _ALIGN_DIM)
    norms = projected.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_masked_mean_ignores_padded_frames() -> None:
    """Padded trailing frames must not change the pooled embedding."""
    encoder = _buildEncoder().eval()
    motion = torch.randn(1, _FRAMES, _BONES, _CHANNELS)
    mask = torch.zeros(1, _FRAMES, dtype=torch.bool)

    padded = torch.cat(
        [motion, torch.randn(1, 4, _BONES, _CHANNELS)], dim=1
    )
    paddedMask = torch.cat(
        [mask, torch.ones(1, 4, dtype=torch.bool)], dim=1
    )

    base = encoder.encodeMotion(motion, mask)
    withPad = encoder.encodeMotion(padded, paddedMask)
    assert torch.allclose(base, withPad, atol=1e-5)
