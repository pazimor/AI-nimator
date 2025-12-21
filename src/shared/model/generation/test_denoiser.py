"""Unit tests for the motion generation denoiser module."""

from __future__ import annotations

import pytest
import torch

from src.shared.model.generation.denoiser import (
    DenoiserBlock,
    MotionDenoiser,
    TagEmbedding,
    TimestepEmbedding,
)
from src.shared.model.generation.losses import (
    accelerationLoss,
    combinedGenerationLoss,
    diffusionLoss,
    geodesicLoss,
    velocityLoss,
)
from src.shared.types import VALID_TAGS, validateTag


class TestTimestepEmbedding:
    """Tests for TimestepEmbedding."""

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        embedDim = 64
        batchSize = 4
        emb = TimestepEmbedding(embedDim)
        timesteps = torch.randint(0, 1000, (batchSize,))
        output = emb(timesteps)
        assert output.shape == (batchSize, embedDim)

    def test_different_timesteps_different_embeddings(self) -> None:
        """Test that different timesteps produce different embeddings."""
        emb = TimestepEmbedding(64)
        t1 = torch.tensor([0])
        t2 = torch.tensor([500])
        e1 = emb(t1)
        e2 = emb(t2)
        assert not torch.allclose(e1, e2)


class TestTagEmbedding:
    """Tests for TagEmbedding."""

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        embedDim = 64
        emb = TagEmbedding(embedDim)
        tags = ["Dance", "Idle", "Combat"]
        output = emb(tags)
        assert output.shape == (3, embedDim)

    def test_all_valid_tags(self) -> None:
        """Test that all valid tags can be embedded."""
        emb = TagEmbedding(64)
        output = emb(VALID_TAGS)
        assert output.shape == (len(VALID_TAGS), 64)

    def test_same_tag_same_embedding(self) -> None:
        """Test that same tag produces same embedding."""
        emb = TagEmbedding(64)
        e1 = emb(["Dance"])
        e2 = emb(["Dance"])
        assert torch.allclose(e1, e2)


class TestDenoiserBlock:
    """Tests for DenoiserBlock."""

    def test_output_shape(self) -> None:
        """Test that output preserves shape."""
        embedDim = 64
        block = DenoiserBlock(embedDim, numHeads=4, condDim=128)
        x = torch.randn(2, 30, embedDim)
        cond = torch.randn(2, 128)
        output = block(x, cond)
        assert output.shape == x.shape


class TestMotionDenoiser:
    """Tests for MotionDenoiser."""

    def test_output_shape(self) -> None:
        """Test that output matches input motion shape."""
        denoiser = MotionDenoiser(
            embedDim=64,
            numHeads=4,
            numLayers=2,
            numBones=65,
        )
        noisyMotion = torch.randn(2, 30, 65, 6)
        textEmb = torch.randn(2, 64)
        tags = ["Dance", "Idle"]
        timesteps = torch.randint(0, 1000, (2,))
        output = denoiser(noisyMotion, textEmb, tags, timesteps)
        assert output.shape == noisyMotion.shape

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the network."""
        denoiser = MotionDenoiser(embedDim=64, numHeads=4, numLayers=2)
        noisyMotion = torch.randn(2, 10, 65, 6)
        textEmb = torch.randn(2, 64)
        tags = ["Dance", "Idle"]
        timesteps = torch.randint(0, 1000, (2,))
        output = denoiser(noisyMotion, textEmb, tags, timesteps)
        loss = output.sum()
        loss.backward()
        # Check that at least some parameters have gradients
        hasGrad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in denoiser.parameters()
        )
        assert hasGrad


class TestLossFunctions:
    """Tests for loss functions."""

    def test_diffusion_loss_shape(self) -> None:
        """Test diffusion loss returns scalar."""
        pred = torch.randn(2, 30, 65, 6)
        target = torch.randn(2, 30, 65, 6)
        loss = diffusionLoss(pred, target)
        assert loss.ndim == 0

    def test_geodesic_loss_shape(self) -> None:
        """Test geodesic loss returns scalar."""
        pred = torch.randn(2, 30, 65, 6)
        target = torch.randn(2, 30, 65, 6)
        loss = geodesicLoss(pred, target)
        assert loss.ndim == 0

    def test_velocity_loss_shape(self) -> None:
        """Test velocity loss returns scalar."""
        motion = torch.randn(2, 30, 65, 6)
        loss = velocityLoss(motion)
        assert loss.ndim == 0

    def test_acceleration_loss_shape(self) -> None:
        """Test acceleration loss returns scalar."""
        motion = torch.randn(2, 30, 65, 6)
        loss = accelerationLoss(motion)
        assert loss.ndim == 0

    def test_combined_loss_returns_components(self) -> None:
        """Test combined loss returns total and components."""
        pred = torch.randn(2, 30, 65, 6)
        target = torch.randn(2, 30, 65, 6)
        total, components = combinedGenerationLoss(pred, target, pred, target)
        assert total.ndim == 0
        assert "loss_diffusion" in components
        assert "loss_geodesic" in components
        assert "loss_velocity" in components


class TestValidateTags:
    """Tests for tag validation."""

    def test_valid_tag_passes(self) -> None:
        """Test that valid tags pass validation."""
        for tag in VALID_TAGS:
            assert validateTag(tag) == tag

    def test_invalid_tag_raises(self) -> None:
        """Test that invalid tags raise ValueError."""
        with pytest.raises(ValueError, match="Invalid tag"):
            validateTag("InvalidTag")
