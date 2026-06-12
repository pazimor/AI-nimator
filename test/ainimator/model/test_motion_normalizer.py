"""Unit tests for :mod:`src.shared.model.generation.motion_normalizer`."""

from __future__ import annotations

import pytest
import torch

from ainimator.model.motion_normalizer import (
    EPSILON_STD,
    MotionNormalizer,
    MotionNormalizerStats,
)


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------
def test_construction_with_valid_dims_creates_buffers() -> None:
    n = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)
    assert n.boneMean.shape == (1, 1, 22, 6)
    assert n.boneStd.shape == (1, 1, 22, 6)
    assert n.globalMean.shape == (1, 1, 3)
    assert n.globalStd.shape == (1, 1, 3)
    # Identity init.
    assert torch.equal(n.boneMean, torch.zeros_like(n.boneMean))
    assert torch.equal(n.boneStd, torch.ones_like(n.boneStd))
    assert torch.equal(n.globalMean, torch.zeros_like(n.globalMean))
    assert torch.equal(n.globalStd, torch.ones_like(n.globalStd))


def test_construction_without_global_branch_skips_globals() -> None:
    n = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=0)
    assert not n.hasGlobalBranch
    assert not hasattr(n, "globalMean")
    assert not hasattr(n, "globalStd")


def test_construction_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="numBones"):
        MotionNormalizer(numBones=0, motionChannels=6, globalChannels=0)
    with pytest.raises(ValueError, match="motionChannels"):
        MotionNormalizer(numBones=22, motionChannels=0, globalChannels=0)
    with pytest.raises(ValueError, match="globalChannels"):
        MotionNormalizer(numBones=22, motionChannels=6, globalChannels=-1)


# ---------------------------------------------------------------------
# Identity transform before fit
# ---------------------------------------------------------------------
def test_normalize_is_identity_before_fit() -> None:
    n = MotionNormalizer(numBones=4, motionChannels=6, globalChannels=3)
    bone = torch.randn(2, 8, 4, 6)
    rtrans = torch.randn(2, 8, 3)
    assert torch.equal(n.normalizeBone(bone), bone)
    assert torch.equal(n.denormalizeBone(bone), bone)
    assert torch.equal(n.normalizeGlobal(rtrans), rtrans)
    assert torch.equal(n.denormalizeGlobal(rtrans), rtrans)


# ---------------------------------------------------------------------
# Fit: stats are correct
# ---------------------------------------------------------------------
def test_fit_from_single_sample_computes_per_channel_stats() -> None:
    n = MotionNormalizer(numBones=2, motionChannels=3, globalChannels=2)
    # Use deterministic data: per-(bone, channel) means and stds we can verify.
    frames = 1000
    torch.manual_seed(0)
    bone = torch.randn(frames, 2, 3) * 2.0 + 1.0
    rtrans = torch.randn(frames, 2) * 5.0 - 3.0
    stats = n.fitFromTensors([bone], [rtrans])

    # Stats are returned as (1, 1, ...) broadcastable buffers.
    assert stats.boneMean.shape == (1, 1, 2, 3)
    assert stats.boneStd.shape == (1, 1, 2, 3)
    assert stats.globalMean is not None and stats.globalStd is not None
    assert stats.globalMean.shape == (1, 1, 2)
    assert stats.globalStd.shape == (1, 1, 2)

    # With 1000 samples, mean ≈ 1.0 and std ≈ 2.0 for bone tensor.
    assert torch.allclose(
        stats.boneMean, torch.full_like(stats.boneMean, 1.0), atol=0.2
    )
    assert torch.allclose(
        stats.boneStd, torch.full_like(stats.boneStd, 2.0), atol=0.2
    )
    # Global tensor: mean ≈ -3.0, std ≈ 5.0.
    assert torch.allclose(
        stats.globalMean, torch.full_like(stats.globalMean, -3.0), atol=0.5
    )
    assert torch.allclose(
        stats.globalStd, torch.full_like(stats.globalStd, 5.0), atol=0.5
    )


def test_fit_supports_batched_samples() -> None:
    """Mixing (F, B, C) and (N, F, B, C) inputs in the same fit call."""
    n = MotionNormalizer(numBones=3, motionChannels=2, globalChannels=0)
    bone1 = torch.randn(8, 3, 2)
    bone2 = torch.randn(4, 8, 3, 2)
    stats = n.fitFromTensors([bone1, bone2])
    assert stats.boneMean.shape == (1, 1, 3, 2)
    # Verify the cumulative mean equals manually flattened.
    flat = torch.cat([bone1, bone2.reshape(-1, 3, 2)], dim=0)
    expectedMean = flat.mean(dim=0).unsqueeze(0).unsqueeze(0)
    assert torch.allclose(stats.boneMean, expectedMean, atol=1e-5)


def test_fit_clamps_zero_std_to_epsilon() -> None:
    """A constant feature has std=0; the normalizer must avoid div by 0."""
    n = MotionNormalizer(numBones=2, motionChannels=2, globalChannels=0)
    bone = torch.full((100, 2, 2), 5.0)
    n.fitFromTensors([bone])
    assert (n.boneStd >= EPSILON_STD).all()


def test_fit_rejects_empty_sample_list() -> None:
    n = MotionNormalizer(numBones=2, motionChannels=2)
    with pytest.raises(ValueError, match="boneSamples"):
        n.fitFromTensors([])


def test_fit_rejects_mismatched_trailing_dims() -> None:
    n = MotionNormalizer(numBones=2, motionChannels=6)
    with pytest.raises(ValueError, match="trailing"):
        n.fitFromTensors([torch.randn(8, 3, 6)])


def test_fit_requires_global_samples_when_branch_active() -> None:
    n = MotionNormalizer(numBones=2, motionChannels=6, globalChannels=3)
    with pytest.raises(ValueError, match="globalSamples"):
        n.fitFromTensors([torch.randn(8, 2, 6)])


# ---------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------
def test_normalize_then_denormalize_is_identity_after_fit() -> None:
    n = MotionNormalizer(numBones=2, motionChannels=3, globalChannels=2)
    torch.manual_seed(0)
    bone = torch.randn(200, 2, 3) * 2.0 + 1.0
    rtrans = torch.randn(200, 2) * 5.0 - 3.0
    n.fitFromTensors([bone], [rtrans])

    sampleBone = torch.randn(2, 8, 2, 3)
    sampleRtrans = torch.randn(2, 8, 2)

    recoveredBone = n.denormalizeBone(n.normalizeBone(sampleBone))
    recoveredRtrans = n.denormalizeGlobal(n.normalizeGlobal(sampleRtrans))

    assert torch.allclose(recoveredBone, sampleBone, atol=1e-5)
    assert torch.allclose(recoveredRtrans, sampleRtrans, atol=1e-5)


def test_normalized_distribution_has_unit_variance_on_training_data() -> None:
    """After fit, normalising the training data must yield mean≈0, std≈1.

    The normalizer expects 4-D input ``(N, F, B, C)`` at the
    normalize/denormalize call sites; the fit method tolerates 3-D
    ``(F, B, C)`` so we wrap the input into a single batch slot before
    feeding it back through the per-frame pipeline.
    """
    n = MotionNormalizer(numBones=2, motionChannels=3)
    torch.manual_seed(0)
    bone = torch.randn(2000, 2, 3) * 2.5 - 1.0
    n.fitFromTensors([bone])
    # Re-shape to 4-D (1, F, B, C) so broadcasting against the
    # (1, 1, B, C) buffers preserves the leading axis.
    normalised = n.normalizeBone(bone.unsqueeze(0))[0]
    assert normalised.mean(dim=0).abs().max().item() < 0.05
    perChannelStd = normalised.std(dim=0, unbiased=False)
    assert (perChannelStd - 1.0).abs().max().item() < 0.05


# ---------------------------------------------------------------------
# Shape validation in normalize / denormalize
# ---------------------------------------------------------------------
def test_normalize_bone_rejects_wrong_trailing_dims() -> None:
    n = MotionNormalizer(numBones=22, motionChannels=6)
    with pytest.raises(ValueError, match="trailing dims"):
        n.normalizeBone(torch.randn(2, 8, 21, 6))


def test_normalize_global_rejects_when_branch_disabled() -> None:
    n = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=0)
    with pytest.raises(RuntimeError, match="globalChannels=0"):
        n.normalizeGlobal(torch.randn(2, 8, 3))


def test_normalize_global_rejects_wrong_last_dim() -> None:
    n = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)
    n.fitFromTensors(
        [torch.randn(8, 22, 6)], [torch.randn(8, 3)]
    )
    with pytest.raises(ValueError, match="last dim"):
        n.normalizeGlobal(torch.randn(2, 8, 4))


# ---------------------------------------------------------------------
# Persistence (state_dict)
# ---------------------------------------------------------------------
def test_state_dict_round_trip_preserves_buffers() -> None:
    a = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)
    a.fitFromTensors(
        [torch.randn(20, 22, 6) * 1.5],
        [torch.randn(20, 3) * 2.0],
    )
    state = a.state_dict()

    b = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)
    b.load_state_dict(state)
    assert torch.equal(a.boneMean, b.boneMean)
    assert torch.equal(a.boneStd, b.boneStd)
    assert torch.equal(a.globalMean, b.globalMean)
    assert torch.equal(a.globalStd, b.globalStd)


def test_config_round_trip_through_dict() -> None:
    a = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)
    payload = a.configToDict()
    b = MotionNormalizer.fromConfigDict(payload)
    assert b.numBones == a.numBones
    assert b.motionChannels == a.motionChannels
    assert b.globalChannels == a.globalChannels
