"""Unit tests for :mod:`src.features.generation.mirror_v2`."""

from __future__ import annotations

import torch

from ainimator.training.full_training_v2 import V2Batch
from ainimator.data.mirror_v2 import (
    isPromptMirrorSafe,
    mirrorBatch,
    mirrorRootTranslationBatch,
    mirrorRotation6dBatch,
)
from ainimator.data.augmentation.mirror import (
    ROT6D_MIRROR_SIGN,
    SMPL22_MIRROR_JOINT_PERMUTATION,
)


# ---------------------------------------------------------------------
# Prompt classifier
# ---------------------------------------------------------------------
def test_prompt_safe_returns_true_for_neutral_prompts() -> None:
    assert isPromptMirrorSafe("a person walking forward")
    assert isPromptMirrorSafe("the woman dances")
    assert isPromptMirrorSafe("")  # empty (cond-mask) is trivially safe


def test_prompt_safe_returns_false_for_left_right_prompts() -> None:
    assert not isPromptMirrorSafe("a person waves their right hand")
    assert not isPromptMirrorSafe("steps to the LEFT")
    assert not isPromptMirrorSafe("counterclockwise spin")


def test_prompt_safe_word_boundary() -> None:
    """'leftover' should not trigger 'left' (word boundary)."""
    assert isPromptMirrorSafe("a person eats leftover food")


# ---------------------------------------------------------------------
# Rotation6d mirror
# ---------------------------------------------------------------------
def test_mirror_rotation6d_double_application_is_identity() -> None:
    """Applying the mirror twice must recover the input."""
    rotation = torch.randn(2, 8, 22, 6)
    mirroredOnce = mirrorRotation6dBatch(rotation)
    mirroredTwice = mirrorRotation6dBatch(mirroredOnce)
    assert torch.allclose(mirroredTwice, rotation, atol=1e-5)


def test_mirror_rotation6d_swaps_left_and_right_bones() -> None:
    """The L/R bone permutation must do what it advertises."""
    rotation = torch.zeros(1, 1, 22, 6)
    # Mark each bone with a distinct rotation to track its position.
    for index in range(22):
        rotation[0, 0, index, 0] = float(index + 1)
    mirrored = mirrorRotation6dBatch(rotation)
    permutation = SMPL22_MIRROR_JOINT_PERMUTATION
    for index in range(22):
        srcIdx = permutation[index]
        # Channel 0 has sign 1 in ROT6D_MIRROR_SIGN, so the swap is
        # the only effect on this channel.
        expected = (index + 1)  # value originally placed at srcIdx is moved to index_of_srcIdx
        assert (
            mirrored[0, 0, permutation.index(index), 0].item()
            == float(expected)
        ), f"bone {index} did not swap correctly"


def test_mirror_rotation6d_rejects_wrong_shape() -> None:
    import pytest

    with pytest.raises(ValueError, match="4-D"):
        mirrorRotation6dBatch(torch.randn(8, 22, 6))


# ---------------------------------------------------------------------
# Root translation mirror
# ---------------------------------------------------------------------
def test_mirror_root_translation_flips_only_x() -> None:
    rtrans = torch.tensor([[[1.0, 2.0, 3.0], [-1.5, 4.0, 0.5]]])
    mirrored = mirrorRootTranslationBatch(rtrans)
    expected = torch.tensor([[[-1.0, 2.0, 3.0], [1.5, 4.0, 0.5]]])
    assert torch.allclose(mirrored, expected)


def test_mirror_root_translation_double_is_identity() -> None:
    rtrans = torch.randn(3, 6, 3)
    twice = mirrorRootTranslationBatch(mirrorRootTranslationBatch(rtrans))
    assert torch.allclose(twice, rtrans)


# ---------------------------------------------------------------------
# Batch-level mirror with probability + prompt skip
# ---------------------------------------------------------------------
def _makeBatch(rawTexts: tuple[str, ...]) -> V2Batch:
    batchSize = len(rawTexts)
    return V2Batch(
        rotation6d=torch.randn(batchSize, 4, 22, 6),
        rootTranslation=torch.randn(batchSize, 4, 3),
        motionMask=torch.ones(batchSize, 4, dtype=torch.bool),
        rawTexts=rawTexts,
    )


def test_mirror_batch_zero_probability_is_no_op() -> None:
    batch = _makeBatch(("a", "b", "c"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    out = mirrorBatch(batch, probability=0.0, generator=generator)
    assert torch.equal(out.rotation6d, batch.rotation6d)
    assert torch.equal(out.rootTranslation, batch.rootTranslation)


def test_mirror_batch_full_probability_mirrors_safe_prompts() -> None:
    batch = _makeBatch(("walking forward", "dancing", "wave right hand"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    out = mirrorBatch(batch, probability=1.0, generator=generator)
    # Sample 0 and 1 are safe → mirrored.  Sample 2 has "right" → kept.
    assert not torch.equal(out.rotation6d[0], batch.rotation6d[0])
    assert not torch.equal(out.rotation6d[1], batch.rotation6d[1])
    assert torch.equal(out.rotation6d[2], batch.rotation6d[2])


def test_mirror_batch_per_sample_probability_is_respected() -> None:
    """Out of many samples with p=0.5, ≈ half should be flipped."""
    batchSize = 64
    batch = _makeBatch(tuple(f"motion {i}" for i in range(batchSize)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    out = mirrorBatch(batch, probability=0.5, generator=generator)
    diffs = (out.rotation6d != batch.rotation6d).any(dim=(1, 2, 3))
    flippedCount = int(diffs.sum().item())
    # Tolerance: 0.5 ± 0.20 over 64 samples (binomial ≈ 4 std).
    assert 20 <= flippedCount <= 44, (
        f"expected ~32/64 mirrored, got {flippedCount}"
    )


def test_mirror_batch_preserves_rawtexts_tuple() -> None:
    batch = _makeBatch(("walking", "dancing"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    out = mirrorBatch(batch, probability=1.0, generator=generator)
    assert out.rawTexts == batch.rawTexts


def test_mirror_batch_rejects_invalid_probability() -> None:
    import pytest

    batch = _makeBatch(("walking",))
    generator = torch.Generator(device="cpu")
    with pytest.raises(ValueError, match="probability"):
        mirrorBatch(batch, probability=-0.1, generator=generator)
    with pytest.raises(ValueError):
        mirrorBatch(batch, probability=1.5, generator=generator)
