"""Unit tests for :mod:`src.shared.model.generation.losses_v2`."""

from __future__ import annotations

import math

import pytest
import torch

from ainimator.model.losses_v2 import (
    DEFAULT_MIN_SNR_GAMMA,
    VELOCITY_SCHEDULE_NONE,
    VELOCITY_SCHEDULE_TIMESTEP,
    ContrastiveMemoryBank,
    combinedLossV2,
    diffusionLossV2,
    minSnrLossWeights,
    perSampleMse,
    textMotionAlignmentLoss,
    textMotionContrastiveLoss,
    velocityXyzLossV2,
)
from ainimator.diffusion.noise_schedule import (
    NoiseSchedule,
    NoiseScheduleConfig,
    PREDICTION_EPSILON,
    PREDICTION_V,
    PREDICTION_X0,
)


# ---------------------------------------------------------------------
# Min-SNR weights
# ---------------------------------------------------------------------
def test_min_snr_weights_disabled_when_gamma_zero() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    timesteps = torch.tensor([10, 50, 90])
    weights = minSnrLossWeights(
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=PREDICTION_X0,
        gamma=0.0,
    )
    assert torch.equal(weights, torch.ones_like(weights))


@pytest.mark.parametrize(
    "predictionMode",
    [PREDICTION_X0, PREDICTION_V, PREDICTION_EPSILON],
)
def test_min_snr_weights_have_correct_shape(predictionMode: str) -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    timesteps = torch.tensor([10, 50, 90])
    weights = minSnrLossWeights(
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=predictionMode,
    )
    assert weights.shape == (3,)
    assert weights.dtype == torch.float32


def test_min_snr_x0_weights_decrease_for_low_t() -> None:
    """For x0-pred, weight = min(SNR, γ) / SNR.

    At low t (SNR very high), weight ≈ γ/SNR (small).
    At high t (SNR < γ), weight = 1.
    """
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=1000))
    lowT = torch.tensor([5])
    highT = torch.tensor([950])
    weightLow = minSnrLossWeights(
        lowT, schedule.alphasCumprod, PREDICTION_X0, gamma=5.0
    )
    weightHigh = minSnrLossWeights(
        highT, schedule.alphasCumprod, PREDICTION_X0, gamma=5.0
    )
    assert weightLow.item() < weightHigh.item()
    assert weightHigh.item() == pytest.approx(1.0, abs=1e-4)


def test_min_snr_v_weights_differ_from_x0() -> None:
    """v-pred uses γ/(SNR+1) factor — different curve from x0-pred."""
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=1000))
    timesteps = torch.tensor([100, 500, 900])
    wX0 = minSnrLossWeights(
        timesteps, schedule.alphasCumprod, PREDICTION_X0, gamma=5.0
    )
    wV = minSnrLossWeights(
        timesteps, schedule.alphasCumprod, PREDICTION_V, gamma=5.0
    )
    # v-pred should not equal x0-pred at any non-trivial t.
    assert not torch.allclose(wX0, wV)
    # All weights remain positive and finite.
    assert (wV > 0).all()
    assert torch.isfinite(wV).all()


def test_min_snr_rejects_unknown_mode() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    with pytest.raises(ValueError, match="predictionMode"):
        minSnrLossWeights(
            torch.tensor([0]),
            schedule.alphasCumprod,
            predictionMode="bogus",
        )


# ---------------------------------------------------------------------
# Per-sample MSE
# ---------------------------------------------------------------------
def test_per_sample_mse_zero_when_predictions_match_target() -> None:
    target = torch.randn(3, 4, 5)
    perSample = perSampleMse(target, target)
    assert torch.allclose(perSample, torch.zeros(3))


def test_per_sample_mse_respects_motion_mask() -> None:
    target = torch.randn(2, 4, 5)
    prediction = target.clone()
    prediction[0, 0, :] = 100.0  # huge error on a masked frame
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 0] = False  # ignore the corrupted frame
    perSample = perSampleMse(prediction, target, motionMask=mask)
    assert torch.allclose(perSample, torch.zeros(2), atol=1e-6)


def test_per_sample_mse_returns_one_value_per_batch() -> None:
    target = torch.randn(5, 8, 22, 6)
    prediction = torch.randn_like(target)
    perSample = perSampleMse(prediction, target)
    assert perSample.shape == (5,)


# ---------------------------------------------------------------------
# Diffusion loss
# ---------------------------------------------------------------------
def test_diffusion_loss_is_zero_when_prediction_equals_target() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    target = torch.randn(2, 4, 22, 6)
    timesteps = torch.tensor([20, 80])
    loss = diffusionLossV2(
        prediction=target,
        target=target,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=PREDICTION_V,
    )
    assert loss.item() == pytest.approx(0.0)


def test_diffusion_loss_is_positive_when_predictions_differ() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    target = torch.randn(2, 4, 22, 6)
    prediction = target + 0.5
    timesteps = torch.tensor([20, 80])
    loss = diffusionLossV2(
        prediction=prediction,
        target=target,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=PREDICTION_V,
    )
    assert loss.item() > 0.0


def test_diffusion_loss_rejects_shape_mismatch() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    with pytest.raises(ValueError, match="shape"):
        diffusionLossV2(
            prediction=torch.randn(2, 4, 22, 6),
            target=torch.randn(2, 4, 22, 5),
            timesteps=torch.tensor([0, 1]),
            alphasCumprod=schedule.alphasCumprod,
            predictionMode=PREDICTION_V,
        )


def test_diffusion_loss_propagates_min_snr_scaling() -> None:
    """gamma=0 should give different (larger or equal) loss than gamma=5."""
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=1000))
    target = torch.randn(8, 16, 22, 6)
    prediction = target + 0.3
    # Force timesteps in the high-SNR (low-t) region where Min-SNR
    # *reduces* weights.
    timesteps = torch.full((8,), 5, dtype=torch.long)

    lossNoMinSnr = diffusionLossV2(
        prediction=prediction,
        target=target,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=PREDICTION_X0,
        gamma=0.0,
    )
    lossWithMinSnr = diffusionLossV2(
        prediction=prediction,
        target=target,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=PREDICTION_X0,
        gamma=5.0,
    )
    assert lossWithMinSnr.item() < lossNoMinSnr.item()


def test_diffusion_loss_supports_motion_mask() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    target = torch.randn(2, 8, 22, 6)
    prediction = target.clone()
    prediction[1, 7, :, :] = 100.0  # garbage on a masked frame
    timesteps = torch.tensor([0, 0])
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, 7] = False
    loss = diffusionLossV2(
        prediction=prediction,
        target=target,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=PREDICTION_V,
        motionMask=mask,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------
# Velocity-XYZ loss
# ---------------------------------------------------------------------
def test_velocity_xyz_loss_zero_when_predictions_match_target() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    rotation = torch.randn(2, 8, 22, 6)
    timesteps = torch.tensor([10, 80])
    loss = velocityXyzLossV2(
        predictedRotation6d=rotation,
        targetRotation6d=rotation,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_velocity_xyz_loss_timestep_schedule_zeroes_high_t() -> None:
    """At the highest t (ᾱ ≈ 0) the timestep schedule should drive the
    weighted loss to nearly zero, even with bad predictions."""
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=1000))
    rotation = torch.randn(4, 8, 22, 6)
    badRotation = torch.randn_like(rotation)

    timestepsHigh = torch.full((4,), 999, dtype=torch.long)
    lossHighScheduled = velocityXyzLossV2(
        predictedRotation6d=badRotation,
        targetRotation6d=rotation,
        timesteps=timestepsHigh,
        alphasCumprod=schedule.alphasCumprod,
        schedule=VELOCITY_SCHEDULE_TIMESTEP,
    )
    lossHighNone = velocityXyzLossV2(
        predictedRotation6d=badRotation,
        targetRotation6d=rotation,
        timesteps=timestepsHigh,
        alphasCumprod=schedule.alphasCumprod,
        schedule=VELOCITY_SCHEDULE_NONE,
    )
    # High-t scheduled loss should be ≪ unscheduled loss.
    assert lossHighScheduled.item() < lossHighNone.item() * 0.01


def test_velocity_xyz_rejects_invalid_schedule() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    rotation = torch.randn(2, 8, 22, 6)
    with pytest.raises(ValueError, match="schedule"):
        velocityXyzLossV2(
            predictedRotation6d=rotation,
            targetRotation6d=rotation,
            timesteps=torch.tensor([0, 0]),
            alphasCumprod=schedule.alphasCumprod,
            schedule="cosine",
        )


def test_velocity_xyz_rejects_shape_mismatch() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    with pytest.raises(ValueError, match="shape"):
        velocityXyzLossV2(
            predictedRotation6d=torch.randn(2, 8, 22, 6),
            targetRotation6d=torch.randn(2, 8, 21, 6),
            timesteps=torch.tensor([0, 0]),
            alphasCumprod=schedule.alphasCumprod,
        )


# ---------------------------------------------------------------------
# Text-motion alignment
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Pool helper
# ---------------------------------------------------------------------
def test_pool_text_embedding_returns_2d_when_mask_is_none() -> None:
    from ainimator.model.losses_v2 import poolTextEmbedding

    hidden = torch.randn(3, 8, 16)
    pooled = poolTextEmbedding(hidden, keyPaddingMask=None)
    assert pooled.shape == (3, 16)
    assert torch.allclose(pooled, hidden.mean(dim=1))


def test_pool_text_embedding_skips_padded_tokens() -> None:
    from ainimator.model.losses_v2 import poolTextEmbedding

    hidden = torch.zeros(2, 4, 8)
    hidden[0, 0] = 5.0  # only this token contributes
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 0] = False  # False == real token in our convention
    mask[1, :] = False
    # Sample 0: only token 0 is real → pooled = 5.0
    # Sample 1: all tokens real but they are zeros → pooled = 0
    pooled = poolTextEmbedding(hidden, keyPaddingMask=mask)
    assert torch.allclose(pooled[0], torch.full((8,), 5.0))
    assert torch.allclose(pooled[1], torch.zeros(8))


def test_pool_text_embedding_rejects_non_3d_input() -> None:
    from ainimator.model.losses_v2 import poolTextEmbedding

    with pytest.raises(ValueError, match="3-D"):
        poolTextEmbedding(torch.randn(4, 8), keyPaddingMask=None)


# ---------------------------------------------------------------------
# Contrastive InfoNCE
# ---------------------------------------------------------------------
def test_contrastive_zero_when_text_motion_aligned_perfectly() -> None:
    from ainimator.model.losses_v2 import (
        textMotionContrastiveLoss,
    )

    text = torch.randn(8, 16)
    # Identical embeddings ⇒ each row's positive is the unique max in
    # its row of the similarity matrix ⇒ cross-entropy → 0 as
    # temperature → 0.  Use a small temperature for a clean signal.
    loss = textMotionContrastiveLoss(text, text, temperature=0.01)
    assert loss.item() < 0.1


def test_contrastive_starts_near_log_batch_size_when_random() -> None:
    """For random embeddings the loss is ≈ log(B) at initialization."""
    from ainimator.model.losses_v2 import (
        textMotionContrastiveLoss,
    )

    torch.manual_seed(0)
    batchSize = 16
    text = torch.randn(batchSize, 64)
    motion = torch.randn(batchSize, 64)
    loss = textMotionContrastiveLoss(text, motion, temperature=1.0)
    expected = math.log(batchSize)
    # Loose tolerance — random init has variance.
    assert abs(loss.item() - expected) < 1.5


def test_contrastive_zero_for_batch_size_one() -> None:
    """A 1-sample batch has no negatives — loss must be 0, not NaN."""
    from ainimator.model.losses_v2 import (
        textMotionContrastiveLoss,
    )

    text = torch.randn(1, 16)
    motion = torch.randn(1, 16)
    loss = textMotionContrastiveLoss(text, motion)
    assert loss.item() == 0.0


def test_contrastive_rejects_shape_mismatch() -> None:
    from ainimator.model.losses_v2 import (
        textMotionContrastiveLoss,
    )

    with pytest.raises(ValueError, match="shape"):
        textMotionContrastiveLoss(torch.randn(4, 16), torch.randn(4, 32))


def test_contrastive_rejects_non_2d_inputs() -> None:
    from ainimator.model.losses_v2 import (
        textMotionContrastiveLoss,
    )

    with pytest.raises(ValueError, match="2-D"):
        textMotionContrastiveLoss(
            torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        )


def test_contrastive_rejects_non_positive_temperature() -> None:
    from ainimator.model.losses_v2 import (
        textMotionContrastiveLoss,
    )

    with pytest.raises(ValueError, match="temperature"):
        textMotionContrastiveLoss(
            torch.randn(4, 8), torch.randn(4, 8), temperature=0.0
        )


def test_contrastive_propagates_gradient_through_both_inputs() -> None:
    from ainimator.model.losses_v2 import (
        textMotionContrastiveLoss,
    )

    text = torch.randn(4, 16, requires_grad=True)
    motion = torch.randn(4, 16, requires_grad=True)
    loss = textMotionContrastiveLoss(text, motion)
    loss.backward()
    assert text.grad is not None
    assert motion.grad is not None
    assert text.grad.abs().sum() > 0
    assert motion.grad.abs().sum() > 0


def test_contrastive_distinct_positives_have_lower_loss_than_collapsed() -> (
    None
):
    """A model where each sample's pair is unique should beat the
    pathological mode-collapse case where all motion embeddings are
    identical."""
    from ainimator.model.losses_v2 import (
        textMotionContrastiveLoss,
    )

    batchSize = 8
    dim = 16
    torch.manual_seed(0)
    # Healthy case: distinct text vectors, motion = text exactly.
    text = torch.randn(batchSize, dim)
    motionHealthy = text.clone()
    lossHealthy = textMotionContrastiveLoss(
        text, motionHealthy, temperature=0.1
    )
    # Collapse case: every motion = the average of text.
    motionCollapsed = text.mean(dim=0, keepdim=True).expand(batchSize, dim)
    lossCollapsed = textMotionContrastiveLoss(
        text, motionCollapsed, temperature=0.1
    )
    assert lossHealthy.item() < lossCollapsed.item() - 0.5


# ---------------------------------------------------------------------
# Original alignment loss (kept for backward compat)
# ---------------------------------------------------------------------
def test_text_motion_alignment_zero_for_identical_embeddings() -> None:
    embedding = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
    loss = textMotionAlignmentLoss(embedding, embedding)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_text_motion_alignment_positive_for_different_embeddings() -> None:
    text = torch.randn(4, 64)
    motion = torch.randn(4, 64)
    loss = textMotionAlignmentLoss(text, motion)
    assert loss.item() > 0.0


def test_text_motion_alignment_rejects_non_2d_inputs() -> None:
    with pytest.raises(ValueError, match="2-D"):
        textMotionAlignmentLoss(
            torch.randn(2, 8, 64),
            torch.randn(2, 8, 64),
        )


def test_text_motion_alignment_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        textMotionAlignmentLoss(
            torch.randn(2, 64),
            torch.randn(2, 32),
        )


# ---------------------------------------------------------------------
# combinedLossV2
# ---------------------------------------------------------------------
def test_combined_loss_aggregates_components() -> None:
    diffusion = torch.tensor(2.0, requires_grad=True)
    velocity = torch.tensor(1.0, requires_grad=True)
    alignment = torch.tensor(0.5, requires_grad=True)
    total, components = combinedLossV2(
        diffusion=diffusion,
        velocityXyz=velocity,
        textAlignment=alignment,
        diffusionWeight=1.0,
        velocityXyzWeight=0.5,
        textAlignmentWeight=0.5,
    )
    expected = 1.0 * 2.0 + 0.5 * 1.0 + 0.5 * 0.5
    assert total.item() == pytest.approx(expected)
    assert components["diffusion"] == pytest.approx(2.0)
    assert components["vel_xyz"] == pytest.approx(1.0)
    assert components["text_alignment"] == pytest.approx(0.5)
    assert components["total"] == pytest.approx(expected)


def test_combined_loss_skips_disabled_components() -> None:
    diffusion = torch.tensor(2.0, requires_grad=True)
    total, components = combinedLossV2(
        diffusion=diffusion,
        velocityXyz=None,
        textAlignment=None,
    )
    assert total.item() == pytest.approx(2.0)
    assert "vel_xyz" not in components
    assert "text_alignment" not in components


def test_combined_loss_is_differentiable() -> None:
    diffusion = torch.tensor(2.0, requires_grad=True)
    velocity = torch.tensor(1.0, requires_grad=True)
    total, _ = combinedLossV2(
        diffusion=diffusion,
        velocityXyz=velocity,
        textAlignment=None,
    )
    total.backward()
    assert diffusion.grad is not None
    assert velocity.grad is not None


# ---------------------------------------------------------------------
# ContrastiveMemoryBank
# ---------------------------------------------------------------------
class TestContrastiveMemoryBank:
    def test_enqueue_dequeue_shapes(self) -> None:
        bank = ContrastiveMemoryBank(embeddingDim=64, bankSize=32)
        text = torch.randn(8, 64)
        motion = torch.randn(8, 64)
        bank.enqueue(text, motion)
        result = bank.dequeue()
        assert result is not None
        negT, negM = result
        assert negT.shape == (8, 64)
        assert negM.shape == (8, 64)
        assert bank.size == 8

    def test_dequeue_empty_returns_none(self) -> None:
        bank = ContrastiveMemoryBank(embeddingDim=64, bankSize=32)
        assert bank.dequeue() is None

    def test_fifo_overwrites_oldest(self) -> None:
        bank = ContrastiveMemoryBank(embeddingDim=4, bankSize=4)
        first = torch.ones(2, 4)
        bank.enqueue(first, first)
        assert bank.size == 2
        second = torch.ones(4, 4) * 2.0
        bank.enqueue(second, second)
        assert bank.size == 4
        result = bank.dequeue()
        assert result is not None
        negT, _ = result
        assert negT.shape == (4, 4)

    def test_dequeued_tensors_are_detached(self) -> None:
        bank = ContrastiveMemoryBank(embeddingDim=16, bankSize=8)
        text = torch.randn(4, 16, requires_grad=True)
        motion = torch.randn(4, 16, requires_grad=True)
        bank.enqueue(text, motion)
        result = bank.dequeue()
        assert result is not None
        negT, negM = result
        assert not negT.requires_grad
        assert not negM.requires_grad

    def test_enqueued_embeddings_are_l2_normalized(self) -> None:
        bank = ContrastiveMemoryBank(embeddingDim=16, bankSize=8)
        text = torch.randn(4, 16) * 10.0
        motion = torch.randn(4, 16) * 10.0
        bank.enqueue(text, motion)
        result = bank.dequeue()
        assert result is not None
        negT, negM = result
        norms_t = negT.norm(dim=-1)
        norms_m = negM.norm(dim=-1)
        assert torch.allclose(norms_t, torch.ones_like(norms_t), atol=1e-5)
        assert torch.allclose(norms_m, torch.ones_like(norms_m), atol=1e-5)


# ---------------------------------------------------------------------
# textMotionContrastiveLoss with negatives
# ---------------------------------------------------------------------
class TestContrastiveLossWithNegatives:
    def test_loss_at_random_init_higher_with_bank(self) -> None:
        B, D, K = 8, 64, 128
        text = torch.randn(B, D)
        motion = torch.randn(B, D)
        neg_text = torch.randn(K, D)
        neg_motion = torch.randn(K, D)
        neg_text = torch.nn.functional.normalize(neg_text, dim=-1)
        neg_motion = torch.nn.functional.normalize(neg_motion, dim=-1)

        loss_no_bank = textMotionContrastiveLoss(text, motion)
        loss_with_bank = textMotionContrastiveLoss(
            text, motion,
            negativeTexts=neg_text,
            negativeMotions=neg_motion,
        )
        assert loss_with_bank > loss_no_bank
        assert loss_with_bank > math.log(B)

    def test_no_gradient_through_negatives(self) -> None:
        B, D = 8, 64
        text = torch.randn(B, D, requires_grad=True)
        motion = torch.randn(B, D, requires_grad=True)
        neg_text = torch.randn(32, D)
        neg_motion = torch.randn(32, D)

        loss = textMotionContrastiveLoss(
            text, motion,
            negativeTexts=neg_text,
            negativeMotions=neg_motion,
        )
        loss.backward()
        assert text.grad is not None
        assert motion.grad is not None

    def test_backward_compat_without_negatives(self) -> None:
        B, D = 8, 64
        text = torch.randn(B, D)
        motion = torch.randn(B, D)
        loss = textMotionContrastiveLoss(text, motion)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_sample_weights_shape_validation(self) -> None:
        B, D = 6, 64
        text = torch.randn(B, D)
        motion = torch.randn(B, D)
        with pytest.raises(ValueError):
            textMotionContrastiveLoss(
                text, motion, sampleWeights=torch.ones(B - 1)
            )

    def test_uniform_sample_weights_match_unweighted(self) -> None:
        B, D = 8, 64
        text = torch.randn(B, D)
        motion = torch.randn(B, D)
        plain = textMotionContrastiveLoss(text, motion)
        weighted = textMotionContrastiveLoss(
            text, motion, sampleWeights=torch.ones(B)
        )
        assert torch.allclose(plain, weighted, atol=1e-5)

    def test_zero_weight_samples_are_ignored_as_positives(self) -> None:
        B, D = 8, 64
        text = torch.randn(B, D)
        motion = torch.randn(B, D)
        weights = torch.ones(B)
        weights[B // 2:] = 0.0
        loss = textMotionContrastiveLoss(
            text, motion, sampleWeights=weights
        )
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
