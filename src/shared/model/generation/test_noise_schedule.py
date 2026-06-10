"""Unit tests for :mod:`src.shared.model.generation.noise_schedule`."""

from __future__ import annotations

import pytest
import torch

from src.shared.model.generation.noise_schedule import (
    NoiseSchedule,
    NoiseScheduleConfig,
    PREDICTION_EPSILON,
    PREDICTION_V,
    PREDICTION_X0,
    SCHEDULE_COSINE,
    SCHEDULE_LINEAR,
)

ATOL = 1e-5


# ---------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------
def test_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="numSteps"):
        NoiseScheduleConfig(numSteps=0)


def test_config_rejects_unknown_schedule_type() -> None:
    with pytest.raises(ValueError, match="scheduleType"):
        NoiseScheduleConfig(scheduleType="exponential")


def test_config_rejects_invalid_linear_endpoints() -> None:
    with pytest.raises(ValueError, match="endpoints"):
        NoiseScheduleConfig(linearBetaStart=0.0, linearBetaEnd=0.02)
    with pytest.raises(ValueError):
        NoiseScheduleConfig(linearBetaStart=0.02, linearBetaEnd=0.01)


def test_config_rejects_invalid_cosine_s() -> None:
    with pytest.raises(ValueError, match="cosineS"):
        NoiseScheduleConfig(cosineS=0.0)
    with pytest.raises(ValueError, match="cosineS"):
        NoiseScheduleConfig(cosineS=0.5)


# ---------------------------------------------------------------------
# Schedule shape & invariants
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "scheduleType", [SCHEDULE_LINEAR, SCHEDULE_COSINE]
)
def test_buffers_have_expected_shape(scheduleType: str) -> None:
    config = NoiseScheduleConfig(numSteps=200, scheduleType=scheduleType)
    schedule = NoiseSchedule(config)

    assert schedule.alphasCumprod.shape == (200,)
    assert schedule.alphas.shape == (200,)
    assert schedule.sigmas.shape == (200,)
    assert schedule.betas.shape == (200,)


@pytest.mark.parametrize(
    "scheduleType", [SCHEDULE_LINEAR, SCHEDULE_COSINE]
)
def test_alphas_cumprod_is_monotone_decreasing(scheduleType: str) -> None:
    schedule = NoiseSchedule(
        NoiseScheduleConfig(numSteps=200, scheduleType=scheduleType)
    )
    diffs = schedule.alphasCumprod.diff()
    assert (diffs <= 0).all(), "alphasCumprod must be non-increasing."


@pytest.mark.parametrize(
    "scheduleType", [SCHEDULE_LINEAR, SCHEDULE_COSINE]
)
def test_alpha_squared_plus_sigma_squared_equals_one(
    scheduleType: str,
) -> None:
    schedule = NoiseSchedule(
        NoiseScheduleConfig(numSteps=200, scheduleType=scheduleType)
    )
    sumOfSquares = schedule.alphas ** 2 + schedule.sigmas ** 2
    assert torch.allclose(
        sumOfSquares, torch.ones_like(sumOfSquares), atol=ATOL
    )


def test_cosine_schedule_starts_near_one_ends_near_zero() -> None:
    schedule = NoiseSchedule(
        NoiseScheduleConfig(numSteps=1000, scheduleType=SCHEDULE_COSINE)
    )
    # At t=0 (after the first cumulative product) ᾱ should still be
    # close to 1; at the last step it should be close to (but > 0).
    assert schedule.alphasCumprod[0].item() > 0.99
    assert schedule.alphasCumprod[-1].item() < 0.01
    assert schedule.alphasCumprod[-1].item() > 0.0


def test_linear_schedule_matches_legacy_endpoints() -> None:
    """Sanity check vs Étape 1 hyperparameters."""
    config = NoiseScheduleConfig(
        numSteps=1000,
        scheduleType=SCHEDULE_LINEAR,
        linearBetaStart=0.0001,
        linearBetaEnd=0.02,
    )
    schedule = NoiseSchedule(config)
    assert schedule.betas[0].item() == pytest.approx(0.0001, rel=1e-4)
    assert schedule.betas[-1].item() == pytest.approx(0.02, rel=1e-4)


# ---------------------------------------------------------------------
# Per-sample alpha/sigma extraction
# ---------------------------------------------------------------------
def test_alpha_sigma_extraction_shapes_without_broadcast() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    timesteps = torch.tensor([10, 50, 90])
    alpha, sigma = schedule.alphaSigma(timesteps)
    assert alpha.shape == (3,)
    assert sigma.shape == (3,)


def test_alpha_sigma_broadcasts_to_target_shape() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    timesteps = torch.tensor([10, 50, 90])
    target = torch.zeros(3, 4, 22, 6)
    alpha, sigma = schedule.alphaSigma(
        timesteps, broadcastShape=target.shape
    )
    # Broadcasts as (3, 1, 1, 1).
    assert alpha.shape == (3, 1, 1, 1)
    assert (alpha * target + sigma * target).shape == target.shape


def test_alpha_sigma_rejects_non_1d_timesteps() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=10))
    with pytest.raises(ValueError, match="1-D"):
        schedule.alphaSigma(torch.tensor([[1, 2]]))


# ---------------------------------------------------------------------
# Forward diffusion (qSample)
# ---------------------------------------------------------------------
def test_q_sample_shape_and_noise_returned() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.randn(2, 16, 22, 6)
    timesteps = torch.tensor([5, 80])
    xT, noise = schedule.qSample(x0, timesteps)
    assert xT.shape == x0.shape
    assert noise.shape == x0.shape


def test_q_sample_uses_provided_noise_unchanged() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.zeros(2, 16, 22, 6)
    fixedNoise = torch.randn_like(x0)
    timesteps = torch.tensor([5, 80])
    _, returnedNoise = schedule.qSample(x0, timesteps, noise=fixedNoise)
    assert torch.equal(returnedNoise, fixedNoise)


def test_q_sample_at_t_zero_recovers_x0() -> None:
    """At t=0 the cosine schedule has ᾱ_0 ≈ 0.999 (not exactly 1).

    The very first timestep already injects a tiny amount of noise to
    keep the schedule numerically well-defined.  We only require the
    output to be close to ``x_0`` up to 1% relative — the strict
    boundary is α(t)² + σ(t)² = 1 (covered elsewhere).
    """
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.randn(2, 16, 22, 6)
    timesteps = torch.tensor([0, 0])
    xT, _ = schedule.qSample(x0, timesteps, noise=torch.zeros_like(x0))
    assert torch.allclose(xT, x0, rtol=1e-2, atol=1e-2)


def test_q_sample_at_high_t_is_dominated_by_noise() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.ones(1, 4, 22, 6) * 100.0
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([99])
    xT, _ = schedule.qSample(x0, timesteps, noise=noise)
    # With α≈0 and σ·noise=0, xT collapses to ~0, far from x0.
    assert xT.abs().max().item() < 50.0


def test_q_sample_rejects_noise_shape_mismatch() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.randn(2, 16, 22, 6)
    badNoise = torch.randn(2, 16, 22, 5)
    with pytest.raises(ValueError, match="shape"):
        schedule.qSample(x0, torch.tensor([0, 1]), noise=badNoise)


# ---------------------------------------------------------------------
# Prediction-target conversions
# ---------------------------------------------------------------------
def test_prediction_target_x0_returns_x0() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.randn(2, 16, 22, 6)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([20, 80])
    target = schedule.predictionTarget(x0, noise, timesteps, PREDICTION_X0)
    assert torch.equal(target, x0)


def test_prediction_target_epsilon_returns_noise() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.randn(2, 16, 22, 6)
    noise = torch.randn_like(x0)
    target = schedule.predictionTarget(
        x0, noise, torch.tensor([20, 80]), PREDICTION_EPSILON
    )
    assert torch.equal(target, noise)


def test_prediction_target_v_matches_definition() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.randn(2, 16, 22, 6)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([20, 80])
    target = schedule.predictionTarget(x0, noise, timesteps, PREDICTION_V)

    alpha, sigma = schedule.alphaSigma(timesteps, broadcastShape=x0.shape)
    expected = alpha * noise - sigma * x0
    assert torch.allclose(target, expected, atol=ATOL)


def test_prediction_target_rejects_unknown_mode() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.zeros(1, 4, 22, 6)
    noise = torch.zeros_like(x0)
    with pytest.raises(ValueError, match="predictionMode"):
        schedule.predictionTarget(x0, noise, torch.tensor([0]), "bogus")


# ---------------------------------------------------------------------
# Round-trip: prediction → x0 / ε
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "predictionMode",
    [PREDICTION_X0, PREDICTION_V, PREDICTION_EPSILON],
)
def test_x0_round_trip_through_prediction_target(
    predictionMode: str,
) -> None:
    """Compute target from (x0, ε), recover x0 from target + xT."""
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.randn(2, 16, 22, 6)
    noise = torch.randn_like(x0)
    # Avoid the very edges where α ≈ 0 or σ ≈ 0 (tested separately).
    timesteps = torch.tensor([10, 80])

    xT, _ = schedule.qSample(x0, timesteps, noise=noise)
    target = schedule.predictionTarget(x0, noise, timesteps, predictionMode)
    x0Recovered = schedule.x0FromPrediction(
        prediction=target,
        xT=xT,
        timesteps=timesteps,
        predictionMode=predictionMode,
    )
    assert torch.allclose(x0Recovered, x0, atol=1e-4)


@pytest.mark.parametrize(
    "predictionMode",
    [PREDICTION_X0, PREDICTION_V, PREDICTION_EPSILON],
)
def test_epsilon_round_trip_through_prediction_target(
    predictionMode: str,
) -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.randn(2, 16, 22, 6)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([10, 80])

    xT, _ = schedule.qSample(x0, timesteps, noise=noise)
    target = schedule.predictionTarget(x0, noise, timesteps, predictionMode)
    epsilonRecovered = schedule.epsilonFromPrediction(
        prediction=target,
        xT=xT,
        timesteps=timesteps,
        predictionMode=predictionMode,
    )
    assert torch.allclose(epsilonRecovered, noise, atol=1e-4)


def test_v_target_consistency_x0_and_epsilon_recovery_agree() -> None:
    """Compute v-target, recover both x0 and ε, verify x_t formula."""
    schedule = NoiseSchedule(NoiseScheduleConfig(numSteps=100))
    x0 = torch.randn(2, 16, 22, 6)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([15, 85])

    xT, _ = schedule.qSample(x0, timesteps, noise=noise)
    v = schedule.predictionTarget(x0, noise, timesteps, PREDICTION_V)
    x0Hat = schedule.x0FromPrediction(v, xT, timesteps, PREDICTION_V)
    epsHat = schedule.epsilonFromPrediction(v, xT, timesteps, PREDICTION_V)
    alpha, sigma = schedule.alphaSigma(timesteps, broadcastShape=x0.shape)
    reconstructed = alpha * x0Hat + sigma * epsHat
    assert torch.allclose(reconstructed, xT, atol=1e-4)


# ---------------------------------------------------------------------
# State-dict round-trip
# ---------------------------------------------------------------------
def test_buffers_round_trip_through_state_dict() -> None:
    schedule = NoiseSchedule(
        NoiseScheduleConfig(numSteps=200, scheduleType=SCHEDULE_COSINE)
    )
    state = schedule.state_dict()
    other = NoiseSchedule(
        NoiseScheduleConfig(numSteps=200, scheduleType=SCHEDULE_COSINE)
    )
    other.load_state_dict(state)
    assert torch.equal(schedule.alphasCumprod, other.alphasCumprod)
    assert torch.equal(schedule.alphas, other.alphas)
    assert torch.equal(schedule.sigmas, other.sigmas)
