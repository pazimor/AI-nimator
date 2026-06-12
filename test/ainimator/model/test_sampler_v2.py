"""Unit tests for :mod:`src.shared.model.generation.sampler_v2`."""

from __future__ import annotations

import pytest
import torch

from ainimator.model.denoiser_v2 import (
    MotionDenoiserV2,
    MotionDenoiserV2Config,
)
from ainimator.diffusion.noise_schedule import (
    NoiseSchedule,
    NoiseScheduleConfig,
    PREDICTION_V,
    PREDICTION_X0,
)
from ainimator.model.sampler_v2 import (
    DDIMSamplerV2,
    SamplerOutput,
)


# ---------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------
def _smallDenoiser(textEmbedDim: int = 64) -> MotionDenoiserV2:
    config = MotionDenoiserV2Config(
        embedDim=64,
        numHeads=4,
        numLayers=2,
        numBones=22,
        motionChannels=6,
        globalChannels=3,
        textEmbedDim=textEmbedDim,
        maxFrames=32,
        dropout=0.0,
    )
    return MotionDenoiserV2(config)


def _shortSchedule(numSteps: int = 100) -> NoiseSchedule:
    return NoiseSchedule(NoiseScheduleConfig(numSteps=numSteps))


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------
def test_sampler_rejects_unknown_prediction_mode() -> None:
    schedule = _shortSchedule()
    with pytest.raises(ValueError, match="predictionMode"):
        DDIMSamplerV2(schedule, predictionMode="bogus")


def test_sampler_exposes_schedule_and_mode() -> None:
    schedule = _shortSchedule()
    sampler = DDIMSamplerV2(schedule, predictionMode=PREDICTION_V)
    assert sampler.schedule is schedule
    assert sampler.predictionMode == PREDICTION_V


# ---------------------------------------------------------------------
# Timestep planning
# ---------------------------------------------------------------------
def test_plan_timesteps_returns_descending_long_tensor() -> None:
    sampler = DDIMSamplerV2(_shortSchedule(numSteps=100))
    plan = sampler.planTimesteps(numSteps=10, device=torch.device("cpu"))
    assert plan.dtype == torch.long
    assert plan.shape == (10,)
    # Strictly descending (the rounded plan can have plateaus only at
    # very small numSteps; 10 is well above that limit).
    assert (plan.diff() < 0).all()
    # Endpoints touch the schedule extremes.
    assert plan[0].item() == 99
    assert plan[-1].item() == 0


def test_plan_timesteps_rejects_zero() -> None:
    sampler = DDIMSamplerV2(_shortSchedule())
    with pytest.raises(ValueError, match="numSteps"):
        sampler.planTimesteps(numSteps=0, device=torch.device("cpu"))


def test_plan_timesteps_rejects_more_than_schedule_length() -> None:
    sampler = DDIMSamplerV2(_shortSchedule(numSteps=50))
    with pytest.raises(ValueError, match="exceeds"):
        sampler.planTimesteps(numSteps=100, device=torch.device("cpu"))


# ---------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------
def test_sample_returns_correct_shapes() -> None:
    schedule = _shortSchedule(numSteps=50)
    denoiser = _smallDenoiser()
    sampler = DDIMSamplerV2(schedule, predictionMode=PREDICTION_V)

    batchSize = 2
    frames = 16
    textTokens = 8
    textHidden = torch.randn(batchSize, textTokens, 64)
    uncondHidden = torch.randn(batchSize, textTokens, 64)
    textMask = torch.zeros(batchSize, textTokens, dtype=torch.bool)

    output = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=textHidden,
        textKeyPaddingMask=textMask,
        unconditionalTextHiddenStates=uncondHidden,
        unconditionalTextKeyPaddingMask=textMask,
        frames=frames,
        numSteps=10,
        cfgScale=2.5,
    )

    assert isinstance(output, SamplerOutput)
    assert output.boneMotion.shape == (
        batchSize, frames, denoiser.config.numBones,
        denoiser.config.motionChannels,
    )
    assert output.globalMotion is not None
    assert output.globalMotion.shape == (
        batchSize, frames, denoiser.config.globalChannels,
    )
    assert torch.isfinite(output.boneMotion).all()
    assert torch.isfinite(output.globalMotion).all()


def test_sample_without_global_branch() -> None:
    config = MotionDenoiserV2Config(
        embedDim=64, numHeads=4, numLayers=2, numBones=22,
        motionChannels=6, globalChannels=0, textEmbedDim=64,
        maxFrames=32, dropout=0.0,
    )
    denoiser = MotionDenoiserV2(config)
    sampler = DDIMSamplerV2(_shortSchedule(numSteps=50))

    output = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=torch.randn(1, 4, 64),
        unconditionalTextHiddenStates=torch.randn(1, 4, 64),
        frames=8,
        numSteps=5,
        cfgScale=2.0,
    )
    assert output.globalMotion is None
    assert output.boneMotion.shape == (1, 8, 22, 6)


# ---------------------------------------------------------------------
# CFG validation
# ---------------------------------------------------------------------
def test_sample_requires_unconditional_when_cfg_active() -> None:
    sampler = DDIMSamplerV2(_shortSchedule(numSteps=50))
    denoiser = _smallDenoiser()
    with pytest.raises(ValueError, match="unconditional"):
        sampler.sample(
            denoiser=denoiser,
            textHiddenStates=torch.randn(1, 4, 64),
            frames=8,
            numSteps=5,
            cfgScale=3.5,
        )


def test_sample_works_without_unconditional_when_cfg_one() -> None:
    """cfgScale=1.0 should disable CFG and skip the unconditional pass."""
    sampler = DDIMSamplerV2(_shortSchedule(numSteps=50))
    denoiser = _smallDenoiser()
    output = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=torch.randn(1, 4, 64),
        frames=8,
        numSteps=5,
        cfgScale=1.0,
    )
    assert output.boneMotion.shape == (1, 8, 22, 6)


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------
def test_sample_is_deterministic_with_fixed_seed_and_eta_zero() -> None:
    schedule = _shortSchedule(numSteps=50)
    denoiser = _smallDenoiser()
    sampler = DDIMSamplerV2(schedule, predictionMode=PREDICTION_V)

    textHidden = torch.randn(1, 4, 64)
    uncondHidden = torch.randn(1, 4, 64)

    out1 = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=textHidden,
        unconditionalTextHiddenStates=uncondHidden,
        frames=8,
        numSteps=10,
        cfgScale=2.5,
        eta=0.0,
        seed=42,
    )
    out2 = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=textHidden,
        unconditionalTextHiddenStates=uncondHidden,
        frames=8,
        numSteps=10,
        cfgScale=2.5,
        eta=0.0,
        seed=42,
    )
    assert torch.allclose(out1.boneMotion, out2.boneMotion, atol=1e-5)
    if out1.globalMotion is not None and out2.globalMotion is not None:
        assert torch.allclose(
            out1.globalMotion, out2.globalMotion, atol=1e-5
        )


def test_sample_changes_with_different_seeds() -> None:
    sampler = DDIMSamplerV2(_shortSchedule(numSteps=50))
    denoiser = _smallDenoiser()
    textHidden = torch.randn(1, 4, 64)
    uncondHidden = torch.randn(1, 4, 64)

    out1 = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=textHidden,
        unconditionalTextHiddenStates=uncondHidden,
        frames=8, numSteps=5, cfgScale=2.0, seed=0,
    )
    out2 = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=textHidden,
        unconditionalTextHiddenStates=uncondHidden,
        frames=8, numSteps=5, cfgScale=2.0, seed=1,
    )
    assert not torch.allclose(out1.boneMotion, out2.boneMotion)


# ---------------------------------------------------------------------
# Eval-mode contract
# ---------------------------------------------------------------------
def test_sample_restores_denoiser_training_mode() -> None:
    """Calling sample() should not silently disable train mode."""
    sampler = DDIMSamplerV2(_shortSchedule(numSteps=50))
    denoiser = _smallDenoiser()
    denoiser.train()
    assert denoiser.training
    sampler.sample(
        denoiser=denoiser,
        textHiddenStates=torch.randn(1, 4, 64),
        unconditionalTextHiddenStates=torch.randn(1, 4, 64),
        frames=8, numSteps=3, cfgScale=2.0,
    )
    assert denoiser.training, "Sampler must restore train() state."


# ---------------------------------------------------------------------
# Sampling consistency at known x0 (sanity)
# ---------------------------------------------------------------------
def test_sample_round_trip_recovers_x0_with_oracle_denoiser() -> None:
    """With an oracle denoiser that returns the true v-target, DDIM
    should converge to the original x0 within numerical precision."""
    schedule = _shortSchedule(numSteps=200)
    sampler = DDIMSamplerV2(schedule, predictionMode=PREDICTION_V)

    # Oracle denoiser implementing exact v-prediction from a fixed x0.
    class OracleDenoiser(torch.nn.Module):
        config = MotionDenoiserV2Config(
            embedDim=8, numHeads=2, numLayers=1, numBones=2,
            motionChannels=2, globalChannels=0, textEmbedDim=8,
            maxFrames=4, dropout=0.0,
        )

        def __init__(
            self,
            schedule: NoiseSchedule,
            x0: torch.Tensor,
        ) -> None:
            super().__init__()
            self._schedule = schedule
            self._x0 = x0
            # Required dummy parameter so the sampler can infer device.
            self._dummyParam = torch.nn.Parameter(torch.zeros(1))

        def forward(
            self,
            noisyMotion: torch.Tensor,
            timesteps: torch.Tensor,
            **kwargs: object,
        ) -> object:
            from ainimator.model.denoiser_v2 import (
                DenoiserOutput,
            )
            alpha, sigma = self._schedule.alphaSigma(
                timesteps, broadcastShape=noisyMotion.shape
            )
            # ε = (xt − α·x0) / σ, then v = α·ε − σ·x0
            epsilon = (
                noisyMotion - alpha * self._x0
            ) / torch.clamp(sigma, min=1e-8)
            v = alpha * epsilon - sigma * self._x0
            return DenoiserOutput(boneOutput=v, globalOutput=None)

    torch.manual_seed(0)
    targetX0 = torch.randn(1, 4, 2, 2) * 0.3
    oracle = OracleDenoiser(schedule=schedule, x0=targetX0)
    output = sampler.sample(
        denoiser=oracle,  # type: ignore[arg-type]
        textHiddenStates=torch.zeros(1, 2, 8),
        frames=4,
        numSteps=200,
        cfgScale=1.0,
        eta=0.0,
        seed=0,
    )
    # 200-step DDIM with an oracle denoiser must land within a small
    # tolerance of the true x0.  We use a generous tolerance because
    # DDIM has a small bias even with the perfect denoiser at finite
    # steps.
    assert torch.allclose(
        output.boneMotion, targetX0, atol=5e-2
    ), (
        f"Oracle round-trip failed: max-error="
        f"{(output.boneMotion - targetX0).abs().max().item():.4f}"
    )


# ---------------------------------------------------------------------
# CFG behaviour
# ---------------------------------------------------------------------
def test_sample_cfg_scale_1_equals_conditional_only() -> None:
    """cfgScale=1.0 must produce the exact same output as no CFG, with
    or without an unconditional embedding present."""
    schedule = _shortSchedule(numSteps=50)
    sampler = DDIMSamplerV2(schedule, predictionMode=PREDICTION_V)
    denoiser = _smallDenoiser()
    textHidden = torch.randn(1, 4, 64)

    out1 = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=textHidden,
        frames=8, numSteps=5, cfgScale=1.0, seed=7,
    )
    out2 = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=textHidden,
        unconditionalTextHiddenStates=torch.randn(1, 4, 64),
        frames=8, numSteps=5, cfgScale=1.0, seed=7,
    )
    # Both should match exactly because the unconditional branch is
    # never invoked when cfgScale=1.0.
    assert torch.allclose(out1.boneMotion, out2.boneMotion, atol=1e-5)


# ---------------------------------------------------------------------
# Eta > 0 path (stochastic DDIM)
# ---------------------------------------------------------------------
def test_sample_with_eta_one_changes_vs_eta_zero() -> None:
    """Stochastic DDIM (eta=1) must produce different output than the
    deterministic DDIM (eta=0) given the same initial noise."""
    schedule = _shortSchedule(numSteps=50)
    sampler = DDIMSamplerV2(schedule, predictionMode=PREDICTION_V)
    denoiser = _smallDenoiser()
    textHidden = torch.randn(1, 4, 64)
    uncondHidden = torch.randn(1, 4, 64)

    torch.manual_seed(99)
    deterministic = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=textHidden,
        unconditionalTextHiddenStates=uncondHidden,
        frames=8, numSteps=10, cfgScale=2.0, eta=0.0, seed=0,
    )
    torch.manual_seed(99)
    stochastic = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=textHidden,
        unconditionalTextHiddenStates=uncondHidden,
        frames=8, numSteps=10, cfgScale=2.0, eta=1.0, seed=0,
    )
    # eta=1 injects stochastic noise at every step → output diverges
    # from eta=0 even with identical initial noise and identical
    # global RNG state at start.
    assert not torch.allclose(
        deterministic.boneMotion, stochastic.boneMotion
    )
