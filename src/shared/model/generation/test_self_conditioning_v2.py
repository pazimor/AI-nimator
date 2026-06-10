"""Unit tests for self-conditioning in :class:`MotionDenoiserV2`."""

from __future__ import annotations

import torch

from src.shared.model.generation.denoiser_v2 import (
    MotionDenoiserV2,
    MotionDenoiserV2Config,
)

_B, _F, _NB = 2, 12, 22


def _config(useSelfConditioning: bool) -> MotionDenoiserV2Config:
    return MotionDenoiserV2Config(
        embedDim=64,
        numHeads=4,
        numLayers=2,
        textEmbedDim=64,
        useSelfConditioning=useSelfConditioning,
    )


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "noisyMotion": torch.randn(_B, _F, _NB, 6),
        "timesteps": torch.randint(0, 1000, (_B,)),
        "textHiddenStates": torch.randn(_B, 5, 64),
        "noisyGlobalFeatures": torch.randn(_B, _F, 3),
        "motionKeyPaddingMask": torch.zeros(_B, _F, dtype=torch.bool),
    }


def test_none_estimate_matches_zeros() -> None:
    """A ``None`` self-cond input must behave like an explicit zero one."""
    denoiser = MotionDenoiserV2(_config(True)).eval()
    inputs = _inputs()
    without = denoiser(**inputs)
    withZeros = denoiser(
        **inputs,
        selfCondBone=torch.zeros_like(inputs["noisyMotion"]),
        selfCondGlobal=torch.zeros_like(inputs["noisyGlobalFeatures"]),
    )
    assert torch.allclose(without.boneOutput, withZeros.boneOutput, atol=1e-6)


def test_zero_init_estimate_has_no_effect() -> None:
    """At init the self-cond projection is zero → estimate is ignored."""
    denoiser = MotionDenoiserV2(_config(True)).eval()
    inputs = _inputs()
    baseline = denoiser(**inputs)
    withEstimate = denoiser(
        **inputs,
        selfCondBone=torch.randn(_B, _F, _NB, 6),
        selfCondGlobal=torch.randn(_B, _F, 3),
    )
    assert torch.allclose(
        baseline.boneOutput, withEstimate.boneOutput, atol=1e-6
    )


def test_disabled_denoiser_ignores_self_cond_arguments() -> None:
    """When disabled the projections are absent and inputs are ignored."""
    denoiser = MotionDenoiserV2(_config(False)).eval()
    assert denoiser.selfCondBoneProj is None
    inputs = _inputs()
    out = denoiser(
        **inputs, selfCondBone=torch.randn(_B, _F, _NB, 6)
    )
    assert out.boneOutput.shape == (_B, _F, _NB, 6)
