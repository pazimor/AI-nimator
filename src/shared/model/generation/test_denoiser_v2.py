"""Unit tests for :mod:`src.shared.model.generation.denoiser_v2`."""

from __future__ import annotations

import pytest
import torch

from src.shared.model.generation.denoiser_v2 import (
    DenoiserOutput,
    MotionDenoiserV2,
    MotionDenoiserV2Config,
)


def _smallConfig(**overrides: object) -> MotionDenoiserV2Config:
    """Build a small config suitable for fast tests."""
    base = dict(
        embedDim=64,
        numHeads=4,
        numLayers=2,
        numBones=22,
        motionChannels=6,
        globalChannels=3,
        textEmbedDim=64,
        maxFrames=32,
        dropout=0.0,
    )
    base.update(overrides)
    return MotionDenoiserV2Config(**base)  # type: ignore[arg-type]


def _dummyBatch(
    config: MotionDenoiserV2Config,
    batchSize: int = 2,
    frames: int = 16,
    textTokens: int = 8,
) -> dict[str, torch.Tensor]:
    return dict(
        noisyMotion=torch.randn(
            batchSize, frames, config.numBones, config.motionChannels
        ),
        timesteps=torch.randint(0, 1000, (batchSize,)),
        textHiddenStates=torch.randn(
            batchSize, textTokens, config.effectiveTextEmbedDim
        ),
        textKeyPaddingMask=torch.zeros(
            batchSize, textTokens, dtype=torch.bool
        ),
        noisyGlobalFeatures=torch.randn(
            batchSize, frames, config.globalChannels
        ),
    )


# ---------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------
def test_config_rejects_non_divisible_embed_dim() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MotionDenoiserV2Config(embedDim=64, numHeads=7)


def test_config_rejects_zero_layers() -> None:
    with pytest.raises(ValueError, match="numLayers"):
        MotionDenoiserV2Config(numLayers=0)


def test_config_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        MotionDenoiserV2Config(dropout=1.5)


def test_config_rejects_negative_global_channels() -> None:
    with pytest.raises(ValueError, match="globalChannels"):
        MotionDenoiserV2Config(globalChannels=-1)


def test_config_effective_text_embed_dim_defaults_to_embed() -> None:
    config = MotionDenoiserV2Config()
    assert config.effectiveTextEmbedDim == config.embedDim


def test_config_total_output_dim_includes_global() -> None:
    config = MotionDenoiserV2Config(
        numBones=22, motionChannels=6, globalChannels=3
    )
    assert config.boneOutputDim == 132
    assert config.totalOutputDim == 135


# ---------------------------------------------------------------------
# Forward shape & contract
# ---------------------------------------------------------------------
def test_forward_returns_correct_shapes() -> None:
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)

    output = denoiser(**batch)

    assert isinstance(output, DenoiserOutput)
    assert output.boneOutput.shape == (
        2, 16, config.numBones, config.motionChannels
    )
    assert output.globalOutput is not None
    assert output.globalOutput.shape == (2, 16, config.globalChannels)


def test_forward_without_global_branch_returns_none() -> None:
    config = _smallConfig(globalChannels=0)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    batch.pop("noisyGlobalFeatures")

    output = denoiser(**batch)

    assert output.boneOutput.shape == (
        2, 16, config.numBones, config.motionChannels
    )
    assert output.globalOutput is None


def test_forward_with_text_projection_handles_dim_mismatch() -> None:
    """When text encoder output dim != denoiser embedDim, project it."""
    config = _smallConfig(textEmbedDim=128)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    batch["textHiddenStates"] = torch.randn(2, 8, 128)

    output = denoiser(**batch)
    assert output.boneOutput.shape == (
        2, 16, config.numBones, config.motionChannels
    )


def test_forward_passes_with_motion_padding_mask() -> None:
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    motionMask = torch.zeros(2, 16, dtype=torch.bool)
    motionMask[0, 12:] = True  # mask out last 4 frames of sample 0

    output = denoiser(**batch, motionKeyPaddingMask=motionMask)
    assert torch.isfinite(output.boneOutput).all()


# ---------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------
def test_forward_rejects_wrong_num_bones() -> None:
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    batch = _dummyBatch(config)
    batch["noisyMotion"] = torch.randn(2, 16, 19, config.motionChannels)
    with pytest.raises(ValueError, match="bones"):
        denoiser(**batch)


def test_forward_rejects_wrong_motion_channels() -> None:
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    batch = _dummyBatch(config)
    batch["noisyMotion"] = torch.randn(2, 16, config.numBones, 4)
    with pytest.raises(ValueError, match="channels per bone"):
        denoiser(**batch)


def test_forward_rejects_frames_exceeding_max() -> None:
    config = _smallConfig(maxFrames=8)
    denoiser = MotionDenoiserV2(config)
    batch = _dummyBatch(config, frames=16)
    with pytest.raises(ValueError, match="maxFrames"):
        denoiser(**batch)


def test_forward_rejects_timestep_batch_mismatch() -> None:
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    batch = _dummyBatch(config)
    batch["timesteps"] = torch.zeros(3, dtype=torch.long)  # wrong batch
    with pytest.raises(ValueError, match="timesteps"):
        denoiser(**batch)


def test_forward_rejects_text_dim_mismatch() -> None:
    config = _smallConfig(textEmbedDim=64)
    denoiser = MotionDenoiserV2(config)
    batch = _dummyBatch(config)
    batch["textHiddenStates"] = torch.randn(2, 8, 96)
    with pytest.raises(ValueError, match="textEmbedDim"):
        denoiser(**batch)


def test_forward_rejects_global_when_disabled() -> None:
    config = _smallConfig(globalChannels=0)
    denoiser = MotionDenoiserV2(config)
    batch = _dummyBatch(config)
    batch["noisyGlobalFeatures"] = torch.randn(2, 16, 3)
    # Replace the now-invalid globalChannels in batch fixture
    with pytest.raises(ValueError, match="globalChannels=0"):
        denoiser(**batch)


# ---------------------------------------------------------------------
# Determinism / output stability
# ---------------------------------------------------------------------
def test_forward_is_deterministic_in_eval_mode() -> None:
    config = _smallConfig()
    torch.manual_seed(0)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    torch.manual_seed(1)
    batch = _dummyBatch(config)

    out1 = denoiser(**batch)
    out2 = denoiser(**batch)
    assert torch.allclose(out1.boneOutput, out2.boneOutput)
    if out1.globalOutput is not None and out2.globalOutput is not None:
        assert torch.allclose(out1.globalOutput, out2.globalOutput)


def test_output_is_finite() -> None:
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    assert torch.isfinite(output.boneOutput).all()
    if output.globalOutput is not None:
        assert torch.isfinite(output.globalOutput).all()


# ---------------------------------------------------------------------
# Param budget
# ---------------------------------------------------------------------
def test_default_config_param_count_within_budget() -> None:
    config = MotionDenoiserV2Config()  # production defaults
    denoiser = MotionDenoiserV2(config)
    paramCount = denoiser.numParameters()
    # Plan target: ~10–15M for the denoiser (text encoder adds ~5M).
    # Combined v2 stack (denoiser + text encoder) lands around ~20M,
    # well below the v2 plan ceiling of 30–40M and the 254M Étape 1.
    assert 10_000_000 <= paramCount <= 16_000_000, (
        f"Default denoiser has {paramCount:,} params; "
        f"expected ~10–15M (combined v2 stack ~20M)."
    )


# ---------------------------------------------------------------------
# Backward / gradient flow
# ---------------------------------------------------------------------
def test_gradient_flows_through_all_blocks() -> None:
    config = _smallConfig(numLayers=3)
    denoiser = MotionDenoiserV2(config)
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    loss = output.boneOutput.sum()
    if output.globalOutput is not None:
        loss = loss + output.globalOutput.sum()
    loss.backward()

    # Check every transformer block received a gradient.
    for block in denoiser.blocks:
        for parameter in block.parameters():
            assert parameter.grad is not None
            assert parameter.grad.abs().sum().item() > 0


def test_text_padding_mask_is_respected() -> None:
    """Changing only fully-masked text tokens must not change output."""
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config, textTokens=10)
    # Mask out tokens [5..10) — they should not influence the output.
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 5:] = True
    batch["textKeyPaddingMask"] = mask

    out1 = denoiser(**batch)

    # Replace masked positions with completely different values.
    perturbedText = batch["textHiddenStates"].clone()
    perturbedText[:, 5:, :] = torch.randn_like(perturbedText[:, 5:, :]) * 100
    batch["textHiddenStates"] = perturbedText

    out2 = denoiser(**batch)
    assert torch.allclose(
        out1.boneOutput, out2.boneOutput, atol=1e-5
    ), "Masked text tokens should not affect output."


# ---------------------------------------------------------------------
# Integration with CustomTextEncoder
# ---------------------------------------------------------------------
def test_integration_with_custom_text_encoder() -> None:
    """End-to-end smoke test: tokenizer → encoder → denoiser."""
    from src.shared.model.text import (
        CustomTextEncoder,
        CustomTokenizer,
        CustomTokenizerConfig,
    )

    corpus = [
        "a person walks forward.",
        "a person crawls slowly.",
        "the woman dances gracefully.",
        "someone runs backward.",
        "a man jumps high.",
        "a person sits down.",
    ]
    tokenizer = CustomTokenizer.train(
        corpus * 4,
        config=CustomTokenizerConfig(vocabSize=128, maxLength=16),
    )
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer,
        hiddenDim=64,
        numLayers=2,
        numHeads=4,
        outputDim=64,  # match denoiser embedDim
    )
    encoder.eval()

    config = _smallConfig(textEmbedDim=64)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()

    encoded = tokenizer.encode(["a person walks forward.", "a man jumps high."])
    encoderOutput = encoder(encoded.inputIds, encoded.attentionMask)
    assert encoderOutput.hiddenStates.shape == (2, 16, 64)

    motion = torch.randn(2, 16, config.numBones, config.motionChannels)
    timesteps = torch.tensor([100, 500])
    globalFeatures = torch.randn(2, 16, config.globalChannels)

    output = denoiser(
        noisyMotion=motion,
        timesteps=timesteps,
        textHiddenStates=encoderOutput.hiddenStates,
        textKeyPaddingMask=encoderOutput.keyPaddingMask,
        noisyGlobalFeatures=globalFeatures,
    )
    assert output.boneOutput.shape == (
        2, 16, config.numBones, config.motionChannels
    )
    assert output.globalOutput is not None
    assert output.globalOutput.shape == (2, 16, config.globalChannels)
    assert torch.isfinite(output.boneOutput).all()
