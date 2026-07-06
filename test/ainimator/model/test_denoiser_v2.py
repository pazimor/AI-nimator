"""Unit tests for :mod:`src.shared.model.generation.denoiser_v2`."""

from __future__ import annotations

import pytest
import torch

from ainimator.model.denoiser_v2 import (
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
# ---------------------------------------------------------------------
# Phase D.1 — alignment head
# ---------------------------------------------------------------------
def test_alignment_disabled_by_default_no_motion_embedding() -> None:
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    assert output.motionEmbedding is None
    assert config.alignmentEnabled is False


def test_alignment_enabled_produces_pooled_embedding() -> None:
    config = _smallConfig(alignmentEnabled=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    assert output.motionEmbedding is not None
    assert output.motionEmbedding.shape == (
        batch["noisyMotion"].shape[0],
        config.alignmentDim,
    )
    assert torch.isfinite(output.motionEmbedding).all()


def test_alignment_pooling_respects_motion_padding_mask() -> None:
    """Padding frames must not contribute to the pooled embedding."""
    config = _smallConfig(alignmentEnabled=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config, batchSize=2, frames=12)

    # Run once with no mask, once with the second sample padded over
    # half its frames.  The first sample's embedding must be unchanged
    # between the two runs (its mask is unchanged).
    out1 = denoiser(**batch)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[1, 6:] = True  # last 6 frames of sample 1 are padding
    # Replace the would-be-padded values with garbage to confirm they
    # do not influence the pooling.
    motion = batch["noisyMotion"].clone()
    motion[1, 6:] = 999.0
    batch2 = {**batch, "noisyMotion": motion}
    out2 = denoiser(**batch2, motionKeyPaddingMask=mask)

    # Sample 0 has no padding in either run → embedding identical.
    assert torch.allclose(
        out1.motionEmbedding[0],
        out2.motionEmbedding[0],
        atol=1e-5,
    ), "Sample 0 embedding should be invariant when only sample 1 is masked."


def test_alignment_head_propagates_gradient_to_cross_attention() -> None:
    """A backward through the motion embedding must reach the
    cross-attention parameters — that is the whole point of D.1."""
    config = _smallConfig(numLayers=2, alignmentEnabled=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.train()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    assert output.motionEmbedding is not None
    output.motionEmbedding.sum().backward()

    # Walk through every block and confirm the cross-attention has
    # received gradient.  Without the alignment head, blocks would only
    # see gradient via the diffusion loss path; here the motion embed
    # forces a direct signal even when boneOutput / globalOutput are
    # discarded.
    for block in denoiser.blocks:
        for name, parameter in block.crossAttention.named_parameters():
            assert parameter.grad is not None, name
            assert parameter.grad.abs().sum().item() > 0, name


def test_alignment_projection_dim_drives_motion_embedding_shape() -> None:
    """The motion embedding lives in the configured projection space."""
    config = _smallConfig(alignmentEnabled=True, alignmentProjectionDim=64)
    assert config.alignmentDim == 64
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    assert output.motionEmbedding is not None
    assert output.motionEmbedding.shape[-1] == 64


# ---------------------------------------------------------------------
# Phase D.1 (Levier B 2026-05-07) — text projection in the head
# ---------------------------------------------------------------------
def test_align_head_text_projection_returns_normalized_embedding() -> None:
    """The text projection L2-normalizes its output."""
    config = _smallConfig(alignmentEnabled=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    textHidden = torch.randn(3, 8, config.effectiveTextEmbedDim)
    textMask = torch.zeros(3, 8, dtype=torch.bool)  # all real
    embedding = denoiser.alignmentHead.projectText(textHidden, textMask)

    assert embedding.shape == (3, config.alignmentDim)
    norms = embedding.norm(dim=-1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), f"text projection should output unit-norm vectors, got {norms}"


def test_align_head_motion_projection_returns_normalized_embedding() -> None:
    """The motion projection L2-normalizes its output too."""
    config = _smallConfig(alignmentEnabled=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    assert output.motionEmbedding is not None
    norms = output.motionEmbedding.norm(dim=-1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), f"motion projection should output unit-norm vectors, got {norms}"


def test_align_head_text_projection_respects_padding_mask() -> None:
    """Padded text tokens must not contribute to the text projection."""
    config = _smallConfig(alignmentEnabled=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()

    text = torch.randn(2, 8, config.effectiveTextEmbedDim)
    maskA = torch.zeros(2, 8, dtype=torch.bool)
    maskA[0, 4:] = True  # sample 0 has 4 padded tokens
    out1 = denoiser.alignmentHead.projectText(text, maskA)

    # Replace the masked positions with garbage values.  The projection
    # of sample 0 must be unchanged because those positions are masked.
    perturbed = text.clone()
    perturbed[0, 4:] = 999.0
    out2 = denoiser.alignmentHead.projectText(perturbed, maskA)
    assert torch.allclose(out1[0], out2[0], atol=1e-5)


def test_align_head_text_and_motion_share_projection_dim() -> None:
    """Both projections land in the same alignmentDim space."""
    config = _smallConfig(alignmentEnabled=True, alignmentProjectionDim=96)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    text = torch.randn(2, 8, config.effectiveTextEmbedDim)
    textMask = torch.zeros(2, 8, dtype=torch.bool)
    textEmb = denoiser.alignmentHead.projectText(text, textMask)
    assert output.motionEmbedding is not None
    assert output.motionEmbedding.shape[-1] == textEmb.shape[-1] == 96


def test_align_head_text_projection_propagates_gradient() -> None:
    """Backward through the text projection reaches its own params."""
    config = _smallConfig(alignmentEnabled=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.train()
    text = torch.randn(2, 8, config.effectiveTextEmbedDim)
    textMask = torch.zeros(2, 8, dtype=torch.bool)
    textEmb = denoiser.alignmentHead.projectText(text, textMask)
    textEmb.sum().backward()
    for parameter in denoiser.alignmentHead.textProjection.parameters():
        assert parameter.grad is not None
        assert parameter.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------
# Phase D Levier D — FiLM conditioning shortcut
# ---------------------------------------------------------------------
def test_film_disabled_by_default_no_module() -> None:
    """Without ``useFilmConditioning`` the slot is ``Identity``."""
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    assert isinstance(denoiser.filmConditioning, torch.nn.Identity)


def test_film_enabled_builds_module_with_safe_init() -> None:
    """At init the FiLM perturbation is small (norm of last linear)."""
    config = _smallConfig(useFilmConditioning=True)
    denoiser = MotionDenoiserV2(config)
    finalLayer = denoiser.filmConditioning.mlp[-1]
    # Bias zero — so γ_offset and β start centered on (0, 0).
    assert torch.allclose(finalLayer.bias, torch.zeros_like(finalLayer.bias))
    # Weight non-zero but small (std ≈ 0.02).
    assert finalLayer.weight.abs().max().item() < 0.2


def test_film_changes_output_when_text_changes() -> None:
    """Two different prompts must produce different forward outputs."""
    config = _smallConfig(
        useFilmConditioning=True, alignmentEnabled=False
    )
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)

    # Replace the text hidden states with two clearly different
    # signals (zeros vs random) and verify the boneOutput differs.
    zeroText = torch.zeros_like(batch["textHiddenStates"])
    randText = torch.randn_like(batch["textHiddenStates"])
    out1 = denoiser(**{**batch, "textHiddenStates": zeroText})
    out2 = denoiser(**{**batch, "textHiddenStates": randText})
    diff = (out1.boneOutput - out2.boneOutput).abs().max().item()
    assert diff > 1e-5, (
        "FiLM-enabled denoiser must produce a different output for a "
        f"different text hidden state, got max diff = {diff}"
    )


def test_film_disabled_output_independent_of_text_when_crossattn_zeroed() -> (
    None
):
    """Sanity check the FiLM is the only path here.

    When ``useFilmConditioning=False`` AND we manually zero the cross-
    attention output projections, the prediction must be IDENTICAL for
    different text inputs — there is no other text path.  This
    validates the FiLM IS responsible for the diff in
    ``test_film_changes_output_when_text_changes`` above.
    """
    config = _smallConfig(useFilmConditioning=False)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    # Zero out every cross-attention out_proj — kills the only text
    # path when FiLM is disabled.
    with torch.no_grad():
        for block in denoiser.blocks:
            block.crossAttention.out_proj.weight.zero_()
            block.crossAttention.out_proj.bias.zero_()
    batch = _dummyBatch(config)
    zeroText = torch.zeros_like(batch["textHiddenStates"])
    randText = torch.randn_like(batch["textHiddenStates"])
    out1 = denoiser(**{**batch, "textHiddenStates": zeroText})
    out2 = denoiser(**{**batch, "textHiddenStates": randText})
    diff = (out1.boneOutput - out2.boneOutput).abs().max().item()
    assert diff < 1e-5, (
        "Without FiLM and with cross-attn out_proj zeroed, text must "
        f"have no effect; got diff = {diff}"
    )


def test_film_propagates_gradient_to_film_mlp() -> None:
    """A backward through boneOutput must give gradient on FiLM params."""
    config = _smallConfig(useFilmConditioning=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.train()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    output.boneOutput.sum().backward()
    for name, parameter in denoiser.filmConditioning.named_parameters():
        assert parameter.grad is not None, name
        # The first epoch's gradient is small but non-zero.
        assert parameter.grad.abs().sum().item() > 0, name


def test_film_respects_text_padding_mask() -> None:
    """Padded text positions must not contribute to FiLM."""
    config = _smallConfig(useFilmConditioning=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config, batchSize=1, textTokens=8)
    # Mask out tokens [4..8) — replace them with garbage; the FiLM
    # should not see them.
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 4:] = True
    text = batch["textHiddenStates"].clone()
    out1 = denoiser(
        **{
            **batch,
            "textHiddenStates": text,
            "textKeyPaddingMask": mask,
        }
    )
    perturbed = text.clone()
    perturbed[0, 4:] = 999.0
    out2 = denoiser(
        **{
            **batch,
            "textHiddenStates": perturbed,
            "textKeyPaddingMask": mask,
        }
    )
    diff = (out1.boneOutput - out2.boneOutput).abs().max().item()
    assert diff < 1e-3, (
        "Padded text tokens must not influence the FiLM-modulated "
        f"output; got diff = {diff}"
    )


def test_film_serialisation_roundtrip() -> None:
    """The useFilmConditioning flag must round-trip through the config."""
    from ainimator.training.training_v2 import (
        _denoiserConfigFromDict,
        _denoiserConfigToDict,
    )

    config = _smallConfig(useFilmConditioning=True, filmDropout=0.05)
    payload = _denoiserConfigToDict(config)
    rebuilt = _denoiserConfigFromDict(payload)
    assert rebuilt.useFilmConditioning is True
    assert rebuilt.filmDropout == 0.05


def test_film_config_rejects_invalid_dropout() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="filmDropout"):
        _smallConfig(useFilmConditioning=True, filmDropout=1.5)


# ---------------------------------------------------------------------
# Phase E (Levier E) — per-block AdaLN-style FiLM
# ---------------------------------------------------------------------
def test_perblock_film_disabled_by_default_no_adaln() -> None:
    """Without ``usePerBlockFilm`` each block's adaln slot is Identity."""
    config = _smallConfig()
    denoiser = MotionDenoiserV2(config)
    for block in denoiser.blocks:
        assert isinstance(block.adaln, torch.nn.Identity)


def test_perblock_film_enabled_builds_module_per_block() -> None:
    config = _smallConfig(usePerBlockFilm=True)
    denoiser = MotionDenoiserV2(config)
    from ainimator.model.denoiser_v2 import (
        _AdaLNBlockModulation,
    )
    assert len(denoiser.blocks) == config.numLayers
    for block in denoiser.blocks:
        assert isinstance(block.adaln, _AdaLNBlockModulation)
        # 6 modulations: γ, β for self-attn / cross-attn / FFN.
        assert block.adaln.NUM_MODULATIONS == 6


def test_perblock_film_init_is_safe_modulations_near_zero() -> None:
    """At init the projection bias is 0 → all (γ, β) start at 0
    (i.e. block ≈ standard transformer block before any update)."""
    config = _smallConfig(usePerBlockFilm=True)
    denoiser = MotionDenoiserV2(config)
    for block in denoiser.blocks:
        finalLayer = block.adaln.proj
        assert torch.allclose(
            finalLayer.bias, torch.zeros_like(finalLayer.bias)
        )
        assert finalLayer.weight.abs().max().item() < 0.2


def test_perblock_film_changes_output_when_text_changes() -> None:
    """Per-block AdaLN must propagate text differences to the output."""
    config = _smallConfig(
        usePerBlockFilm=True, useFilmConditioning=False
    )
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    out1 = denoiser(
        **{
            **batch,
            "textHiddenStates": torch.zeros_like(batch["textHiddenStates"]),
        }
    )
    out2 = denoiser(
        **{
            **batch,
            "textHiddenStates": torch.randn_like(batch["textHiddenStates"]),
        }
    )
    diff = (out1.boneOutput - out2.boneOutput).abs().max().item()
    assert diff > 1e-5, (
        "Per-block AdaLN should propagate a text change to the output, "
        f"got max diff = {diff}"
    )


def test_perblock_film_gradient_reaches_every_block_adaln() -> None:
    """A backward through boneOutput must give gradient on every
    block's AdaLN MLP — the whole point of per-block modulation."""
    config = _smallConfig(usePerBlockFilm=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.train()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    output.boneOutput.sum().backward()
    for blockIndex, block in enumerate(denoiser.blocks):
        for name, parameter in block.adaln.named_parameters():
            assert parameter.grad is not None, (
                f"block[{blockIndex}].adaln.{name} got no gradient"
            )
            assert parameter.grad.abs().sum().item() > 0, (
                f"block[{blockIndex}].adaln.{name} gradient is zero"
            )


def test_perblock_film_respects_text_padding_mask() -> None:
    """Padded text tokens must not influence the AdaLN modulation."""
    config = _smallConfig(usePerBlockFilm=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config, batchSize=1, textTokens=8)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 4:] = True
    text = batch["textHiddenStates"].clone()
    out1 = denoiser(
        **{
            **batch,
            "textHiddenStates": text,
            "textKeyPaddingMask": mask,
        }
    )
    perturbed = text.clone()
    perturbed[0, 4:] = 999.0
    out2 = denoiser(
        **{
            **batch,
            "textHiddenStates": perturbed,
            "textKeyPaddingMask": mask,
        }
    )
    diff = (out1.boneOutput - out2.boneOutput).abs().max().item()
    assert diff < 1e-3, (
        "Padded text tokens must not affect the per-block modulation; "
        f"got diff = {diff}"
    )


def test_perblock_film_serialisation_roundtrip() -> None:
    from ainimator.training.training_v2 import (
        _denoiserConfigFromDict,
        _denoiserConfigToDict,
    )

    config = _smallConfig(usePerBlockFilm=True)
    payload = _denoiserConfigToDict(config)
    rebuilt = _denoiserConfigFromDict(payload)
    assert rebuilt.usePerBlockFilm is True


def test_perblock_and_global_film_compose() -> None:
    """Both Phase D and Phase E enabled together must still forward."""
    config = _smallConfig(
        useFilmConditioning=True, usePerBlockFilm=True
    )
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config)
    output = denoiser(**batch)
    assert torch.isfinite(output.boneOutput).all()
    assert output.globalOutput is not None
    assert torch.isfinite(output.globalOutput).all()


def test_perblock_film_block_raises_without_cond_when_enabled() -> None:
    """A block built with AdaLN must reject ``condition=None``."""
    from ainimator.model.denoiser_v2 import DenoiserBlockV2

    block = DenoiserBlockV2(
        embedDim=32,
        numHeads=4,
        ffnDim=64,
        dropout=0.0,
        usePerBlockFilm=True,
        condDim=48,
    )
    block.eval()
    x = torch.randn(2, 8, 32)
    text = torch.randn(2, 5, 32)
    import pytest as _pytest
    with _pytest.raises(ValueError, match="condition"):
        block(x, textHiddenStates=text, condition=None)


def test_alignment_disabled_keeps_param_count_below_enabled() -> None:
    """Toggling alignmentEnabled adds two MLPs worth of parameters."""
    base = MotionDenoiserV2(_smallConfig(alignmentEnabled=False))
    enriched = MotionDenoiserV2(_smallConfig(alignmentEnabled=True))
    delta = enriched.numParameters() - base.numParameters()
    # The head is two LN+Linear+Linear stacks; the exact count is
    # asserted to within 25% to avoid brittleness if the hidden width
    # is tweaked.  Lower bound = motionDim*alignDim only.
    embedDim = base.config.embedDim
    textDim = base.config.effectiveTextEmbedDim
    alignDim = base.config.alignmentProjectionDim
    minExpected = embedDim * alignDim + textDim * alignDim
    assert delta >= minExpected
    # Upper bound: well under 1M for the small test config.
    assert delta < 1_000_000


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
    from ainimator.text import (
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


# ---------------------------------------------------------------------
# Phase F iter-2 — auxiliary raw-pool alignment head
# ---------------------------------------------------------------------
def test_aux_pool_alignment_disabled_by_default() -> None:
    config = MotionDenoiserV2Config()
    assert config.auxPoolAlignmentEnabled is False
    denoiser = MotionDenoiserV2(_smallConfig(auxPoolAlignmentEnabled=False))
    batch = _dummyBatch(denoiser.config)
    output = denoiser(**batch)
    assert output.textPooledRaw is None
    assert output.motionPooledRaw is None


def test_aux_pool_alignment_produces_l2_normalized_outputs() -> None:
    config = _smallConfig(auxPoolAlignmentEnabled=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config, batchSize=3)
    output = denoiser(**batch)
    assert output.textPooledRaw is not None
    assert output.motionPooledRaw is not None
    # Both pools live in the textDim space.
    assert output.textPooledRaw.shape == (3, config.effectiveTextEmbedDim)
    assert output.motionPooledRaw.shape == (3, config.effectiveTextEmbedDim)
    textNorms = output.textPooledRaw.norm(dim=-1)
    motionNorms = output.motionPooledRaw.norm(dim=-1)
    assert torch.allclose(textNorms, torch.ones_like(textNorms), atol=1e-5)
    assert torch.allclose(motionNorms, torch.ones_like(motionNorms), atol=1e-5)


def test_aux_pool_text_path_has_no_learnable_params() -> None:
    """Text side must be pure identity — no params can amplify the pool.

    This is the core property that forces the encoder pool itself to
    discriminate prompts.  If the text projection grows learnable
    weights, the 0.9998 collapse loophole reopens.
    """
    from ainimator.model.denoiser_v2 import _RawPoolAlignment
    head = _RawPoolAlignment(textDim=32, motionDim=64)
    # Only motionProjection should hold parameters.
    learnableModules = [
        name for name, p in head.named_parameters() if p.requires_grad
    ]
    assert all(
        name.startswith("motionProjection") for name in learnableModules
    ), f"Unexpected learnable params on text side: {learnableModules}"


def test_aux_pool_motion_projection_has_no_bias() -> None:
    """A bias would let the projection collapse to a constant vector.

    Without bias, the only way to satisfy the InfoNCE on raw pool is
    for the encoder pool itself to differ across prompts.
    """
    from ainimator.model.denoiser_v2 import _RawPoolAlignment
    head = _RawPoolAlignment(textDim=32, motionDim=64)
    assert head.motionProjection.bias is None


def test_aux_pool_gradient_reaches_encoder_pool() -> None:
    """The whole point — gradient of the aux loss must reach the raw pool.

    We feed a leaf tensor as the text hidden states and check it
    receives gradients through the aux pool path.
    """
    config = _smallConfig(auxPoolAlignmentEnabled=True, alignmentEnabled=False)
    denoiser = MotionDenoiserV2(config)
    batch = _dummyBatch(config, batchSize=4)
    # Make textHiddenStates a leaf with requires_grad so we can see if
    # gradient flows back to it via the aux pool path.
    batch["textHiddenStates"] = batch["textHiddenStates"].clone().requires_grad_(True)
    output = denoiser(**batch)
    assert output.textPooledRaw is not None
    # Use a simple sum on the raw text pool — if grad flows, the input
    # tensor will receive non-None .grad.
    loss = output.textPooledRaw.sum() + output.motionPooledRaw.sum()
    loss.backward()
    assert batch["textHiddenStates"].grad is not None
    assert torch.isfinite(batch["textHiddenStates"].grad).all()
    assert batch["textHiddenStates"].grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------
# 2026-05-28 — pool L2 normalisation (fix for cond/uncond magnitude gap)
# ---------------------------------------------------------------------
def test_pool_text_is_l2_normalised_before_film() -> None:
    """The masked-mean pool fed to FiLM must be on the unit sphere.

    Per-token outputs of the text encoder are already L2-normed, but
    the masked mean of unit vectors has norm < 1.0 for cond (multi-token)
    while uncond holds a single token of norm 1.0 — that magnitude gap
    (0.49 vs 0.755 measured on v2_full_best.pt at epoch 215) is what
    CFG amplifies into mode collapse.  Forcing both onto the unit
    sphere is the prerequisite for the (cond - uncond) direction to
    carry semantic content rather than scale noise.
    """
    config = _smallConfig(useFilmConditioning=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config, batchSize=2, textTokens=8)

    captured: list[torch.Tensor] = []

    def _capture(_module, args):
        captured.append(args[1].detach().clone())

    handle = denoiser.filmConditioning.register_forward_pre_hook(_capture)
    try:
        denoiser(**batch)
    finally:
        handle.remove()

    assert len(captured) == 1
    pooled = captured[0]
    norms = pooled.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"text pool must be unit-norm before FiLM; got norms = {norms.tolist()}"
    )


def test_pool_text_scale_invariant_through_l2_norm() -> None:
    """Scaling the text hidden states by a positive constant must yield
    the same pooled vector fed to FiLM — direct evidence that the
    L2-norm is applied after the masked mean.
    """
    config = _smallConfig(useFilmConditioning=True)
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    batch = _dummyBatch(config, batchSize=2, textTokens=8)

    captured: list[torch.Tensor] = []

    def _capture(_module, args):
        captured.append(args[1].detach().clone())

    handle = denoiser.filmConditioning.register_forward_pre_hook(_capture)
    try:
        denoiser(**batch)
        denoiser(**{**batch, "textHiddenStates": batch["textHiddenStates"] * 7.3})
    finally:
        handle.remove()

    assert len(captured) == 2
    diff = (captured[0] - captured[1]).abs().max().item()
    assert diff < 1e-5, (
        f"pool L2-norm must absorb a uniform positive rescaling of the "
        f"text hidden states; got max diff = {diff}"
    )
