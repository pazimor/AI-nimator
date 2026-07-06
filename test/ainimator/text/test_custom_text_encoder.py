"""Unit tests for :mod:`src.shared.model.text.custom_text_encoder`."""

from __future__ import annotations

import pytest
import torch

from ainimator.text.custom_text_encoder import (
    CustomTextEncoder,
    CustomTextEncoderConfig,
    TextEncoderOutput,
)
from ainimator.text.custom_tokenizer import (
    CustomTokenizer,
    CustomTokenizerConfig,
)

# Reuse the same small corpus used to test the tokenizer.
MOTION_CORPUS: list[str] = [
    "a person walks forward.",
    "a person walks backward.",
    "the person walks slowly.",
    "the person walks quickly.",
    "a man crawls on the floor.",
    "a man crawls forward slowly.",
    "the woman dances in a circle.",
    "the woman dances gracefully.",
    "a person jumps high.",
    "a person jumps over an obstacle.",
    "the person sits down on a chair.",
    "the person sits and stands up.",
    "a person waves their right hand.",
    "a person waves their left hand.",
    "someone runs forward.",
    "someone runs backward then stops.",
]


def _trainTokenizer(maxLength: int = 16, vocabSize: int = 256) -> CustomTokenizer:
    config = CustomTokenizerConfig(
        vocabSize=vocabSize,
        minFrequency=2,
        maxLength=maxLength,
    )
    return CustomTokenizer.train(MOTION_CORPUS, config=config)


# ---------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------
def test_config_rejects_non_divisible_hidden_dim() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CustomTextEncoderConfig(vocabSize=100, hiddenDim=64, numHeads=7)


def test_config_rejects_zero_layers() -> None:
    with pytest.raises(ValueError, match="numLayers"):
        CustomTextEncoderConfig(vocabSize=100, numLayers=0)


def test_config_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        CustomTextEncoderConfig(vocabSize=100, dropout=1.5)


def test_config_effective_ffn_dim_defaults_to_4x_hidden() -> None:
    config = CustomTextEncoderConfig(vocabSize=100, hiddenDim=64)
    assert config.effectiveFfnDim == 256


def test_config_effective_output_dim_defaults_to_hidden() -> None:
    config = CustomTextEncoderConfig(vocabSize=100, hiddenDim=64)
    assert config.effectiveOutputDim == 64
    config = CustomTextEncoderConfig(
        vocabSize=100, hiddenDim=64, outputDim=128
    )
    assert config.effectiveOutputDim == 128


# ---------------------------------------------------------------------
# Forward shape & contracts
# ---------------------------------------------------------------------
def test_forward_returns_per_token_hidden_states() -> None:
    tokenizer = _trainTokenizer(maxLength=16, vocabSize=256)
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer, hiddenDim=32, numLayers=2, numHeads=4
    )
    encoder.eval()
    batch = tokenizer.encode(["a person walks forward.", "someone runs."])

    output = encoder(batch.inputIds, batch.attentionMask)

    assert isinstance(output, TextEncoderOutput)
    assert output.hiddenStates.shape == (2, 16, 32)
    assert output.keyPaddingMask.shape == (2, 16)
    assert output.keyPaddingMask.dtype == torch.bool


def test_forward_with_output_projection_changes_dim() -> None:
    tokenizer = _trainTokenizer(maxLength=16, vocabSize=256)
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer,
        hiddenDim=32,
        numLayers=2,
        numHeads=4,
        outputDim=128,
    )
    encoder.eval()
    batch = tokenizer.encode("a person walks forward.")

    output = encoder(batch.inputIds, batch.attentionMask)

    assert output.hiddenStates.shape == (1, 16, 128)
    assert encoder.outputDim == 128


def test_key_padding_mask_marks_padding_positions() -> None:
    tokenizer = _trainTokenizer(maxLength=16, vocabSize=256)
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer, hiddenDim=32, numLayers=2, numHeads=4
    )
    encoder.eval()
    batch = tokenizer.encode("a person walks forward.")

    output = encoder(batch.inputIds, batch.attentionMask)
    realLen = int(batch.lengths[0].item())

    # Real-token positions should be False, padding positions True.
    assert not output.keyPaddingMask[0, :realLen].any()
    assert output.keyPaddingMask[0, realLen:].all()


def test_forward_rejects_non_2d_input_ids() -> None:
    tokenizer = _trainTokenizer()
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer, hiddenDim=32, numLayers=2, numHeads=4
    )
    inputIds = torch.zeros(3, dtype=torch.long)
    attentionMask = torch.ones(3, dtype=torch.float32)
    with pytest.raises(ValueError, match="2-D"):
        encoder(inputIds, attentionMask)


def test_forward_rejects_mask_shape_mismatch() -> None:
    tokenizer = _trainTokenizer(maxLength=16, vocabSize=256)
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer, hiddenDim=32, numLayers=2, numHeads=4
    )
    inputIds = torch.zeros((2, 16), dtype=torch.long)
    badMask = torch.ones((2, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="does not match"):
        encoder(inputIds, badMask)


def test_forward_rejects_sequence_longer_than_max_length() -> None:
    tokenizer = _trainTokenizer(maxLength=8, vocabSize=256)
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer, hiddenDim=32, numLayers=2, numHeads=4
    )
    encoder._config = CustomTextEncoderConfig(  # narrow to test the guard
        vocabSize=tokenizer.vocabSize,
        maxLength=4,
        hiddenDim=32,
        numLayers=2,
        numHeads=4,
    )
    inputIds = torch.zeros((1, 8), dtype=torch.long)
    attentionMask = torch.ones((1, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="exceeds configured"):
        encoder(inputIds, attentionMask)


# ---------------------------------------------------------------------
# Padding invariance
# ---------------------------------------------------------------------
def test_padding_does_not_affect_real_token_outputs() -> None:
    """Adding more padding to the right must not change real-token outputs."""
    # Tokenizer with short max_length — the encoder is sized to max_length=32
    # so we can pad up to that ceiling without overflowing.
    tokenizer = _trainTokenizer(maxLength=16, vocabSize=256)
    config = CustomTextEncoderConfig(
        vocabSize=tokenizer.vocabSize,
        maxLength=32,
        hiddenDim=32,
        numLayers=2,
        numHeads=4,
        padTokenId=tokenizer.padTokenId,
        dropout=0.0,
    )
    encoder = CustomTextEncoder(config)
    encoder.eval()

    text = "a person walks forward."
    shortBatch = tokenizer.encode(text)
    realLen = int(shortBatch.lengths[0].item())

    padId = tokenizer.padTokenId
    extraPad = 8  # 16 + 8 = 24 ≤ encoder.maxLength=32
    paddedIds = torch.cat(
        [
            shortBatch.inputIds,
            torch.full(
                (1, extraPad),
                fill_value=padId,
                dtype=shortBatch.inputIds.dtype,
            ),
        ],
        dim=1,
    )
    paddedMask = torch.cat(
        [
            shortBatch.attentionMask,
            torch.zeros(
                (1, extraPad),
                dtype=shortBatch.attentionMask.dtype,
            ),
        ],
        dim=1,
    )

    out1 = encoder(shortBatch.inputIds, shortBatch.attentionMask)
    out2 = encoder(paddedIds, paddedMask)

    diff = (
        out1.hiddenStates[0, :realLen]
        - out2.hiddenStates[0, :realLen]
    ).abs().max()
    assert diff.item() < 1e-5


# ---------------------------------------------------------------------
# Param budget
# ---------------------------------------------------------------------
def test_default_config_param_count_within_budget() -> None:
    config = CustomTextEncoderConfig(vocabSize=8000)
    encoder = CustomTextEncoder(config)
    paramCount = encoder.numParameters()
    # The plan targets 5–10M params for the default config.
    assert 4_000_000 <= paramCount <= 10_000_000, (
        f"Default encoder has {paramCount:,} params, expected 4–10M."
    )


def test_only_trainable_parameters_are_counted_by_default() -> None:
    encoder = CustomTextEncoder(CustomTextEncoderConfig(vocabSize=200))
    total = encoder.numParameters(trainableOnly=False)
    trainable = encoder.numParameters(trainableOnly=True)
    assert total == trainable  # Nothing is frozen by default


# ---------------------------------------------------------------------
# Backward / gradient flow
# ---------------------------------------------------------------------
def test_gradient_flows_through_all_blocks() -> None:
    tokenizer = _trainTokenizer(maxLength=16, vocabSize=256)
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer, hiddenDim=32, numLayers=3, numHeads=4
    )
    batch = tokenizer.encode(["a person walks forward."])
    output = encoder(batch.inputIds, batch.attentionMask)
    output.hiddenStates.sum().backward()

    # Embedding gradients should be non-zero on at least the real tokens.
    grad = encoder.tokenEmbedding.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0
    # Each transformer block should also receive gradient.
    for block in encoder.blocks:
        for parameter in block.parameters():
            assert parameter.grad is not None


# ---------------------------------------------------------------------
# Integration with tokenizer
# ---------------------------------------------------------------------
def test_from_tokenizer_matches_vocab_and_pad_id() -> None:
    tokenizer = _trainTokenizer(maxLength=16, vocabSize=300)
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer, hiddenDim=32, numLayers=2, numHeads=4
    )
    assert encoder.config.vocabSize == tokenizer.vocabSize
    assert encoder.config.maxLength == tokenizer.config.maxLength
    assert encoder.config.padTokenId == tokenizer.padTokenId


def test_encode_texts_helper_returns_correct_shape() -> None:
    tokenizer = _trainTokenizer(maxLength=16, vocabSize=256)
    encoder = CustomTextEncoder.fromTokenizer(
        tokenizer, hiddenDim=32, numLayers=2, numHeads=4
    )
    encoder.eval()
    output = encoder.encodeTexts(
        tokenizer,
        ["a person walks forward.", "someone runs backward then stops."],
    )
    assert output.hiddenStates.shape == (2, 16, 32)


# ---------------------------------------------------------------------
# Phase F — null embedding + L2-normalisation
# ---------------------------------------------------------------------
def _buildSmallEncoder(
    useNullEmbedding: bool = True,
    l2NormalizeOutput: bool = True,
    outputDim: int = 0,
) -> tuple[CustomTokenizer, CustomTextEncoder]:
    tokenizer = _trainTokenizer(maxLength=16, vocabSize=256)
    config = CustomTextEncoderConfig(
        vocabSize=tokenizer.vocabSize,
        maxLength=tokenizer.config.maxLength,
        hiddenDim=32,
        numLayers=2,
        numHeads=4,
        ffnDim=64,
        dropout=0.0,
        padTokenId=tokenizer.padTokenId,
        outputDim=outputDim,
        useNullEmbedding=useNullEmbedding,
        l2NormalizeOutput=l2NormalizeOutput,
    )
    encoder = CustomTextEncoder(config=config)
    encoder.eval()
    return tokenizer, encoder


def test_null_embedding_is_registered_parameter() -> None:
    _, encoder = _buildSmallEncoder(useNullEmbedding=True)
    assert encoder.nullEmbedding is not None
    assert encoder.nullEmbedding.requires_grad
    assert encoder.nullEmbedding.shape == (1, 1, encoder.outputDim)


def test_null_embedding_can_be_disabled() -> None:
    _, encoder = _buildSmallEncoder(useNullEmbedding=False)
    assert encoder.nullEmbedding is None
    with pytest.raises(RuntimeError, match="useNullEmbedding=False"):
        encoder.forwardNull(batchSize=2)


def test_forward_null_returns_expected_shapes() -> None:
    _, encoder = _buildSmallEncoder()
    out = encoder.forwardNull(batchSize=4)
    assert out.hiddenStates.shape == (4, 1, encoder.outputDim)
    assert out.keyPaddingMask.shape == (4, 1)
    # All positions are real (not padding) — keyPaddingMask is all False.
    assert not out.keyPaddingMask.any().item()


def test_forward_null_is_constant_across_batch() -> None:
    _, encoder = _buildSmallEncoder()
    out = encoder.forwardNull(batchSize=8)
    first = out.hiddenStates[0]
    assert torch.allclose(out.hiddenStates, first.expand_as(out.hiddenStates))


def test_forward_null_distinct_from_empty_string_encoding() -> None:
    # Sanity check that the null embedding is NOT a function of the
    # encoder forward path on EMPTY_PROMPT="" — that was the whole point
    # of the change.
    tokenizer, encoder = _buildSmallEncoder()
    emptyBatch = tokenizer.encode([""])
    encOutput = encoder(emptyBatch.inputIds, emptyBatch.attentionMask)
    nullOutput = encoder.forwardNull(batchSize=1)
    # Pool both to a single (1, D) vector and compare.
    encMask = (~encOutput.keyPaddingMask).float().unsqueeze(-1)
    encPooled = (encOutput.hiddenStates * encMask).sum(dim=1) / encMask.sum(
        dim=1
    ).clamp(min=1.0)
    nullPooled = nullOutput.hiddenStates.squeeze(1)
    cosine = torch.nn.functional.cosine_similarity(encPooled, nullPooled).item()
    # They should NOT be identical.  With random init the two vectors
    # live in independent subspaces, so cosine ≪ 1.
    assert abs(cosine) < 0.95, (
        f"Null embedding cosine to empty-string encoding is {cosine:.4f}, "
        "which suggests the null token collapsed onto the BOS embedding."
    )


def test_l2_normalise_output_has_unit_norm() -> None:
    tokenizer, encoder = _buildSmallEncoder(l2NormalizeOutput=True)
    batch = tokenizer.encode(["a person walks forward."])
    output = encoder(batch.inputIds, batch.attentionMask)
    norms = output.hiddenStates.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_l2_normalise_output_can_be_disabled() -> None:
    tokenizer, encoder = _buildSmallEncoder(l2NormalizeOutput=False)
    batch = tokenizer.encode(["a person walks forward."])
    output = encoder(batch.inputIds, batch.attentionMask)
    norms = output.hiddenStates.norm(dim=-1)
    # With L2 disabled, norms are NOT all 1.0.
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


def test_forward_null_respects_l2_normalisation_setting() -> None:
    _, encoder = _buildSmallEncoder(l2NormalizeOutput=True)
    out = encoder.forwardNull(batchSize=2)
    norms = out.hiddenStates.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
