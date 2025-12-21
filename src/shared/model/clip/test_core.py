"""Tests for ClipModel loss components and logit scale handling."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from src.shared.constants.clip import LOGIT_SCALE_MAX
from src.shared.model.clip.core import ClipModel


@dataclass
class _DummyEncoderOutput:
    """Container mimicking Hugging Face outputs."""

    last_hidden_state: torch.Tensor


class _DummyTextEncoder(torch.nn.Module):
    """Minimal encoder returning trainable hidden states."""

    def __init__(self, vocabSize: int, hiddenSize: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabSize, hiddenSize)
        self.config = SimpleNamespace(hidden_size=hiddenSize)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> _DummyEncoderOutput:
        del attention_mask
        hidden = self.embedding(input_ids)
        return _DummyEncoderOutput(last_hidden_state=hidden)


def _buildTestModel(embedDim: int = 8) -> ClipModel:
    """Return a ClipModel backed by the dummy encoder."""
    dummyEncoder = _DummyTextEncoder(vocabSize=32, hiddenSize=16)
    tokenizerStub = object()
    return ClipModel(
        tokenizer=tokenizerStub,
        textEncoder=dummyEncoder,
        freezeTextEncoder=False,
        embedDim=embedDim,
    )


def test_clip_loss_returns_components() -> None:
    """clipLoss must expose its component metrics."""
    model = _buildTestModel()
    batchSize = 2
    inputIds = torch.randint(0, 16, (batchSize, 5))
    attentionMask = torch.ones_like(inputIds)
    motion = torch.randn(batchSize, 4, 3, 6)

    outputs = model(
        textInputIds=inputIds,
        textAttentionMask=attentionMask,
        motionInput=motion,
        computeLoss=True,
    )

    assert "clip_loss" in outputs
    assert "loss_text_contrastive" in outputs
    assert "loss_motion_cosine" in outputs
    lossValue = float(outputs["clip_loss"])
    assert lossValue > 0.0


def test_logit_scale_clamped_to_constant() -> None:
    """logitScale parameter must honor LOGIT_SCALE_MAX."""
    model = _buildTestModel()
    with torch.no_grad():
        model.logitScale.fill_(LOGIT_SCALE_MAX + 10.0)
    batchSize = 2
    inputIds = torch.randint(0, 16, (batchSize, 5))
    attentionMask = torch.ones_like(inputIds)
    motion = torch.randn(batchSize, 4, 3, 6)

    outputs = model(
        textInputIds=inputIds,
        textAttentionMask=attentionMask,
        motionInput=motion,
    )
    clampedValue = float(outputs["logit_scale"].max())
    assert clampedValue <= float(torch.exp(torch.tensor(LOGIT_SCALE_MAX))) + 1e-5
