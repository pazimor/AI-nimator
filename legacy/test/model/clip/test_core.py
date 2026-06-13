"""Tests for ClipModel loss components and logit scale handling."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from ainimator.core.constants.clip import LOGIT_SCALE_MAX
from ainimator.model.clip.core import ClipModel
from ainimator.model.clip.motion_input import buildMotionInputFromBatch
from ainimator.geometry.components.kinematics import JointXyzComponent
from ainimator.geometry.components.root import RootTranslationComponent


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


def _buildTestModel(
    embedDim: int = 8,
    motionComponents: tuple[object, ...] | None = None,
) -> ClipModel:
    """Return a ClipModel backed by the dummy encoder."""
    dummyEncoder = _DummyTextEncoder(vocabSize=32, hiddenSize=16)
    tokenizerStub = object()
    return ClipModel(
        tokenizer=tokenizerStub,
        textEncoder=dummyEncoder,
        freezeTextEncoder=False,
        embedDim=embedDim,
        motionComponents=motionComponents,
        numBones=3,
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
    assert "loss_motion_contrastive" in outputs
    assert "loss_motion_cosine" in outputs
    lossValue = float(outputs["clip_loss"])
    assert lossValue > 0.0


def test_clip_loss_accepts_multi_positive_mask() -> None:
    """clipLoss must support more than one positive per query."""
    model = _buildTestModel()
    batchSize = 2
    inputIds = torch.randint(0, 16, (batchSize, 5))
    attentionMask = torch.ones_like(inputIds)
    motion = torch.randn(batchSize, 4, 3, 6)
    positiveMask = torch.ones(batchSize, batchSize, dtype=torch.bool)

    outputs = model(
        textInputIds=inputIds,
        textAttentionMask=attentionMask,
        motionInput=motion,
        positiveMask=positiveMask,
        computeLoss=True,
    )

    assert "clip_loss" in outputs
    assert float(outputs["clip_loss"]) > 0.0


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


def test_encode_pooled_text_matches_encode_text_projection() -> None:
    """encodePooledText must match encodeText after the same mean pooling."""
    model = _buildTestModel()
    inputIds = torch.randint(0, 16, (2, 5))
    attentionMask = torch.ones_like(inputIds)

    with torch.no_grad():
        hidden = model.textEncoder(
            input_ids=inputIds,
            attention_mask=attentionMask,
        ).last_hidden_state
    pooled = model._maskedMean(hidden, attentionMask)
    embedsFromText, _ = model.encodeText(inputIds, attentionMask)
    embedsFromPooled, _ = model.encodePooledText(pooled)

    assert torch.allclose(embedsFromText, embedsFromPooled, atol=1e-5)


def test_forward_accepts_pooled_text_without_tokens() -> None:
    """forward must support pooledText-only inputs."""
    model = _buildTestModel()
    inputIds = torch.randint(0, 16, (2, 5))
    attentionMask = torch.ones_like(inputIds)
    with torch.no_grad():
        hidden = model.textEncoder(
            input_ids=inputIds,
            attention_mask=attentionMask,
        ).last_hidden_state
    pooled = model._maskedMean(hidden, attentionMask)
    motion = torch.randn(2, 4, 3, 6)

    outputs = model(
        textInputIds=None,
        textAttentionMask=None,
        motionInput=motion,
        pooledText=pooled,
        computeLoss=True,
    )

    assert "clip_loss" in outputs


def test_encode_motion_ignores_masked_padding_frames() -> None:
    """encodeMotion must ignore padded tail frames when motionMask is set."""
    torch.manual_seed(0)
    model = _buildTestModel()
    model.eval()

    validFrames = torch.randn(2, 3, 6)
    motion = torch.zeros(2, 4, 3, 6)
    motion[0, :2] = validFrames
    motion[1, :2] = validFrames
    motion[1, 2:] = torch.randn(2, 3, 6)
    motionMask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )

    embeds = model.encodeMotion(motion, motionMask=motionMask)

    assert torch.allclose(embeds[0], embeds[1], atol=1e-5)


def test_encode_motion_ignores_masked_padding_with_global_branch() -> None:
    """Padding masks must also hold when global motion features use a late branch."""
    torch.manual_seed(0)
    components = (JointXyzComponent(), RootTranslationComponent())
    model = _buildTestModel(motionComponents=components)
    model.eval()

    jointValid = torch.randn(2, 3, 3)
    rootValid = torch.randn(2, 3)
    batch = {
        "joint_xyz": torch.zeros(2, 4, 3, 3),
        "root_translation": torch.zeros(2, 4, 3),
    }
    batch["joint_xyz"][0, :2] = jointValid
    batch["joint_xyz"][1, :2] = jointValid
    batch["joint_xyz"][1, 2:] = torch.randn(2, 3, 3)
    batch["root_translation"][0, :2] = rootValid
    batch["root_translation"][1, :2] = rootValid
    batch["root_translation"][1, 2:] = torch.randn(2, 3)
    motionMask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )
    motion = buildMotionInputFromBatch(
        batch=batch,
        components=components,
        numBones=3,
    )

    embeds = model.encodeMotion(motion, motionMask=motionMask)

    assert torch.allclose(embeds[0], embeds[1], atol=1e-5)
