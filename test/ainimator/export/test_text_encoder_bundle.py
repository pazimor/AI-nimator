"""Tests for the in-engine text-encoder bundle extension (B7 / A7.1).

Three locks (``apps/spec/text_encoding.md`` §4):

* the pooled encoder ONNX graph matches the torch masked-mean path
  (fast, synthetic ``CustomTextEncoder`` — no HF download);
* a bundle exported with ``encoderArtifactPath`` ships
  ``text_encoder.onnx`` + ``tokenizer/`` + the manifest section, and
  the ONNX embedding matches ``encodeTextToPooled`` (gated on the
  canonical CLIP artifact being present locally);
* fail-fast guards: a promptless controller refuses an encoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ainimator.core.constants.controller import (
    NUM_SMPL22_BONES,
    PhaseMode,
    ROOT_LOCAL_MOTION_CHANNELS,
)
from ainimator.core.types.controller import ControllerV2Config
from ainimator.export.bundle import (
    MANIFEST_FILENAME,
    TEXT_ENCODER_FILENAME,
    TOKENIZER_SUBDIR,
    exportControllerBundle,
)
from ainimator.export.onnx import (
    ONNX_PARITY_TOLERANCE,
    exportPooledTextEncoder,
)
from ainimator.model.controller_v2 import MotionController
from ainimator.model.motion_normalizer import MotionNormalizer
from ainimator.text.custom_text_encoder import (
    CustomTextEncoder,
    CustomTextEncoderConfig,
)

_CLIP_ARTIFACT = Path("output/clip_text_artifact")
_MAX_LENGTH = 16


def _tinyTextController(promptEmbChannels: int) -> MotionController:
    """Minimal text-conditioned controller for bundle tests."""
    cfg = ControllerV2Config(
        embedDim=64,
        numHeads=4,
        numLayers=2,
        numBones=NUM_SMPL22_BONES,
        contextFrames=1,
        phaseMode=PhaseMode.NONE,
        promptEmbChannels=promptEmbChannels,
    )
    return MotionController(cfg)


def _tinyNorms() -> tuple[MotionNormalizer, MotionNormalizer]:
    """Unit-fitted normalizers (mirrors test_bundle helpers)."""
    state = MotionNormalizer(
        NUM_SMPL22_BONES, 6, ROOT_LOCAL_MOTION_CHANNELS
    )
    delta = MotionNormalizer(
        NUM_SMPL22_BONES, 6, ROOT_LOCAL_MOTION_CHANNELS
    )
    bone = torch.randn(32, NUM_SMPL22_BONES, 6)
    root = torch.randn(32, ROOT_LOCAL_MOTION_CHANNELS)
    state.fitFromTensors([bone], [root])
    delta.fitFromTensors(
        [bone[1:] - bone[:-1]], [root[1:] - root[:-1]]
    )
    return state, delta


def test_pooled_export_matches_torch(tmp_path: Path) -> None:
    """ONNX pooled output == torch masked-mean pooling (synthetic)."""
    onnxruntime = pytest.importorskip("onnxruntime")
    encoder = CustomTextEncoder(
        CustomTextEncoderConfig(
            vocabSize=64,
            maxLength=_MAX_LENGTH,
            hiddenDim=32,
            numLayers=1,
            numHeads=4,
        )
    ).eval()
    onnxPath = exportPooledTextEncoder(
        encoder, tmp_path / "text_encoder.onnx", maxLength=_MAX_LENGTH
    )
    inputIds = torch.randint(0, 64, (2, _MAX_LENGTH))
    attentionMask = torch.ones(2, _MAX_LENGTH)
    attentionMask[0, 5:] = 0.0
    attentionMask[1, 9:] = 0.0
    with torch.no_grad():
        output = encoder(inputIds, attentionMask)
        realMask = attentionMask.unsqueeze(-1)
        expected = (output.hiddenStates * realMask).sum(dim=1)
        expected = expected / realMask.sum(dim=1).clamp(min=1.0)
    session = onnxruntime.InferenceSession(str(onnxPath))
    (actual,) = session.run(
        None,
        {
            "input_ids": inputIds.numpy(),
            "attention_mask": attentionMask.numpy(),
        },
    )
    maxDiff = float(
        (torch.from_numpy(actual) - expected).abs().max()
    )
    assert maxDiff < ONNX_PARITY_TOLERANCE


def test_encoder_refused_for_promptless_controller(
    tmp_path: Path,
) -> None:
    """A controller with promptEmbChannels=0 rejects an encoder."""
    controller = _tinyTextController(promptEmbChannels=0)
    state, delta = _tinyNorms()
    with pytest.raises(ValueError, match="promptEmbChannels == 0"):
        exportControllerBundle(
            controller=controller,
            stateNorm=state,
            deltaNorm=delta,
            controlMean=torch.zeros(2),
            controlStd=torch.ones(2),
            outputDir=tmp_path / "bundle",
            encoderArtifactPath=tmp_path / "missing_artifact",
        )


@pytest.mark.skipif(
    not _CLIP_ARTIFACT.exists(),
    reason="canonical CLIP artifact not present locally",
)
def test_bundle_ships_text_encoder(tmp_path: Path) -> None:
    """A7.1 bundle: files + manifest section + embedding parity."""
    onnxruntime = pytest.importorskip("onnxruntime")
    from ainimator.training.controller_training_v2 import (
        encodeTextToPooled,
        loadFrozenTextEncoder,
    )

    controller = _tinyTextController(promptEmbChannels=512)
    state, delta = _tinyNorms()
    bundleDir = exportControllerBundle(
        controller=controller,
        stateNorm=state,
        deltaNorm=delta,
        controlMean=torch.zeros(2),
        controlStd=torch.ones(2),
        outputDir=tmp_path / "bundle",
        encoderArtifactPath=_CLIP_ARTIFACT,
    )
    assert (bundleDir / TEXT_ENCODER_FILENAME).exists()
    assert (bundleDir / TOKENIZER_SUBDIR / "vocab.json").exists()
    assert (bundleDir / TOKENIZER_SUBDIR / "merges.txt").exists()

    manifest = json.loads(
        (bundleDir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    section = manifest["text_encoder"]
    assert section["file"] == TEXT_ENCODER_FILENAME
    assert section["embedding_channels"] == 512
    assert section["tokenizer"]["type"] == "clip-bpe"
    assert manifest["bundle_version"] == "A7.1"

    schema = json.loads(
        Path("apps/spec/manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(manifest, schema)

    device = torch.device("cpu")
    encoder, tokenizer = loadFrozenTextEncoder(_CLIP_ARTIFACT, device)
    prompt = "a person dances"
    expected = encodeTextToPooled([prompt], encoder, tokenizer, device)
    encoded = tokenizer.encode([prompt])
    session = onnxruntime.InferenceSession(
        str(bundleDir / TEXT_ENCODER_FILENAME)
    )
    (actual,) = session.run(
        None,
        {
            "input_ids": encoded.inputIds.numpy(),
            "attention_mask": encoded.attentionMask.numpy(),
        },
    )
    maxDiff = float(
        (torch.from_numpy(actual) - expected).abs().max()
    )
    assert maxDiff < ONNX_PARITY_TOLERANCE
    assert section["tokenizer"]["max_length"] == (
        tokenizer.config.maxLength
    )
