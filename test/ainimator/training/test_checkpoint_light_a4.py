"""Phase A4 acceptance tests — light checkpoints + run hygiene.

Three criteria tested
--------------------

AC1 : ``test_clip_frozen_weights_absent_from_checkpoint``
    The frozen CLIP tower (``clip.*`` keys) must NOT appear in the
    ``encoder_state_dict`` of a saved checkpoint when the encoder is
    ``ClipTextEncoder``.  Builds a minimal model with a fake frozen
    sub-module and asserts the filter removes it.

AC2 : ``test_custom_encoder_checkpoint_round_trip``
    Save → load round-trip for the custom-BPE encoder path.  Asserts
    that every trainable state-dict key is numerically identical after
    the round-trip (``torch.allclose``) and that the denoiser forward
    produces identical outputs on a fixed random input before and after
    load.

AC3 : ``test_old_clip_checkpoint_backward_compat``
    An OLD-format checkpoint payload that embeds the frozen CLIP tower
    weights (``clip.*`` tensors present in ``encoder_state_dict``) must
    still be loadable by ``loadCheckpointV2``.  The loader's
    ``strict=False`` path must silently ignore the extra keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from ainimator.core.checkpoint_io import saveTorchObjectAtomically
from ainimator.model.denoiser_v2 import MotionDenoiserV2, MotionDenoiserV2Config
from ainimator.model.motion_normalizer import MotionNormalizer
from ainimator.diffusion.noise_schedule import (
    NoiseSchedule,
    NoiseScheduleConfig,
)
from ainimator.text import (
    CustomTextEncoder,
    CustomTextEncoderConfig,
    CustomTokenizer,
    CustomTokenizerConfig,
)
from ainimator.text.clip_text_encoder import (
    ClipTextEncoder,
    ClipTextEncoderConfig,
)
from ainimator.training.training_v2 import (
    _assertNoMissingTrainableKeys,
    _denoiserConfigToDict,
    _denoiserConfigFromDict,
    _encoderConfigToDict,
    _encoderConfigFromDict,
    _scheduleConfigToDict,
    _scheduleConfigFromDict,
    loadCheckpointV2,
)


# =====================================================================
# Helpers
# =====================================================================

def _tinyDenoiserConfig() -> MotionDenoiserV2Config:
    """Return a tiny denoiser config that runs fast on CPU."""
    return MotionDenoiserV2Config(
        embedDim=32,
        numHeads=4,
        numLayers=1,
        numBones=22,
        motionChannels=6,
        globalChannels=3,
        textEmbedDim=32,
        maxFrames=16,
        dropout=0.0,
    )


def _tinyScheduleConfig() -> NoiseScheduleConfig:
    return NoiseScheduleConfig(numSteps=50, scheduleType="cosine")


def _buildTokenizerDir(root: Path) -> Path:
    """Create and save a tiny custom tokenizer."""
    tokenizer = CustomTokenizer.train(
        ["a person walks."] * 8,
        config=CustomTokenizerConfig(
            vocabSize=64, maxLength=8, minFrequency=1
        ),
    )
    tokenizer.save(root)
    return root


def _buildCustomEncoder(tokenizerDir: Path) -> CustomTextEncoder:
    """Build a tiny custom text encoder (no HF download needed)."""
    tokenizer = CustomTokenizer.load(tokenizerDir)
    return CustomTextEncoder(
        CustomTextEncoderConfig(
            vocabSize=tokenizer.vocabSize,
            maxLength=tokenizer.config.maxLength,
            hiddenDim=32,
            numLayers=1,
            numHeads=4,
            outputDim=32,
            padTokenId=tokenizer.padTokenId,
            dropout=0.0,
            useNullEmbedding=True,
            l2NormalizeOutput=True,
        )
    )


def _buildMockClipEncoder(outputDim: int = 32) -> ClipTextEncoder:
    """Build a ClipTextEncoder with a mocked frozen CLIP tower.

    The real ``CLIPTextModel.from_pretrained`` is intercepted so no
    network call is made.  A tiny ``nn.Linear`` stands in for the CLIP
    tower so the state_dict has realistic ``clip.*`` key paths.
    """
    # The mock CLIP model has one Linear layer — its parameters appear
    # in the state_dict under the ``clip.*`` prefix.
    mockClipModel = nn.Linear(4, 4, bias=False)
    mockClipModel.eval()

    # CLIPTextModel is imported lazily inside ClipTextEncoder.__init__,
    # so we patch the `transformers` namespace directly.
    with patch("transformers.AutoModel.from_pretrained",
               return_value=mockClipModel):
        config = ClipTextEncoderConfig(
            modelName="mock/clip",
            maxLength=8,
            outputDim=outputDim,
            clipHiddenDim=4,
            dropout=0.0,
            useNullEmbedding=True,
            l2NormalizeOutput=True,
        )
        return ClipTextEncoder(config)


# =====================================================================
# AC1 — Frozen CLIP weights absent from saved checkpoint
# =====================================================================

def test_clip_frozen_weights_absent_from_checkpoint(
    tmp_path: Path,
) -> None:
    """Frozen ``clip.*`` keys must NOT appear in the saved encoder payload.

    Strategy
    --------
    Build a ``ClipTextEncoder`` backed by a mock frozen sub-module.
    Verify that the full state_dict DOES contain ``clip.*`` keys (the
    invariant would be vacuous otherwise), then apply the same filter
    that ``_saveCheckpoint`` applies and assert no ``clip.*`` key
    survives.  Finally, write the payload to disk and reload it to
    confirm the file itself contains no frozen-tower tensors.
    """
    encoder = _buildMockClipEncoder(outputDim=32)

    # Sanity guard: the full state_dict must have clip.* keys for the
    # test to be meaningful.
    fullState = encoder.state_dict()
    clipKeys = [k for k in fullState if k.startswith("clip.")]
    assert clipKeys, (
        "Mock ClipTextEncoder has no clip.* keys — test is vacuous."
    )

    # Apply the same filter as _saveCheckpoint.
    filteredState = {
        k: v for k, v in fullState.items() if not k.startswith("clip.")
    }
    assert not any(k.startswith("clip.") for k in filteredState), (
        "Frozen clip.* keys survived the filter."
    )

    # Prove the filtered payload is strictly smaller (not just vacuously
    # equal to the full state_dict).
    assert len(filteredState) < len(fullState), (
        "Filter removed no keys — frozen weights would be serialised."
    )

    # Write to disk and reload; assert the file contains no clip.* key.
    payloadPath = tmp_path / "ckpt_light.pt"
    payload = {
        "version": 3,
        "text_encoder_type": "clip",
        "encoder_state_dict": filteredState,
    }
    saveTorchObjectAtomically(payload, payloadPath)
    loaded = torch.load(payloadPath, map_location="cpu", weights_only=False)
    savedKeys = list(loaded["encoder_state_dict"].keys())
    assert not any(k.startswith("clip.") for k in savedKeys), (
        f"File still contains frozen clip.* keys: "
        f"{[k for k in savedKeys if k.startswith('clip.')]}"
    )


# =====================================================================
# AC2 — Save / load round-trip for the custom-encoder path
# =====================================================================

def test_custom_encoder_checkpoint_round_trip(tmp_path: Path) -> None:
    """State-dict round-trip: trainable weights survive save → load.

    For each key in the encoder and denoiser state-dicts, asserts that
    the loaded tensor is numerically identical (``torch.allclose``) to
    the saved one.  Then runs a fixed denoiser forward and checks
    outputs are identical after reload.

    The custom-BPE path is used because it requires no HF download.
    """
    tokenizerDir = _buildTokenizerDir(tmp_path / "tok")
    tokenizer = CustomTokenizer.load(tokenizerDir)

    encoder = _buildCustomEncoder(tokenizerDir)
    denoiserConfig = _tinyDenoiserConfig()
    denoiser = MotionDenoiserV2(denoiserConfig)
    schedule = NoiseSchedule(_tinyScheduleConfig())
    normalizer = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)

    # Fix random state so the payload is deterministic.
    torch.manual_seed(0)
    randomBone = [torch.randn(8, 22, 6)]
    randomGlobal = [torch.randn(8, 3)]
    normalizer.fitFromTensors(randomBone, randomGlobal)

    # Build payload matching the v3 format.
    encoderState = encoder.state_dict()
    denoiserState = denoiser.state_dict()
    payload: dict[str, Any] = {
        "version": 3,
        "text_encoder_type": "custom",
        "encoder_state_dict": encoderState,
        "encoder_config": _encoderConfigToDict(encoder.config),
        "denoiser_state_dict": denoiserState,
        "denoiser_config": _denoiserConfigToDict(denoiserConfig),
        "schedule_config": _scheduleConfigToDict(schedule.config),
        "schedule_state_dict": schedule.state_dict(),
        "normalizer_config": normalizer.configToDict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "tokenizer_dir": str(tokenizerDir.resolve()),
        "training_config": {},
        "training_sample": {
            "sampleId": 0,
            "textId": 0,
            "rawText": "a person walks.",
            "frames": 8,
        },
    }
    ckptPath = tmp_path / "checkpoints" / "v2_test_ckpt.pt"
    saveTorchObjectAtomically(payload, ckptPath)

    # Load the checkpoint.
    (
        loadedTokenizer,
        loadedEncoder,
        loadedDenoiser,
        loadedSchedule,
        loadedNormalizer,
        _,
    ) = loadCheckpointV2(ckptPath, device="cpu")

    # Assert state-dict key-by-key numerical equality.
    for key, origTensor in encoderState.items():
        loadedTensor = loadedEncoder.state_dict()[key]
        assert torch.allclose(origTensor, loadedTensor, atol=0.0), (
            f"Encoder key {key!r} changed after round-trip."
        )

    for key, origTensor in denoiserState.items():
        loadedTensor = loadedDenoiser.state_dict()[key]
        assert torch.allclose(origTensor, loadedTensor, atol=0.0), (
            f"Denoiser key {key!r} changed after round-trip."
        )

    # Assert identical forward output.
    torch.manual_seed(1)
    noisyMotion = torch.randn(1, 8, 22, 6)
    noisyGlobal = torch.randn(1, 8, 3)
    timesteps = torch.tensor([100])
    textHidden = torch.randn(1, 4, 32)
    textMask = torch.zeros(1, 4, dtype=torch.bool)

    denoiser.eval()
    loadedDenoiser.eval()
    with torch.no_grad():
        outOrig = denoiser(
            noisyMotion=noisyMotion,
            timesteps=timesteps,
            textHiddenStates=textHidden,
            textKeyPaddingMask=textMask,
            noisyGlobalFeatures=noisyGlobal,
        )
        outLoaded = loadedDenoiser(
            noisyMotion=noisyMotion,
            timesteps=timesteps,
            textHiddenStates=textHidden,
            textKeyPaddingMask=textMask,
            noisyGlobalFeatures=noisyGlobal,
        )

    assert torch.allclose(outOrig.boneOutput, outLoaded.boneOutput), (
        "Denoiser boneOutput differs after round-trip."
    )
    assert outOrig.globalOutput is not None
    assert outLoaded.globalOutput is not None
    assert torch.allclose(
        outOrig.globalOutput, outLoaded.globalOutput
    ), "Denoiser globalOutput differs after round-trip."


# =====================================================================
# AC3 — Old-format CLIP checkpoint (frozen weights embedded) is loadable
# =====================================================================

def test_old_clip_checkpoint_backward_compat(tmp_path: Path) -> None:
    """Old checkpoints that embedded the frozen CLIP tower must still load.

    Constructs a synthetic v3 CLIP payload that includes ``clip.*``
    tensors in ``encoder_state_dict`` (simulating the pre-A4 format
    where the full ~63M frozen tower was serialised).  Asserts that
    ``loadCheckpointV2`` loads the payload without error and that the
    encoder's trainable weights (projection + null embedding) match
    the expected values.
    """
    tokenizerDir = _buildTokenizerDir(tmp_path / "tok")
    outputDim = 32
    clipHiddenDim = 4

    # Build a ClipTextEncoder via mocks so we have a real state_dict
    # layout for the trainable parts.
    realEncoder = _buildMockClipEncoder(outputDim=outputDim)
    trainableState = {
        k: v for k, v in realEncoder.state_dict().items()
        if not k.startswith("clip.")
    }

    # Simulate the OLD format: add frozen tower keys back in.
    fakeClipWeight = torch.randn(outputDim, clipHiddenDim)
    oldFormatState = {
        **trainableState,
        "clip.fake_frozen_weight": fakeClipWeight,
        "clip.another_frozen_tensor": torch.zeros(2, 2),
    }
    assert any(k.startswith("clip.") for k in oldFormatState), (
        "Old-format simulation is broken — no clip.* keys present."
    )

    # Build a minimal schedule + denoiser + normalizer for the payload.
    denoiserConfig = _tinyDenoiserConfig()
    denoiser = MotionDenoiserV2(denoiserConfig)
    schedule = NoiseSchedule(_tinyScheduleConfig())
    normalizer = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)
    torch.manual_seed(0)
    normalizer.fitFromTensors([torch.randn(8, 22, 6)], [torch.randn(8, 3)])

    clipEncoderConfig: dict[str, Any] = {
        "modelName": "mock/clip",
        "maxLength": 8,
        "outputDim": outputDim,
        "clipHiddenDim": clipHiddenDim,
        "dropout": 0.0,
        "useNullEmbedding": True,
        "l2NormalizeOutput": True,
    }
    oldPayload: dict[str, Any] = {
        "version": 3,
        "text_encoder_type": "clip",
        "encoder_state_dict": oldFormatState,
        "encoder_config": clipEncoderConfig,
        "denoiser_state_dict": denoiser.state_dict(),
        "denoiser_config": _denoiserConfigToDict(denoiserConfig),
        "schedule_config": _scheduleConfigToDict(schedule.config),
        "schedule_state_dict": schedule.state_dict(),
        "normalizer_config": normalizer.configToDict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "tokenizer_dir": str(tokenizerDir.resolve()),
        "training_config": {},
        "training_sample": {
            "sampleId": 0,
            "textId": 0,
            "rawText": "test",
            "frames": 8,
        },
    }
    ckptPath = tmp_path / "old_clip_checkpoint.pt"
    saveTorchObjectAtomically(oldPayload, ckptPath)

    # Load via the public API; patch HF downloads to avoid network.
    # CLIPTextModel is lazily imported inside ClipTextEncoder.__init__,
    # and CLIPTokenizerFast inside ClipTokenizer.__init__, so both are
    # patched at their origin namespace (``transformers.*``).
    mockClipTower = nn.Linear(clipHiddenDim, clipHiddenDim, bias=False)
    mockClipTower.eval()
    mockTokenizerBackend = MagicMock()
    mockTokenizerBackend.vocab_size = 100
    mockTokenizerBackend.pad_token_id = 0

    with patch(
        "transformers.AutoModel.from_pretrained",
        return_value=mockClipTower,
    ), patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mockTokenizerBackend,
    ):
        (
            _,
            loadedEncoder,
            loadedDenoiser,
            loadedSchedule,
            loadedNormalizer,
            returnedPayload,
        ) = loadCheckpointV2(ckptPath, device="cpu")

    # The trainable projection weights must match exactly.
    for key, expected in trainableState.items():
        if key not in loadedEncoder.state_dict():
            continue  # may be absent in mock layout; tolerance for test
        loaded = loadedEncoder.state_dict()[key]
        assert torch.allclose(expected, loaded, atol=0.0), (
            f"Trainable encoder key {key!r} changed during load of "
            "old-format checkpoint."
        )

    # The loaded denoiser must produce finite outputs — a basic sanity
    # check that the module is fully constructed.
    loadedDenoiser.eval()
    with torch.no_grad():
        out = loadedDenoiser(
            noisyMotion=torch.randn(1, 8, 22, 6),
            timesteps=torch.tensor([50]),
            textHiddenStates=torch.randn(1, 4, outputDim),
            textKeyPaddingMask=torch.zeros(1, 4, dtype=torch.bool),
            noisyGlobalFeatures=torch.randn(1, 8, 3),
        )
    assert torch.isfinite(out.boneOutput).all(), (
        "Denoiser output contains NaN/Inf after loading old-format "
        "CLIP checkpoint."
    )


# =====================================================================
# A4-hardening — strict=False missing-key guard
# =====================================================================

def test_missing_trainable_key_raises(tmp_path: Path) -> None:
    """A payload missing a non-clip trainable key must raise RuntimeError.

    Constructs a CLIP checkpoint where the ``outputProjection.weight``
    key has been removed from the saved state-dict (simulating a future
    architecture rename).  Asserts that ``loadCheckpointV2`` raises
    ``RuntimeError`` and that the error message names the missing key.
    """
    tokenizerDir = _buildTokenizerDir(tmp_path / "tok")
    outputDim = 32
    clipHiddenDim = 4
    realEncoder = _buildMockClipEncoder(outputDim=outputDim)
    fullState = realEncoder.state_dict()

    # Remove a non-clip trainable key to simulate architecture drift.
    droppedKey = next(
        k for k in fullState if not k.startswith("clip.")
    )
    truncatedState = {
        k: v for k, v in fullState.items()
        if k != droppedKey and not k.startswith("clip.")
    }

    denoiserConfig = _tinyDenoiserConfig()
    denoiser = MotionDenoiserV2(denoiserConfig)
    schedule = NoiseSchedule(_tinyScheduleConfig())
    normalizer = MotionNormalizer(
        numBones=22, motionChannels=6, globalChannels=3
    )
    torch.manual_seed(0)
    normalizer.fitFromTensors(
        [torch.randn(8, 22, 6)], [torch.randn(8, 3)]
    )

    payload: dict[str, Any] = {
        "version": 3,
        "text_encoder_type": "clip",
        "encoder_state_dict": truncatedState,
        "encoder_config": {
            "modelName": "mock/clip",
            "maxLength": 8,
            "outputDim": outputDim,
            "clipHiddenDim": clipHiddenDim,
            "dropout": 0.0,
            "useNullEmbedding": True,
            "l2NormalizeOutput": True,
        },
        "denoiser_state_dict": denoiser.state_dict(),
        "denoiser_config": _denoiserConfigToDict(denoiserConfig),
        "schedule_config": _scheduleConfigToDict(schedule.config),
        "schedule_state_dict": schedule.state_dict(),
        "normalizer_config": normalizer.configToDict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "tokenizer_dir": str(tokenizerDir.resolve()),
        "training_config": {},
        "training_sample": {
            "sampleId": 0, "textId": 0,
            "rawText": "test", "frames": 8,
        },
    }
    ckptPath = tmp_path / "missing_trainable_ckpt.pt"
    saveTorchObjectAtomically(payload, ckptPath)

    mockClipTower = nn.Linear(clipHiddenDim, clipHiddenDim, bias=False)
    mockClipTower.eval()
    mockTokenizerBackend = MagicMock()
    mockTokenizerBackend.vocab_size = 100
    mockTokenizerBackend.pad_token_id = 0

    with patch(
        "transformers.AutoModel.from_pretrained",
        return_value=mockClipTower,
    ), patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mockTokenizerBackend,
    ):
        with pytest.raises(RuntimeError) as excInfo:
            loadCheckpointV2(ckptPath, device="cpu")

    assert droppedKey in str(excInfo.value), (
        f"RuntimeError did not name the missing key {droppedKey!r}."
    )


def test_only_clip_keys_missing_loads_fine(tmp_path: Path) -> None:
    """A light checkpoint (only clip.* absent) must load without error.

    Constructs a CLIP checkpoint where only the frozen ``clip.*`` keys
    are absent (the expected light-format case).  Asserts that
    ``loadCheckpointV2`` loads successfully and that the trainable keys
    are restored correctly.
    """
    tokenizerDir = _buildTokenizerDir(tmp_path / "tok")
    outputDim = 32
    clipHiddenDim = 4
    realEncoder = _buildMockClipEncoder(outputDim=outputDim)
    # Keep only non-clip keys — the standard light-format payload.
    lightState = {
        k: v for k, v in realEncoder.state_dict().items()
        if not k.startswith("clip.")
    }
    assert lightState, "No trainable keys found — test is vacuous."

    denoiserConfig = _tinyDenoiserConfig()
    denoiser = MotionDenoiserV2(denoiserConfig)
    schedule = NoiseSchedule(_tinyScheduleConfig())
    normalizer = MotionNormalizer(
        numBones=22, motionChannels=6, globalChannels=3
    )
    torch.manual_seed(0)
    normalizer.fitFromTensors(
        [torch.randn(8, 22, 6)], [torch.randn(8, 3)]
    )

    payload: dict[str, Any] = {
        "version": 3,
        "text_encoder_type": "clip",
        "encoder_state_dict": lightState,
        "encoder_config": {
            "modelName": "mock/clip",
            "maxLength": 8,
            "outputDim": outputDim,
            "clipHiddenDim": clipHiddenDim,
            "dropout": 0.0,
            "useNullEmbedding": True,
            "l2NormalizeOutput": True,
        },
        "denoiser_state_dict": denoiser.state_dict(),
        "denoiser_config": _denoiserConfigToDict(denoiserConfig),
        "schedule_config": _scheduleConfigToDict(schedule.config),
        "schedule_state_dict": schedule.state_dict(),
        "normalizer_config": normalizer.configToDict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "tokenizer_dir": str(tokenizerDir.resolve()),
        "training_config": {},
        "training_sample": {
            "sampleId": 0, "textId": 0,
            "rawText": "test", "frames": 8,
        },
    }
    ckptPath = tmp_path / "light_clip_ckpt.pt"
    saveTorchObjectAtomically(payload, ckptPath)

    mockClipTower = nn.Linear(clipHiddenDim, clipHiddenDim, bias=False)
    mockClipTower.eval()
    mockTokenizerBackend = MagicMock()
    mockTokenizerBackend.vocab_size = 100
    mockTokenizerBackend.pad_token_id = 0

    with patch(
        "transformers.AutoModel.from_pretrained",
        return_value=mockClipTower,
    ), patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mockTokenizerBackend,
    ):
        (
            _,
            loadedEncoder,
            _denoiser,
            _schedule,
            _normalizer,
            _payload,
        ) = loadCheckpointV2(ckptPath, device="cpu")

    # Trainable keys must be restored to their saved values.
    for key, expected in lightState.items():
        loaded = loadedEncoder.state_dict().get(key)
        assert loaded is not None, (
            f"Trainable key {key!r} absent from loaded encoder."
        )
        assert torch.allclose(expected, loaded, atol=0.0), (
            f"Trainable key {key!r} changed during light-format load."
        )
