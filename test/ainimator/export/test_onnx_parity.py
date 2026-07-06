"""ONNX parity tests for the encoder and denoiser (Phase A8).

These tests export both models to ONNX, run inference with ONNXRuntime
(CPU) and PyTorch on IDENTICAL inputs, and assert that the outputs match
within :data:`TOLERANCE`.

The tests are marked with the ``onnx`` pytest mark so they can be
filtered independently, but they run in the default suite (no
``--run-slow`` gate) because they complete in seconds on CPU.

CI contract (§2.10): these tests must stay green.  Any future PR that
breaks ONNX export is refused.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ainimator.model.denoiser_v2 import (
    MotionDenoiserV2,
    MotionDenoiserV2Config,
)
from ainimator.text.custom_text_encoder import (
    CustomTextEncoder,
    CustomTextEncoderConfig,
)
from ainimator.export.onnx import (
    exportEncoder,
    exportDenoiser,
    ONNX_PARITY_TOLERANCE,
)

# ---------------------------------------------------------------------------
# pytest mark — lets CI run "pytest -m onnx" independently if desired
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.onnx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_EMBED_DIM = 64
_TEXT_DIM = 64
_NUM_LAYERS = 2
_NUM_HEADS = 4
_NUM_BONES = 22
_MOTION_CH = 6
_GLOBAL_CH = 3
_FRAMES = 8
_TEXT_LEN = 12
_BATCH = 2
_VOCAB_SIZE = 200


def _makeEncoder() -> CustomTextEncoder:
    """Build a minimal encoder for tests."""
    config = CustomTextEncoderConfig(
        vocabSize=_VOCAB_SIZE,
        maxLength=_TEXT_LEN,
        hiddenDim=_EMBED_DIM,
        numLayers=_NUM_LAYERS,
        numHeads=_NUM_HEADS,
        dropout=0.0,
        outputDim=_TEXT_DIM,
        useNullEmbedding=True,
        l2NormalizeOutput=True,
    )
    encoder = CustomTextEncoder(config)
    encoder.eval()
    return encoder


def _makeDenoiser() -> MotionDenoiserV2:
    """Build a minimal denoiser for tests."""
    config = MotionDenoiserV2Config(
        embedDim=_EMBED_DIM,
        numHeads=_NUM_HEADS,
        numLayers=_NUM_LAYERS,
        numBones=_NUM_BONES,
        motionChannels=_MOTION_CH,
        globalChannels=_GLOBAL_CH,
        textEmbedDim=_TEXT_DIM,
        maxFrames=32,
        dropout=0.0,
        useFilmConditioning=False,
        usePerBlockFilm=False,
    )
    denoiser = MotionDenoiserV2(config)
    denoiser.eval()
    return denoiser


def _ortSession(onnxPath: Path) -> "onnxruntime.InferenceSession":
    """Create an ONNXRuntime CPU session."""
    import onnxruntime

    return onnxruntime.InferenceSession(
        str(onnxPath),
        providers=["CPUExecutionProvider"],
    )


def _maxAbsDiff(
    torchOut: torch.Tensor,
    ortOut: np.ndarray,
) -> float:
    """Return the max absolute difference between torch and ORT outputs."""
    npTorch = torchOut.detach().numpy()
    return float(np.abs(npTorch - ortOut).max())


# ---------------------------------------------------------------------------
# Encoder parity
# ---------------------------------------------------------------------------
class TestEncoderOnnxParity:
    """Export encoder → run ORT → compare to PyTorch."""

    def test_encoder_export_produces_valid_onnx(self) -> None:
        """onnx.checker must pass on the exported encoder graph."""
        import onnx

        encoder = _makeEncoder()
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "encoder.onnx"
            exportEncoder(
                encoder=encoder,
                outputPath=outPath,
                batchSize=_BATCH,
                textLen=_TEXT_LEN,
            )
            assert outPath.exists(), "Export did not produce a file."
            model = onnx.load(str(outPath))
            onnx.checker.check_model(model)  # raises on invalid graph

    def test_encoder_ort_vs_torch_hidden_states(self) -> None:
        """ORT hidden-states must match PyTorch within tolerance."""
        encoder = _makeEncoder()
        inputIds = torch.zeros(_BATCH, _TEXT_LEN, dtype=torch.long)
        attMask = torch.ones(_BATCH, _TEXT_LEN, dtype=torch.float32)

        with torch.no_grad():
            torchOut = encoder(inputIds, attMask)
        torchHidden = torchOut.hiddenStates  # (B, T, D)

        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "encoder.onnx"
            exportEncoder(
                encoder=encoder,
                outputPath=outPath,
                batchSize=_BATCH,
                textLen=_TEXT_LEN,
            )
            session = _ortSession(outPath)
            feeds = {
                "inputIds": inputIds.numpy(),
                "attentionMask": attMask.numpy(),
            }
            ortHidden, _ortMask = session.run(None, feeds)

        diff = _maxAbsDiff(torchHidden, ortHidden)
        assert diff <= ONNX_PARITY_TOLERANCE, (
            f"Encoder hidden-states max abs diff = {diff:.6f} "
            f"exceeds tolerance {ONNX_PARITY_TOLERANCE}."
        )

    def test_encoder_ort_vs_torch_key_padding_mask(self) -> None:
        """ORT key-padding-mask must match PyTorch within tolerance."""
        encoder = _makeEncoder()
        inputIds = torch.zeros(_BATCH, _TEXT_LEN, dtype=torch.long)
        attMask = torch.ones(_BATCH, _TEXT_LEN, dtype=torch.float32)

        with torch.no_grad():
            torchOut = encoder(inputIds, attMask)
        torchMask = torchOut.keyPaddingMask.to(torch.float32)

        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "encoder.onnx"
            exportEncoder(
                encoder=encoder,
                outputPath=outPath,
                batchSize=_BATCH,
                textLen=_TEXT_LEN,
            )
            session = _ortSession(outPath)
            feeds = {
                "inputIds": inputIds.numpy(),
                "attentionMask": attMask.numpy(),
            }
            _ortHidden, ortMask = session.run(None, feeds)

        diff = _maxAbsDiff(torchMask, ortMask)
        assert diff <= ONNX_PARITY_TOLERANCE, (
            f"Encoder mask max abs diff = {diff:.6f} "
            f"exceeds tolerance {ONNX_PARITY_TOLERANCE}."
        )

    def test_encoder_dynamic_axes_different_batch(self) -> None:
        """Exported encoder runs with a different batch size."""
        encoder = _makeEncoder()
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "encoder.onnx"
            exportEncoder(
                encoder=encoder,
                outputPath=outPath,
                batchSize=_BATCH,
                textLen=_TEXT_LEN,
            )
            session = _ortSession(outPath)
            newBatch = 3
            inputIds = torch.zeros(newBatch, _TEXT_LEN, dtype=torch.long)
            attMask = torch.ones(newBatch, _TEXT_LEN, dtype=torch.float32)
            feeds = {
                "inputIds": inputIds.numpy(),
                "attentionMask": attMask.numpy(),
            }
            ortHidden, _ = session.run(None, feeds)
            assert ortHidden.shape == (newBatch, _TEXT_LEN, _TEXT_DIM)

    def test_encoder_dynamic_axes_different_text_len(self) -> None:
        """Exported encoder runs with a different text length."""
        encoder = _makeEncoder()
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "encoder.onnx"
            exportEncoder(
                encoder=encoder,
                outputPath=outPath,
                batchSize=_BATCH,
                textLen=_TEXT_LEN,
            )
            session = _ortSession(outPath)
            newLen = 6
            inputIds = torch.zeros(_BATCH, newLen, dtype=torch.long)
            attMask = torch.ones(_BATCH, newLen, dtype=torch.float32)
            feeds = {
                "inputIds": inputIds.numpy(),
                "attentionMask": attMask.numpy(),
            }
            ortHidden, _ = session.run(None, feeds)
            assert ortHidden.shape == (_BATCH, newLen, _TEXT_DIM)


    def test_encoder_partial_mask_ort_vs_torch_hidden_states(
        self,
    ) -> None:
        """ORT must match PyTorch under a REAL partial padding mask.

        Uses B=3 rows with DIFFERENT numbers of padded positions so a
        row-uniform shortcut cannot accidentally pass.  This exercises the
        ``_OnnxMHAReplace`` additive-bias path (True→-inf) which is the
        subtlest part of the ONNX export and was untested for the
        non-trivial masked case.

        Mask design (attentionMask, 1.0=real 0.0=pad):
        * row 0 : positions 0-7 real, 8-11 padded  (4 padded)
        * row 1 : positions 0-4 real, 5-11 padded  (7 padded)
        * row 2 : all 12 positions real             (0 padded)

        inputIds at padded positions are valid token ids so only the mask
        drives the masking (not the embedding table lookup).
        """
        _MASK_BATCH = 3
        _PAD_ROW0 = 4   # last 4 positions padded in row 0
        _PAD_ROW1 = 7   # last 7 positions padded in row 1
        _REAL_LEN_0 = _TEXT_LEN - _PAD_ROW0   # 8
        _REAL_LEN_1 = _TEXT_LEN - _PAD_ROW1   # 5

        encoder = _makeEncoder()

        inputIds = torch.ones(
            _MASK_BATCH, _TEXT_LEN, dtype=torch.long
        )

        attMask = _buildPartialMask(
            batchSize=_MASK_BATCH,
            textLen=_TEXT_LEN,
            realLengths=[_REAL_LEN_0, _REAL_LEN_1, _TEXT_LEN],
        )

        with torch.no_grad():
            torchOut = encoder(inputIds, attMask)
        torchHidden = torchOut.hiddenStates  # (B, T, D)

        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "encoder_partial.onnx"
            exportEncoder(
                encoder=encoder,
                outputPath=outPath,
                batchSize=_MASK_BATCH,
                textLen=_TEXT_LEN,
            )
            session = _ortSession(outPath)
            feeds = {
                "inputIds": inputIds.numpy(),
                "attentionMask": attMask.numpy(),
            }
            ortHidden, _ortMask = session.run(None, feeds)

        diff = _maxAbsDiff(torchHidden, ortHidden)
        assert diff <= ONNX_PARITY_TOLERANCE, (
            f"Encoder hidden-states max abs diff under partial masking "
            f"= {diff:.6f} exceeds tolerance {ONNX_PARITY_TOLERANCE}.  "
            f"This likely indicates a mask polarity regression in "
            f"_OnnxMHAReplace."
        )


def _buildPartialMask(
    batchSize: int,
    textLen: int,
    realLengths: list[int],
) -> torch.Tensor:
    """Build an attention mask with row-varying real token counts.

    Parameters
    ----------
    batchSize : int
        Number of rows.  Must equal ``len(realLengths)``.
    textLen : int
        Total sequence length.
    realLengths : list[int]
        Number of real (non-padding) tokens in each row.  Remaining
        positions are set to 0.0 (padding).

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(batchSize, textLen)`` with 1.0 on
        real positions and 0.0 on padded positions.
    """
    if len(realLengths) != batchSize:
        raise ValueError(
            f"realLengths length {len(realLengths)} != "
            f"batchSize {batchSize}."
        )
    mask = torch.zeros(batchSize, textLen, dtype=torch.float32)
    for rowIdx, realLen in enumerate(realLengths):
        mask[rowIdx, :realLen] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Denoiser step parity
# ---------------------------------------------------------------------------
class TestDenoiserOnnxParity:
    """Export denoiser-step → run ORT → compare to PyTorch."""

    def _makeInputs(self) -> dict[str, torch.Tensor]:
        """Build concrete example inputs for one denoiser step."""
        return {
            "noisyMotion": torch.randn(
                _BATCH, _FRAMES, _NUM_BONES, _MOTION_CH
            ),
            "noisyGlobal": torch.randn(_BATCH, _FRAMES, _GLOBAL_CH),
            "timesteps": torch.zeros(_BATCH, dtype=torch.long),
            "textHiddenStates": torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
            "textKeyPaddingMask": torch.zeros(
                _BATCH, _TEXT_LEN, dtype=torch.float32
            ),
            "motionKeyPaddingMask": torch.zeros(
                _BATCH, _FRAMES, dtype=torch.float32
            ),
        }

    def test_denoiser_export_produces_valid_onnx(self) -> None:
        """onnx.checker must pass on the exported denoiser graph."""
        import onnx

        denoiser = _makeDenoiser()
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "denoiser_step.onnx"
            exportDenoiser(
                denoiser=denoiser,
                outputPath=outPath,
                batchSize=_BATCH,
                frames=_FRAMES,
                textLen=_TEXT_LEN,
            )
            assert outPath.exists(), "Export did not produce a file."
            model = onnx.load(str(outPath))
            onnx.checker.check_model(model)

    def test_denoiser_ort_vs_torch_bone_output(self) -> None:
        """ORT boneOutput must match PyTorch within tolerance."""
        denoiser = _makeDenoiser()
        inputs = self._makeInputs()

        with torch.no_grad():
            torchResult = denoiser(
                noisyMotion=inputs["noisyMotion"],
                timesteps=inputs["timesteps"],
                textHiddenStates=inputs["textHiddenStates"],
                textKeyPaddingMask=inputs["textKeyPaddingMask"].bool(),
                noisyGlobalFeatures=inputs["noisyGlobal"],
                motionKeyPaddingMask=inputs["motionKeyPaddingMask"].bool(),
            )

        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "denoiser_step.onnx"
            exportDenoiser(
                denoiser=denoiser,
                outputPath=outPath,
                batchSize=_BATCH,
                frames=_FRAMES,
                textLen=_TEXT_LEN,
            )
            session = _ortSession(outPath)
            feeds = {
                key: inputs[key].numpy() for key in inputs
            }
            ortBone, _ortGlobal = session.run(None, feeds)

        diff = _maxAbsDiff(torchResult.boneOutput, ortBone)
        assert diff <= ONNX_PARITY_TOLERANCE, (
            f"Denoiser boneOutput max abs diff = {diff:.6f} "
            f"exceeds tolerance {ONNX_PARITY_TOLERANCE}."
        )

    def test_denoiser_ort_vs_torch_global_output(self) -> None:
        """ORT globalOutput must match PyTorch within tolerance."""
        denoiser = _makeDenoiser()
        inputs = self._makeInputs()

        with torch.no_grad():
            torchResult = denoiser(
                noisyMotion=inputs["noisyMotion"],
                timesteps=inputs["timesteps"],
                textHiddenStates=inputs["textHiddenStates"],
                textKeyPaddingMask=inputs["textKeyPaddingMask"].bool(),
                noisyGlobalFeatures=inputs["noisyGlobal"],
                motionKeyPaddingMask=inputs["motionKeyPaddingMask"].bool(),
            )

        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "denoiser_step.onnx"
            exportDenoiser(
                denoiser=denoiser,
                outputPath=outPath,
                batchSize=_BATCH,
                frames=_FRAMES,
                textLen=_TEXT_LEN,
            )
            session = _ortSession(outPath)
            feeds = {
                key: inputs[key].numpy() for key in inputs
            }
            _ortBone, ortGlobal = session.run(None, feeds)

        assert torchResult.globalOutput is not None
        diff = _maxAbsDiff(torchResult.globalOutput, ortGlobal)
        assert diff <= ONNX_PARITY_TOLERANCE, (
            f"Denoiser globalOutput max abs diff = {diff:.6f} "
            f"exceeds tolerance {ONNX_PARITY_TOLERANCE}."
        )

    def test_denoiser_dynamic_axes_different_batch(self) -> None:
        """Exported denoiser-step runs with a different batch size."""
        denoiser = _makeDenoiser()
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "denoiser_step.onnx"
            exportDenoiser(
                denoiser=denoiser,
                outputPath=outPath,
                batchSize=_BATCH,
                frames=_FRAMES,
                textLen=_TEXT_LEN,
            )
            session = _ortSession(outPath)
            newBatch = 3
            feeds = {
                "noisyMotion": torch.randn(
                    newBatch, _FRAMES, _NUM_BONES, _MOTION_CH
                ).numpy(),
                "noisyGlobal": torch.randn(
                    newBatch, _FRAMES, _GLOBAL_CH
                ).numpy(),
                "timesteps": np.zeros(newBatch, dtype=np.int64),
                "textHiddenStates": torch.randn(
                    newBatch, _TEXT_LEN, _TEXT_DIM
                ).numpy(),
                "textKeyPaddingMask": np.zeros(
                    (newBatch, _TEXT_LEN), dtype=np.float32
                ),
                "motionKeyPaddingMask": np.zeros(
                    (newBatch, _FRAMES), dtype=np.float32
                ),
            }
            ortBone, ortGlobal = session.run(None, feeds)
            assert ortBone.shape == (
                newBatch, _FRAMES, _NUM_BONES, _MOTION_CH
            )
            assert ortGlobal.shape == (newBatch, _FRAMES, _GLOBAL_CH)

    def test_denoiser_dynamic_axes_different_frames(self) -> None:
        """Exported denoiser-step runs with a different frame count."""
        denoiser = _makeDenoiser()
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "denoiser_step.onnx"
            exportDenoiser(
                denoiser=denoiser,
                outputPath=outPath,
                batchSize=_BATCH,
                frames=_FRAMES,
                textLen=_TEXT_LEN,
            )
            session = _ortSession(outPath)
            newFrames = 16
            feeds = {
                "noisyMotion": torch.randn(
                    _BATCH, newFrames, _NUM_BONES, _MOTION_CH
                ).numpy(),
                "noisyGlobal": torch.randn(
                    _BATCH, newFrames, _GLOBAL_CH
                ).numpy(),
                "timesteps": np.zeros(_BATCH, dtype=np.int64),
                "textHiddenStates": torch.randn(
                    _BATCH, _TEXT_LEN, _TEXT_DIM
                ).numpy(),
                "textKeyPaddingMask": np.zeros(
                    (_BATCH, _TEXT_LEN), dtype=np.float32
                ),
                "motionKeyPaddingMask": np.zeros(
                    (_BATCH, newFrames), dtype=np.float32
                ),
            }
            ortBone, ortGlobal = session.run(None, feeds)
            assert ortBone.shape == (
                _BATCH, newFrames, _NUM_BONES, _MOTION_CH
            )
            assert ortGlobal.shape == (_BATCH, newFrames, _GLOBAL_CH)


# ---------------------------------------------------------------------------
# Controller parity (Goal A, phase A5)
# ---------------------------------------------------------------------------
from ainimator.core.constants.controller import PhaseMode  # noqa: E402
from ainimator.core.types.controller import (  # noqa: E402
    ControllerV2Config,
)
from ainimator.model.controller_v2 import MotionController  # noqa: E402
from ainimator.export.onnx import exportController  # noqa: E402

_CONTEXT = 1


def _makeController(phaseMode: PhaseMode = PhaseMode.NONE) -> MotionController:
    """Build a minimal controller for export tests."""
    config = ControllerV2Config(
        embedDim=_EMBED_DIM,
        numHeads=_NUM_HEADS,
        numLayers=_NUM_LAYERS,
        numBones=_NUM_BONES,
        motionChannels=_MOTION_CH,
        globalChannels=_GLOBAL_CH,
        contextFrames=_CONTEXT,
        phaseMode=phaseMode,
        dropout=0.0,
    )
    model = MotionController(config)
    model.eval()
    return model


class TestControllerOnnxParity:
    """Export controller → run ORT → compare to PyTorch (CI §2.10)."""

    def test_controller_export_produces_valid_onnx(self) -> None:
        """onnx.checker must pass on the exported controller graph."""
        import onnx

        controller = _makeController()
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "controller.onnx"
            exportController(controller, outPath, batchSize=_BATCH)
            assert outPath.exists()
            onnx.checker.check_model(onnx.load(str(outPath)))

    def test_controller_ort_vs_torch(self) -> None:
        """ORT delta outputs must match PyTorch within tolerance."""
        controller = _makeController()
        bone = torch.randn(_BATCH, _CONTEXT, _NUM_BONES, _MOTION_CH)
        control = torch.randn(_BATCH, controller.config.controlChannels)
        glob = torch.randn(_BATCH, _CONTEXT, _GLOBAL_CH)
        with torch.no_grad():
            out = controller(bone, control, globalWindow=glob)

        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "controller.onnx"
            exportController(controller, outPath, batchSize=_BATCH)
            session = _ortSession(outPath)
            feeds = {
                "bone_window": bone.numpy(),
                "control": control.numpy(),
                "global_window": glob.numpy(),
            }
            ortBone, ortGlobal = session.run(None, feeds)

        assert _maxAbsDiff(out.boneDelta, ortBone) <= ONNX_PARITY_TOLERANCE
        assert out.globalDelta is not None
        assert (
            _maxAbsDiff(out.globalDelta, ortGlobal) <= ONNX_PARITY_TOLERANCE
        )

    def test_controller_dynamic_batch(self) -> None:
        """Exported controller runs with a different batch size."""
        controller = _makeController()
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "controller.onnx"
            exportController(controller, outPath, batchSize=_BATCH)
            session = _ortSession(outPath)
            newBatch = 5
            feeds = {
                "bone_window": torch.randn(
                    newBatch, _CONTEXT, _NUM_BONES, _MOTION_CH
                ).numpy(),
                "control": torch.randn(
                    newBatch, controller.config.controlChannels
                ).numpy(),
                "global_window": torch.randn(
                    newBatch, _CONTEXT, _GLOBAL_CH
                ).numpy(),
            }
            ortBone, ortGlobal = session.run(None, feeds)
            assert ortBone.shape == (newBatch, _NUM_BONES, _MOTION_CH)
            assert ortGlobal.shape == (newBatch, _GLOBAL_CH)

    def test_controller_with_explicit_phase_exports(self) -> None:
        """A phase-conditioned controller exports and matches PyTorch."""
        controller = _makeController(PhaseMode.EXPLICIT)
        bone = torch.randn(_BATCH, _CONTEXT, _NUM_BONES, _MOTION_CH)
        control = torch.randn(_BATCH, controller.config.controlChannels)
        glob = torch.randn(_BATCH, _CONTEXT, _GLOBAL_CH)
        phase = torch.randn(_BATCH, controller.config.phaseChannels)
        with torch.no_grad():
            out = controller(bone, control, globalWindow=glob, phase=phase)

        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "controller_phase.onnx"
            exportController(controller, outPath, batchSize=_BATCH)
            session = _ortSession(outPath)
            feeds = {
                "bone_window": bone.numpy(),
                "control": control.numpy(),
                "global_window": glob.numpy(),
                "phase": phase.numpy(),
            }
            ortBone, _ortGlobal = session.run(None, feeds)
        assert _maxAbsDiff(out.boneDelta, ortBone) <= ONNX_PARITY_TOLERANCE

    def test_controller_with_prompt_emb_exports(self) -> None:
        """A controller with promptEmbChannels exports and matches PyTorch."""
        _PROMPT_DIM = 32
        config = ControllerV2Config(
            embedDim=_EMBED_DIM,
            numHeads=_NUM_HEADS,
            numLayers=_NUM_LAYERS,
            numBones=_NUM_BONES,
            motionChannels=_MOTION_CH,
            globalChannels=_GLOBAL_CH,
            contextFrames=_CONTEXT,
            phaseMode=PhaseMode.NONE,
            dropout=0.0,
            promptEmbChannels=_PROMPT_DIM,
        )
        controller = MotionController(config)
        controller.eval()
        bone = torch.randn(_BATCH, _CONTEXT, _NUM_BONES, _MOTION_CH)
        control = torch.randn(_BATCH, controller.config.controlChannels)
        glob = torch.randn(_BATCH, _CONTEXT, _GLOBAL_CH)
        promptEmb = torch.randn(_BATCH, _PROMPT_DIM)
        with torch.no_grad():
            out = controller(
                bone, control, globalWindow=glob, promptEmb=promptEmb
            )
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "controller_prompt.onnx"
            exportController(controller, outPath, batchSize=_BATCH)
            session = _ortSession(outPath)
            feeds = {
                "bone_window": bone.numpy(),
                "control": control.numpy(),
                "global_window": glob.numpy(),
                "prompt_emb": promptEmb.numpy(),
            }
            ortBone, ortGlobal = session.run(None, feeds)
        assert _maxAbsDiff(out.boneDelta, ortBone) <= ONNX_PARITY_TOLERANCE
        assert out.globalDelta is not None
        assert (
            _maxAbsDiff(out.globalDelta, ortGlobal) <= ONNX_PARITY_TOLERANCE
        )

    def test_controller_without_prompt_emb_unaffected(self) -> None:
        """A controller without promptEmbChannels is unaffected (regression)."""
        controller = _makeController(PhaseMode.NONE)
        bone = torch.randn(_BATCH, _CONTEXT, _NUM_BONES, _MOTION_CH)
        control = torch.randn(_BATCH, controller.config.controlChannels)
        glob = torch.randn(_BATCH, _CONTEXT, _GLOBAL_CH)
        with torch.no_grad():
            out = controller(bone, control, globalWindow=glob)
        with tempfile.TemporaryDirectory() as tmpDir:
            outPath = Path(tmpDir) / "controller_no_prompt.onnx"
            exportController(controller, outPath, batchSize=_BATCH)
            session = _ortSession(outPath)
            feeds = {
                "bone_window": bone.numpy(),
                "control": control.numpy(),
                "global_window": glob.numpy(),
            }
            ortBone, ortGlobal = session.run(None, feeds)
        assert _maxAbsDiff(out.boneDelta, ortBone) <= ONNX_PARITY_TOLERANCE
        assert out.globalDelta is not None
        assert (
            _maxAbsDiff(out.globalDelta, ortGlobal) <= ONNX_PARITY_TOLERANCE
        )
