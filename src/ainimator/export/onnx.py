"""ONNX export utilities for AI-nimator v2 (Phase A8).

Exports the text encoder (full encode path) and the denoiser (one
denoising step).  The DDIM sampling loop lives in Python and is
**not** exported — the exported denoiser graph is one forward call.

Design constraints (§2.10)
--------------------------
* No data-dependent control flow inside ``forward()``.
* No ``.item()`` on the graph path.
* Dynamic axes declared for batch, frames, and text length.
* DDIM loop stays outside the exported graph.
* Masks are concrete float32 tensors (never None) so the traced graph
  always uses the mask-aware code path.

Both exported graphs are verified with ``onnx.checker.check_model``
before the file is written.  Use :func:`exportEncoder` and
:func:`exportDenoiser`.

Why the MHA replacement is needed
----------------------------------
PyTorch 2.9+ internally lowers ``nn.MultiheadAttention`` to
``aten::_native_multi_head_attention``, which has no symbolic in the
legacy TorchScript ONNX exporter.  The fix (:class:`_OnnxMHAReplace`)
is **behavior-preserving**: it extracts the same weights and re-runs
the identical math via ``F.scaled_dot_product_attention``.  The
replacement happens only on deep-copy wrappers used for export — the
original model is never mutated.  Parity with the original
``nn.MultiheadAttention`` is verified in the parity tests.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from ainimator.text.custom_text_encoder import (
    CustomTextEncoder,
    TextEncoderOutput,
)
from ainimator.model.denoiser_v2 import (
    DenoiserOutput,
    MotionDenoiserV2,
)
from ainimator.model.controller_v2 import MotionController

logger = logging.getLogger(__name__)

# Tolerance used in the parity test (max abs diff torch vs ORT).
ONNX_PARITY_TOLERANCE: float = 1e-3

# Default opset version for all exports.
ONNX_OPSET_VERSION: int = 17


# -----------------------------------------------------------------------
# ONNX-traceable MultiheadAttention replacement
# -----------------------------------------------------------------------
class _OnnxMHAReplace(nn.Module):
    """ONNX-traceable drop-in for ``nn.MultiheadAttention``.

    ``aten::_native_multi_head_attention`` (PyTorch 2.9+) has no ONNX
    symbolic in the legacy exporter.  This module extracts the same
    weight tensors and recomputes the identical attention via
    ``F.scaled_dot_product_attention`` which IS lowerable to ONNX ops.

    Numerically identical to the original for ``batch_first=True`` when
    the model is in ``eval()`` mode.  Only dropout is affected: this
    module disables dropout at trace time (``self.training=False``),
    which is correct for export.

    Parameters
    ----------
    mha : nn.MultiheadAttention
        Original module to replace.  Must have ``batch_first=True``.
    """

    def __init__(self, mha: nn.MultiheadAttention) -> None:
        super().__init__()
        if not mha.batch_first:
            raise ValueError(
                "_OnnxMHAReplace requires batch_first=True."
            )
        embed_dim = mha.embed_dim
        self._embedDim: int = embed_dim
        self._numHeads: int = mha.num_heads
        self._headDim: int = embed_dim // mha.num_heads
        self._dropoutP: float = float(mha.dropout)
        # Share the same parameter tensors (no copy needed — we are
        # operating on a deep-copied model so there are no aliasing issues
        # between the training model and the export wrapper).
        self.in_proj_weight = nn.Parameter(
            mha.in_proj_weight.data.clone()
        )
        self.in_proj_bias: nn.Parameter | None = (
            nn.Parameter(mha.in_proj_bias.data.clone())
            if mha.in_proj_bias is not None
            else None
        )
        self.out_proj_weight = nn.Parameter(
            mha.out_proj.weight.data.clone()
        )
        self.out_proj_bias: nn.Parameter | None = (
            nn.Parameter(mha.out_proj.bias.data.clone())
            if mha.out_proj.bias is not None
            else None
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """Run multi-head attention via SDPA.

        Parameters
        ----------
        query : torch.Tensor
            ``(B, Tq, E)`` float32.
        key : torch.Tensor
            ``(B, Tk, E)`` float32.
        value : torch.Tensor
            ``(B, Tk, E)`` float32.
        key_padding_mask : torch.Tensor, optional
            ``(B, Tk)`` bool — True means padding (will be -inf).
        need_weights : bool
            Ignored; kept for API compatibility.

        Returns
        -------
        tuple[torch.Tensor, None]
            Output ``(B, Tq, E)`` and ``None`` for the weight tensor
            (matching ``nn.MultiheadAttention`` return signature).
        """
        batchSize, seqQ, _ = query.shape
        _, seqK, _ = key.shape
        embedDim = self._embedDim
        numHeads = self._numHeads
        headDim = self._headDim

        biasQ = (
            self.in_proj_bias[:embedDim]
            if self.in_proj_bias is not None
            else None
        )
        biasK = (
            self.in_proj_bias[embedDim : 2 * embedDim]
            if self.in_proj_bias is not None
            else None
        )
        biasV = (
            self.in_proj_bias[2 * embedDim :]
            if self.in_proj_bias is not None
            else None
        )

        queryProj = F.linear(
            query, self.in_proj_weight[:embedDim], biasQ
        )
        keyProj = F.linear(
            key, self.in_proj_weight[embedDim : 2 * embedDim], biasK
        )
        valueProj = F.linear(
            value, self.in_proj_weight[2 * embedDim :], biasV
        )

        # Reshape for multi-head: (B, T, E) -> (B, H, T, D)
        queryProj = queryProj.view(
            batchSize, seqQ, numHeads, headDim
        ).transpose(1, 2)
        keyProj = keyProj.view(
            batchSize, seqK, numHeads, headDim
        ).transpose(1, 2)
        valueProj = valueProj.view(
            batchSize, seqK, numHeads, headDim
        ).transpose(1, 2)

        # Build additive attention bias from key_padding_mask.
        attnBias: torch.Tensor | None = None
        if key_padding_mask is not None:
            attnBias = torch.zeros(
                batchSize, 1, 1, seqK,
                dtype=query.dtype,
                device=query.device,
            )
            attnBias = attnBias.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        dropoutP = self._dropoutP if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            queryProj,
            keyProj,
            valueProj,
            attn_mask=attnBias,
            dropout_p=dropoutP,
        )
        # Merge heads: (B, H, Tq, D) -> (B, Tq, E)
        attended = attended.transpose(1, 2).reshape(
            batchSize, seqQ, embedDim
        )
        output = F.linear(
            attended, self.out_proj_weight, self.out_proj_bias
        )
        return output, None


def _replaceAllMHA(module: nn.Module) -> nn.Module:
    """Recursively replace every ``nn.MultiheadAttention`` in *module*.

    Operates **in-place** on the passed module (which should be a deep
    copy of the original model).  Returns the module for chaining.

    Parameters
    ----------
    module : nn.Module
        Root module to walk.  All ``nn.MultiheadAttention`` descendants
        are replaced with :class:`_OnnxMHAReplace`.

    Returns
    -------
    nn.Module
        The (mutated) input module.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, _OnnxMHAReplace(child))
        else:
            _replaceAllMHA(child)
    return module


# -----------------------------------------------------------------------
# Encoder wrapper
# -----------------------------------------------------------------------
class _EncoderWrapper(nn.Module):
    """Thin wrapper around CustomTextEncoder for ONNX tracing.

    The real ``forward`` returns a :class:`TextEncoderOutput` dataclass.
    ONNX export requires the graph to output plain tensors, so this
    wrapper unpacks it into ``(hiddenStates, keyPaddingMask)``.

    The mask output is cast to ``float32`` because bool tensors are
    less portable across ONNX runtimes; callers may cast back after
    inference.
    """

    def __init__(self, encoder: CustomTextEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        inputIds: torch.Tensor,
        attentionMask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens; return ``(hiddenStates, keyPaddingMask_float)``.

        Parameters
        ----------
        inputIds : torch.Tensor
            Long tensor ``(B, T)`` — token ids.
        attentionMask : torch.Tensor
            Float tensor ``(B, T)`` — 1.0 real, 0.0 padding.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``hiddenStates`` of shape ``(B, T, D)`` and
            ``keyPaddingMask`` cast to float32 ``(B, T)``.
        """
        out: TextEncoderOutput = self.encoder(inputIds, attentionMask)
        return out.hiddenStates, out.keyPaddingMask.to(torch.float32)


# -----------------------------------------------------------------------
# Denoiser wrapper — one step
# -----------------------------------------------------------------------
class _DenoiserStepWrapper(nn.Module):
    """Thin wrapper for exporting one denoiser step via ONNX.

    The full :class:`MotionDenoiserV2` signature has many Optional
    arguments.  For ONNX export we lock down one concrete signature that
    covers the default production path:

    * ``noisyMotion``       — bones, always present
    * ``noisyGlobal``       — root translation, always present
    * ``timesteps``         — long ``(B,)``
    * ``textHiddenStates``  — ``(B, T, D_text)``
    * ``textKeyPaddingMask``— float32 ``(B, T)`` (True=pad, cast to
                              bool internally to avoid ONNX bool issues)
    * ``motionKeyPaddingMask`` — float32 ``(B, F)`` same convention

    Self-conditioning inputs are EXCLUDED: they are zero-initialised at
    the first DDIM step; the graph is exported without them.

    Outputs: ``(boneOutput, globalOutput)`` as plain tensors.
    """

    def __init__(self, denoiser: MotionDenoiserV2) -> None:
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        noisyMotion: torch.Tensor,
        noisyGlobal: torch.Tensor,
        timesteps: torch.Tensor,
        textHiddenStates: torch.Tensor,
        textKeyPaddingMask: torch.Tensor,
        motionKeyPaddingMask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one denoising step; return ``(boneOutput, globalOutput)``.

        Parameters
        ----------
        noisyMotion : torch.Tensor
            ``(B, F, numBones, motionChannels)`` float32.
        noisyGlobal : torch.Tensor
            ``(B, F, globalChannels)`` float32.
        timesteps : torch.Tensor
            ``(B,)`` long — diffusion timestep indices.
        textHiddenStates : torch.Tensor
            ``(B, T, D_text)`` float32 from the encoder.
        textKeyPaddingMask : torch.Tensor
            ``(B, T)`` float32 — 1.0 == padding (cast to bool).
        motionKeyPaddingMask : torch.Tensor
            ``(B, F)`` float32 — 1.0 == padding (cast to bool).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``boneOutput`` ``(B, F, numBones, motionChannels)`` and
            ``globalOutput`` ``(B, F, globalChannels)``.
        """
        textMask = textKeyPaddingMask.bool()
        motionMask = motionKeyPaddingMask.bool()
        out: DenoiserOutput = self.denoiser(
            noisyMotion=noisyMotion,
            timesteps=timesteps,
            textHiddenStates=textHiddenStates,
            textKeyPaddingMask=textMask,
            noisyGlobalFeatures=noisyGlobal,
            motionKeyPaddingMask=motionMask,
        )
        # globalOutput is never None when globalChannels > 0 (production)
        assert out.globalOutput is not None, (
            "Denoiser returned None globalOutput; "
            "rebuild with globalChannels > 0."
        )
        return out.boneOutput, out.globalOutput


# -----------------------------------------------------------------------
# Controller wrapper — one frame (Goal C, phase C5)
# -----------------------------------------------------------------------
class _ControllerStepWrapper(nn.Module):
    """Thin wrapper for exporting a single controller forward via ONNX.

    The autoregressive rollout loop stays in the engine; only this one
    frame-to-delta forward — the graph the NPU accelerates — is exported
    (ROADMAP_DETERMINIST §2.1 truth #10).  The :class:`ControllerOutput`
    dataclass is flattened to plain tensors.

    The ``phase`` input is included only when the model uses phase
    conditioning (``phaseChannels > 0``); ``style`` is excluded (deferred
    C3).
    """

    def __init__(self, controller: MotionController) -> None:
        super().__init__()
        self.controller = controller

    def forward(
        self,
        boneWindow: torch.Tensor,
        control: torch.Tensor,
        globalWindow: torch.Tensor,
        phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one forward; return ``(boneDelta, globalDelta)``."""
        out = self.controller(
            boneWindow, control, globalWindow=globalWindow, phase=phase
        )
        assert out.globalDelta is not None, (
            "Controller returned None globalDelta; "
            "rebuild with globalChannels > 0."
        )
        return out.boneDelta, out.globalDelta


# -----------------------------------------------------------------------
# Public export functions
# -----------------------------------------------------------------------
def exportEncoder(
    encoder: CustomTextEncoder,
    outputPath: Path,
    batchSize: int = 1,
    textLen: int = 16,
    opsetVersion: int = ONNX_OPSET_VERSION,
) -> Path:
    """Export a :class:`CustomTextEncoder` to ONNX.

    Creates a deep copy of the encoder with all ``nn.MultiheadAttention``
    replaced by :class:`_OnnxMHAReplace` for export compatibility.  The
    original model is never mutated.

    Parameters
    ----------
    encoder : CustomTextEncoder
        Model to export (eval mode is set; weights unchanged).
    outputPath : Path
        Destination ``.onnx`` file path.
    batchSize : int
        Concrete batch size used as the tracing example.
    textLen : int
        Concrete sequence length used as the tracing example.
    opsetVersion : int
        ONNX opset version.  Default is :data:`ONNX_OPSET_VERSION`.

    Returns
    -------
    Path
        The written ``.onnx`` file path (same as ``outputPath``).
    """
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    exportReady = _replaceAllMHA(copy.deepcopy(encoder))
    wrapper = _EncoderWrapper(exportReady)  # type: ignore[arg-type]
    wrapper.eval()

    inputIds = torch.zeros(
        batchSize, textLen, dtype=torch.long
    )
    attentionMask = torch.ones(
        batchSize, textLen, dtype=torch.float32
    )
    exampleInputs = (inputIds, attentionMask)

    dynamicAxes: dict[str, dict[int, str]] = {
        "inputIds": {0: "batch", 1: "text_len"},
        "attentionMask": {0: "batch", 1: "text_len"},
        "hiddenStates": {0: "batch", 1: "text_len"},
        "keyPaddingMask": {0: "batch", 1: "text_len"},
    }
    outputNames = ["hiddenStates", "keyPaddingMask"]
    inputNames = ["inputIds", "attentionMask"]

    _runExport(
        wrapper=wrapper,
        exampleInputs=exampleInputs,
        outputPath=outputPath,
        inputNames=inputNames,
        outputNames=outputNames,
        dynamicAxes=dynamicAxes,
        opsetVersion=opsetVersion,
    )
    _verifyModel(outputPath)
    logger.info("Encoder exported and verified: %s", outputPath)
    return outputPath


def exportDenoiser(
    denoiser: MotionDenoiserV2,
    outputPath: Path,
    batchSize: int = 1,
    frames: int = 32,
    textLen: int = 16,
    opsetVersion: int = ONNX_OPSET_VERSION,
) -> Path:
    """Export one denoiser step to ONNX.

    The DDIM sampling loop stays in Python.  Only the single forward
    call — the graph the NPU would accelerate — is exported.

    Creates a deep copy of the denoiser with all ``nn.MultiheadAttention``
    replaced by :class:`_OnnxMHAReplace`.  The original model is never
    mutated.

    Parameters
    ----------
    denoiser : MotionDenoiserV2
        Model to export (eval mode is set; weights unchanged).
    outputPath : Path
        Destination ``.onnx`` file path.
    batchSize : int
        Concrete batch size for the tracing example.
    frames : int
        Concrete frame count for the tracing example.
    textLen : int
        Concrete text length for the tracing example.
    opsetVersion : int
        ONNX opset version.  Default is :data:`ONNX_OPSET_VERSION`.

    Returns
    -------
    Path
        The written ``.onnx`` file path (same as ``outputPath``).
    """
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    exportReady = _replaceAllMHA(copy.deepcopy(denoiser))
    wrapper = _DenoiserStepWrapper(exportReady)  # type: ignore[arg-type]
    wrapper.eval()

    cfg = denoiser.config
    noisyMotion = torch.randn(
        batchSize, frames, cfg.numBones, cfg.motionChannels
    )
    noisyGlobal = torch.randn(
        batchSize, frames, cfg.globalChannels
    )
    timesteps = torch.zeros(batchSize, dtype=torch.long)
    textHiddenStates = torch.randn(
        batchSize, textLen, cfg.effectiveTextEmbedDim
    )
    textKeyPaddingMask = torch.zeros(
        batchSize, textLen, dtype=torch.float32
    )
    motionKeyPaddingMask = torch.zeros(
        batchSize, frames, dtype=torch.float32
    )
    exampleInputs = (
        noisyMotion,
        noisyGlobal,
        timesteps,
        textHiddenStates,
        textKeyPaddingMask,
        motionKeyPaddingMask,
    )

    inputNames = [
        "noisyMotion",
        "noisyGlobal",
        "timesteps",
        "textHiddenStates",
        "textKeyPaddingMask",
        "motionKeyPaddingMask",
    ]
    outputNames = ["boneOutput", "globalOutput"]
    dynamicAxes: dict[str, dict[int, str]] = {
        "noisyMotion": {0: "batch", 1: "frames"},
        "noisyGlobal": {0: "batch", 1: "frames"},
        "timesteps": {0: "batch"},
        "textHiddenStates": {0: "batch", 2: "text_len"},
        "textKeyPaddingMask": {0: "batch", 1: "text_len"},
        "motionKeyPaddingMask": {0: "batch", 1: "frames"},
        "boneOutput": {0: "batch", 1: "frames"},
        "globalOutput": {0: "batch", 1: "frames"},
    }

    _runExport(
        wrapper=wrapper,
        exampleInputs=exampleInputs,
        outputPath=outputPath,
        inputNames=inputNames,
        outputNames=outputNames,
        dynamicAxes=dynamicAxes,
        opsetVersion=opsetVersion,
    )
    _verifyModel(outputPath)
    logger.info(
        "Denoiser step exported and verified: %s", outputPath
    )
    return outputPath


def exportController(
    controller: MotionController,
    outputPath: Path,
    batchSize: int = 1,
    opsetVersion: int = ONNX_OPSET_VERSION,
) -> Path:
    """Export a single controller forward to ONNX (Goal C, phase C5).

    The rollout loop stays in Python/the engine; only one frame-to-delta
    forward is exported.  A deep copy with all ``nn.MultiheadAttention``
    replaced by :class:`_OnnxMHAReplace` is used so the original model is
    never mutated.  Dynamic axes: batch and the context-window length.

    Parameters
    ----------
    controller : MotionController
        Model to export (eval mode is set; weights unchanged).
    outputPath : Path
        Destination ``.onnx`` file path.
    batchSize : int
        Concrete batch size for the tracing example.
    opsetVersion : int
        ONNX opset version.

    Returns
    -------
    Path
        The written ``.onnx`` file path.
    """
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    exportReady = _replaceAllMHA(copy.deepcopy(controller))
    wrapper = _ControllerStepWrapper(exportReady)  # type: ignore[arg-type]
    wrapper.eval()

    config = controller.config
    window = config.contextFrames
    boneWindow = torch.randn(
        batchSize, window, config.numBones, config.motionChannels
    )
    control = torch.randn(batchSize, config.controlChannels)
    globalWindow = torch.randn(batchSize, window, config.globalChannels)

    inputNames = ["bone_window", "control", "global_window"]
    dynamicAxes: dict[str, dict[int, str]] = {
        "bone_window": {0: "batch", 1: "context"},
        "control": {0: "batch"},
        "global_window": {0: "batch", 1: "context"},
        "bone_delta": {0: "batch"},
        "global_delta": {0: "batch"},
    }
    exampleInputs: tuple[Any, ...] = (boneWindow, control, globalWindow)
    if config.phaseChannels > 0:
        phase = torch.randn(batchSize, config.phaseChannels)
        exampleInputs = (boneWindow, control, globalWindow, phase)
        inputNames.append("phase")
        dynamicAxes["phase"] = {0: "batch"}

    _runExport(
        wrapper=wrapper,
        exampleInputs=exampleInputs,
        outputPath=outputPath,
        inputNames=inputNames,
        outputNames=["bone_delta", "global_delta"],
        dynamicAxes=dynamicAxes,
        opsetVersion=opsetVersion,
    )
    _verifyModel(outputPath)
    logger.info("Controller forward exported and verified: %s", outputPath)
    return outputPath


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------
def _runExport(
    wrapper: nn.Module,
    exampleInputs: tuple[Any, ...],
    outputPath: Path,
    inputNames: list[str],
    outputNames: list[str],
    dynamicAxes: dict[str, dict[int, str]],
    opsetVersion: int,
) -> None:
    """Run ``torch.onnx.export`` with consistent settings.

    Uses ``dynamo=False`` (TorchScript tracer) because ``dynamo=True``
    requires the optional ``onnxscript`` package.

    Parameters
    ----------
    wrapper : nn.Module
        Traced module (eval mode expected by caller).
    exampleInputs : tuple
        Concrete example inputs used for tracing.
    outputPath : Path
        Destination path for the ``.onnx`` file.
    inputNames : list[str]
        Names for the graph input nodes.
    outputNames : list[str]
        Names for the graph output nodes.
    dynamicAxes : dict
        Dynamic-axes spec passed to ``torch.onnx.export``.
    opsetVersion : int
        ONNX opset version.
    """
    with torch.no_grad():
        torch.onnx.export(  # type: ignore[call-overload]
            wrapper,
            exampleInputs,
            str(outputPath),
            input_names=inputNames,
            output_names=outputNames,
            dynamic_axes=dynamicAxes,
            opset_version=opsetVersion,
            do_constant_folding=True,
            # Use the legacy TorchScript-based path (dynamo=False) so
            # this works without onnxscript installed.  The new dynamo
            # path requires onnxscript which is not in the dependency
            # list; torch 2.9+ defaults to dynamo=True.
            dynamo=False,
        )


def _verifyModel(path: Path) -> None:
    """Load and validate the ONNX model with ``onnx.checker``.

    Parameters
    ----------
    path : Path
        Path to the ``.onnx`` file to validate.

    Raises
    ------
    onnx.checker.ValidationError
        If the model graph is malformed.
    """
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
