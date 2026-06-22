"""Autoregressive deterministic motion controller (Goal C, phase C1).

This is the second generation engine of AI-nimator, parallel to the
diffusion denoiser (ROADMAP_DETERMINIST §1).  Instead of *sampling* a
full animation from a text prompt, it *regresses* the motion one frame
at a time from a control signal:

    f(state_window, control, [phase]) → Δstate

A single forward produces the next-frame delta — no schedule, no DDIM
loop — which makes the ONNX/NPU path trivial (ROADMAP_DETERMINIST §2.1
truth #10, promoted to an acquis here).

Design
------
* The controller operates on the **lean** representation only:
  rotation6d (132) + root_translation (3).  It predicts ``Δstate``;
  FK-derivable signals (velocities, contacts) are supervised at the
  loss, never emitted as redundant channels (§2.2).
* Conditioning (control signal + locomotor phase [+ style, deferred])
  drives every block through DiT-style **AdaLN** modulation.  The
  projection is initialised with a small non-zero ``filmInitStd`` so a
  gradient path control→output exists from epoch 1 — this is the same
  escape from posterior collapse used by the denoiser, and it is what
  keeps ``control_sensitivity`` (§4) above zero.
* The backbone is a stack of pre-norm self-attention blocks over the
  context window (length ``contextFrames``); for C1 the window is 1 so
  the block reduces to a conditioned per-frame residual MLP.

Z-normalization (§2.1 truth #3) is the caller's responsibility: the
controller consumes a normalized state window and emits a normalized
delta.  Statistics live in :class:`MotionNormalizer` instances owned by
the training loop (one for the state, one for the delta).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ainimator.core.types.controller import ControllerV2Config
from ainimator.model.denoiser_v2 import SinusoidalPositionalEncoding


# ---------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ControllerOutput:
    """Structured next-frame prediction of :class:`MotionController`.

    Attributes
    ----------
    boneDelta : torch.Tensor
        Predicted bone-scoped delta, shape ``(B, numBones, motionCh)``
        (normalized rotation6d delta).
    globalDelta : torch.Tensor or None
        Predicted global delta, shape ``(B, globalChannels)``, or
        ``None`` when ``globalChannels == 0``.
    """

    boneDelta: torch.Tensor
    globalDelta: torch.Tensor | None


# ---------------------------------------------------------------------
# AdaLN modulation generator (DiT-style, small non-zero init)
# ---------------------------------------------------------------------
class _ControllerModulation(nn.Module):
    """Produce per-block AdaLN ``(γ, β)`` pairs from the condition.

    Two sub-layers per block (self-attn, FFN) → four modulations.  The
    projection uses a small non-zero init so the conditioning gradient
    is alive from the first step (anti-collapse, §4).
    """

    NUM_MODULATIONS: int = 4  # γ, β for self-attn and FFN

    def __init__(self, condDim: int, embedDim: int, initStd: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(condDim)
        self.proj = nn.Linear(condDim, self.NUM_MODULATIONS * embedDim)
        nn.init.normal_(self.proj.weight, std=initStd)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(γ_self, β_self, γ_ffn, β_ffn)``, each ``(B, D)``."""
        modulations = self.proj(self.norm(condition))
        gammaSelf, betaSelf, gammaFfn, betaFfn = modulations.chunk(
            self.NUM_MODULATIONS, dim=-1
        )
        return gammaSelf, betaSelf, gammaFfn, betaFfn


# ---------------------------------------------------------------------
# Controller block
# ---------------------------------------------------------------------
class ControllerBlock(nn.Module):
    """Pre-norm self-attention + FFN, AdaLN-modulated by the control.

    The self-attention runs over the context window axis; with
    ``contextFrames == 1`` it is a no-op mixer and the block behaves as a
    conditioned residual MLP.  No cross-attention: the controller has no
    text stream — conditioning enters exclusively through AdaLN.
    """

    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        ffnDim: int,
        dropout: float,
        condDim: int,
        filmInitStd: float,
    ) -> None:
        super().__init__()
        self.normSelf = nn.LayerNorm(embedDim)
        self.selfAttention = nn.MultiheadAttention(
            embed_dim=embedDim,
            num_heads=numHeads,
            dropout=dropout,
            batch_first=True,
        )
        self.normFfn = nn.LayerNorm(embedDim)
        self.feedForward = nn.Sequential(
            nn.Linear(embedDim, ffnDim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffnDim, embedDim),
        )
        self.dropout = nn.Dropout(dropout)
        self.modulation = _ControllerModulation(
            condDim=condDim, embedDim=embedDim, initStd=filmInitStd
        )

    @staticmethod
    def _modulate(
        normalized: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``(1 + γ) · normalized + β`` with window-axis broadcast."""
        return normalized * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """Run the block on ``x`` of shape ``(B, K, embedDim)``."""
        gammaSelf, betaSelf, gammaFfn, betaFfn = self.modulation(condition)

        hidden = self._modulate(self.normSelf(x), gammaSelf, betaSelf)
        attended, _ = self.selfAttention(
            hidden, hidden, hidden, need_weights=False
        )
        x = x + self.dropout(attended)

        hidden = self._modulate(self.normFfn(x), gammaFfn, betaFfn)
        x = x + self.dropout(self.feedForward(hidden))
        return x


# ---------------------------------------------------------------------
# Top-level controller
# ---------------------------------------------------------------------
class MotionController(nn.Module):
    """Autoregressive deterministic controller ``f(window, ctrl) → Δ``.

    Inputs (all in normalized space)
    --------------------------------
    * ``boneWindow``   : ``(B, K, numBones, motionChannels)``
    * ``globalWindow`` : ``(B, K, globalChannels)`` or ``None``
    * ``control``      : ``(B, controlChannels)``
    * ``phase``        : ``(B, phaseChannels)`` or ``None``
    * ``style``        : ``(B, styleChannels)`` or ``None`` (deferred C3)

    where ``K == config.contextFrames``.

    Output
    ------
    :class:`ControllerOutput` — the next-frame normalized delta.
    """

    def __init__(self, config: ControllerV2Config) -> None:
        super().__init__()
        self._config = config
        embedDim = config.embedDim
        ffnDim = embedDim * 4

        self.boneProj = nn.Linear(config.motionChannels, embedDim)
        if config.globalChannels > 0:
            self.globalProj: nn.Module = nn.Linear(
                config.globalChannels, embedDim
            )
        else:
            self.globalProj = None  # type: ignore[assignment]

        frameInputDim = config.numBones * embedDim
        if config.globalChannels > 0:
            frameInputDim += embedDim
        self.frameProj = nn.Linear(frameInputDim, embedDim)

        # Sized to maxFrames (not contextFrames) so the context-window
        # axis can stay a dynamic ONNX axis (C5) without resizing the PE
        # buffer.  The buffer is non-learned, so the cost is negligible.
        self.posEncoder = SinusoidalPositionalEncoding(
            embedDim, maxLen=max(config.maxFrames, config.contextFrames)
        )
        self.embedDropout = nn.Dropout(config.dropout)

        # Conditioning encoder: raw (control [+phase] [+style]) → embedDim.
        self.condEncoder = nn.Sequential(
            nn.Linear(config.conditioningChannels, embedDim),
            nn.SiLU(),
            nn.Linear(embedDim, embedDim),
        )

        self.blocks = nn.ModuleList(
            [
                ControllerBlock(
                    embedDim=embedDim,
                    numHeads=config.numHeads,
                    ffnDim=ffnDim,
                    dropout=config.dropout,
                    condDim=embedDim,
                    filmInitStd=config.filmInitStd,
                )
                for _ in range(config.numLayers)
            ]
        )
        self.outputNorm = nn.LayerNorm(embedDim)
        self.outputProjection = nn.Linear(embedDim, config.totalOutputDim)
        self._initWeights()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def config(self) -> ControllerV2Config:
        return self._config

    def numParameters(self, trainableOnly: bool = True) -> int:
        """Return the (trainable) parameter count."""
        if trainableOnly:
            return sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        boneWindow: torch.Tensor,
        control: torch.Tensor,
        globalWindow: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        style: torch.Tensor | None = None,
    ) -> ControllerOutput:
        """Predict the next-frame normalized delta from a state window."""
        self._validateInputs(boneWindow, control, globalWindow, phase, style)
        batchSize, window = boneWindow.shape[0], boneWindow.shape[1]

        boneTokens = self.boneProj(boneWindow)
        boneFlat = boneTokens.reshape(
            batchSize, window, self._config.numBones * self._config.embedDim
        )
        if self.globalProj is not None and globalWindow is not None:
            frameInput = torch.cat(
                [boneFlat, self.globalProj(globalWindow)], dim=-1
            )
        else:
            frameInput = boneFlat
        frameHidden = self.frameProj(frameInput)

        frameHidden = self.posEncoder(frameHidden)
        frameHidden = self.embedDropout(frameHidden)

        condition = self.condEncoder(
            self._assembleConditioning(control, phase, style)
        )
        for block in self.blocks:
            frameHidden = block(frameHidden, condition)

        lastFrame = self.outputNorm(frameHidden[:, -1, :])
        flatDelta = self.outputProjection(lastFrame)
        return self._splitOutput(flatDelta, batchSize)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _assembleConditioning(
        self,
        control: torch.Tensor,
        phase: torch.Tensor | None,
        style: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenate the active conditioning channels into ``(B, C)``."""
        parts = [control]
        if self._config.phaseChannels > 0 and phase is not None:
            parts.append(phase)
        if self._config.styleChannels > 0 and style is not None:
            parts.append(style)
        return torch.cat(parts, dim=-1)

    def _splitOutput(
        self, flatDelta: torch.Tensor, batchSize: int
    ) -> ControllerOutput:
        """Split the flat output into bone / global deltas."""
        boneDim = self._config.boneOutputDim
        boneDelta = flatDelta[..., :boneDim].view(
            batchSize, self._config.numBones, self._config.motionChannels
        )
        globalDelta: torch.Tensor | None = None
        if self._config.globalChannels > 0:
            globalDelta = flatDelta[..., boneDim:]
        return ControllerOutput(boneDelta=boneDelta, globalDelta=globalDelta)

    def _validateInputs(
        self,
        boneWindow: torch.Tensor,
        control: torch.Tensor,
        globalWindow: torch.Tensor | None,
        phase: torch.Tensor | None,
        style: torch.Tensor | None,
    ) -> None:
        """Validate input ranks and channel widths."""
        if boneWindow.ndim != 4:
            raise ValueError(
                "boneWindow must be 4-D (B, K, bones, C); got "
                f"{tuple(boneWindow.shape)}."
            )
        _, window, bones, channels = boneWindow.shape
        if window != self._config.contextFrames:
            raise ValueError(
                f"window K={window} != contextFrames="
                f"{self._config.contextFrames}."
            )
        if bones != self._config.numBones:
            raise ValueError(
                f"boneWindow has {bones} bones; config expects "
                f"{self._config.numBones}."
            )
        if channels != self._config.motionChannels:
            raise ValueError(
                f"boneWindow has {channels} channels; config expects "
                f"{self._config.motionChannels}."
            )
        self._validateConditioning(control, globalWindow, phase, style)

    def _validateConditioning(
        self,
        control: torch.Tensor,
        globalWindow: torch.Tensor | None,
        phase: torch.Tensor | None,
        style: torch.Tensor | None,
    ) -> None:
        """Validate control / global / phase / style widths."""
        if control.shape[-1] != self._config.controlChannels:
            raise ValueError(
                f"control has {control.shape[-1]} channels; config "
                f"expects {self._config.controlChannels}."
            )
        if globalWindow is not None and (
            globalWindow.shape[-1] != self._config.globalChannels
        ):
            raise ValueError(
                f"globalWindow has {globalWindow.shape[-1]} channels; "
                f"config expects {self._config.globalChannels}."
            )
        if self._config.phaseChannels > 0 and phase is None:
            raise ValueError(
                "phase is required when phaseMode is not 'none'."
            )
        if self._config.styleChannels > 0 and style is None:
            raise ValueError(
                "style is required when styleLatentEnabled is True."
            )

    def _initWeights(self) -> None:
        """Xavier / LayerNorm init; re-apply the modulation custom init.

        The generic loop sees each modulation's projection as a plain
        ``nn.Linear`` and would overwrite its small non-zero init, so the
        AdaLN projections are re-initialised afterwards (same pattern as
        the denoiser's self-conditioning projections).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, _ControllerModulation):
                nn.init.normal_(
                    module.proj.weight, std=self._config.filmInitStd
                )
                nn.init.zeros_(module.proj.bias)


# ---------------------------------------------------------------------
# ONNX export wrapper (single forward — the ROADMAP §2.10 acquis)
# ---------------------------------------------------------------------
class ControllerForwardWrapper(nn.Module):
    """Tuple-output wrapper around :class:`MotionController` for export.

    ``torch.onnx.export`` cannot return the :class:`ControllerOutput`
    dataclass, so this thin module flattens the forward to a tuple
    ``(boneDelta, globalDelta)`` (or just ``(boneDelta,)`` when there is
    no global branch).  It wraps a *single* forward — the autoregressive
    rollout loop stays outside the graph (ROADMAP_DETERMINIST §2.1).
    """

    def __init__(self, controller: MotionController) -> None:
        super().__init__()
        self.controller = controller

    def forward(
        self,
        boneWindow: torch.Tensor,
        control: torch.Tensor,
        globalWindow: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return the next-frame delta(s) as a plain tuple."""
        output = self.controller(
            boneWindow, control, globalWindow=globalWindow, phase=phase
        )
        if output.globalDelta is None:
            return (output.boneDelta,)
        return output.boneDelta, output.globalDelta
