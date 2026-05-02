"""Compact diffusion denoiser for AI-nimator v2.

This module is the v2 rewrite of :mod:`src.shared.model.generation.denoiser`.
It deliberately drops several components of the legacy stack:

* **Spatial GCN** and **SpatioTemporalMix** pre-blocks — never ablation-
  tested, large parameter cost, no measured gain on the ~9k–50k corpus.
* **FiLM** and **AdaLN** conditioning paths — the AdaLN was already dead
  (kept for checkpoint compat), and FiLM let the model rely on a global
  scalar modulation which competed with the cross-attention.
* **Pooled-text + textAdapter MLP** — replaced by the per-token
  cross-attention into the :class:`CustomTextEncoder` output.

What stays
----------
* Sinusoidal positional encoding for frame order (MDM-style).
* Timestep embedding (sinusoidal + MLP).  Injected once via additive
  broadcast into the motion tokens — no FiLM, no prepend hack.
* Pre-norm transformer blocks: self-attn → cross-attn → FFN, each with
  its own LayerNorm and dropout, residual everywhere.
* Output split into bone-scoped and global-scoped predictions.

Target dimensions (matches v2 plan)
-----------------------------------
* embedDim:        384
* numLayers:         4
* numHeads:          8
* numBones:         22 (SMPL-22)
* motionChannels:    6 (rotation6d)
* globalChannels:    3 (root_translation)

Approximate parameter budget (production defaults)
--------------------------------------------------
* boneProj                          (6 → 384):           ≈    2.7k
* globalProj                        (3 → 384):           ≈    1.5k
* frameProj           (22*384 + 384 = 8832 → 384):       ≈  3.39M
* timestepEmbed (sinusoidal + 384→1536→384 MLP):         ≈  592k
* sinusoidal PE (no params):                             ≈    0
* per DenoiserBlockV2 (self-attn + cross-attn + FFN):
    self-attn  (Q,K,V,O = 4 × 384² + biases):            ≈  591k
    cross-attn (Q,K,V,O = 4 × 384² + biases):            ≈  591k
    FFN        (384 → 1536 → 384 + biases):              ≈ 1.18M
    3 × LayerNorm:                                       ≈    2.3k
    block total:                                         ≈  2.36M
* 4 blocks stack:                                        ≈  9.45M
* output norm + projection (384 → 22*6+3 = 135):         ≈   53k
* **Total ≈ 14.1M trainable params** (denoiser only).

Combined with the v2 :class:`CustomTextEncoder` (~5.2M) the entire text-
to-motion stack lands at ~19M params — well below the legacy 254M and
the v2 plan budget of ~30–40M.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class MotionDenoiserV2Config:
    """Configuration for :class:`MotionDenoiserV2`.

    Parameters
    ----------
    embedDim : int
        Hidden width of the denoiser.  ``384`` is the v2 default.
    numHeads : int
        Number of attention heads.  Must divide ``embedDim``.
    numLayers : int
        Number of stacked :class:`DenoiserBlockV2` blocks.
    numBones : int
        Skeleton size (SMPL-22 → 22).
    motionChannels : int
        Channels per bone.  ``6`` for rotation6d.
    globalChannels : int
        Global channels per frame (root_translation = 3).  Set ``0`` to
        disable the global branch entirely.
    textEmbedDim : int
        Width of the incoming text hidden states.  ``0`` means "same as
        ``embedDim``" — when the text encoder's ``outputDim`` already
        matches the denoiser, no projection is inserted.
    maxFrames : int
        Maximum motion length used to size the sinusoidal PE buffer.
    dropout : float
        Dropout probability applied inside attention and FFN sublayers.
    """

    embedDim: int = 384
    numHeads: int = 8
    numLayers: int = 4
    numBones: int = 22
    motionChannels: int = 6
    globalChannels: int = 3
    textEmbedDim: int = 0  # 0 → same as embedDim
    maxFrames: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.embedDim % self.numHeads != 0:
            raise ValueError(
                f"embedDim ({self.embedDim}) must be divisible by "
                f"numHeads ({self.numHeads})."
            )
        if self.numLayers < 1:
            raise ValueError("numLayers must be >= 1.")
        if self.numBones < 1:
            raise ValueError("numBones must be >= 1.")
        if self.motionChannels < 1:
            raise ValueError("motionChannels must be >= 1.")
        if self.globalChannels < 0:
            raise ValueError("globalChannels must be >= 0.")
        if self.maxFrames < 1:
            raise ValueError("maxFrames must be >= 1.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")

    @property
    def effectiveTextEmbedDim(self) -> int:
        """Text-side dimension after applying the default."""
        return self.textEmbedDim if self.textEmbedDim > 0 else self.embedDim

    @property
    def boneOutputDim(self) -> int:
        """Number of channels predicted per frame for the bone branch."""
        return self.numBones * self.motionChannels

    @property
    def totalOutputDim(self) -> int:
        """Total per-frame output width (bone + global)."""
        return self.boneOutputDim + self.globalChannels


# ---------------------------------------------------------------------
# Helpers reused from the legacy module — re-implemented here to keep
# v2 self-contained (legacy denoiser.py will eventually be archived).
# ---------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + small MLP."""

    def __init__(self, embedDim: int, maxPeriod: int = 10000) -> None:
        super().__init__()
        self.embedDim = embedDim
        self.maxPeriod = maxPeriod
        self.mlp = nn.Sequential(
            nn.Linear(embedDim, embedDim * 4),
            nn.SiLU(),
            nn.Linear(embedDim * 4, embedDim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed a 1-D tensor of timesteps into ``(batch, embedDim)``."""
        half = self.embedDim // 2
        freqs = torch.exp(
            -math.log(self.maxPeriod)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedDim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return self.mlp(embedding)


class SinusoidalPositionalEncoding(nn.Module):
    """Additive non-learned sinusoidal positional encoding."""

    def __init__(self, embedDim: int, maxLen: int) -> None:
        super().__init__()
        pe = torch.zeros(maxLen, embedDim)
        position = torch.arange(0, maxLen, dtype=torch.float32).unsqueeze(1)
        divTerm = torch.exp(
            torch.arange(0, embedDim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(embedDim, 1))
        )
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, maxLen, embedDim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the slice ``pe[:, :T, :]`` to ``x`` of shape ``(B, T, D)``."""
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds the PE buffer "
                f"size {self.pe.shape[1]}."
            )
        return x + self.pe[:, : x.shape[1], :].to(x.dtype)


# ---------------------------------------------------------------------
# Denoiser block (v2)
# ---------------------------------------------------------------------
class DenoiserBlockV2(nn.Module):
    """Pre-norm transformer block: self-attn → cross-attn → FFN.

    Differences with the legacy :class:`DenoiserBlock`:

    * No FiLM, no AdaLN.  Conditioning flows exclusively through the
      cross-attention onto text tokens.
    * Cross-attention is **mandatory** — passing ``None`` for
      ``textHiddenStates`` raises rather than silently degrading to a
      pure self-attention block.  This forces the caller to be explicit
      about ablations; if you really need to disable text, train with a
      zero-text-embed dropout (see ``cond-mask-prob`` in the YAML
      config).
    """

    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        ffnDim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.normSelf = nn.LayerNorm(embedDim)
        self.selfAttention = nn.MultiheadAttention(
            embed_dim=embedDim,
            num_heads=numHeads,
            dropout=dropout,
            batch_first=True,
        )

        self.normCross = nn.LayerNorm(embedDim)
        self.crossAttention = nn.MultiheadAttention(
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

    def forward(
        self,
        x: torch.Tensor,
        textHiddenStates: torch.Tensor,
        textKeyPaddingMask: torch.Tensor | None = None,
        motionKeyPaddingMask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attn + cross-attn + FFN with pre-norm residuals."""
        # 1. Self-attention over motion frames.
        h = self.normSelf(x)
        h, _ = self.selfAttention(
            h,
            h,
            h,
            key_padding_mask=motionKeyPaddingMask,
            need_weights=False,
        )
        x = x + self.dropout(h)

        # 2. Cross-attention onto the text token sequence.
        h = self.normCross(x)
        h, _ = self.crossAttention(
            query=h,
            key=textHiddenStates,
            value=textHiddenStates,
            key_padding_mask=textKeyPaddingMask,
            need_weights=False,
        )
        x = x + self.dropout(h)

        # 3. Position-wise feed-forward.
        h = self.normFfn(x)
        h = self.feedForward(h)
        x = x + self.dropout(h)

        return x


# ---------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DenoiserOutput:
    """Structured output of :meth:`MotionDenoiserV2.forward`.

    Attributes
    ----------
    boneOutput : torch.Tensor
        Predicted bone-scoped features of shape
        ``(B, F, numBones, motionChannels)`` — typically rotation6d.
    globalOutput : torch.Tensor or None
        Predicted global features of shape ``(B, F, globalChannels)``,
        or ``None`` when ``globalChannels == 0``.  Typically the
        root_translation channels.
    """

    boneOutput: torch.Tensor
    globalOutput: torch.Tensor | None


# ---------------------------------------------------------------------
# Top-level denoiser
# ---------------------------------------------------------------------
class MotionDenoiserV2(nn.Module):
    """Compact text-conditioned diffusion denoiser.

    Inputs
    ------
    * ``noisyMotion``         : ``(B, F, numBones, motionChannels)``
    * ``timesteps``           : ``(B,)`` long
    * ``textHiddenStates``    : ``(B, T_text, D_text)`` from the text encoder
    * ``textKeyPaddingMask``  : ``(B, T_text)`` bool, ``True == pad``
    * ``noisyGlobalFeatures`` : ``(B, F, globalChannels)`` or ``None``
    * ``motionKeyPaddingMask``: ``(B, F)`` bool, ``True == pad`` — optional

    Output
    ------
    :class:`DenoiserOutput`.

    The forward pass mirrors a denoising step in either x0- or
    v-prediction mode; the choice is *not* baked into the architecture
    (it lives in the loss / sampler).
    """

    def __init__(self, config: MotionDenoiserV2Config) -> None:
        super().__init__()
        self._config = config

        embedDim = config.embedDim
        ffnDim = embedDim * 4

        # --- Input projections ---------------------------------------
        # Per-bone shared projection: motionChannels -> embedDim.  A
        # single Linear is broadcast across the bones dimension by
        # PyTorch (input shape (B, F, B, C) → output (B, F, B, embedDim)).
        # Sharing weights across bones gives a useful inductive bias and
        # keeps the parameter count tiny (motionChannels × embedDim).
        self.boneProj = nn.Linear(config.motionChannels, embedDim)
        if config.globalChannels > 0:
            self.globalProj = nn.Linear(config.globalChannels, embedDim)
        else:
            # Register `None` to keep the attribute typed.
            self.globalProj = None  # type: ignore[assignment]

        # Per-frame fusion: concatenate (numBones × embedDim) [+ embedDim
        # for global] then project back to embedDim.  This is where the
        # bone tokens collapse into a single per-frame token used by the
        # transformer stack.
        frameInputDim = config.numBones * embedDim
        if config.globalChannels > 0:
            frameInputDim += embedDim
        self.frameProj = nn.Linear(frameInputDim, embedDim)

        # --- Conditioning embeddings ---------------------------------
        self.timestepEmbed = TimestepEmbedding(embedDim)
        self.posEncoder = SinusoidalPositionalEncoding(
            embedDim, maxLen=config.maxFrames
        )
        self.embedDropout = nn.Dropout(config.dropout)

        # --- Optional text-side projection ---------------------------
        if config.effectiveTextEmbedDim != embedDim:
            self.textProjection: nn.Module = nn.Sequential(
                nn.Linear(config.effectiveTextEmbedDim, embedDim),
                nn.LayerNorm(embedDim),
            )
        else:
            self.textProjection = nn.Identity()

        # --- Transformer stack ---------------------------------------
        self.blocks = nn.ModuleList(
            [
                DenoiserBlockV2(
                    embedDim=embedDim,
                    numHeads=config.numHeads,
                    ffnDim=ffnDim,
                    dropout=config.dropout,
                )
                for _ in range(config.numLayers)
            ]
        )
        self.outputNorm = nn.LayerNorm(embedDim)
        self.outputProjection = nn.Linear(embedDim, config.totalOutputDim)

        self._initWeights()

    # ------------------------------------------------------------------
    # Properties / introspection
    # ------------------------------------------------------------------
    @property
    def config(self) -> MotionDenoiserV2Config:
        return self._config

    def numParameters(self, trainableOnly: bool = True) -> int:
        """Return the total (trainable) parameter count."""
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
        noisyMotion: torch.Tensor,
        timesteps: torch.Tensor,
        textHiddenStates: torch.Tensor,
        textKeyPaddingMask: torch.Tensor | None = None,
        noisyGlobalFeatures: torch.Tensor | None = None,
        motionKeyPaddingMask: torch.Tensor | None = None,
    ) -> DenoiserOutput:
        """Predict clean (or v-target) motion features from noisy input."""
        self._validateInputs(
            noisyMotion=noisyMotion,
            timesteps=timesteps,
            textHiddenStates=textHiddenStates,
            noisyGlobalFeatures=noisyGlobalFeatures,
        )

        batchSize, frames, numBones, _ = noisyMotion.shape

        # --- Project bone features -----------------------------------
        # (B, F, B_bones, motionChannels) → (B, F, B_bones, embedDim)
        boneTokens = self.boneProj(noisyMotion)
        # Flatten bones into the per-frame channel axis.
        boneFlat = boneTokens.reshape(
            batchSize, frames, numBones * self._config.embedDim
        )

        # --- Optionally add global features --------------------------
        if (
            self.globalProj is not None
            and noisyGlobalFeatures is not None
        ):
            globalH = self.globalProj(noisyGlobalFeatures)
            frameInput = torch.cat([boneFlat, globalH], dim=-1)
        else:
            frameInput = boneFlat
        frameH = self.frameProj(frameInput)  # (B, F, embedDim)

        # --- Inject timestep ----------------------------------------
        # Single broadcast addition: every frame gets the same scalar
        # timestep modulation.  This replaces the legacy FiLM path.
        timestepH = self.timestepEmbed(timesteps)  # (B, embedDim)
        frameH = frameH + timestepH.unsqueeze(1)

        # --- Add temporal positional encoding -----------------------
        frameH = self.posEncoder(frameH)
        frameH = self.embedDropout(frameH)

        # --- Project text once -------------------------------------
        textH = self.textProjection(textHiddenStates)

        # --- Transformer stack ---------------------------------------
        for block in self.blocks:
            frameH = block(
                frameH,
                textHiddenStates=textH,
                textKeyPaddingMask=textKeyPaddingMask,
                motionKeyPaddingMask=motionKeyPaddingMask,
            )

        # --- Output projection ---------------------------------------
        frameH = self.outputNorm(frameH)
        flatOutput = self.outputProjection(frameH)  # (B, F, totalOutputDim)

        boneFlatDim = self._config.boneOutputDim
        boneOutput = flatOutput[..., :boneFlatDim].view(
            batchSize,
            frames,
            self._config.numBones,
            self._config.motionChannels,
        )
        globalOutput: torch.Tensor | None = None
        if self._config.globalChannels > 0:
            globalOutput = flatOutput[..., boneFlatDim:]

        return DenoiserOutput(
            boneOutput=boneOutput,
            globalOutput=globalOutput,
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _validateInputs(
        self,
        noisyMotion: torch.Tensor,
        timesteps: torch.Tensor,
        textHiddenStates: torch.Tensor,
        noisyGlobalFeatures: torch.Tensor | None,
    ) -> None:
        if noisyMotion.ndim != 4:
            raise ValueError(
                "noisyMotion must be 4-D (B, F, B_bones, C); got shape "
                f"{tuple(noisyMotion.shape)}."
            )
        batchSize, frames, numBones, channels = noisyMotion.shape
        if numBones != self._config.numBones:
            raise ValueError(
                f"noisyMotion has {numBones} bones but config expects "
                f"{self._config.numBones}."
            )
        if channels != self._config.motionChannels:
            raise ValueError(
                f"noisyMotion has {channels} channels per bone but "
                f"config expects {self._config.motionChannels}."
            )
        if frames > self._config.maxFrames:
            raise ValueError(
                f"frames={frames} exceeds configured maxFrames="
                f"{self._config.maxFrames}."
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != batchSize:
            raise ValueError(
                "timesteps must be 1-D of length batchSize; got shape "
                f"{tuple(timesteps.shape)} for batchSize={batchSize}."
            )
        if textHiddenStates.ndim != 3 or textHiddenStates.shape[0] != batchSize:
            raise ValueError(
                "textHiddenStates must be 3-D (B, T_text, D_text) with "
                f"matching batch size; got shape "
                f"{tuple(textHiddenStates.shape)}."
            )
        if textHiddenStates.shape[-1] != self._config.effectiveTextEmbedDim:
            raise ValueError(
                f"textHiddenStates last dim "
                f"{textHiddenStates.shape[-1]} does not match config "
                f"textEmbedDim {self._config.effectiveTextEmbedDim}."
            )
        if (
            noisyGlobalFeatures is not None
            and self._config.globalChannels == 0
        ):
            raise ValueError(
                "noisyGlobalFeatures was provided but globalChannels=0."
            )
        if noisyGlobalFeatures is not None:
            if (
                noisyGlobalFeatures.shape[0] != batchSize
                or noisyGlobalFeatures.shape[1] != frames
                or noisyGlobalFeatures.shape[2]
                != self._config.globalChannels
            ):
                raise ValueError(
                    "noisyGlobalFeatures has unexpected shape "
                    f"{tuple(noisyGlobalFeatures.shape)}; expected "
                    f"({batchSize}, {frames}, "
                    f"{self._config.globalChannels})."
                )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initWeights(self) -> None:
        """Standard Xavier / LayerNorm init."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
