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
import torch.nn.functional as F

from ainimator.model.motion_alignment_encoder import (
    MotionAlignmentEncoder,
    MotionAlignmentEncoderConfig,
)


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
    # Phase D.1 — when True the denoiser exposes a pooled
    # ``motionEmbedding`` in :class:`DenoiserOutput` consumed by the
    # text↔motion contrastive loss.  Adds two small MLPs (motion and
    # text projections) and zero cost at inference (no extra forward
    # pass when the head is unused).
    alignmentEnabled: bool = False
    # Phase D.1 (Levier B refinement, 2026-05-07) — projection
    # bottleneck for the contrastive alignment.  Following the
    # SimCLR / CLIP pattern: project both modalities into a shared
    # smaller-dim space then L2-normalize.  ``128`` is the CLIP
    # default and forces the head to compress useful information.
    alignmentProjectionDim: int = 128
    # Phase D.4 (Levier D, 2026-05-07) — FiLM conditioning shortcut.
    # When the cross-attention collapses (its residual contribution
    # ≈ 0), the text signal stops reaching the diffusion features.
    # FiLM gives the text a *multiplicative* path that the network
    # cannot zero out — the modulation `h * (1 + γ) + β` is computed
    # from `(text_pooled, timestep_embed)` once and broadcast across
    # all frames.  Enabled with ``useFilmConditioning=True``; default
    # off for backward compat with pre-Levier-D checkpoints.
    useFilmConditioning: bool = False
    filmDropout: float = 0.0  # dropout inside the FiLM MLP
    # Phase E (Levier E, 2026-05-08) — per-block AdaLN-style FiLM.
    # In addition to (or instead of) the single global FiLM above,
    # each :class:`DenoiserBlockV2` gets its own modulation generator
    # driven by ``(text_pooled, timestep_embed)`` that conditions
    # every pre-norm in the block.  This is the DiT / SD3 standard
    # for conditional diffusion transformers and is the standard fix
    # for posterior-collapse on cross-attention.
    usePerBlockFilm: bool = False
    filmInitStd: float = 0.02
    # Phase F iter-2 (2026-05-14) — auxiliary contrastive loss applied
    # directly to the raw masked-mean pool of the text encoder hidden
    # states (no learnable projection on text side).  Forces the encoder
    # pool itself to discriminate prompts — without it, the SimCLR-style
    # alignment head's 2-layer MLP can amplify micro-differences in the
    # raw pool (cond↔uncond cos-sim 0.9998) into well-separated alignment
    # vectors, satisfying the main contrastive loss while leaving the
    # pooled vector consumed by FiLM/AdaLN collapsed → cfg_sim → 1.0.
    auxPoolAlignmentEnabled: bool = False
    # 2026-06-01 — generated-motion (x0) contrastive.  When True the
    # denoiser owns a small TMR-style :class:`MotionAlignmentEncoder`
    # that embeds the *reconstructed x0* (not the noisy input) so the
    # text↔motion contrastive forces the predicted motion — not just the
    # input it reads — to be classifiable to its prompt.  Built only when
    # enabled to keep older checkpoints loadable.  See
    # ``motion_alignment_encoder.py`` for the loophole this closes.
    x0AlignmentEnabled: bool = False
    x0AlignmentEmbedDim: int = 256
    x0AlignmentNumLayers: int = 2
    x0AlignmentNumHeads: int = 4
    # 2026-06-02 — self-conditioning (Chen et al. 2022, "Analog Bits").
    # The denoiser optionally receives its own previous x0 estimate
    # (detached) as an extra input projected and added to the bone /
    # global tokens.  At training a coin flip decides whether to run a
    # first no-grad pass to produce the estimate; at sampling each DDIM
    # step feeds the previous step's x0.  This narrows the train/sampling
    # exposure gap diagnosed on 2026-06-02 (the model denoised from x_t,
    # which leaks the answer, so it never had to use the text; at sampling
    # from pure noise it fell back to the unconditional mode).
    useSelfConditioning: bool = False

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
        if self.alignmentProjectionDim < 1:
            raise ValueError("alignmentProjectionDim must be >= 1.")
        if not (0.0 <= self.filmDropout < 1.0):
            raise ValueError("filmDropout must be in [0, 1).")

    @property
    def effectiveTextEmbedDim(self) -> int:
        """Text-side dimension after applying the default."""
        return self.textEmbedDim if self.textEmbedDim > 0 else self.embedDim

    @property
    def alignmentDim(self) -> int:
        """Shared projection dim of the contrastive alignment head."""
        return self.alignmentProjectionDim

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
        usePerBlockFilm: bool = False,
        condDim: int = 0,
        filmInitStd: float = 0.02,
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

        if usePerBlockFilm:
            if condDim <= 0:
                raise ValueError(
                    "DenoiserBlockV2: usePerBlockFilm requires "
                    "condDim > 0."
                )
            self.adaln: nn.Module = _AdaLNBlockModulation(
                condDim=condDim, embedDim=embedDim, initStd=filmInitStd
            )
        else:
            self.adaln = nn.Identity()

    @staticmethod
    def _applyModulation(
        normalized: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``(1 + γ) · normalized + β`` with frame-axis broadcast."""
        return normalized * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        textHiddenStates: torch.Tensor,
        textKeyPaddingMask: torch.Tensor | None = None,
        motionKeyPaddingMask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attn + cross-attn + FFN with pre-norm residuals.

        When ``condition`` is provided and the block was built with
        ``usePerBlockFilm=True``, the three pre-norms are modulated by
        a FiLM (γ, β) pair derived from the condition vector.
        """
        if isinstance(self.adaln, _AdaLNBlockModulation):
            if condition is None:
                raise ValueError(
                    "DenoiserBlockV2: a per-block FiLM was configured "
                    "but no condition vector was passed to forward."
                )
            (
                gammaSelf,
                betaSelf,
                gammaCross,
                betaCross,
                gammaFfn,
                betaFfn,
            ) = self.adaln(condition)
        else:
            gammaSelf = betaSelf = None
            gammaCross = betaCross = None
            gammaFfn = betaFfn = None

        # 1. Self-attention over motion frames (with optional AdaLN).
        h = self.normSelf(x)
        if gammaSelf is not None and betaSelf is not None:
            h = self._applyModulation(h, gammaSelf, betaSelf)
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
        if gammaCross is not None and betaCross is not None:
            h = self._applyModulation(h, gammaCross, betaCross)
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
        if gammaFfn is not None and betaFfn is not None:
            h = self._applyModulation(h, gammaFfn, betaFfn)
        h = self.feedForward(h)
        x = x + self.dropout(h)

        return x


# ---------------------------------------------------------------------
# Per-block AdaLN modulation (Phase E, Levier E 2026-05-08)
# ---------------------------------------------------------------------
class _AdaLNBlockModulation(nn.Module):
    """Per-block AdaLN-style FiLM modulation generator.

    DiT / Stable Diffusion 3 standard: each transformer block consumes
    a conditioning vector ``c`` (here ``c = concat(text_pooled,
    timestep_embed)``) through a small MLP that produces a set of
    modulation tensors used to FiLM-modulate each pre-norm in the
    block.

    For our :class:`DenoiserBlockV2` we have three sub-layers
    (self-attn, cross-attn, FFN) and therefore six modulations:
    ``(γ_self, β_self, γ_cross, β_cross, γ_ffn, β_ffn)``.

    Why this works against the cross-attn collapse
    ----------------------------------------------
    Global FiLM (Levier D) modulates the input *once*; the network can
    still ignore the modulation if its self-attn / cross-attn / FFN
    happen to be locally invariant to it.  Per-block AdaLN injects the
    text signal at **every** sub-layer, so the gradient pathway from
    text → output exists 12 times for a 4-block stack — the network
    cannot satisfy the diffusion objective without picking some of
    those signals up, and that is what breaks the posterior collapse.

    Init strategy
    -------------
    Same as Levier D: small non-zero ``std=0.02`` on the projection
    weight, zeros on the bias.  At init the modulation is ``γ ≈ 0``
    and ``β ≈ 0`` (so the block behaves like a standard transformer
    block) but the gradient through the cond pathway is non-zero from
    epoch 1.
    """

    NUM_MODULATIONS: int = 6  # γ, β for self-attn / cross-attn / FFN

    def __init__(
        self, condDim: int, embedDim: int, initStd: float = 0.02
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(condDim)
        self.proj = nn.Linear(condDim, self.NUM_MODULATIONS * embedDim)
        nn.init.normal_(self.proj.weight, std=initStd)
        nn.init.zeros_(self.proj.bias)
        self.embedDim = embedDim

    def forward(
        self, condition: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return the 6 modulation tensors, each shape ``(B, embedDim)``.

        Parameters
        ----------
        condition : torch.Tensor
            Conditioning vector of shape ``(B, condDim)``.  Typically
            the concatenation of the masked-mean-pooled text encoder
            output and the sinusoidal timestep embedding.
        """
        modulations = self.proj(self.norm(condition))
        gammaSelf, betaSelf, gammaCross, betaCross, gammaFfn, betaFfn = (
            modulations.chunk(self.NUM_MODULATIONS, dim=-1)
        )
        return (
            gammaSelf,
            betaSelf,
            gammaCross,
            betaCross,
            gammaFfn,
            betaFfn,
        )


# ---------------------------------------------------------------------
# FiLM conditioning shortcut (Phase D Levier D, 2026-05-07)
# ---------------------------------------------------------------------
class _TextFilmConditioning(nn.Module):
    """Multiplicative text+timestep modulation of the per-frame features.

    Why
    ---
    The cross-attention path can collapse silently — when the diffusion
    task is solvable from motion alone, the cross-attn residual
    ``x = x + dropout(attended)`` ends up with ``attended ≈ 0`` and the
    text stops reaching the prediction.  Diagnostic on the post-Levier-B
    225-epoch ACCAD run confirmed this: ``cfg_sim ≈ 0.9996`` even
    though the encoder produced clearly different cond/uncond
    embeddings (sim 0.47).

    FiLM (Feature-wise Linear Modulation) wraps the per-frame features
    in a multiplicative + additive transform driven by the
    conditioning vector.  Because it multiplies the features, it
    cannot be neutralised the way an additive cross-attn residual can:
    the network cannot make ``γ`` exactly equal to a constant for all
    samples without losing the diffusion signal too.

    Implementation
    --------------
    * Conditioning vector is the concatenation of ``text_pooled`` and
      ``timestep_embed`` — the standard MDM/DiT convention.
    * Output of the MLP is split into ``γ_offset`` and ``β`` of shape
      ``(B, embedDim)``, broadcast across the frame axis.
    * Applied as ``h * (1 + γ_offset) + β`` so at zero output the
      transform is the identity (numerically safe), but the small
      non-zero init lets the FiLM perturb from epoch 1 — that
      perturbation is exactly the gradient pathway we want, and a
      pure zero-init (DiT AdaLN-zero style) would just reproduce the
      collapse problem we are trying to escape.
    """

    def __init__(
        self,
        textDim: int,
        timestepDim: int,
        embedDim: int,
        dropout: float = 0.0,
        initStd: float = 0.02,
    ) -> None:
        super().__init__()
        condDim = textDim + timestepDim
        hidden = 4 * embedDim
        self.conditionNorm = nn.LayerNorm(condDim)
        self.mlp = nn.Sequential(
            nn.Linear(condDim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * embedDim),
        )
        nn.init.normal_(self.mlp[-1].weight, std=initStd)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        frameH: torch.Tensor,
        textPooled: torch.Tensor,
        timestepEmbed: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate ``frameH`` of shape ``(B, F, D)`` by FiLM.

        Parameters
        ----------
        frameH : torch.Tensor
            Per-frame features after the input projection + timestep
            injection.  Shape ``(B, F, embedDim)``.
        textPooled : torch.Tensor
            Mean-pooled text features over real tokens.  Shape
            ``(B, textDim)``.
        timestepEmbed : torch.Tensor
            Sinusoidal-MLP embedding of the diffusion timestep.  Shape
            ``(B, timestepDim)``.
        """
        cond = torch.cat([textPooled, timestepEmbed], dim=-1)
        cond = self.conditionNorm(cond)
        gammaBeta = self.mlp(cond)  # (B, 2 * embedDim)
        gammaOffset, beta = gammaBeta.chunk(2, dim=-1)
        gammaOffset = gammaOffset.unsqueeze(1)  # (B, 1, embedDim)
        beta = beta.unsqueeze(1)
        return frameH * (1.0 + gammaOffset) + beta


# ---------------------------------------------------------------------
# Alignment head (Phase D.1, Levier B 2026-05-07)
# ---------------------------------------------------------------------
class _MotionTextAlignmentHead(nn.Module):
    """SimCLR / CLIP-style projection heads for both modalities.

    Takes the post-block motion features ``frameH`` of shape
    ``(B, F, embedDim)`` and the text encoder hidden states of shape
    ``(B, T, textDim)``.  Each modality is pooled (masked-mean) and
    pushed through its own 2-layer MLP into a shared smaller
    ``alignmentDim`` (default 128).  Both outputs are L2-normalized so
    the downstream InfoNCE loss operates on a unit sphere where the
    dot product equals the cosine similarity.

    Why a separate projection head per modality
    -------------------------------------------
    The pre-Levier-B head only projected motion; text was used raw via
    a free-standing ``poolTextEmbedding``.  The asymmetry forced the
    contrastive loss to align two spaces with very different
    statistics, and the model bypassed the difficulty by collapsing
    the motion projection to a near-constant vector — InfoNCE then sat
    at exactly ``log(B)`` (random chance) across the entire run.

    SimCLR / CLIP solved this by giving each modality its own
    projection MLP into a small shared space.  The MLP capacity lets
    each modality learn the discrimination it needs; the bottleneck
    (smaller alignmentDim) prevents trivial solutions from carrying
    enough information to satisfy the loss.
    """

    def __init__(
        self,
        motionDim: int,
        textDim: int,
        alignmentDim: int,
    ) -> None:
        super().__init__()
        self.motionProjection = nn.Sequential(
            nn.LayerNorm(motionDim),
            nn.Linear(motionDim, alignmentDim),
            nn.GELU(),
            nn.Linear(alignmentDim, alignmentDim),
        )
        self.textProjection = nn.Sequential(
            nn.LayerNorm(textDim),
            nn.Linear(textDim, alignmentDim),
            nn.GELU(),
            nn.Linear(alignmentDim, alignmentDim),
        )

    @staticmethod
    def _maskedMean(
        sequence: torch.Tensor,
        keyPaddingMask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Masked mean over the time/token axis (axis 1)."""
        if keyPaddingMask is None:
            return sequence.mean(dim=1)
        realMask = (
            (~keyPaddingMask).to(sequence.dtype).unsqueeze(-1)
        )
        return (sequence * realMask).sum(dim=1) / torch.clamp(
            realMask.sum(dim=1), min=1.0
        )

    def projectMotion(
        self,
        frameH: torch.Tensor,
        motionKeyPaddingMask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool, project and L2-normalize the motion features."""
        pooled = self._maskedMean(frameH, motionKeyPaddingMask)
        projected = self.motionProjection(pooled)
        return torch.nn.functional.normalize(projected, dim=-1)

    def projectText(
        self,
        textHiddenStates: torch.Tensor,
        textKeyPaddingMask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool, project and L2-normalize the text hidden states."""
        pooled = self._maskedMean(textHiddenStates, textKeyPaddingMask)
        projected = self.textProjection(pooled)
        return torch.nn.functional.normalize(projected, dim=-1)

    def forward(
        self,
        frameH: torch.Tensor,
        motionKeyPaddingMask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Backward-compat alias for :meth:`projectMotion`.

        Existing callers that built the head with the pre-Levier-B
        single-projection signature can keep using ``head(frameH, mask)``
        unchanged — they just won't see the text projection.
        """
        return self.projectMotion(frameH, motionKeyPaddingMask)


# ---------------------------------------------------------------------
# Raw-pool alignment (Phase F iter-2, 2026-05-14)
# ---------------------------------------------------------------------
class _RawPoolAlignment(nn.Module):
    """Minimal-capacity contrastive head on raw masked-mean pools.

    Companion to :class:`_MotionTextAlignmentHead`.  Where the main head
    inserts a 2-layer MLP per modality (LayerNorm → Linear → GELU →
    Linear) that has enough capacity to amplify near-zero pool
    differences into a discriminative space, this auxiliary head is
    deliberately **identity on the text side**: the text path is just
    ``L2-normalize(pool)``.  No learnable parameters can warp the text
    space.

    Why this matters
    ----------------
    Diagnostic on a 49-epoch Phase-F run showed:
    * encoder pool cond↔uncond cos-sim = **0.9998** (raw pool collapsed)
    * alignment-head text cos-sim ≈ 0.3 (post-MLP, well separated)
    * cfg_sim = 1.0000 exactly (denoiser sees identical pools for
      cond vs null → identical outputs)

    The MLP in the main head provides a *loophole*: the contrastive
    loss is satisfied without the encoder learning a discriminative
    pool, and FiLM/AdaLN — which consume the **raw** pool — see no
    text variation.  Forcing a second contrastive on the raw pool
    (with no text-side capacity) closes the loophole.

    Motion side keeps a single bias-free Linear because the modalities
    have different dimensions and the projection has to map from
    ``embedDim`` (motion) to ``textDim`` (text).  A linear projection
    can rotate but cannot warp the unit sphere, so amplifying tiny
    raw-pool differences still requires the encoder pool to be
    discriminative.
    """

    def __init__(self, textDim: int, motionDim: int) -> None:
        super().__init__()
        # text side: pure identity — no learnable params.  Output dim
        # is ``textDim`` so the InfoNCE operates on the encoder's
        # native pool space.
        self.motionProjection = nn.Linear(motionDim, textDim, bias=False)

    def projectTextPool(self, pooled: torch.Tensor) -> torch.Tensor:
        """L2-normalize the raw text pool — identity then norm."""
        return torch.nn.functional.normalize(pooled, dim=-1)

    def projectMotionPool(self, pooled: torch.Tensor) -> torch.Tensor:
        """Linear-project the raw motion pool then L2-normalize."""
        return torch.nn.functional.normalize(
            self.motionProjection(pooled), dim=-1
        )


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
    motionEmbedding : torch.Tensor or None
        Pooled per-sample motion embedding of shape ``(B, embedDim)``
        consumed by the text↔motion contrastive loss (Phase D.1).  Only
        produced when the denoiser was built with ``alignmentEnabled``;
        otherwise ``None``.  This is **not** used by the sampler.
    """

    boneOutput: torch.Tensor
    globalOutput: torch.Tensor | None
    motionEmbedding: torch.Tensor | None = None
    # Phase F iter-2 — raw masked-mean pools, L2-normalized, consumed
    # by the auxiliary pool contrastive loss.  ``textPooledRaw`` is the
    # identity-projected encoder pool (no learnable params on text
    # side); ``motionPooledRaw`` is the bias-free Linear-projected pool
    # of the post-block motion features.  Both are ``None`` unless the
    # denoiser was built with ``auxPoolAlignmentEnabled=True``.
    textPooledRaw: torch.Tensor | None = None
    motionPooledRaw: torch.Tensor | None = None


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

        # Self-conditioning input projections (Analog Bits).  Built only
        # when enabled; project the *previous x0 estimate* and add it to
        # the corresponding tokens.  Zero-init so the network starts as if
        # self-conditioning were absent and learns to exploit it.
        if config.useSelfConditioning:
            self.selfCondBoneProj: nn.Module = nn.Linear(
                config.motionChannels, embedDim
            )
            nn.init.zeros_(self.selfCondBoneProj.weight)
            nn.init.zeros_(self.selfCondBoneProj.bias)
            if config.globalChannels > 0:
                self.selfCondGlobalProj: nn.Module = nn.Linear(
                    config.globalChannels, embedDim
                )
                nn.init.zeros_(self.selfCondGlobalProj.weight)
                nn.init.zeros_(self.selfCondGlobalProj.bias)
            else:
                self.selfCondGlobalProj = None  # type: ignore[assignment]
        else:
            self.selfCondBoneProj = None  # type: ignore[assignment]
            self.selfCondGlobalProj = None  # type: ignore[assignment]

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
        # Phase E — per-block AdaLN condition vector is the
        # concatenation of the masked-mean-pooled text encoder output
        # and the timestep embedding; both have width ``embedDim`` (the
        # text path projection in :attr:`textProjection` always maps to
        # ``embedDim``, even when ``textProjection`` is the identity).
        perBlockCondDim = (
            embedDim + config.effectiveTextEmbedDim
            if config.usePerBlockFilm
            else 0
        )
        self.blocks = nn.ModuleList(
            [
                DenoiserBlockV2(
                    embedDim=embedDim,
                    numHeads=config.numHeads,
                    ffnDim=ffnDim,
                    dropout=config.dropout,
                    usePerBlockFilm=config.usePerBlockFilm,
                    condDim=perBlockCondDim,
                    filmInitStd=config.filmInitStd,
                )
                for _ in range(config.numLayers)
            ]
        )
        self.outputNorm = nn.LayerNorm(embedDim)
        self.outputProjection = nn.Linear(embedDim, config.totalOutputDim)

        # Phase D.1 — alignment head for the text↔motion contrastive
        # loss.  Only built when explicitly enabled to keep older
        # checkpoints loadable without unexpected key mismatches.
        if config.alignmentEnabled:
            self.alignmentHead: nn.Module = _MotionTextAlignmentHead(
                motionDim=embedDim,
                textDim=config.effectiveTextEmbedDim,
                alignmentDim=config.alignmentProjectionDim,
            )
        else:
            self.alignmentHead = nn.Identity()

        # Phase F iter-2 (2026-05-14) — auxiliary raw-pool alignment.
        # Only the motion side carries learnable parameters (a single
        # bias-free Linear); the text side is identity + L2-norm.
        if config.auxPoolAlignmentEnabled:
            self.auxPoolAlignment: nn.Module = _RawPoolAlignment(
                textDim=config.effectiveTextEmbedDim,
                motionDim=embedDim,
            )
        else:
            self.auxPoolAlignment = nn.Identity()

        # Phase D.4 (Levier D) — FiLM conditioning shortcut on
        # ``(text_pooled, timestep_embed)``.  Built only when enabled.
        if config.useFilmConditioning:
            self.filmConditioning: nn.Module = _TextFilmConditioning(
                textDim=config.effectiveTextEmbedDim,
                timestepDim=embedDim,
                embedDim=embedDim,
                dropout=config.filmDropout,
                initStd=config.filmInitStd,
            )
        else:
            self.filmConditioning = nn.Identity()

        # 2026-06-01 — generated-motion (x0) alignment encoder.  Owned by
        # the denoiser so it rides the existing optimizer / EMA / save
        # plumbing, but it is fed the reconstructed x0 from the training
        # loop (not the forward pass) — see ``encodeGeneratedMotion``.
        if config.x0AlignmentEnabled:
            self.x0AlignmentEncoder: nn.Module = MotionAlignmentEncoder(
                MotionAlignmentEncoderConfig(
                    numBones=config.numBones,
                    motionChannels=config.motionChannels,
                    embedDim=config.x0AlignmentEmbedDim,
                    numLayers=config.x0AlignmentNumLayers,
                    numHeads=config.x0AlignmentNumHeads,
                    alignmentDim=config.alignmentProjectionDim,
                    textDim=config.effectiveTextEmbedDim,
                    maxFrames=config.maxFrames,
                    dropout=config.dropout,
                )
            )
        else:
            self.x0AlignmentEncoder = nn.Identity()

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
    # Generated-motion (x0) alignment — see motion_alignment_encoder.py
    # ------------------------------------------------------------------
    def encodeGeneratedMotion(
        self,
        rotation6d: torch.Tensor,
        motionKeyPaddingMask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed a reconstructed-x0 motion into the alignment space.

        Raises if the x0 alignment encoder was not built (the caller
        must gate on ``config.x0AlignmentEnabled``).
        """
        if not self._config.x0AlignmentEnabled:
            raise RuntimeError(
                "x0 alignment encoder is disabled; build the denoiser "
                "with x0AlignmentEnabled=True to use it."
            )
        return self.x0AlignmentEncoder.encodeMotion(
            rotation6d, motionKeyPaddingMask
        )

    def projectGeneratedMotionText(
        self, pooledText: torch.Tensor
    ) -> torch.Tensor:
        """Project pooled text into the x0 alignment space."""
        if not self._config.x0AlignmentEnabled:
            raise RuntimeError(
                "x0 alignment encoder is disabled; build the denoiser "
                "with x0AlignmentEnabled=True to use it."
            )
        return self.x0AlignmentEncoder.projectText(pooledText)

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
        selfCondBone: torch.Tensor | None = None,
        selfCondGlobal: torch.Tensor | None = None,
    ) -> DenoiserOutput:
        """Predict clean (or v-target) motion features from noisy input.

        ``selfCondBone`` / ``selfCondGlobal`` are the (detached) previous
        x0 estimate used for self-conditioning; ignored unless the
        denoiser was built with ``useSelfConditioning=True``.  ``None``
        is treated as a zero estimate (the first-pass / disabled case).
        """
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
        if self.selfCondBoneProj is not None:
            estimate = (
                selfCondBone
                if selfCondBone is not None
                else torch.zeros_like(noisyMotion)
            )
            boneTokens = boneTokens + self.selfCondBoneProj(estimate)
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
            if self.selfCondGlobalProj is not None:
                estimateGlobal = (
                    selfCondGlobal
                    if selfCondGlobal is not None
                    else torch.zeros_like(noisyGlobalFeatures)
                )
                globalH = globalH + self.selfCondGlobalProj(estimateGlobal)
            frameInput = torch.cat([boneFlat, globalH], dim=-1)
        else:
            frameInput = boneFlat
        frameH = self.frameProj(frameInput)  # (B, F, embedDim)

        # --- Inject timestep ----------------------------------------
        # Single broadcast addition: every frame gets the same scalar
        # timestep modulation.  This replaces the legacy v1 FiLM path
        # that conditioned on the diffusion step alone.
        timestepH = self.timestepEmbed(timesteps)  # (B, embedDim)
        frameH = frameH + timestepH.unsqueeze(1)

        # --- Pool text once (re-used by FiLM + per-block AdaLN) -----
        # When either Levier D (global FiLM) or Levier E (per-block
        # AdaLN) is enabled we need a (B, textDim) text summary; both
        # paths pool the same way so we compute it once and share.
        needPooledText = (
            self._config.useFilmConditioning
            or self._config.usePerBlockFilm
            or self._config.auxPoolAlignmentEnabled
        )
        textPooled = (
            self._poolMaskedMean(textHiddenStates, textKeyPaddingMask)
            if needPooledText
            else None
        )
        # 2026-05-28 — L2-normalise the pool so cond and uncond branches
        # enter FiLM / AdaLN / aux-pool with the same magnitude.  Per-token
        # outputs are already L2-normed inside the text encoder, but the
        # masked mean of unit vectors has norm < 1.0 for cond (many tokens
        # pointing in different directions) while uncond holds a single
        # learnable token of norm 1.0 — the resulting magnitude gap
        # (~0.49 vs 0.755 measured at epoch 215) is what CFG amplifies
        # into mode collapse.  Forcing both onto the unit sphere makes
        # the (cond - uncond) direction informative rather than scale-
        # driven.
        if textPooled is not None:
            textPooled = F.normalize(textPooled, p=2.0, dim=-1, eps=1e-8)

        # --- Phase D Levier D — global FiLM shortcut ----------------
        if self._config.useFilmConditioning and textPooled is not None:
            frameH = self.filmConditioning(
                frameH, textPooled, timestepH
            )

        # --- Add temporal positional encoding -----------------------
        frameH = self.posEncoder(frameH)
        frameH = self.embedDropout(frameH)

        # --- Project text once -------------------------------------
        textH = self.textProjection(textHiddenStates)

        # --- Phase E Levier E — per-block AdaLN condition vector ----
        # Same source as the global FiLM (text_pooled, timestep_embed)
        # but produced once and passed identically to every block; the
        # per-block AdaLN MLPs differ so each block learns its own
        # conditioning style.
        perBlockCondition: torch.Tensor | None = None
        if self._config.usePerBlockFilm and textPooled is not None:
            perBlockCondition = torch.cat(
                [textPooled, timestepH], dim=-1
            )

        # --- Transformer stack ---------------------------------------
        for block in self.blocks:
            frameH = block(
                frameH,
                textHiddenStates=textH,
                textKeyPaddingMask=textKeyPaddingMask,
                motionKeyPaddingMask=motionKeyPaddingMask,
                condition=perBlockCondition,
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

        # Phase D.1 — pooled and projected motion embedding for the
        # contrastive alignment loss.  Built from the post-block
        # ``frameH`` so the gradient flows back through every
        # cross-attention layer.  L2-normalized inside the head so the
        # downstream InfoNCE consumes unit-sphere vectors directly.
        motionEmbedding: torch.Tensor | None = None
        if self._config.alignmentEnabled:
            motionEmbedding = self.alignmentHead.projectMotion(
                frameH, motionKeyPaddingMask
            )

        # Phase F iter-2 — auxiliary raw-pool projections for the
        # aux contrastive loss.  Text path is identity + L2-norm so the
        # encoder pool itself must discriminate prompts; motion path is
        # a bias-free Linear from embedDim to textDim then L2-norm.
        textPooledRaw: torch.Tensor | None = None
        motionPooledRaw: torch.Tensor | None = None
        if (
            self._config.auxPoolAlignmentEnabled
            and textPooled is not None
        ):
            textPooledRaw = self.auxPoolAlignment.projectTextPool(
                textPooled
            )
            motionPooledMean = self._poolMaskedMean(
                frameH, motionKeyPaddingMask
            )
            motionPooledRaw = self.auxPoolAlignment.projectMotionPool(
                motionPooledMean
            )

        return DenoiserOutput(
            boneOutput=boneOutput,
            globalOutput=globalOutput,
            motionEmbedding=motionEmbedding,
            textPooledRaw=textPooledRaw,
            motionPooledRaw=motionPooledRaw,
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
    # Pool helper (shared by FiLM + alignment head)
    # ------------------------------------------------------------------
    @staticmethod
    def _poolMaskedMean(
        sequence: torch.Tensor,
        keyPaddingMask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Masked-mean pool ``(B, T, D) → (B, D)``.

        Padded positions (where ``keyPaddingMask == True``) are
        excluded.  Used by the FiLM conditioning to consume the text
        encoder hidden states without spending a separate projection.
        """
        if keyPaddingMask is None:
            return sequence.mean(dim=1)
        realMask = (
            (~keyPaddingMask).to(sequence.dtype).unsqueeze(-1)
        )
        return (sequence * realMask).sum(dim=1) / torch.clamp(
            realMask.sum(dim=1), min=1.0
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
        # Self-conditioning projections must start at zero so the network
        # behaves identically to a no-self-cond denoiser at init and then
        # learns to use the previous x0 estimate (re-zeroed here because
        # the generic Xavier loop above would otherwise overwrite them).
        if self._config.useSelfConditioning:
            for projection in (self.selfCondBoneProj, self.selfCondGlobalProj):
                if isinstance(projection, nn.Linear):
                    nn.init.zeros_(projection.weight)
                    nn.init.zeros_(projection.bias)
