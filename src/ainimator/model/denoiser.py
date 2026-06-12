"""Diffusion denoising network for motion generation."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from ainimator.model.layers.normalization import AdaLN, FiLM
from ainimator.model.layers.spatial_gcn import SpatialGCNBlock


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding for diffusion models.

    Converts scalar timesteps to high-dimensional embeddings.
    """

    def __init__(self, embedDim: int, maxPeriod: int = 10000) -> None:
        """
        Initialize TimestepEmbedding.

        Parameters
        ----------
        embedDim : int
            Dimension of the output embedding.
        maxPeriod : int, optional
            Maximum period for sinusoidal frequencies, by default 10000.
        """
        super().__init__()
        self.embedDim = embedDim
        self.maxPeriod = maxPeriod
        self.mlp = nn.Sequential(
            nn.Linear(embedDim, embedDim * 4),
            nn.SiLU(),
            nn.Linear(embedDim * 4, embedDim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps into sinusoidal representations.

        Parameters
        ----------
        timesteps : torch.Tensor
            Tensor of timesteps shaped (batch_size,).

        Returns
        -------
        torch.Tensor
            Timestep embeddings shaped (batch_size, embedDim).
        """
        half = self.embedDim // 2
        freqs = torch.exp(
            -math.log(self.maxPeriod)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedDim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return self.mlp(embedding)


class SinusoidalPositionalEncoding(nn.Module):
    """Additive sinusoidal positional encoding (MDM-style)."""

    def __init__(self, embedDim: int, dropout: float = 0.1, maxLen: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(maxLen, embedDim)
        position = torch.arange(0, maxLen, dtype=torch.float32).unsqueeze(1)
        divTerm = torch.exp(
            torch.arange(0, embedDim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedDim)
        )
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        # (1, maxLen, embedDim) for batch_first usage
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding: x shape (batch, seqLen, dim)."""
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class DenoiserBlock(nn.Module):
    """
    Single denoising transformer block (MDM-style) with cross-attention.

    Architecture:
      1. Self-attention residual (motion ↔ motion).
      2. FiLM conditioning (timestep modulation).
      3. Cross-attention residual (motion → text tokens).
      4. Feed-forward residual.

    The conditioning is split: timestep goes through FiLM (a global scalar
    modulation, well-suited to a per-sample scalar) and text tokens go
    through cross-attention (each motion frame attends to each prompt
    token, the standard MDM/MotionDiffuse design).  Before this refactor
    text was pooled into a single vector and mixed into ``cond`` for FiLM,
    which provided no per-token attention and led to posterior collapse —
    the network ignored the prompt and produced the dataset's mean
    distribution regardless of input.
    """

    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        condDim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedDim)
        self.attention = nn.MultiheadAttention(
            embedDim, numHeads, dropout=dropout, batch_first=True,
        )
        self.filmCondition = FiLM(embedDim, condDim)
        # Cross-attention onto text tokens.  query=motion, key=value=text.
        # normCross is applied to the query (pre-norm transformer design,
        # consistent with norm1/norm2).
        self.normCross = nn.LayerNorm(embedDim)
        self.crossAttention = nn.MultiheadAttention(
            embedDim, numHeads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embedDim)
        self.ffn = nn.Sequential(
            nn.Linear(embedDim, embedDim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedDim * 4, embedDim),
        )
        # adalnCondition kept dead for checkpoint compatibility — see the
        # historical comment in ``forward`` below.  Its weights are zero on
        # fresh init and ignored on resume.
        self.adalnCondition = AdaLN(embedDim, condDim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        textTokens: Optional[torch.Tensor] = None,
        textKeyPaddingMask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DenoiserBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped (batch_size, seq_len, embedDim).
        cond : torch.Tensor
            Conditioning tensor shaped (batch_size, [seq_len,] condDim).
            Carries the timestep (and optionally per-frame PE) modulation
            consumed by FiLM.
        textTokens : Optional[torch.Tensor]
            Projected text token sequence shaped
            (batch_size, textSeqLen, embedDim).  When None, cross-attention
            is skipped (backward-compat for ablations or pretraining).
        textKeyPaddingMask : Optional[torch.Tensor]
            Key padding mask aligned with ``textTokens`` (True = pad),
            shaped (batch_size, textSeqLen).
        mask : Optional[torch.Tensor], optional
            Self-attention key padding mask (True = pad).

        Returns
        -------
        torch.Tensor
            Output tensor shaped (batch_size, seq_len, embedDim).
        """
        # 1. Self-attention with residual + FiLM conditioning.
        # FiLM applies ``x * (1 + gamma) + beta`` which is residual-safe at
        # zero-init (gamma=0, beta=0 -> identity), so the attention residual
        # keeps its magnitude.
        h = self.norm1(x)
        h, _ = self.attention(
            h, h, h, key_padding_mask=mask, need_weights=False,
        )
        h = x + self.dropout(h)
        h = self.filmCondition(h, cond)

        # 2. Cross-attention residual onto text tokens.
        # Pre-norm on the query, residual add on the original h so the
        # block is identity-safe at init (the cross-attn out_proj receives
        # gradient and pulls toward useful text alignment over training).
        if textTokens is not None:
            qCross = self.normCross(h)
            crossOut, _ = self.crossAttention(
                query=qCross,
                key=textTokens,
                value=textTokens,
                key_padding_mask=textKeyPaddingMask,
                need_weights=False,
            )
            h = h + self.dropout(crossOut)

        # 3. Feed-forward residual with pre-LayerNorm.
        h = h + self.dropout(self.ffn(self.norm2(h)))

        # IMPORTANT: the AdaLN module is intentionally NOT applied here.
        # AdaLN returns ``layernorm(h) * (1 + gamma) + beta`` which, applied
        # as the final op of the block, *replaces* h with its normalised
        # version and **breaks the residual chain** — the per-frame magnitude
        # carried by ``x`` is erased at every block, stacking to a total
        # collapse over ``numLayers`` blocks.  The zero-init of DiT's
        # AdaLN-Zero is meant to be *inside* a residual (``x + scale * f(x)``
        # with ``scale`` zero-init), not to replace the block output.
        # The AdaLN module is kept in ``__init__`` for checkpoint
        # compatibility; its weights remain at zero when loaded fresh and are
        # ignored when resuming from checkpoints trained before this fix.
        # h = self.adalnCondition(h, cond)  # structural bug — see comment

        return h
class SpatioTemporalMixBlock(nn.Module):
    """
    Local spatio-temporal mixing over (frames, bones) right after split.
    """

    def __init__(self, embedDim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.depthwiseConv = nn.Conv2d(
            in_channels=embedDim,
            out_channels=embedDim,
            kernel_size=(3, 3),
            padding=(1, 1),
            groups=embedDim,
        )
        self.pointwiseConv = nn.Conv2d(
            in_channels=embedDim,
            out_channels=embedDim,
            kernel_size=1,
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedDim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply depthwise separable 2D conv on temporal and skeletal axes.
        """
        h = x.permute(0, 3, 1, 2)
        h = self.depthwiseConv(h)
        h = self.activation(h)
        h = self.pointwiseConv(h)
        h = h.permute(0, 2, 3, 1)
        h = self.dropout(h)
        h = self.norm(h)
        return x + h


class MotionDenoiser(nn.Module):
    """
    Main diffusion denoiser network for motion generation.

    Takes noisy motion features, text embedding, and timestep to predict
    clean motion (x0). Supports bone-scoped features (per bone per frame)
    and optional global-scoped features (per frame).
    """

    def __init__(
        self,
        embedDim: int = 64,
        numHeads: int = 4,
        numLayers: int = 6,
        numBones: int = 65,
        motionChannels: int = 6,
        globalChannels: int = 0,
        dropout: float = 0.1,
        numSpatialLayers: int = 1,
        numSpatioTemporalLayers: int = 1,
        textEmbedDim: Optional[int] = None,
    ) -> None:
        """
        Initialize MotionDenoiser.

        Parameters
        ----------
        embedDim : int, optional
            Hidden dimension, by default 64.
        numHeads : int, optional
            Number of attention heads, by default 4.
        numLayers : int, optional
            Number of denoising blocks, by default 6.
        numBones : int, optional
            Number of skeleton bones, by default 65.
        motionChannels : int, optional
            Total channels per bone for all bone-scoped features, by default 6.
        globalChannels : int, optional
            Total channels for all global-scoped features, by default 0.
        dropout : float, optional
            Dropout rate, by default 0.1.
        numSpatialLayers : int, optional
            Number of spatial GCN blocks, by default 1.
        numSpatioTemporalLayers : int, optional
            Number of local spatio-temporal mixing blocks.
        textEmbedDim : Optional[int], optional
            Dimension of the incoming text embedding before projection.
            Defaults to ``embedDim`` when omitted.
        """
        super().__init__()
        if embedDim % numHeads != 0:
            raise ValueError(
                "Generation embedDim must be divisible by numHeads "
                f"(embedDim={embedDim}, numHeads={numHeads})."
            )
        self.embedDim = embedDim
        self.textEmbedDim = (
            embedDim if textEmbedDim is None else int(textEmbedDim)
        )
        self.numBones = numBones
        self.motionChannels = motionChannels
        self.globalChannels = globalChannels

        # Input projections
        self.boneProj = nn.Linear(motionChannels, embedDim)
        frameProjInputDim = numBones * embedDim
        if globalChannels > 0:
            self.globalProj: Optional[nn.Linear] = nn.Linear(globalChannels, embedDim)
            frameProjInputDim += embedDim
        else:
            self.globalProj = None
        self.frameProj = nn.Linear(frameProjInputDim, embedDim)
        self.textProj = nn.Linear(self.textEmbedDim, embedDim)
        self.textAdapter = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim * 2),
            nn.SiLU(),
            nn.Linear(embedDim * 2, embedDim),
        )
        # Token-level projection used by cross-attention in DenoiserBlock.
        # Maps the raw XLM-R last_hidden_state (shape (B, seqLen, textHiddenSize))
        # into the denoiser's embedding space (B, seqLen, embedDim) so each
        # motion frame can attend to each prompt token directly.  Before this
        # refactor only the pooled vector reached the denoiser, which made
        # the network ignore the prompt (posterior collapse — see plan).
        self.textTokenProj = nn.Sequential(
            nn.Linear(self.textEmbedDim, embedDim),
            nn.LayerNorm(embedDim),
        )

        # Conditioning embeddings
        self.timestepEmbed = TimestepEmbedding(embedDim)

        # Conditioning dimension: timestep
        condDim = embedDim

        # Sinusoidal positional encoding for temporal frame ordering (MDM-style)
        self.sequencePosEncoder = SinusoidalPositionalEncoding(
            embedDim, dropout=dropout,
        )

        # Spatial blocks (GCN over bones per frame)
        self.spatialBlocks = nn.ModuleList(
            [
                SpatialGCNBlock(
                    numBones=numBones,
                    embedDim=embedDim,
                    dropout=dropout,
                )
                for _ in range(numSpatialLayers)
            ]
        )

        self.spatioTemporalBlocks = nn.ModuleList(
            [
                SpatioTemporalMixBlock(embedDim=embedDim, dropout=dropout)
                for _ in range(numSpatioTemporalLayers)
            ]
        )

        # Denoising blocks
        self.blocks = nn.ModuleList([
            DenoiserBlock(embedDim, numHeads, condDim, dropout)
            for _ in range(numLayers)
        ])

        # Output projection: bone features + global features
        outputDim = numBones * motionChannels + globalChannels
        self.outputNorm = nn.LayerNorm(embedDim)
        self.outputProj = nn.Linear(embedDim, outputDim)

        # Per-frame residual skip from the pre-transformer motion embedding
        # straight to the output.  When self-attention collapses to a
        # mean-pose representation, this path preserves the per-frame
        # structure that was present in ``motionH`` before the blocks.
        #
        # Previously zero-initialised so a resumed checkpoint kept its old
        # behaviour — but that means a FRESH run has *no* live path from
        # ``noisyMotion`` to ``output`` until the zero-init weights pick up
        # a gradient, which for an overfit run delays learning by thousands
        # of steps and lets the transformer collapse to a mean pose before
        # the skip ever becomes useful.  A small Xavier init gives a
        # non-trivial (but still small enough to not dominate) per-frame
        # path from step 0.  Resumed checkpoints override this init via
        # ``load_state_dict``, so production runs keep their trained skip.
        self.motionSkipProj = nn.Linear(embedDim, outputDim)
        nn.init.xavier_uniform_(self.motionSkipProj.weight, gain=1e-2)
        nn.init.zeros_(self.motionSkipProj.bias)

    def forward(
        self,
        noisyMotion: torch.Tensor,
        textEmbedding: torch.Tensor,
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        noisyGlobalFeatures: Optional[torch.Tensor] = None,
        textTokens: Optional[torch.Tensor] = None,
        textTokenMask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict clean motion features from noisy input.

        Parameters
        ----------
        noisyMotion : torch.Tensor
            Noisy bone-scoped features shaped (batch, frames, bones, motionChannels).
        textEmbedding : torch.Tensor
            Pooled text embedding shaped (batch, textEmbedDim).  Used for the
            global FiLM modulation channel and the conditioning prepend token.
        timesteps : torch.Tensor
            Diffusion timesteps shaped (batch,).
        mask : Optional[torch.Tensor], optional
            Motion temporal key padding mask (True = pad).
        noisyGlobalFeatures : Optional[torch.Tensor], optional
            Noisy global-scoped features shaped (batch, frames, globalChannels).
        textTokens : Optional[torch.Tensor], optional
            Per-token text representation shaped
            (batch, textSeqLen, textEmbedDim) — usually the last_hidden_state
            from the frozen XLM-Roberta encoder.  When provided, every
            DenoiserBlock applies a cross-attention residual onto these tokens
            (motion → text).  When ``None``, the cross-attention path is
            skipped (legacy / ablation behaviour).
        textTokenMask : Optional[torch.Tensor], optional
            Attention mask for ``textTokens`` (1 = valid, 0 = pad), shaped
            (batch, textSeqLen).  Internally inverted to the ``True = pad``
            convention required by ``nn.MultiheadAttention``.

        Returns
        -------
        tuple[torch.Tensor, Optional[torch.Tensor]]
            Predicted bone features shaped (batch, frames, bones, motionChannels)
            and predicted global features shaped (batch, frames, globalChannels)
            or None when globalChannels is 0.
        """
        batch, frames, bones, channels = noisyMotion.shape

        # Bone-wise projection, then spatial and spatio-temporal mixing.
        boneH = self.boneProj(noisyMotion)
        for block in self.spatialBlocks:
            boneH = block(boneH)
        for block in self.spatioTemporalBlocks:
            boneH = block(boneH)

        # Flatten per-frame features after spatial mixing.
        motionH = boneH.reshape(batch, frames, bones * self.embedDim)

        # Incorporate global features when present.
        if self.globalProj is not None and noisyGlobalFeatures is not None:
            globalH = self.globalProj(noisyGlobalFeatures)
            motionH = torch.cat([motionH, globalH], dim=-1)

        motionH = self.frameProj(motionH)
        textH = self.textProj(textEmbedding)
        textH = textH + self.textAdapter(textH)

        # Add sinusoidal positional encoding so the model knows frame order.
        # Without this, self-attention is permutation-equivariant and
        # cannot learn any temporal structure (MDM's key design choice).
        h = self.sequencePosEncoder(motionH)

        # Build conditioning token: timestep + text (MDM-style prepend).
        condToken = self.timestepEmbed(timesteps) + textH  # (batch, embedDim)

        # Build PER-FRAME conditioning for FiLM/AdaLN.  Previously the same
        # ``cond`` vector was broadcast to every token, so every frame
        # received identical gamma/beta and the only way to produce
        # per-frame variation was via self-attention on PE.  On a
        # pure-transformer overfit run (no spatial/spatio-temporal
        # blocks) that collapses to a mean-pose output.  By adding the
        # sinusoidal PE to the conditioning path itself we give each
        # frame its own modulation; FiLM/AdaLN already broadcast over
        # any leading dims, so a (B, 1+F, embedDim) cond works as-is and
        # stays backward-compatible with resumed checkpoints.
        framePE = self.sequencePosEncoder.pe[:, :frames, :].to(h.dtype)
        condFrames = condToken.unsqueeze(1) + framePE  # (batch, frames, embedDim)
        condSeq = torch.cat([condToken.unsqueeze(1), condFrames], dim=1)  # (B, 1+F, embedDim)

        # Prepend conditioning token to the sequence (MDM-style).
        xseq = torch.cat([condToken.unsqueeze(1), h], dim=1)  # (batch, 1+frames, embedDim)

        # Extend mask for the prepended conditioning token (always valid).
        if mask is not None:
            condMask = torch.zeros(
                batch, 1, dtype=torch.bool, device=mask.device,
            )
            mask = torch.cat([condMask, mask], dim=1)

        # Project text tokens for cross-attention (motion → tokens).  Each
        # block takes the same projected sequence; the projection lives at
        # the model level (not per-block) so the parameter count stays
        # bounded and the same cross-attention "vocabulary" is shared
        # across depths.  textKeyPaddingMask follows nn.MultiheadAttention's
        # convention: True = pad.  ``textTokenMask`` from the dataloader is
        # 1=valid / 0=pad, so we invert it.
        projectedTextTokens: Optional[torch.Tensor] = None
        textKeyPaddingMask: Optional[torch.Tensor] = None
        if textTokens is not None:
            projectedTextTokens = self.textTokenProj(textTokens)
            if textTokenMask is not None:
                textKeyPaddingMask = ~textTokenMask.bool()

        # Apply denoising blocks
        for block in self.blocks:
            xseq = block(
                xseq,
                condSeq,
                textTokens=projectedTextTokens,
                textKeyPaddingMask=textKeyPaddingMask,
                mask=mask,
            )

        # Remove the conditioning token from the output.
        h = xseq[:, 1:]

        # Output projection.  The motionSkip path is zero-initialised and
        # feeds the pre-transformer motion embedding directly into the
        # output so per-frame structure is preserved even when the
        # attention output is temporally flat.
        h = self.outputNorm(h)
        output = self.outputProj(h) + self.motionSkipProj(motionH)

        # Split into bone and global predictions.
        boneFlatDim = bones * self.motionChannels
        boneOutput = output[..., :boneFlatDim].view(batch, frames, bones, self.motionChannels)
        globalOutput: Optional[torch.Tensor] = None
        if self.globalChannels > 0:
            globalOutput = output[..., boneFlatDim:]

        return boneOutput, globalOutput
