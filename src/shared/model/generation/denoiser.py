"""Diffusion denoising network for motion generation."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from src.shared.model.layers.normalization import AdaLN, FiLM
from src.shared.model.layers.spatial_gcn import SpatialGCNBlock


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
    Single denoising transformer block (MDM-style).

    Self-attention with FiLM conditioning and a feed-forward transform,
    both wrapped in residual connections.
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
        self.norm2 = nn.LayerNorm(embedDim)
        self.ffn = nn.Sequential(
            nn.Linear(embedDim, embedDim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedDim * 4, embedDim),
        )
        self.adalnCondition = AdaLN(embedDim, condDim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DenoiserBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped (batch_size, seq_len, embedDim).
        cond : torch.Tensor
            Conditioning tensor shaped (batch_size, condDim).
        mask : Optional[torch.Tensor], optional
            Key padding mask (True = pad), by default None.

        Returns
        -------
        torch.Tensor
            Output tensor shaped (batch_size, seq_len, embedDim).
        """
        # Self-attention with residual + FiLM conditioning
        h = self.norm1(x)
        h, _ = self.attention(
            h, h, h, key_padding_mask=mask, need_weights=False,
        )
        h = x + self.dropout(h)
        h = self.filmCondition(h, cond)

        # Feed-forward with residual + AdaLN conditioning
        h = h + self.dropout(self.ffn(self.norm2(h)))
        h = self.adalnCondition(h, cond)

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

    Takes noisy motion, text embedding, and timestep to predict noise.
    """

    def __init__(
        self,
        embedDim: int = 64,
        numHeads: int = 4,
        numLayers: int = 6,
        numBones: int = 65,
        motionChannels: int = 6,
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
            Channels per bone (6D rotation), by default 6.
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

        # Input projections
        self.boneProj = nn.Linear(motionChannels, embedDim)
        self.frameProj = nn.Linear(numBones * embedDim, embedDim)
        self.textProj = nn.Linear(self.textEmbedDim, embedDim)
        self.textAdapter = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim * 2),
            nn.SiLU(),
            nn.Linear(embedDim * 2, embedDim),
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

        # Output projection
        self.outputNorm = nn.LayerNorm(embedDim)
        self.outputProj = nn.Linear(embedDim, numBones * motionChannels)

    def forward(
        self,
        noisyMotion: torch.Tensor,
        textEmbedding: torch.Tensor,
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict noise from noisy motion.

        Parameters
        ----------
        noisyMotion : torch.Tensor
            Noisy motion shaped (batch, frames, bones, 6).
        textEmbedding : torch.Tensor
            Text embedding shaped (batch, textEmbedDim).
        timesteps : torch.Tensor
            Diffusion timesteps shaped (batch,).
        mask : Optional[torch.Tensor], optional
            Temporal mask, by default None.

        Returns
        -------
        torch.Tensor
            Predicted noise shaped (batch, frames, bones, 6).
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
        motionH = self.frameProj(motionH)
        textH = self.textProj(textEmbedding)
        textH = textH + self.textAdapter(textH)

        # Add sinusoidal positional encoding so the model knows frame order.
        # Without this, self-attention is permutation-equivariant and
        # cannot learn any temporal structure (MDM's key design choice).
        h = self.sequencePosEncoder(motionH)

        # Build conditioning token: timestep + text (MDM-style prepend).
        condToken = self.timestepEmbed(timesteps) + textH  # (batch, embedDim)
        cond = condToken  # per-layer FiLM/AdaLN conditioning

        # Prepend conditioning token to the sequence (MDM-style).
        xseq = torch.cat([condToken.unsqueeze(1), h], dim=1)  # (batch, 1+frames, embedDim)

        # Extend mask for the prepended conditioning token (always valid).
        if mask is not None:
            condMask = torch.zeros(
                batch, 1, dtype=torch.bool, device=mask.device,
            )
            mask = torch.cat([condMask, mask], dim=1)

        # Apply denoising blocks
        for block in self.blocks:
            xseq = block(xseq, cond, mask)

        # Remove the conditioning token from the output.
        h = xseq[:, 1:]

        # Output projection
        h = self.outputNorm(h)
        output = self.outputProj(h)

        # Reshape to (batch, frames, bones, channels)
        return output.view(batch, frames, bones, channels)
