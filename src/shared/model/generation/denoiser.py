"""Diffusion denoising network for motion generation."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.shared.model.layers.attention import MultiHeadAttention
from src.shared.model.layers.normalization import AdaLN, FiLM
from src.shared.model.layers.positional import RoPE
from src.shared.model.layers.spatial_gcn import SpatialGCNBlock
from src.shared.model.layers.temporal import TemporalLayer
from src.shared.model.layers.transform import TransformLayer
from src.shared.constants.rotation import ROTATION_CHANNELS_ROT6D
from src.shared.types.generation import VALID_TAGS
from src.shared.types.network import (
    DEFAULT_SPATIOTEMPORAL_MODE,
    SPATIOTEMPORAL_MODE_FACTORIZED,
)


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


class TagEmbedding(nn.Module):
    """
    Learnable embedding for categorical motion tags.

    Maps the 9 valid tags to dense embeddings. Supports None/missing tags
    by using a learnable default embedding.
    """

    def __init__(self, embedDim: int) -> None:
        """
        Initialize TagEmbedding.

        Parameters
        ----------
        embedDim : int
            Dimension of the output embedding.
        """
        super().__init__()
        self.embedDim = embedDim
        self.numTags = len(VALID_TAGS)
        self.tagToIdx = {tag: idx for idx, tag in enumerate(VALID_TAGS)}
        # +1 for the default/no-tag embedding at index 0
        self.embedding = nn.Embedding(self.numTags + 1, embedDim)
        # Index 0 is reserved for "no tag" / None
        self.defaultIdx = self.numTags

    def forward(self, tags: Optional[list[Optional[str]]]) -> torch.Tensor:
        """
        Embed a batch of tag strings.

        Parameters
        ----------
        tags : Optional[list[Optional[str]]]
            List of tag strings, can contain None for missing tags.
            If the entire list is None, returns default embeddings.

        Returns
        -------
        torch.Tensor
            Tag embeddings shaped (batch_size, embedDim).
        """
        device = self.embedding.weight.device
        
        if tags is None:
            # Return a single default embedding (will be broadcast)
            return self.embedding(torch.tensor([self.defaultIdx], device=device))
        
        indices = []
        for tag in tags:
            if tag is None or tag == "":
                indices.append(self.defaultIdx)
            else:
                indices.append(self.tagToIdx.get(tag, self.defaultIdx))
        
        indexTensor = torch.tensor(indices, dtype=torch.long, device=device)
        return self.embedding(indexTensor)


class DenoiserBlock(nn.Module):
    """
    Single denoising transformer block.

    Combines TemporalLayer, RoPE, MHA, and TransformLayer with conditioning.
    """

    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        condDim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize DenoiserBlock.

        Parameters
        ----------
        embedDim : int
            Hidden dimension of the block.
        numHeads : int
            Number of attention heads.
        condDim : int
            Dimension of conditioning embeddings (tag + timestep).
        dropout : float, optional
            Dropout rate, by default 0.1.
        """
        super().__init__()
        self.temporal = TemporalLayer(embedDim, numHeads, dropout)
        self.filmCondition = FiLM(embedDim, condDim)
        self.rope = RoPE(embedDim)
        self.attention = MultiHeadAttention(embedDim, numHeads, dropout)
        self.transform = TransformLayer(embedDim, embedDim * 4, embedDim, dropout)
        self.adalnCondition = AdaLN(embedDim, condDim)
        self.norm = nn.LayerNorm(embedDim)

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
            Temporal mask, by default None.

        Returns
        -------
        torch.Tensor
            Output tensor shaped (batch_size, seq_len, embedDim).
        """
        # Temporal layer with FiLM conditioning
        h = self.temporal(x, mask)
        h = self.filmCondition(h, cond)

        # RoPE positional encoding
        h = self.rope(h)

        # Multi-head attention for temporal relationships
        h = h + self.attention(self.norm(h))

        # Transform layer with AdaLN conditioning
        h = self.adalnCondition(h, cond)
        h = self.transform(h)

        return h


class MotionDenoiser(nn.Module):
    """
    Main diffusion denoiser network for motion generation.

    Takes noisy motion, text embedding, tag, and timestep to predict noise.
    """

    def __init__(
        self,
        embedDim: int = 64,
        numHeads: int = 4,
        numLayers: int = 6,
        spatiotemporalMode: str = DEFAULT_SPATIOTEMPORAL_MODE,
        numBones: int = 65,
        motionChannels: int = ROTATION_CHANNELS_ROT6D,
        dropout: float = 0.1,
        numSpatialLayers: int = 1,
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
        spatiotemporalMode : str, optional
            Spatio-temporal strategy, by default "flat".
        numBones : int, optional
            Number of skeleton bones, by default 65.
        motionChannels : int, optional
            Channels per bone, by default 6.
        dropout : float, optional
            Dropout rate, by default 0.1.
        numSpatialLayers : int, optional
            Number of spatial GCN blocks, by default 1.
        """
        super().__init__()
        self.embedDim = embedDim
        self.numBones = numBones
        self.motionChannels = motionChannels
        self.spatiotemporalMode = spatiotemporalMode

        # Input projections
        self.boneProj = nn.Linear(motionChannels, embedDim)
        self.frameProj = nn.Linear(numBones * embedDim, embedDim)
        self.textProj = nn.Linear(embedDim, embedDim)

        # Conditioning embeddings
        self.timestepEmbed = TimestepEmbedding(embedDim)
        self.tagEmbed = TagEmbedding(embedDim)

        # Conditioning dimension: tag + timestep
        condDim = embedDim * 2

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

        # Factorized temporal blocks (per bone sequence).
        self.boneTemporalBlocks = nn.ModuleList([])
        if spatiotemporalMode == SPATIOTEMPORAL_MODE_FACTORIZED:
            self.boneTemporalBlocks = nn.ModuleList(
                [
                    DenoiserBlock(embedDim, numHeads, condDim, dropout)
                    for _ in range(numLayers)
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
        tags: Optional[list[Optional[str]]],
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict noise from noisy motion.

        Parameters
        ----------
        noisyMotion : torch.Tensor
            Noisy motion shaped (batch, frames, bones, channels).
        textEmbedding : torch.Tensor
            CLIP text embedding shaped (batch, embedDim).
        tags : Optional[list[Optional[str]]]
            List of tag strings for the batch. Can be None or contain None elements.
        timesteps : torch.Tensor
            Diffusion timesteps shaped (batch,).
        mask : Optional[torch.Tensor], optional
            Temporal mask, by default None.

        Returns
        -------
        torch.Tensor
            Predicted noise shaped (batch, frames, bones, channels).
        """
        batch, frames, bones, channels = noisyMotion.shape

        # Bone-wise projection + spatial GCN.
        boneH = self.boneProj(noisyMotion)
        for block in self.spatialBlocks:
            boneH = block(boneH)

        # Get conditioning embeddings
        timeEmb = self.timestepEmbed(timesteps)
        tagEmb = self.tagEmbed(tags)
        cond = torch.cat([timeEmb, tagEmb], dim=-1)

        # Optional factorized temporal modeling per bone.
        boneH = self._applyFactorizedTemporal(boneH, cond, mask)

        # Flatten per-frame features after spatial mixing.
        motionH = boneH.reshape(batch, frames, bones * self.embedDim)
        motionH = self.frameProj(motionH)
        textH = self.textProj(textEmbedding)

        # Expand text embedding to sequence length and add
        textH = textH.unsqueeze(1).expand(-1, frames, -1)
        h = motionH + textH

        # Apply denoising blocks
        for block in self.blocks:
            h = block(h, cond, mask)

        # Output projection
        h = self.outputNorm(h)
        output = self.outputProj(h)

        # Reshape to (batch, frames, bones, channels)
        return output.view(batch, frames, bones, channels)

    def _applyFactorizedTemporal(
        self,
        boneH: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply temporal blocks per bone sequence when enabled.

        Parameters
        ----------
        boneH : torch.Tensor
            Bone embeddings shaped (batch, frames, bones, embedDim).
        cond : torch.Tensor
            Conditioning embeddings shaped (batch, condDim).
        mask : Optional[torch.Tensor]
            Temporal padding mask shaped (batch, frames).

        Returns
        -------
        torch.Tensor
            Updated bone embeddings shaped (batch, frames, bones, embedDim).
        """
        if (
            self.spatiotemporalMode != SPATIOTEMPORAL_MODE_FACTORIZED
            or not self.boneTemporalBlocks
        ):
            return boneH
        batch, frames, bones, embedDim = boneH.shape
        boneSeq = boneH.permute(0, 2, 1, 3).reshape(
            batch * bones,
            frames,
            embedDim,
        )
        condExpanded = cond.repeat_interleave(bones, dim=0)
        maskExpanded = self._repeatMask(mask, bones)
        for block in self.boneTemporalBlocks:
            boneSeq = block(boneSeq, condExpanded, maskExpanded)
        boneSeq = boneSeq.reshape(batch, bones, frames, embedDim)
        return boneSeq.permute(0, 2, 1, 3)

    def _repeatMask(
        self,
        mask: Optional[torch.Tensor],
        bones: int,
    ) -> Optional[torch.Tensor]:
        """
        Repeat temporal mask per bone when factorized.

        Parameters
        ----------
        mask : Optional[torch.Tensor]
            Temporal padding mask shaped (batch, frames).
        bones : int
            Number of bones.

        Returns
        -------
        Optional[torch.Tensor]
            Repeated mask shaped (batch * bones, frames).
        """
        if mask is None:
            return None
        return mask.repeat_interleave(bones, dim=0)
