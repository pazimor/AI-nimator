"""Diffusion denoising network for motion generation."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.shared.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL22_HIERARCHY,
    SMPL24_BONE_ORDER,
)
from src.shared.model.layers.attention import MultiHeadAttention
from src.shared.model.layers.normalization import AdaLN, FiLM
from src.shared.model.layers.positional import RoPE
from src.shared.model.layers.spatial_gcn import SpatialGCNBlock
from src.shared.model.layers.temporal import TemporalLayer
from src.shared.model.layers.transform import TransformLayer


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
            Dimension of conditioning embeddings (timestep).
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
        attnMask = None
        if mask is not None:
            # Convert key padding mask (True = pad) to attention keep-mask
            # expected by MultiHeadAttention (True = keep, False = mask).
            attnMask = (~mask).unsqueeze(1).unsqueeze(2)
        h = h + self.attention(self.norm(h), mask=attnMask)

        # Transform layer with AdaLN conditioning
        h = self.adalnCondition(h, cond)
        h = self.transform(h)

        return h


def _buildHierarchyAdjacency(numBones: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build directed hierarchy adjacency matrices for parent/child aggregation.
    """
    if numBones == len(SMPL22_BONE_ORDER):
        boneOrder = SMPL22_BONE_ORDER
        hierarchy: dict[str, Optional[str]] = dict(SMPL22_HIERARCHY)
    elif numBones == len(SMPL24_BONE_ORDER):
        boneOrder = SMPL24_BONE_ORDER
        hierarchy = dict(SMPL22_HIERARCHY)
        hierarchy["leftHand"] = "leftWrist"
        hierarchy["rightHand"] = "rightWrist"
    else:
        identity = torch.eye(numBones, dtype=torch.float32)
        return identity, identity

    index = {name: idx for idx, name in enumerate(boneOrder)}
    parentAdj = torch.eye(numBones, dtype=torch.float32)
    childAdj = torch.eye(numBones, dtype=torch.float32)

    for child, parent in hierarchy.items():
        if parent is None:
            continue
        childIdx = index.get(child)
        parentIdx = index.get(parent)
        if childIdx is None or parentIdx is None:
            continue
        parentAdj[childIdx, parentIdx] = 1.0
        childAdj[parentIdx, childIdx] = 1.0

    parentDegree = parentAdj.sum(dim=1, keepdim=True).clamp(min=1.0)
    childDegree = childAdj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return parentAdj / parentDegree, childAdj / childDegree


class BoneHierarchyBlock(nn.Module):
    """
    Directed hierarchy mixing block over bones for each frame.
    """

    def __init__(
        self,
        numBones: int,
        embedDim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        parentAdj, childAdj = _buildHierarchyAdjacency(numBones)
        self.register_buffer("parentAdjacency", parentAdj)
        self.register_buffer("childAdjacency", childAdj)
        self.selfLinear = nn.Linear(embedDim, embedDim)
        self.parentLinear = nn.Linear(embedDim, embedDim)
        self.childLinear = nn.Linear(embedDim, embedDim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedDim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply directed parent/child aggregation.
        """
        batch, frames, bones, embedDim = x.shape
        h = x.reshape(batch * frames, bones, embedDim)
        parentMix = torch.matmul(self.parentAdjacency, h)
        childMix = torch.matmul(self.childAdjacency, h)
        mixed = (
            self.selfLinear(h)
            + self.parentLinear(parentMix)
            + self.childLinear(childMix)
        )
        mixed = self.activation(mixed)
        mixed = self.dropout(mixed)
        mixed = self.norm(mixed)
        mixed = mixed.reshape(batch, frames, bones, embedDim)
        return x + mixed


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
        numHierarchyLayers: int = 1,
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
        numHierarchyLayers : int, optional
            Number of directed hierarchy blocks after bone split.
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

        self.hierarchyBlocks = nn.ModuleList(
            [
                BoneHierarchyBlock(
                    numBones=numBones,
                    embedDim=embedDim,
                    dropout=dropout,
                )
                for _ in range(numHierarchyLayers)
            ]
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

        # Bone-wise projection, hierarchy, spatial and spatio-temporal mixing.
        boneH = self.boneProj(noisyMotion)
        for block in self.hierarchyBlocks:
            boneH = block(boneH)
        for block in self.spatialBlocks:
            boneH = block(boneH)
        for block in self.spatioTemporalBlocks:
            boneH = block(boneH)

        # Flatten per-frame features after spatial mixing.
        motionH = boneH.reshape(batch, frames, bones * self.embedDim)
        motionH = self.frameProj(motionH)
        textH = self.textProj(textEmbedding)
        textH = textH + self.textAdapter(textH)

        # Expand text embedding to sequence length and add
        textH = textH.unsqueeze(1).expand(-1, frames, -1)
        h = motionH + textH

        # Get conditioning embeddings
        cond = self.timestepEmbed(timesteps)

        # Apply denoising blocks
        for block in self.blocks:
            h = block(h, cond, mask)

        # Output projection
        h = self.outputNorm(h)
        output = self.outputProj(h)

        # Reshape to (batch, frames, bones, channels)
        return output.view(batch, frames, bones, channels)
