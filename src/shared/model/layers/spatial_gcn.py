"""Spatial GCN layers for skeleton-aware motion modeling."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.shared.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL22_HIERARCHY,
    SMPL24_BONE_ORDER,
)


def _build_adjacency(numBones: int) -> torch.Tensor:
    """
    Build a normalized adjacency matrix for SMPL skeletons.

    Uses SMPL-22 hierarchy by default. For SMPL-24, hands attach to wrists.
    Falls back to identity if the bone count is unknown.
    """
    if numBones == len(SMPL22_BONE_ORDER):
        boneOrder = SMPL22_BONE_ORDER
        hierarchy: Dict[str, Optional[str]] = dict(SMPL22_HIERARCHY)
    elif numBones == len(SMPL24_BONE_ORDER):
        boneOrder = SMPL24_BONE_ORDER
        hierarchy = dict(SMPL22_HIERARCHY)
        hierarchy["leftHand"] = "leftWrist"
        hierarchy["rightHand"] = "rightWrist"
    else:
        return torch.eye(numBones, dtype=torch.float32)

    index = {name: idx for idx, name in enumerate(boneOrder)}
    adjacency = torch.eye(numBones, dtype=torch.float32)
    for child, parent in hierarchy.items():
        if parent is None:
            continue
        childIdx = index.get(child)
        parentIdx = index.get(parent)
        if childIdx is None or parentIdx is None:
            continue
        adjacency[childIdx, parentIdx] = 1.0
        adjacency[parentIdx, childIdx] = 1.0

    # Row-normalize to keep activations stable.
    degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adjacency / degree


class SpatialGCNBlock(nn.Module):
    """Skeleton-aware spatial graph convolution per frame."""

    def __init__(
        self,
        numBones: int,
        embedDim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        adjacency = _build_adjacency(numBones)
        self.register_buffer("adjacency", adjacency)
        self.linear = nn.Linear(embedDim, embedDim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedDim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial GCN.

        Parameters
        ----------
        x : torch.Tensor
            Tensor shaped (batch, frames, bones, embedDim).
        """
        batch, frames, bones, embedDim = x.shape
        h = x.reshape(batch * frames, bones, embedDim)
        h = torch.matmul(self.adjacency, h)
        h = self.linear(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.norm(h)
        h = h.reshape(batch, frames, bones, embedDim)
        return x + h
