"""Lightweight motion encoder for the generated-motion (x0) contrastive.

Rationale
---------
The Phase D/F text↔motion contrastive heads pool the denoiser's
post-block hidden state ``frameH``, which is computed from the **noisy
input** ``x_t``.  At low noise ``x_t`` already contains the answer, so
the contrastive can be satisfied without the *predicted* output
``boneOutput`` depending on the prompt at all.  The 2026-05-31 diagnostic
showed exactly this generation loophole: cross-prompt cosine similarity
of the sampled motion was 0.98 at cfg=1.0 (``p(motion|text) ≈
p(motion)``) while the encoder pool discriminated prompts cleanly
(enc_pair_sim ≈ 0.22).

This module is a self-contained TMR-style encoder that maps a *motion*
tensor (rotation6d) to a pooled, L2-normalised embedding **without ever
seeing the noisy input or the text**.  The training loop feeds it the
reconstructed ``x0`` so the contrastive forces the generated motion to
be classifiable to its prompt.  The contrastive on this embedding is
weighted toward high timesteps (where the prediction cannot simply copy
the noisy input) — the opposite of the FK losses' low-t schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MotionAlignmentEncoderConfig:
    """Configuration for :class:`MotionAlignmentEncoder`.

    Parameters
    ----------
    numBones, motionChannels : int
        Skeleton size and per-bone channel count (SMPL-22 rotation6d →
        22 and 6).
    embedDim : int
        Hidden width of the small transformer encoder.
    numLayers, numHeads : int
        Depth and head count of the transformer encoder.
    alignmentDim : int
        Shared contrastive space dimension (matches the denoiser
        alignment head, default 128).
    textDim : int
        Width of the pooled text embedding fed to :meth:`projectText`.
    maxFrames : int
        Maximum motion length used to size the positional buffer.
    dropout : float
        Dropout inside the transformer encoder.
    """

    numBones: int = 22
    motionChannels: int = 6
    embedDim: int = 256
    numLayers: int = 2
    numHeads: int = 4
    alignmentDim: int = 128
    textDim: int = 384
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
        if self.alignmentDim < 1:
            raise ValueError("alignmentDim must be >= 1.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")

    @property
    def motionInputDim(self) -> int:
        """Flattened per-frame motion width (numBones × motionChannels)."""
        return self.numBones * self.motionChannels


def _buildSinusoidalTable(maxLen: int, dim: int) -> torch.Tensor:
    """Return a ``(maxLen, dim)`` non-learned sinusoidal PE table."""
    position = torch.arange(maxLen, dtype=torch.float32).unsqueeze(1)
    divTerm = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    table = torch.zeros(maxLen, dim, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * divTerm)
    table[:, 1::2] = torch.cos(position * divTerm[: table[:, 1::2].shape[1]])
    return table


def _projectionMlp(inputDim: int, alignmentDim: int) -> nn.Sequential:
    """Two-layer SimCLR/CLIP-style projection head into ``alignmentDim``."""
    return nn.Sequential(
        nn.LayerNorm(inputDim),
        nn.Linear(inputDim, alignmentDim),
        nn.GELU(),
        nn.Linear(alignmentDim, alignmentDim),
    )


class MotionAlignmentEncoder(nn.Module):
    """TMR-style encoder mapping rotation6d motion → alignment embedding.

    Two public entry points feed the symmetric InfoNCE contrastive on
    the *generated* motion:

    * :meth:`encodeMotion` — pooled, L2-normalised embedding of a motion
      tensor (the reconstructed ``x0``).
    * :meth:`projectText` — pooled text embedding projected into the
      same alignment space and L2-normalised.

    The module is deliberately isolated (no dependency on the denoiser's
    internal state) so the contrastive cannot leak the noisy input back
    into the motion embedding.
    """

    def __init__(self, config: MotionAlignmentEncoderConfig) -> None:
        super().__init__()
        self._config = config
        embedDim = config.embedDim

        self.inputProjection = nn.Linear(config.motionInputDim, embedDim)
        self.register_buffer(
            "positionalTable",
            _buildSinusoidalTable(config.maxFrames, embedDim),
            persistent=False,
        )
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=embedDim,
            nhead=config.numHeads,
            dim_feedforward=embedDim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoderLayer, num_layers=config.numLayers
        )
        self.motionProjection = _projectionMlp(embedDim, config.alignmentDim)
        self.textProjection = _projectionMlp(
            config.textDim, config.alignmentDim
        )

    @property
    def config(self) -> MotionAlignmentEncoderConfig:
        return self._config

    @staticmethod
    def _maskedMean(
        sequence: torch.Tensor,
        keyPaddingMask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Masked mean over the frame axis (axis 1), ``True == pad``."""
        if keyPaddingMask is None:
            return sequence.mean(dim=1)
        realMask = (~keyPaddingMask).to(sequence.dtype).unsqueeze(-1)
        return (sequence * realMask).sum(dim=1) / torch.clamp(
            realMask.sum(dim=1), min=1.0
        )

    def encodeMotion(
        self,
        rotation6d: torch.Tensor,
        motionKeyPaddingMask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``(B, F, numBones, 6)`` motion → ``(B, alignmentDim)``.

        The output is L2-normalised so the downstream InfoNCE operates on
        a unit sphere.  ``motionKeyPaddingMask`` is ``(B, F)`` with
        ``True`` marking padded frames.
        """
        batchSize, frames, _, _ = rotation6d.shape
        flat = rotation6d.reshape(batchSize, frames, -1)
        hidden = self.inputProjection(flat)
        hidden = hidden + self.positionalTable[:frames].unsqueeze(0)
        encoded = self.transformer(
            hidden, src_key_padding_mask=motionKeyPaddingMask
        )
        pooled = self._maskedMean(encoded, motionKeyPaddingMask)
        return F.normalize(self.motionProjection(pooled), dim=-1)

    def projectText(self, pooledText: torch.Tensor) -> torch.Tensor:
        """Project a pooled text embedding into the alignment space."""
        return F.normalize(self.textProjection(pooledText), dim=-1)
