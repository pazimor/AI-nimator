"""Batch-level dataclasses shared across training and data layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class V2Batch:
    """A padded batch ready to feed the v2 training step.

    Attributes
    ----------
    rotation6d : torch.Tensor
        ``(B, F, 22, 6)`` float32 — already on the target device.
    rootTranslation : torch.Tensor
        ``(B, F, 3)`` float32.
    motionMask : torch.Tensor
        ``(B, F)`` bool, ``True`` on real frames, ``False`` on padding.
    rawTexts : tuple[str, ...]
        Source prompt for each sample.  Tokenised inside the training
        step so we can apply ``cond-mask-prob`` at the string level.
    """

    rotation6d: torch.Tensor
    rootTranslation: torch.Tensor
    motionMask: torch.Tensor
    rawTexts: tuple[str, ...]
