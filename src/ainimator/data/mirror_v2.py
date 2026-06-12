"""Batch-level SMPL-22 mirror augmentation for v2 training.

The legacy ``src.shared.augmentation.mirror`` operates on **single
sample dicts**.  For the v2 training loop we need a fast, batch-aware
helper that mirrors :class:`V2Batch` tensors in-place on the active
device.

Key differences vs the legacy module
------------------------------------
* **Per-sample probability** — each sample in the batch is independently
  mirrored with probability ``p`` so half the batch keeps its original
  orientation.  Doubles the effective dataset size without losing the
  original signal.
* **Prompt-aware skipping** — prompts that mention "left" / "right"
  cannot be safely mirrored without rewriting the text.  Such samples
  are skipped to keep the cross-attention from learning a corrupted
  text↔motion correlation.
* **No padding shift** — only frames already marked real by
  ``motionMask`` are touched.  Padded frames are zero anyway and
  mirroring zeros yields zeros, but this lets the helper be a no-op on
  fully-padded samples.

The actual rotation and translation transforms reuse the constants
defined in the legacy module
(``SMPL22_MIRROR_JOINT_PERMUTATION``, ``ROT6D_MIRROR_SIGN``).
"""

from __future__ import annotations

import re
from dataclasses import replace

import torch

from ainimator.training.full_training_v2 import V2Batch
from ainimator.data.augmentation.mirror import (
    ROT6D_MIRROR_SIGN,
    SMPL22_MIRROR_JOINT_PERMUTATION,
)

# Words that flip meaning under a left↔right swap.  When any of these
# appears in the prompt, the sample is skipped from mirroring.
_LEFT_RIGHT_PATTERN = re.compile(
    r"\b(left|right|leftward|rightward|lefthand|righthand|leftleg|"
    r"rightleg|clockwise|counterclockwise|counter-clockwise)\b",
    re.IGNORECASE,
)


def isPromptMirrorSafe(text: str) -> bool:
    """Return ``True`` when the prompt has no left/right semantics.

    A simple regex check on common L/R words.  False positives (e.g.
    "Heinrich Heine") are rare in motion prompts; false negatives are
    impossible by construction given the word list.
    """
    if not text:
        # Empty prompts (used by cond-mask-prob) are trivially safe.
        return True
    return _LEFT_RIGHT_PATTERN.search(text) is None


def mirrorRotation6dBatch(rotation: torch.Tensor) -> torch.Tensor:
    """Mirror a ``(B, F, 22, 6)`` rotation6d batch along the X axis.

    Applies the rotation6d sign flip and then permutes the bones
    according to SMPL-22 left↔right mapping.  Non-contiguous tensors
    are handled by ``index_select``.
    """
    if rotation.ndim != 4 or rotation.shape[-1] != 6:
        raise ValueError(
            "rotation must be 4-D (B, F, B_bones, 6); got shape "
            f"{tuple(rotation.shape)}."
        )
    signs = torch.tensor(
        ROT6D_MIRROR_SIGN, device=rotation.device, dtype=rotation.dtype
    )
    flipped = rotation * signs
    permIndex = torch.as_tensor(
        SMPL22_MIRROR_JOINT_PERMUTATION,
        device=rotation.device,
        dtype=torch.long,
    )
    return flipped.index_select(dim=2, index=permIndex)


def mirrorRootTranslationBatch(rtrans: torch.Tensor) -> torch.Tensor:
    """Mirror a ``(B, F, 3)`` root translation batch by flipping X."""
    if rtrans.ndim != 3 or rtrans.shape[-1] != 3:
        raise ValueError(
            "rtrans must be 3-D (B, F, 3); got shape "
            f"{tuple(rtrans.shape)}."
        )
    signs = torch.tensor(
        [-1.0, 1.0, 1.0], device=rtrans.device, dtype=rtrans.dtype
    )
    return rtrans * signs


def mirrorBatch(
    batch: V2Batch,
    probability: float,
    generator: torch.Generator,
) -> V2Batch:
    """Mirror each sample in ``batch`` with the given probability.

    Samples whose prompt contains a left/right word are never mirrored
    even when the coin flip says yes — preserving the text↔motion
    semantics of the cross-attention.

    The ``generator`` argument lives on CPU (matching
    :class:`TrainingRandomState.cpuGenerator`) so the per-sample coin
    flips are deterministic across MPS / CUDA / CPU runs given a fixed
    seed.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must be in [0, 1].")
    if probability == 0.0:
        return batch

    batchSize = batch.rotation6d.shape[0]
    coins = torch.rand(
        (batchSize,),
        generator=generator,
        device=torch.device("cpu"),
    )
    # Build the (B,) bool mask of samples to flip.
    mirrorFlags: list[bool] = []
    for index in range(batchSize):
        coin = float(coins[index].item())
        wantsMirror = coin < probability
        safeText = isPromptMirrorSafe(batch.rawTexts[index])
        mirrorFlags.append(wantsMirror and safeText)
    if not any(mirrorFlags):
        return batch

    flipMask = torch.tensor(
        mirrorFlags, device=batch.rotation6d.device, dtype=torch.bool
    )
    rotation = batch.rotation6d.clone()
    rtrans = batch.rootTranslation.clone()

    rotationMirrored = mirrorRotation6dBatch(rotation)
    rtransMirrored = mirrorRootTranslationBatch(rtrans)

    flipMaskRot = flipMask.view(batchSize, 1, 1, 1)
    flipMaskRt = flipMask.view(batchSize, 1, 1)

    rotation = torch.where(flipMaskRot, rotationMirrored, rotation)
    rtrans = torch.where(flipMaskRt, rtransMirrored, rtrans)

    return replace(
        batch,
        rotation6d=rotation,
        rootTranslation=rtrans,
    )
