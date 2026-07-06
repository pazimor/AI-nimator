"""Unit tests for ``ainimator.health.evaluation`` helpers.

Isolated tests — no checkpoints, no model components.  They guard the
length-alignment fix that lets fidelity/retrieval/distinctness compare
generated motions (fixed frame count) against ground-truth motions
(variable, possibly shorter) without a tensor-size crash.
"""

from __future__ import annotations

import torch

from ainimator.health.evaluation import alignVectors, cosineSim

_JOINT_DIMS = 66  # 22 SMPL joints × 3 (xyz) — one frame's worth


def test_align_vectors_truncates_to_common_length() -> None:
    """Mismatched FK vectors are truncated to the shortest length."""
    longVector = torch.arange(200 * _JOINT_DIMS, dtype=torch.float32)
    shortVector = torch.arange(176 * _JOINT_DIMS, dtype=torch.float32)
    aligned = alignVectors([longVector, shortVector])
    assert all(v.numel() == 176 * _JOINT_DIMS for v in aligned)
    # Truncation keeps the shared prefix, not a resampling.
    assert torch.equal(aligned[0], longVector[: 176 * _JOINT_DIMS])


def test_align_vectors_noop_when_equal() -> None:
    """Equal-length vectors are returned unchanged."""
    first = torch.ones(100 * _JOINT_DIMS)
    second = torch.zeros(100 * _JOINT_DIMS)
    aligned = alignVectors([first, second])
    assert aligned[0].numel() == aligned[1].numel() == 100 * _JOINT_DIMS


def test_aligned_vectors_are_cosine_comparable() -> None:
    """After alignment, cosineSim no longer raises on size mismatch."""
    gen = torch.randn(200 * _JOINT_DIMS)
    groundTruth = torch.randn(176 * _JOINT_DIMS)
    alignedGen, alignedGt = alignVectors([gen, groundTruth])
    value = cosineSim(alignedGen, alignedGt)
    assert -1.0 <= value <= 1.0


def test_align_vectors_empty() -> None:
    """An empty list is returned unchanged."""
    assert alignVectors([]) == []
