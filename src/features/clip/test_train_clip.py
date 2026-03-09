"""Tests for CLIP training utilities."""

from __future__ import annotations

import torch

from src.features.clip.train_clip import _buildPositiveMask


def test_build_positive_mask_marks_shared_text_ids_as_positives() -> None:
    """Links with the same text_id must not be treated as negatives."""
    batch = {
        "sample_id": torch.tensor([11, 12, 13], dtype=torch.long),
        "text_id": torch.tensor([7, 7, 9], dtype=torch.long),
    }

    positiveMask = _buildPositiveMask(batch, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(positiveMask.cpu(), expected)
