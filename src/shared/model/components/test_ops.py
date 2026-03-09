"""Tests for shared motion tensor operations."""

from __future__ import annotations

import math

import torch

from src.shared.model.components.ops import (
    temporalAngleDifference,
    temporalDifference,
)


def test_temporalDifference_keeps_length_and_repeats_first_delta() -> None:
    values = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 5.0],
            [5.0, 9.0],
        ]
    )
    diff = temporalDifference(values)
    expected = torch.tensor(
        [
            [2.0, 4.0],
            [2.0, 4.0],
            [3.0, 4.0],
        ]
    )
    assert torch.equal(diff, expected)


def test_temporalAngleDifference_wraps_across_pi_boundary() -> None:
    angles = torch.tensor(
        [
            [math.pi - 0.1],
            [-math.pi + 0.1],
        ]
    )
    diff = temporalAngleDifference(angles)
    expected = torch.tensor([[0.2], [0.2]])
    assert torch.allclose(diff, expected, atol=1e-5)
