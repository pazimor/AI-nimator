"""Tests for shared motion tensor operations."""

from __future__ import annotations

import math

import torch

from ainimator.geometry.components.ops import (
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


def test_orthonormalizeRot6d_is_idempotent_on_valid_rotations() -> None:
    from ainimator.geometry.components.ops import (
        orthonormalizeRot6d,
        sixdToRotationMatrix,
    )

    generator = torch.Generator().manual_seed(0)
    raw = torch.randn(4, 22, 6, generator=generator)
    valid = orthonormalizeRot6d(raw)

    reprojected = orthonormalizeRot6d(valid)
    assert torch.allclose(reprojected, valid, atol=1e-6)

    # The projected 6D must describe the SAME rotation as the full
    # Gram-Schmidt matrix path (first two columns of the matrix).
    matrices = sixdToRotationMatrix(raw)
    expected = torch.cat(
        [matrices[..., :, 0], matrices[..., :, 1]], dim=-1
    )
    assert torch.allclose(valid, expected, atol=1e-6)


def test_orthonormalizeRot6d_restores_orthonormality_of_drifted_input() -> None:
    from ainimator.geometry.components.ops import orthonormalizeRot6d

    generator = torch.Generator().manual_seed(1)
    drifted = orthonormalizeRot6d(torch.randn(8, 6, generator=generator))
    drifted = drifted + 0.3 * torch.randn(8, 6, generator=generator)

    projected = orthonormalizeRot6d(drifted)
    b1 = projected[..., :3]
    b2 = projected[..., 3:]
    assert torch.allclose(b1.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(b2.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(
        (b1 * b2).sum(dim=-1), torch.zeros(8), atol=1e-5
    )
