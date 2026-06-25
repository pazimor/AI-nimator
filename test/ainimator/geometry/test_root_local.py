"""Tests for the root-local motion conversion utilities (§2.2.a)."""

from __future__ import annotations

import math

import torch
import pytest

from ainimator.geometry.root_local import (
    absoluteToRootLocalDeltas,
    aimDirectionFromPelvisYaw,
    pelvisYawFromRot6d,
    rootLocalDeltasToAbsolute,
)


def _identityRot6d(frames: int, numBones: int = 22) -> torch.Tensor:
    """Return identity rot6d (cols: [1,0,0], [0,1,0]) for all frames/bones."""
    rot = torch.zeros(frames, numBones, 6)
    rot[..., 0] = 1.0  # col0.x
    rot[..., 4] = 1.0  # col1.y
    return rot


def _yawOnlyRot6d(frames: int, yaw: float, numBones: int = 22) -> torch.Tensor:
    """Return rot6d with pelvis facing direction rotated by ``yaw`` rad."""
    rot = _identityRot6d(frames, numBones)
    # col0 = [cos yaw, 0, sin yaw] in the ground plane (Y-up).
    rot[:, 0, 0] = math.cos(yaw)  # col0.x
    rot[:, 0, 1] = 0.0            # col0.y
    rot[:, 0, 2] = math.sin(yaw)  # col0.z
    # col1 stays [0, 1, 0] (up axis)
    return rot


# -------------------------------------------------------------------------
# pelvisYawFromRot6d
# -------------------------------------------------------------------------

def test_pelvis_yaw_identity_is_zero() -> None:
    rot = _identityRot6d(5)
    pelvis = rot[:, 0, :]  # (5, 6)
    yaw = pelvisYawFromRot6d(pelvis)
    assert torch.allclose(yaw, torch.zeros(5), atol=1e-5)


def test_pelvis_yaw_90_deg() -> None:
    rot = _yawOnlyRot6d(3, yaw=math.pi / 2)
    pelvis = rot[:, 0, :]
    yaw = pelvisYawFromRot6d(pelvis)
    assert torch.allclose(yaw, torch.full((3,), math.pi / 2), atol=1e-5)


# -------------------------------------------------------------------------
# absoluteToRootLocalDeltas
# -------------------------------------------------------------------------

def test_local_deltas_shape() -> None:
    frames = 10
    rot = _identityRot6d(frames)
    root = torch.zeros(frames, 3)
    deltas = absoluteToRootLocalDeltas(root, rot)
    assert deltas.shape == (frames, 4)


def test_local_deltas_frame0_is_zero() -> None:
    """Frame 0 has no predecessor, so its delta must be zero."""
    frames = 8
    rot = _identityRot6d(frames)
    root = torch.cumsum(torch.ones(frames, 3) * 0.1, dim=0)
    deltas = absoluteToRootLocalDeltas(root, rot)
    assert torch.allclose(deltas[0], torch.zeros(4), atol=1e-5)


def test_local_deltas_forward_motion_identity_yaw() -> None:
    """With identity yaw, forward motion (+X) appears as (Δfwd, 0, 0, 0)."""
    frames = 5
    rot = _identityRot6d(frames)
    root = torch.zeros(frames, 3)
    root[:, 0] = torch.arange(frames, dtype=torch.float32) * 0.1  # +X motion

    deltas = absoluteToRootLocalDeltas(root, rot)
    # Frame 0: zero (no predecessor).
    # Frames 1+: Δforward ≈ 0.1, Δlateral ≈ 0, Δheight ≈ 0, Δyaw ≈ 0.
    assert torch.allclose(deltas[1:, 0], torch.full((frames - 1,), 0.1), atol=1e-5)
    assert torch.allclose(deltas[1:, 1:], torch.zeros(frames - 1, 3), atol=1e-5)


def test_local_deltas_90deg_yaw_lateral_motion() -> None:
    """With 90-deg yaw (facing +Z), +X world motion appears as lateral."""
    frames = 5
    rot = _yawOnlyRot6d(frames, yaw=math.pi / 2)
    root = torch.zeros(frames, 3)
    root[:, 0] = torch.arange(frames, dtype=torch.float32) * 0.1  # +X world

    deltas = absoluteToRootLocalDeltas(root, rot)
    # Local forward is now +Z (world); +X world = lateral in local frame.
    # Δlateral[1:] ≈ -0.1 (right of a character facing +Z is -X local).
    assert torch.allclose(
        deltas[1:, 1], torch.full((frames - 1,), -0.1), atol=1e-4
    )
    assert torch.allclose(deltas[1:, 0], torch.zeros(frames - 1), atol=1e-4)


# -------------------------------------------------------------------------
# rootLocalDeltasToAbsolute (round-trip)
# -------------------------------------------------------------------------

def test_round_trip_identity_yaw() -> None:
    """absoluteToRootLocalDeltas → rootLocalDeltasToAbsolute = identity."""
    frames = 8
    rot = _identityRot6d(frames)
    root = torch.zeros(frames, 3)
    root[:, 0] = torch.arange(frames, dtype=torch.float32) * 0.05
    root[:, 1] = torch.sin(torch.linspace(0, math.pi, frames)) * 0.2

    deltas = absoluteToRootLocalDeltas(root, rot)
    # Integrate from frame 0.
    seedPos = root[0]
    seedYaw = torch.tensor(0.0)
    reconstructed, _ = rootLocalDeltasToAbsolute(
        seedPos, seedYaw, deltas[1:]
    )
    # Frames 1..F-1 should match the original.
    assert torch.allclose(reconstructed, root[1:], atol=1e-4)


def test_round_trip_varying_yaw() -> None:
    """Round-trip with a turning sequence."""
    frames = 10
    # Yaw ramps from 0 to π/2.
    yaws = torch.linspace(0, math.pi / 2, frames)
    rot = torch.zeros(frames, 22, 6)
    rot[..., 1] = 0.0
    rot[..., 4] = 1.0
    for i, y in enumerate(yaws):
        rot[i, 0, 0] = math.cos(float(y))
        rot[i, 0, 2] = math.sin(float(y))

    root = torch.zeros(frames, 3)
    root[:, 0] = torch.arange(frames, dtype=torch.float32) * 0.05

    deltas = absoluteToRootLocalDeltas(root, rot)
    seedPos = root[0]
    seedYaw = yaws[0]
    reconstructed, _ = rootLocalDeltasToAbsolute(
        seedPos, seedYaw, deltas[1:]
    )
    assert torch.allclose(reconstructed, root[1:], atol=1e-4)


# -------------------------------------------------------------------------
# aimDirectionFromPelvisYaw
# -------------------------------------------------------------------------

def test_aim_direction_identity_is_facing_plus_x() -> None:
    """Identity rot6d → character faces +X → aim = (1, 0)."""
    frames = 5
    rot = _identityRot6d(frames)
    aim = aimDirectionFromPelvisYaw(rot)
    assert aim.shape == (frames, 2)
    assert torch.allclose(aim[:, 0], torch.ones(frames), atol=1e-5)
    assert torch.allclose(aim[:, 1], torch.zeros(frames), atol=1e-5)


def test_aim_direction_is_unit_norm() -> None:
    rot = _yawOnlyRot6d(8, yaw=1.2)
    aim = aimDirectionFromPelvisYaw(rot)
    norms = aim.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_aim_direction_decoupled_from_velocity() -> None:
    """Aim from pelvis yaw != unit(velocity) when facing ≠ moving dir."""
    frames = 6
    # Pelvis faces +Z (yaw = π/2).
    rot = _yawOnlyRot6d(frames, yaw=math.pi / 2)
    # Character moves along +X.
    root = torch.zeros(frames, 3)
    root[:, 0] = torch.arange(frames, dtype=torch.float32) * 0.1

    aim = aimDirectionFromPelvisYaw(rot)
    # Aim should point toward +Z: cos(π/2) ≈ 0, sin(π/2) ≈ 1.
    assert aim[0, 0].abs() < 1e-4  # cos(π/2) ≈ 0
    assert abs(float(aim[0, 1].item()) - 1.0) < 1e-4  # sin(π/2) ≈ 1
