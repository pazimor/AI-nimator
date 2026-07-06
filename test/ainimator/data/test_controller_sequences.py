"""Phase A1 tests — autoregressive sequence builder."""

from __future__ import annotations

import pytest
import torch

from ainimator.data.controller_sequences import (
    ControllerSequenceConfig,
    buildControllerSequences,
    deriveFootContacts,
    deriveGaitPhase,
)


def _clip(frames: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    rotation6d = torch.randn(frames, 22, 6)
    rootTranslation = torch.randn(frames, 3)
    return rotation6d, rootTranslation


def test_transition_count_context_one() -> None:
    rot, root = _clip(10)
    batch = buildControllerSequences(
        rot, root, ControllerSequenceConfig(contextFrames=1)
    )
    assert batch.numTransitions == 9
    assert batch.boneWindow.shape == (9, 1, 22, 6)
    # globalWindow holds root-local motion deltas (4 channels, not 3).
    assert batch.globalWindow.shape == (9, 1, 4)
    assert batch.control.shape == (9, 2)


def test_transition_count_context_three() -> None:
    rot, root = _clip(10)
    batch = buildControllerSequences(
        rot, root, ControllerSequenceConfig(contextFrames=3)
    )
    assert batch.numTransitions == 7
    assert batch.boneWindow.shape == (7, 3, 22, 6)


def test_delta_targets_are_consistent() -> None:
    rot, root = _clip(8)
    batch = buildControllerSequences(
        rot, root, ControllerSequenceConfig(contextFrames=1)
    )
    # target delta == next - last-of-window
    lastBone = batch.boneWindow[:, -1, :, :]
    assert torch.allclose(
        batch.targetBoneDelta, batch.targetBoneNext - lastBone, atol=1e-5
    )


def test_control_is_local_planar_velocity() -> None:
    """Control (vx, vz) equals the root-local planar delta (Δfwd, Δlat)."""
    rot, root = _clip(6)
    batch = buildControllerSequences(
        rot, root, ControllerSequenceConfig(contextFrames=1)
    )
    # targetGlobalDelta is (N, 4): (Δforward, Δlateral, Δheight, Δyaw).
    # Control (no aim) = first two channels = (Δfwd, Δlat) in local frame.
    expected = batch.targetGlobalDelta[:, :2]
    assert torch.allclose(batch.control, expected, atol=1e-5)


def test_aim_direction_appends_two_channels() -> None:
    rot, root = _clip(6)
    batch = buildControllerSequences(
        rot,
        root,
        ControllerSequenceConfig(contextFrames=1, useAimDirection=True),
    )
    assert batch.control.shape[-1] == 4
    aim = batch.control[:, 2:]
    norms = aim.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_aim_direction_is_decoupled_from_locomotion() -> None:
    """Aim (pelvis yaw) must differ from locomotion direction (velocity).

    When the character faces one direction while moving in another, aim
    and locomotion velocity are not collinear — the new implementation
    decouples them via pelvis yaw (ROADMAP_DETERMINIST §2.2.b).
    """
    frames = 10
    # Pelvis yaw points toward +Z (col0 of rot matrix = [0, 0, 1]).
    # In rot6d, col0 is the first 3 channels of the 6-channel vector.
    rot = torch.zeros(frames, 22, 6)
    rot[:, 0, 0] = 0.0  # col0.x
    rot[:, 0, 1] = 0.0  # col0.y
    rot[:, 0, 2] = 1.0  # col0.z  → facing +Z
    # col1 (channels 3,4,5) = orthogonal; e.g. (0, 1, 0)
    rot[:, 0, 3] = 0.0
    rot[:, 0, 4] = 1.0
    rot[:, 0, 5] = 0.0

    # Character moves along +X (world frame).
    root = torch.zeros(frames, 3)
    root[:, 0] = torch.arange(frames, dtype=torch.float32) * 0.1

    batch = buildControllerSequences(
        rot, root,
        ControllerSequenceConfig(contextFrames=1, useAimDirection=True),
    )
    # Aim should point roughly toward +Z (cos≈0, sin≈1) for all frames.
    aim = batch.control[:, 2:]
    # cos θ ≈ 0, sin θ ≈ 1 (facing +Z in the XZ ground plane)
    assert aim[:, 1].mean() > 0.5, "Aim should point toward +Z (sin > 0)"


def test_clip_too_short_raises() -> None:
    rot, root = _clip(2)
    with pytest.raises(ValueError, match="too short"):
        buildControllerSequences(
            rot, root, ControllerSequenceConfig(contextFrames=4)
        )


# ---------------------------------------------------------------------
# A2 — foot contacts + gait phase
# ---------------------------------------------------------------------
def test_foot_contacts_shape_and_range() -> None:
    rot, root = _clip(12)
    contacts = deriveFootContacts(rot, root)
    assert contacts.shape == (12, 2)
    assert set(contacts.unique().tolist()) <= {0.0, 1.0}


def test_foot_contacts_all_when_thresholds_huge() -> None:
    rot, root = _clip(8)
    contacts = deriveFootContacts(
        rot, root, heightThreshold=1e9, speedThreshold=1e9
    )
    assert float(contacts.mean()) == 1.0


def test_gait_phase_is_unit_circle() -> None:
    contacts = torch.zeros(20, 2)
    # alternating strikes: left at 0,8,16 ; right at 4,12
    for frame in (0, 8, 16):
        contacts[frame:frame + 2, 0] = 1.0
    for frame in (4, 12):
        contacts[frame:frame + 2, 1] = 1.0
    phase = deriveGaitPhase(contacts)
    assert phase.shape == (20, 2)
    norms = phase.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_gait_phase_constant_without_strikes() -> None:
    contacts = torch.zeros(10, 2)
    phase = deriveGaitPhase(contacts)
    # No strikes → phase held at 0 → (cos,sin) = (1, 0) everywhere.
    assert torch.allclose(phase[:, 0], torch.ones(10), atol=1e-6)
    assert torch.allclose(phase[:, 1], torch.zeros(10), atol=1e-6)


def test_builder_emits_phase_and_contacts() -> None:
    rot, root = _clip(10)
    batch = buildControllerSequences(
        rot,
        root,
        ControllerSequenceConfig(
            contextFrames=1, emitPhase=True, emitContacts=True
        ),
    )
    assert batch.phase is not None
    assert batch.phase.shape == (9, 2)
    assert batch.contactTarget is not None
    assert batch.contactTarget.shape == (9, 2)


def test_builder_omits_phase_by_default() -> None:
    rot, root = _clip(10)
    batch = buildControllerSequences(
        rot, root, ControllerSequenceConfig(contextFrames=1)
    )
    assert batch.phase is None
    assert batch.contactTarget is None
