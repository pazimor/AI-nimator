"""Tests for :mod:`src.cli.generate_animation_v2`.

These tests focus on the **rotation conversion** and **AnimationSample**
construction — the parts that actually produce the .dae payload.  The
DDIM sampler and AnimationRebuilder are already covered elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.cli.generate_animation_v2 import (
    _buildAnimationSampleV2,
    _buildRebuilder,
    _mapAxisAnglesToSmpl24,
)
from src.shared.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL24_BONE_ORDER,
)
from src.shared.quaternion import Rotation


# ---------------------------------------------------------------------
# _mapAxisAnglesToSmpl24
# ---------------------------------------------------------------------
def test_map_axis_angles_to_smpl24_pads_with_zeros() -> None:
    frames = 4
    axisAngles = np.random.RandomState(0).randn(
        frames, len(SMPL22_BONE_ORDER), 3
    ).astype(np.float32)
    result = _mapAxisAnglesToSmpl24(axisAngles, SMPL22_BONE_ORDER)
    assert result.shape == (frames, len(SMPL24_BONE_ORDER), 3)
    # Bones present in SMPL-22 must be copied verbatim.
    smpl24Index = {b: i for i, b in enumerate(SMPL24_BONE_ORDER)}
    for srcIdx, name in enumerate(SMPL22_BONE_ORDER):
        np.testing.assert_array_equal(
            result[:, smpl24Index[name]], axisAngles[:, srcIdx]
        )
    # The two missing bones should be all zeros.
    extraBones = set(SMPL24_BONE_ORDER) - set(SMPL22_BONE_ORDER)
    for boneName in extraBones:
        assert (result[:, smpl24Index[boneName]] == 0).all()


def test_map_axis_angles_rejects_bone_count_mismatch() -> None:
    axisAngles = np.zeros((2, 19, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Bone count"):
        _mapAxisAnglesToSmpl24(axisAngles, SMPL22_BONE_ORDER)


# ---------------------------------------------------------------------
# _buildAnimationSampleV2 — rotation6d → axis-angle round-trip
# ---------------------------------------------------------------------
def test_build_sample_round_trips_rotation_through_axis_angle() -> None:
    """The rotation6d → axis-angle → rotation6d round-trip must
    preserve the underlying rotation up to numerical precision.

    This catches any regression that would silently corrupt
    orientations between the v2 sampler output and the .dae export."""
    frames = 4
    bones = len(SMPL22_BONE_ORDER)
    torch.manual_seed(0)
    randomAxisAngle = torch.randn(frames, bones, 3) * 0.4
    rot6d = Rotation(randomAxisAngle, kind="axis_angle").rot6d.float()

    outputPath = Path("/tmp/_sample_unit.dae")
    sample = _buildAnimationSampleV2(
        boneRotation6d=rot6d,
        rootTranslation=None,
        fps=30,
        outputPath=outputPath,
        extras={},
    )
    # Sample axis-angles flatten (frames, 24*3) and pad missing bones.
    flat = sample.axisAngles
    assert flat.shape == (frames, len(SMPL24_BONE_ORDER) * 3)
    smpl24 = flat.reshape(frames, len(SMPL24_BONE_ORDER), 3)
    smpl24Index = {b: i for i, b in enumerate(SMPL24_BONE_ORDER)}
    for srcIdx, name in enumerate(SMPL22_BONE_ORDER):
        recoveredAxisAngle = smpl24[:, smpl24Index[name]]
        # Compare via rotation matrices to absorb the axis-angle ↔
        # quaternion ambiguity (ε vs 2π−ε on the same axis).
        ref = Rotation(
            randomAxisAngle[:, srcIdx], kind="axis_angle"
        ).matrix
        recovered = Rotation(
            torch.from_numpy(recoveredAxisAngle), kind="axis_angle"
        ).matrix
        assert torch.allclose(ref, recovered, atol=1e-5), (
            f"Bone {name!r} drift: max="
            f"{(ref - recovered).abs().max().item():.2e}"
        )


def test_build_sample_attaches_root_translation_to_extras() -> None:
    frames = 6
    rot6d = torch.zeros(frames, len(SMPL22_BONE_ORDER), 6)
    rot6d[..., 0] = 1.0
    rot6d[..., 4] = 1.0  # identity rotation6d.
    trans = torch.tensor(
        [[0.1, 0.2, 0.3]] * frames, dtype=torch.float32
    )

    sample = _buildAnimationSampleV2(
        boneRotation6d=rot6d,
        rootTranslation=trans,
        fps=24,
        outputPath=Path("/tmp/x.dae"),
        extras={},
    )
    assert "trans" in sample.extras
    transOut = np.asarray(sample.extras["trans"])
    assert transOut.shape == (frames, 3)
    np.testing.assert_allclose(transOut, trans.numpy(), rtol=0, atol=1e-6)


def test_build_sample_rejects_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="3-D"):
        _buildAnimationSampleV2(
            boneRotation6d=torch.zeros(4, 22 * 6),
            rootTranslation=None,
            fps=30,
            outputPath=Path("/tmp/x.dae"),
            extras={},
        )


def test_build_sample_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="6-channel"):
        _buildAnimationSampleV2(
            boneRotation6d=torch.zeros(4, 22, 5),
            rootTranslation=None,
            fps=30,
            outputPath=Path("/tmp/x.dae"),
            extras={},
        )


def test_build_sample_rejects_translation_shape_mismatch() -> None:
    rot6d = torch.zeros(4, 22, 6)
    rot6d[..., 0] = 1.0
    rot6d[..., 4] = 1.0
    badTrans = torch.zeros(3, 3)  # wrong frame count
    with pytest.raises(ValueError, match="rootTranslation"):
        _buildAnimationSampleV2(
            boneRotation6d=rot6d,
            rootTranslation=badTrans,
            fps=30,
            outputPath=Path("/tmp/x.dae"),
            extras={},
        )


# ---------------------------------------------------------------------
# _buildRebuilder
# ---------------------------------------------------------------------
def test_build_rebuilder_returns_usable_instance(tmp_path: Path) -> None:
    rebuilder = _buildRebuilder(tmp_path / "out.dae")
    # The rebuilder must be ready for exportCollada — we don't run
    # the full export here (that path is covered by the legacy tests),
    # but we sanity-check the dataclass plumbing.
    assert rebuilder.config.paths.outputRoot == tmp_path.resolve()


# ---------------------------------------------------------------------
# End-to-end: build sample then export through real rebuilder
# ---------------------------------------------------------------------
def test_full_export_writes_valid_dae(tmp_path: Path) -> None:
    frames = 4
    bones = len(SMPL22_BONE_ORDER)
    torch.manual_seed(0)
    rot6d = Rotation(
        torch.randn(frames, bones, 3) * 0.2, kind="axis_angle"
    ).rot6d.float()
    trans = torch.zeros(frames, 3)
    outputPath = tmp_path / "anim.dae"

    sample = _buildAnimationSampleV2(
        boneRotation6d=rot6d,
        rootTranslation=trans,
        fps=24,
        outputPath=outputPath,
        extras={"prompt": "test"},
    )
    rebuilder = _buildRebuilder(outputPath)
    rebuilder.exportCollada(sample, outputPath)

    assert outputPath.exists()
    # Quick sanity: file is non-empty XML containing a SMPL bone name.
    content = outputPath.read_text(encoding="utf-8")
    assert "<COLLADA" in content
    assert "pelvis" in content
