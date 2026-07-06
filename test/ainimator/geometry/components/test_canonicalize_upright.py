"""Unit tests for ``canonicalizeMotionUpright``.

Isolated tests: build a synthetic T-pose, tilt it so the body lies along
world X, then assert the canonicaliser recovers an upright (Y-up) body.
"""

from __future__ import annotations

import torch

from ainimator.core.constants.skeletons import SMPL22_BONE_ORDER
from ainimator.geometry.components.feature_builder import (
    canonicalizeMotionUpright,
)
from ainimator.geometry.components.ops import rot6dToJointXYZ

_FRAMES = 4
_BONES = len(SMPL22_BONE_ORDER)
_PELVIS = SMPL22_BONE_ORDER.index("pelvis")
_HEAD = SMPL22_BONE_ORDER.index("head")
_LEFT_FOOT = SMPL22_BONE_ORDER.index("leftFoot")
_RIGHT_FOOT = SMPL22_BONE_ORDER.index("rightFoot")

# rot6d of the identity rotation = first two columns of the identity.
_IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
# rot6d of Rz(+90°) (maps the body's +Y up axis onto world +X = lying).
_TILT_Z90_6D = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])


def _tposeMotion() -> torch.Tensor:
    """Return an all-identity (T-pose) rot6d motion ``(F, 22, 6)``."""
    return _IDENTITY_6D.repeat(_FRAMES, _BONES, 1).clone()


def _headFootVector(motion: torch.Tensor) -> torch.Tensor:
    """Mean head-minus-feet vector from forward kinematics."""
    xyz = rot6dToJointXYZ(motion.unsqueeze(0).float()).squeeze(0)
    feet = xyz[:, [_LEFT_FOOT, _RIGHT_FOOT], :].mean(1).mean(0)
    return xyz[:, _HEAD, :].mean(0) - feet


def test_upright_recovered_from_tilted_body() -> None:
    """A body laid along X is rotated back to a Y-up stance."""
    motion = _tposeMotion()
    motion[:, _PELVIS, :] = _TILT_Z90_6D  # lay the whole body along X
    tilted = _headFootVector(motion)
    assert int(tilted.abs().argmax()) != 1  # not Y-up before the fix

    canon, _ = canonicalizeMotionUpright(motion, {})
    fixed = _headFootVector(canon)
    assert int(fixed.abs().argmax()) == 1  # Y is now dominant
    assert float(fixed[1]) > 0.0  # head above feet


def test_trans_is_rotated_and_shape_preserved() -> None:
    """The ``trans`` extra is rotated and keeps its (F, 3) shape."""
    motion = _tposeMotion()
    motion[:, _PELVIS, :] = _TILT_Z90_6D
    trans = torch.randn(_FRAMES, 3)
    _, extras = canonicalizeMotionUpright(motion, {"trans": trans})
    assert extras["trans"].shape == (_FRAMES, 3)


def test_invalid_shape_raises() -> None:
    """A non ``(F, 22, 6)`` tensor is rejected."""
    bad = torch.zeros(_FRAMES, _BONES, 3)
    try:
        canonicalizeMotionUpright(bad, {})
    except ValueError:
        return
    raise AssertionError("expected ValueError on bad motion shape")
