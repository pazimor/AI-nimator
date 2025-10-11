"""Skeleton layout definitions shared by animation exporters."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

SMPL22_PARENTS: Dict[str, Optional[str]] = {
    "pelvis": None,
    "spine1": "pelvis",
    "spine2": "spine1",
    "spine3": "spine2",
    "neck": "spine3",
    "head": "neck",
    "clavicle_l": "spine3",
    "upperarm_l": "clavicle_l",
    "lowerarm_l": "upperarm_l",
    "hand_l": "lowerarm_l",
    "clavicle_r": "spine3",
    "upperarm_r": "clavicle_r",
    "lowerarm_r": "upperarm_r",
    "hand_r": "lowerarm_r",
    "thigh_l": "pelvis",
    "calf_l": "thigh_l",
    "foot_l": "calf_l",
    "toe_l": "foot_l",
    "thigh_r": "pelvis",
    "calf_r": "thigh_r",
    "foot_r": "calf_r",
    "toe_r": "foot_r",
}


_SMPL22_OFFSET_DATA: Dict[str, Tuple[float, float, float]] = {
    "pelvis": (0.0, 0.0, 0.0),
    "spine1": (0.0, 0.09, 0.0),
    "spine2": (0.0, 0.11, 0.0),
    "spine3": (0.0, 0.13, 0.0),
    "neck": (0.0, 0.11, 0.0),
    "head": (0.0, 0.18, 0.02),
    "clavicle_l": (0.07, 0.05, 0.02),
    "upperarm_l": (0.18, 0.0, 0.0),
    "lowerarm_l": (0.26, -0.01, 0.0),
    "hand_l": (0.18, -0.01, 0.0),
    "clavicle_r": (-0.07, 0.05, 0.02),
    "upperarm_r": (-0.18, 0.0, 0.0),
    "lowerarm_r": (-0.26, -0.01, 0.0),
    "hand_r": (-0.18, -0.01, 0.0),
    "thigh_l": (0.09, -0.08, 0.01),
    "calf_l": (0.0, -0.45, 0.0),
    "foot_l": (0.0, -0.05, 0.11),
    "toe_l": (0.0, 0.0, 0.16),
    "thigh_r": (-0.09, -0.08, 0.01),
    "calf_r": (0.0, -0.45, 0.0),
    "foot_r": (0.0, -0.05, 0.11),
    "toe_r": (0.0, 0.0, 0.16),
}


SMPL24_PARENTS: Dict[str, Optional[str]] = {
    "pelvis": None,
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "head": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
}


_SMPL24_OFFSET_DATA: Dict[str, Tuple[float, float, float]] = {
    "pelvis": (0.0, 0.0, 0.0),
    "spine1": (0.0, 0.09, 0.0),
    "spine2": (0.0, 0.11, 0.0),
    "spine3": (0.0, 0.13, 0.0),
    "neck": (0.0, 0.11, 0.0),
    "head": (0.0, 0.18, 0.02),
    "left_collar": (0.07, 0.05, 0.02),
    "right_collar": (-0.07, 0.05, 0.02),
    "left_shoulder": (0.15, 0.0, 0.0),
    "right_shoulder": (-0.15, 0.0, 0.0),
    "left_elbow": (0.25, -0.01, 0.0),
    "right_elbow": (-0.25, -0.01, 0.0),
    "left_wrist": (0.23, -0.01, 0.0),
    "right_wrist": (-0.23, -0.01, 0.0),
    "left_hand": (0.18, -0.01, 0.0),
    "right_hand": (-0.18, -0.01, 0.0),
    "left_hip": (0.09, -0.08, 0.01),
    "right_hip": (-0.09, -0.08, 0.01),
    "left_knee": (0.0, -0.45, 0.0),
    "right_knee": (0.0, -0.45, 0.0),
    "left_ankle": (0.0, -0.05, 0.11),
    "right_ankle": (0.0, -0.05, 0.11),
    "left_foot": (0.0, 0.0, 0.16),
    "right_foot": (0.0, 0.0, 0.16),
}


def _normalise_offsets(data: Dict[str, Tuple[float, float, float]]) -> Dict[str, np.ndarray]:
    return {name: np.array(offset, dtype=np.float32) for name, offset in data.items()}


SMPL22_OFFSETS = _normalise_offsets(_SMPL22_OFFSET_DATA)
SMPL24_OFFSETS = _normalise_offsets(_SMPL24_OFFSET_DATA)


def resolve_skeleton(bones: Tuple[str, ...]) -> Tuple[Dict[str, Optional[str]], Dict[str, np.ndarray]]:
    """Return parent and offset mapping matching a known skeleton layout.

    Parameters
    ----------
    bones:
        Ordered tuple of joint names describing the clip to export.

    Returns
    -------
    tuple[dict[str, Optional[str]], dict[str, np.ndarray]]
        Parent mapping and offset vectors (relative to parent). Unknown bones
        fall back to a flat hierarchy with zero offsets.
    """

    ordered = tuple(bones)
    if not ordered:
        return {}, {}
    bone_set = set(ordered)
    if len(ordered) == len(SMPL22_PARENTS) and bone_set == set(SMPL22_PARENTS):
        return SMPL22_PARENTS.copy(), SMPL22_OFFSETS.copy()
    if len(ordered) == len(SMPL24_PARENTS) and bone_set == set(SMPL24_PARENTS):
        return SMPL24_PARENTS.copy(), SMPL24_OFFSETS.copy()
    parents: Dict[str, Optional[str]] = {ordered[0]: None}
    offsets: Dict[str, np.ndarray] = {ordered[0]: np.zeros(3, dtype=np.float32)}
    for index, bone in enumerate(ordered[1:], start=1):
        parent = ordered[index - 1]
        parents[bone] = parent
        offsets[bone] = np.zeros(3, dtype=np.float32)
    return parents, offsets


__all__ = [
    "resolve_skeleton",
    "SMPL22_PARENTS",
    "SMPL22_OFFSETS",
    "SMPL24_PARENTS",
    "SMPL24_OFFSETS",
]
