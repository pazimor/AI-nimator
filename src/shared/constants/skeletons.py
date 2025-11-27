"""Canonical skeleton definitions used across dataset tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

# Canonical SMPL-22 order enforced across the project.
SMPL22_BONE_ORDER: List[str] = [
    "pelvis",
    "leftHip",
    "rightHip",
    "spine1",
    "leftKnee",
    "rightKnee",
    "spine2",
    "leftAnkle",
    "rightAnkle",
    "spine3",
    "leftFoot",
    "rightFoot",
    "neck",
    "leftCollar",
    "rightCollar",
    "head",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
]

# SMPL-24 extends SMPL-22 with hands.
SMPL24_BONE_ORDER: List[str] = SMPL22_BONE_ORDER + ["leftHand", "rightHand"]

# Aliases collected from the AMASS, HumanML3D and MotionX datasets.
SMPL_BONE_ALIASES: Dict[str, Set[str]] = {
    "pelvis": {"pelvis", "root", "hip", "hips"},
    "leftHip": {"leftupleg", "left_upleg", "lhip", "lefthip"},
    "rightHip": {"rightupleg", "right_upleg", "rhip", "righthip"},
    "spine1": {"spine1", "spine_1", "spine"},
    "leftKnee": {"leftleg", "left_leg", "lknee", "leftknee"},
    "rightKnee": {"rightleg", "right_leg", "rknee", "rightknee"},
    "spine2": {"spine2", "spine_2", "spine1a"},
    "leftAnkle": {"leftfoot", "left_foot", "lankle", "leftankle"},
    "rightAnkle": {"rightfoot", "right_foot", "rankle", "rightankle"},
    "spine3": {"spine3", "spine_3", "chest"},
    "leftFoot": {"lefttoe", "left_toe", "lefttoebase"},
    "rightFoot": {"righttoe", "right_toe", "righttoebase"},
    "neck": {"neck", "neck1", "neck_1"},
    "leftCollar": {"leftcollar", "left_collar", "lclavicle"},
    "rightCollar": {"rightcollar", "right_collar", "rclavicle"},
    "head": {"head", "headtop", "head_top"},
    "leftShoulder": {"leftshoulder", "left_shoulder", "lshoulder"},
    "rightShoulder": {"rightshoulder", "right_shoulder", "rshoulder"},
    "leftElbow": {"leftarm", "left_arm", "lelbow", "leftelbow"},
    "rightElbow": {"rightarm", "right_arm", "relbow", "rightelbow"},
    "leftWrist": {"lefthand", "left_hand", "lwrist", "leftwrist"},
    "rightWrist": {"righthand", "right_hand", "rwrist", "rightwrist"},
    "leftHand": {"lefthandtip", "left_hand_tip"},
    "rightHand": {"righthandtip", "right_hand_tip"},
}


@dataclass(frozen=True)
class SkeletonDefinition:
    """Describe a skeleton variant."""

    name: str
    bones: List[str]


SMPL22_SKELETON = SkeletonDefinition(name="SMPL-22", bones=SMPL22_BONE_ORDER)
SMPL24_SKELETON = SkeletonDefinition(name="SMPL-24", bones=SMPL24_BONE_ORDER)


# Parent-child relationships for SMPL-22
SMPL22_HIERARCHY: Dict[str, str | None] = {
    "pelvis": None,
    "leftHip": "pelvis",
    "rightHip": "pelvis",
    "spine1": "pelvis",
    "leftKnee": "leftHip",
    "rightKnee": "rightHip",
    "spine2": "spine1",
    "leftAnkle": "leftKnee",
    "rightAnkle": "rightKnee",
    "spine3": "spine2",
    "leftFoot": "leftAnkle",
    "rightFoot": "rightAnkle",
    "neck": "spine3",
    "leftCollar": "spine3",
    "rightCollar": "spine3",
    "head": "neck",
    "leftShoulder": "leftCollar",
    "rightShoulder": "rightCollar",
    "leftElbow": "leftShoulder",
    "rightElbow": "rightShoulder",
    "leftWrist": "leftElbow",
    "rightWrist": "rightElbow",
}

# Approximate default offsets (in meters) for Mean SMPL shape in T-Pose.
# These are used when exact subject shape is not available.
SMPL22_DEFAULT_OFFSETS: Dict[str, List[float]] = {
    "pelvis": [0.0, 0.0, 0.0],
    "leftHip": [0.07, -0.04, 0.0],
    "rightHip": [-0.07, -0.04, 0.0],
    "spine1": [0.0, 0.1, 0.02],
    "leftKnee": [0.0, -0.40, 0.0],
    "rightKnee": [0.0, -0.40, 0.0],
    "spine2": [0.0, 0.15, -0.02],
    "leftAnkle": [0.0, -0.42, 0.0],
    "rightAnkle": [0.0, -0.42, 0.0],
    "spine3": [0.0, 0.15, 0.0],
    "leftFoot": [0.0, -0.06, 0.12],
    "rightFoot": [0.0, -0.06, 0.12],
    "neck": [0.0, 0.12, 0.0],
    "leftCollar": [0.06, 0.08, -0.02],
    "rightCollar": [-0.06, 0.08, -0.02],
    "head": [0.0, 0.12, 0.04],
    "leftShoulder": [0.12, 0.0, 0.0],
    "rightShoulder": [-0.12, 0.0, 0.0],
    "leftElbow": [0.26, 0.0, 0.0],
    "rightElbow": [-0.26, 0.0, 0.0],
    "leftWrist": [0.24, 0.0, 0.0],
    "rightWrist": [-0.24, 0.0, 0.0],
}

