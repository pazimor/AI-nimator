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

