"""
SMPL-22 left/right mirror augmentation.

Doubles the effective training set by flipping motion across the YZ plane
(x → -x) and swapping left/right joint indices.  Used train-only via a
per-sample probability; val/overfit paths are untouched.

Scope note
----------
Text prompts are *not* swapped here — that would require re-tokenizing
inside the augmentation path, which is tangled with the tokenizer-owning
base dataset.  A motion described as "left hand waves" will therefore be
presented to the denoiser both as-is and mirrored ("now the right hand
waves while the text still says left").  This is the same simplification
used by the original MDM reference impl; empirically the CLIP embedding
tolerates the ambiguity because locomotion / gross-body cues dominate.

Mirror semantics
----------------
For a rotation matrix ``R`` expressed in world or parent-local frame,
mirroring across the YZ plane is ``R' = M R M`` with ``M = diag(-1, 1, 1)``.
In a 6D representation ``[a1, a2]`` (first two columns of ``R``) this
simplifies to flipping the signs of ``a1[1], a1[2], a2[0]`` — i.e.
multiplying by the fixed vector ``[1, -1, -1, -1, 1, 1]``.  The SMPL
T-pose bone offsets are bilaterally symmetric, so mirroring the local
rotations and swapping left/right joint indices produces a
kinematically valid mirrored motion with no need to re-run FK.
"""

from __future__ import annotations

from typing import Dict

import torch

from ainimator.core.constants.skeletons import SMPL22_BONE_ORDER

__all__ = [
    "SMPL22_LEFT_RIGHT_JOINT_PAIRS",
    "SMPL22_MIRROR_JOINT_PERMUTATION",
    "ROT6D_MIRROR_SIGN",
    "mirrorMotionSample",
]

# Left/right joint index pairs in SMPL-22 order.  Central joints
# (pelvis/spines/neck/head) are not in the list — they keep their own
# index after the permutation is built.
_LEFT_RIGHT_NAMES = (
    ("leftHip", "rightHip"),
    ("leftKnee", "rightKnee"),
    ("leftAnkle", "rightAnkle"),
    ("leftFoot", "rightFoot"),
    ("leftCollar", "rightCollar"),
    ("leftShoulder", "rightShoulder"),
    ("leftElbow", "rightElbow"),
    ("leftWrist", "rightWrist"),
)

SMPL22_LEFT_RIGHT_JOINT_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (SMPL22_BONE_ORDER.index(left), SMPL22_BONE_ORDER.index(right))
    for left, right in _LEFT_RIGHT_NAMES
)


def _buildMirrorPermutation() -> tuple[int, ...]:
    perm = list(range(len(SMPL22_BONE_ORDER)))
    for left, right in SMPL22_LEFT_RIGHT_JOINT_PAIRS:
        perm[left], perm[right] = right, left
    return tuple(perm)


SMPL22_MIRROR_JOINT_PERMUTATION: tuple[int, ...] = _buildMirrorPermutation()

# Sign pattern applied per channel to rot6d = [a1(3), a2(3)].
# Derivation: mirror matrix M = diag(-1, 1, 1), R' = M R M flips
# R[0, 1], R[0, 2], R[1, 0], R[2, 0].  Columns 0 and 1 of R are a1 and
# a2, so we flip a1[1], a1[2], a2[0].
ROT6D_MIRROR_SIGN: tuple[int, ...] = (1, -1, -1, -1, 1, 1)

# Feature indices (inside foot_contact / hand_contact tensors) for left/right.
# Must match feature_builder.FOOT_CONTACT_INDICES / HAND_CONTACT_INDICES order.
_FOOT_CONTACT_PERMUTATION = (2, 3, 0, 1)  # (LA, LF, RA, RF) -> (RA, RF, LA, LF)
_HAND_CONTACT_PERMUTATION = (1, 0)        # (LW, RW) -> (RW, LW)

# End effectors in feature_builder are stored flattened as
# (LW, RW, LA, RA) × 3.  To mirror we permute the joint axis (LW<->RW,
# LA<->RA) AND flip x.  Easier to operate on the un-flattened view.
_END_EFFECTOR_PERMUTATION = (1, 0, 3, 2)


def _mirrorJointTensor(
    tensor: torch.Tensor,
    permutation: tuple[int, ...],
) -> torch.Tensor:
    """
    Mirror a per-joint tensor: flip joint order AND the x component.

    Accepts shape (T, J, 3); non-joint tensors are returned unchanged.
    """
    if tensor.dim() != 3 or tensor.shape[-1] != 3:
        return tensor
    indexTensor = torch.as_tensor(
        permutation,
        device=tensor.device,
        dtype=torch.long,
    )
    permuted = tensor.index_select(dim=1, index=indexTensor)
    signX = torch.tensor(
        [-1.0, 1.0, 1.0],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return permuted * signX


def _mirrorXyzTimeTensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flip x on a (T, 3) tensor (root translation, root velocity, ...)."""
    if tensor.dim() != 2 or tensor.shape[-1] != 3:
        return tensor
    signX = torch.tensor(
        [-1.0, 1.0, 1.0],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return tensor * signX


def _mirrorRotation6d(motion: torch.Tensor) -> torch.Tensor:
    """Mirror rot6d motion (T, 22, 6) — sign-flip channels + swap L/R bones."""
    if motion.dim() != 3 or motion.shape[-1] != 6:
        return motion
    signs = torch.tensor(
        ROT6D_MIRROR_SIGN,
        device=motion.device,
        dtype=motion.dtype,
    )
    flipped = motion * signs
    permIndex = torch.as_tensor(
        SMPL22_MIRROR_JOINT_PERMUTATION,
        device=motion.device,
        dtype=torch.long,
    )
    return flipped.index_select(dim=1, index=permIndex)


def _mirrorContact(
    tensor: torch.Tensor,
    permutation: tuple[int, ...],
) -> torch.Tensor:
    if tensor.dim() != 2 or tensor.shape[-1] != len(permutation):
        return tensor
    indexTensor = torch.as_tensor(
        permutation,
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(dim=1, index=indexTensor)


def _mirrorEndEffectorVelocity(tensor: torch.Tensor) -> torch.Tensor:
    """End-effector velocity is stored flat (T, 4*3).  Unflatten, mirror, reflat."""
    if tensor.dim() != 2 or tensor.shape[-1] != 12:
        return tensor
    frames = tensor.shape[0]
    reshaped = tensor.view(frames, 4, 3)
    mirrored = _mirrorJointTensor(reshaped, _END_EFFECTOR_PERMUTATION)
    return mirrored.reshape(frames, 12)


def _mirrorYaw(tensor: torch.Tensor) -> torch.Tensor:
    """root_yaw is an angle around Y; mirroring X flips its sign."""
    return -tensor


def mirrorMotionSample(sample: Dict[str, object]) -> Dict[str, object]:
    """
    Return a mirrored copy of a training sample (text fields kept as-is).

    Only tensors that have a geometric interpretation are transformed; the
    tokenized prompt (``input_ids`` / ``attention_mask`` / ``pooled_text`` /
    ``generation_text_embedding``) is passed through unchanged — see the
    module docstring for the rationale.
    """
    mirrored: Dict[str, object] = dict(sample)

    motion = sample.get("motion")
    if isinstance(motion, torch.Tensor):
        mirrored["motion"] = _mirrorRotation6d(motion)

    jointXyz = sample.get("joint_xyz")
    if isinstance(jointXyz, torch.Tensor):
        mirrored["joint_xyz"] = _mirrorJointTensor(
            jointXyz,
            SMPL22_MIRROR_JOINT_PERMUTATION,
        )
    jointVelocity = sample.get("joint_velocity")
    if isinstance(jointVelocity, torch.Tensor):
        mirrored["joint_velocity"] = _mirrorJointTensor(
            jointVelocity,
            SMPL22_MIRROR_JOINT_PERMUTATION,
        )
    for key in ("root_translation", "root_velocity"):
        value = sample.get(key)
        if isinstance(value, torch.Tensor):
            mirrored[key] = _mirrorXyzTimeTensor(value)
    footContact = sample.get("foot_contact")
    if isinstance(footContact, torch.Tensor):
        mirrored["foot_contact"] = _mirrorContact(
            footContact,
            _FOOT_CONTACT_PERMUTATION,
        )
    handContact = sample.get("hand_contact")
    if isinstance(handContact, torch.Tensor):
        mirrored["hand_contact"] = _mirrorContact(
            handContact,
            _HAND_CONTACT_PERMUTATION,
        )
    endEffVel = sample.get("end_effector_velocity")
    if isinstance(endEffVel, torch.Tensor):
        mirrored["end_effector_velocity"] = _mirrorEndEffectorVelocity(endEffVel)
    rootYaw = sample.get("root_yaw")
    if isinstance(rootYaw, torch.Tensor):
        mirrored["root_yaw"] = _mirrorYaw(rootYaw)
    rootYawVel = sample.get("root_yaw_velocity")
    if isinstance(rootYawVel, torch.Tensor):
        mirrored["root_yaw_velocity"] = _mirrorYaw(rootYawVel)
    # pelvis_height is along Y (unchanged by YZ-plane mirror).

    return mirrored
