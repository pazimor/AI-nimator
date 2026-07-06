"""Build component tensors from a base motion window."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence

import torch

from ainimator.core.constants.skeletons import SMPL22_BONE_ORDER
from ainimator.geometry.components.base import MotionComponent
from ainimator.geometry.components.ops import (
    rot6dToJointXYZ,
    sixdToRotationMatrix,
    temporalAngleDifference,
    temporalDifference,
)

FOOT_CONTACT_THRESHOLD = 0.002
HAND_CONTACT_THRESHOLD = 0.002
ROOT_TRANSLATION_KEY = "trans"
EPSILON = 1e-8

LEFT_ANKLE_INDEX = SMPL22_BONE_ORDER.index("leftAnkle")
LEFT_FOOT_INDEX = SMPL22_BONE_ORDER.index("leftFoot")
RIGHT_ANKLE_INDEX = SMPL22_BONE_ORDER.index("rightAnkle")
RIGHT_FOOT_INDEX = SMPL22_BONE_ORDER.index("rightFoot")
LEFT_WRIST_INDEX = SMPL22_BONE_ORDER.index("leftWrist")
RIGHT_WRIST_INDEX = SMPL22_BONE_ORDER.index("rightWrist")
PELVIS_INDEX = SMPL22_BONE_ORDER.index("pelvis")
NECK_INDEX = SMPL22_BONE_ORDER.index("neck")
LEFT_HIP_INDEX = SMPL22_BONE_ORDER.index("leftHip")
RIGHT_HIP_INDEX = SMPL22_BONE_ORDER.index("rightHip")
FOOT_CONTACT_INDICES = (
    LEFT_ANKLE_INDEX,
    LEFT_FOOT_INDEX,
    RIGHT_ANKLE_INDEX,
    RIGHT_FOOT_INDEX,
)
HAND_CONTACT_INDICES = (
    LEFT_WRIST_INDEX,
    RIGHT_WRIST_INDEX,
)
END_EFFECTOR_INDICES = (
    LEFT_WRIST_INDEX,
    RIGHT_WRIST_INDEX,
    LEFT_ANKLE_INDEX,
    RIGHT_ANKLE_INDEX,
)


def buildMotionFeatureTensors(
    motion: torch.Tensor,
    extras: Mapping[str, object],
    enabledComponents: Sequence[MotionComponent],
) -> dict[str, torch.Tensor]:
    """
    Build all auxiliary component tensors requested by the active config.

    Parameters
    ----------
    motion : torch.Tensor
        Rotation tensor shaped (frames, bones, 6).
    extras : Mapping[str, object]
        Sliced top-level extras aligned to the same frame window.
    enabledComponents : Sequence[MotionComponent]
        Active components from the network config.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from sample key to feature tensor. The base ``motion`` tensor
        is not duplicated here.
    """
    enabledKeys = {component.key for component in enabledComponents}
    features: dict[str, torch.Tensor] = {}
    if not enabledKeys or enabledKeys == {"rotation6d"}:
        return features

    rootTranslation: torch.Tensor | None = None
    jointXyz: torch.Tensor | None = None
    jointVelocity: torch.Tensor | None = None
    rootYaw: torch.Tensor | None = None

    if enabledKeys & {
        "root_translation",
        "root_velocity",
    }:
        rootTranslation = _extractRootTranslation(
            extras=extras,
            frameCount=motion.shape[0],
            dtype=motion.dtype,
        )
    if enabledKeys & {
        "joint_xyz",
        "joint_velocity",
        "end_effector_velocity",
        "foot_contact",
        "hand_contact",
        "pelvis_height",
    }:
        jointXyz = rot6dToJointXYZ(motion.unsqueeze(0)).squeeze(0)
    if enabledKeys & {
        "joint_velocity",
        "end_effector_velocity",
        "foot_contact",
        "hand_contact",
    }:
        if jointXyz is None:
            raise RuntimeError("joint_xyz must be computed before velocities.")
        jointVelocity = temporalDifference(jointXyz)
    if enabledKeys & {"root_yaw", "root_yaw_velocity"}:
        rootYaw = _computeRootYaw(motion)

    if "root_translation" in enabledKeys:
        if rootTranslation is None:
            raise RuntimeError("root_translation was not computed.")
        features["root_translation"] = rootTranslation
    if "root_velocity" in enabledKeys:
        if rootTranslation is None:
            raise RuntimeError("root_translation is required for root_velocity.")
        features["root_velocity"] = temporalDifference(rootTranslation)
    if "pelvis_height" in enabledKeys:
        if rootTranslation is not None:
            features["pelvis_height"] = rootTranslation[:, 1:2]
        elif jointXyz is not None:
            features["pelvis_height"] = jointXyz[:, PELVIS_INDEX, 1:2]
        else:
            features["pelvis_height"] = torch.zeros(
                motion.shape[0],
                1,
                dtype=motion.dtype,
            )
    if "root_yaw" in enabledKeys:
        if rootYaw is None:
            raise RuntimeError("root_yaw was not computed.")
        features["root_yaw"] = rootYaw
    if "root_yaw_velocity" in enabledKeys:
        if rootYaw is None:
            raise RuntimeError("root_yaw is required for root_yaw_velocity.")
        features["root_yaw_velocity"] = temporalAngleDifference(rootYaw)
    if "joint_xyz" in enabledKeys:
        if jointXyz is None:
            raise RuntimeError("joint_xyz was not computed.")
        features["joint_xyz"] = jointXyz
    if "joint_velocity" in enabledKeys:
        if jointVelocity is None:
            raise RuntimeError("joint_velocity was not computed.")
        features["joint_velocity"] = jointVelocity
    if "end_effector_velocity" in enabledKeys:
        if jointVelocity is None:
            raise RuntimeError("joint_velocity was not computed.")
        endEffectorVelocity = jointVelocity[:, END_EFFECTOR_INDICES, :]
        features["end_effector_velocity"] = endEffectorVelocity.reshape(
            endEffectorVelocity.shape[0],
            -1,
        )
    if "foot_contact" in enabledKeys:
        if jointXyz is None:
            raise RuntimeError("joint_xyz is required for foot_contact.")
        features["foot_contact"] = _buildContactFeatures(
            jointXyz=jointXyz,
            indices=FOOT_CONTACT_INDICES,
            threshold=FOOT_CONTACT_THRESHOLD,
        )
    if "hand_contact" in enabledKeys:
        if jointXyz is None:
            raise RuntimeError("joint_xyz is required for hand_contact.")
        features["hand_contact"] = _buildContactFeatures(
            jointXyz=jointXyz,
            indices=HAND_CONTACT_INDICES,
            threshold=HAND_CONTACT_THRESHOLD,
        )
    return features


def _extractRootTranslation(
    extras: Mapping[str, object],
    frameCount: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Extract root translation anchored at the first frame (origin).

    Training targets in absolute world coordinates forced the model to
    memorise per-sample trajectories that do not transfer across prompts:
    on a 1482-sample ACCAD run, rtrans loss generalised 3–4× worse than
    diff / xyz / vel_xyz and dominated the aux gradient, blocking bone_d
    convergence.  Anchoring subtracts ``trans[0]`` so every sample starts
    at the origin — the model only has to learn relative displacement,
    which is directly shared across samples.  The transformation
    preserves all motion information (frame-to-frame velocity is
    invariant to constant subtraction), so ``root_velocity`` and any
    downstream FK-derived features remain unchanged.  Inference export
    re-anchors by default (see GenerationOutputOptions.anchorRootTranslation).
    """
    rawTranslation = extras.get(ROOT_TRANSLATION_KEY)
    if rawTranslation is None:
        raise KeyError(
            "Animation extras are missing 'trans', required by root-motion "
            "components. Rebuild the converted dataset with translation extras."
        )
    translation = torch.as_tensor(rawTranslation, dtype=dtype)
    if translation.dim() != 2 or translation.shape[1] != 3:
        raise ValueError(
            "Expected root translation shape (frames, 3), got "
            f"{tuple(translation.shape)}."
        )
    if translation.shape[0] != frameCount:
        raise ValueError(
            "Root translation frame count mismatch: "
            f"{translation.shape[0]} vs motion {frameCount}."
        )
    # Anchor to origin: every training sample starts at (0, 0, 0).
    translation = translation - translation[0:1]
    return translation


def canonicalizeMotionFacing(
    motion: torch.Tensor,
    extras: MutableMapping[str, object] | Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Rotate motion + extras so frame 0 always faces world +Z.

    Phase 1.3 (2026-05-17) — facing-direction canonicalisation.  Removes
    one degree of freedom the model otherwise has to memorise (the
    starting yaw of every sample).  Two things happen:

    1. The pelvis yaw at frame 0 (``yaw0``) is extracted from the pelvis
       6D rotation via :func:`_computeRootYaw`.
    2. A rotation of ``-yaw0`` around the world Y axis is applied to
       BOTH the pelvis 6D rotation (every frame) AND the root
       translation (every frame).  All downstream joints inherit the
       new pelvis frame via forward kinematics, so they do not need a
       separate transform.

    The result is bit-identical motion content but in a canonical
    frame: every sample begins facing +Z, which dramatically reduces
    the entropy of the input distribution the diffusion model has to
    capture.  This is the standard HumanML3D / MDM pre-processing step.

    Parameters
    ----------
    motion : torch.Tensor
        Rotation6d tensor shaped ``(frames, 22, 6)``.
    extras : Mapping[str, object]
        Top-level extras (may include ``"trans"``).  The returned dict
        is a shallow copy with ``"trans"`` rotated in place if present.

    Returns
    -------
    tuple[torch.Tensor, dict[str, object]]
        Canonicalised ``(motion, extras)``.  Other extras are passed
        through untouched.
    """
    if motion.dim() != 3 or motion.shape[1] != len(SMPL22_BONE_ORDER) or motion.shape[2] != 6:
        raise ValueError(
            "canonicalizeMotionFacing expects motion of shape "
            f"(frames, 22, 6); got {tuple(motion.shape)}."
        )

    yaw0 = _computeRootYaw(motion)[0, 0]  # scalar — yaw at frame 0
    cosA = torch.cos(-yaw0)
    sinA = torch.sin(-yaw0)
    # Rotation around the world Y axis by -yaw0:
    #     [[ cos, 0, sin],
    #      [   0, 1,   0],
    #      [-sin, 0, cos]]
    rotY = torch.stack(
        [
            torch.stack([cosA, torch.zeros_like(cosA), sinA]),
            torch.stack(
                [torch.zeros_like(cosA), torch.ones_like(cosA), torch.zeros_like(cosA)]
            ),
            torch.stack([-sinA, torch.zeros_like(cosA), cosA]),
        ]
    ).to(dtype=motion.dtype, device=motion.device)

    # --- Pelvis rotation: left-multiply pelvis matrix by rotY ----
    motion = motion.clone()
    pelvisRotation = motion[:, PELVIS_INDEX, :]  # (F, 6)
    pelvisMatrix = sixdToRotationMatrix(pelvisRotation)  # (F, 3, 3)
    pelvisMatrixCanon = rotY.unsqueeze(0) @ pelvisMatrix
    # Re-pack to 6D as the first two columns of the rotation matrix.
    pelvisCanon6d = torch.cat(
        [pelvisMatrixCanon[..., 0], pelvisMatrixCanon[..., 1]],
        dim=-1,
    )
    motion[:, PELVIS_INDEX, :] = pelvisCanon6d

    # --- Root translation: rotate every frame by rotY ------------
    extrasOut: dict[str, object] = dict(extras)
    raw = extrasOut.get(ROOT_TRANSLATION_KEY)
    if raw is not None:
        translation = torch.as_tensor(raw, dtype=motion.dtype)
        if translation.dim() != 2 or translation.shape[1] != 3:
            raise ValueError(
                "canonicalizeMotionFacing expects 'trans' of shape "
                f"(frames, 3); got {tuple(translation.shape)}."
            )
        translationCanon = translation @ rotY.t()
        extrasOut[ROOT_TRANSLATION_KEY] = translationCanon

    return motion, extrasOut


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    """Return ``vector`` scaled to unit length (eps-guarded)."""
    return vector / (vector.norm() + EPSILON)


def _buildUprightFrame(joints: torch.Tensor) -> torch.Tensor:
    """Build the frame-0 body rotation that maps the body to Y-up/+Z.

    The body axes are derived from the rest-stable skeleton structure at
    frame 0: ``up`` = pelvis→neck (torso), ``right`` = left→right hip
    (orthogonalised against ``up``), ``forward`` = right × up.  The
    returned matrix has these axes as ROWS, so left-multiplying the
    pelvis rotation by it sends body-up → world +Y and body-forward →
    world +Z.

    Parameters
    ----------
    joints : torch.Tensor
        Forward-kinematics joint positions, shape ``(F, 22, 3)``.

    Returns
    -------
    torch.Tensor
        A ``(3, 3)`` rotation matrix.
    """
    first = joints[0]
    up = _normalize(first[NECK_INDEX] - first[PELVIS_INDEX])
    right = first[RIGHT_HIP_INDEX] - first[LEFT_HIP_INDEX]
    right = _normalize(right - (right @ up) * up)
    forward = _normalize(torch.cross(right, up, dim=0))
    return torch.stack([right, up, forward], dim=0)


def _applyRootRotation(
    motion: torch.Tensor,
    extras: Mapping[str, object],
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Left-multiply the pelvis rotation + root translation by ``rotation``.

    Parameters
    ----------
    motion : torch.Tensor
        Rotation6d motion, shape ``(F, 22, 6)``.
    extras : Mapping[str, object]
        Extras possibly holding ``"trans"`` (F, 3).
    rotation : torch.Tensor
        World rotation matrix ``(3, 3)`` to apply.

    Returns
    -------
    tuple[torch.Tensor, dict[str, object]]
        Rotated ``(motion, extras)``.
    """
    motion = motion.clone()
    pelvisMatrix = sixdToRotationMatrix(motion[:, PELVIS_INDEX, :])
    pelvisCanon = rotation.unsqueeze(0) @ pelvisMatrix
    motion[:, PELVIS_INDEX, :] = torch.cat(
        [pelvisCanon[..., 0], pelvisCanon[..., 1]], dim=-1
    )
    extrasOut: dict[str, object] = dict(extras)
    raw = extrasOut.get(ROOT_TRANSLATION_KEY)
    if raw is not None:
        translation = torch.as_tensor(raw, dtype=motion.dtype)
        extrasOut[ROOT_TRANSLATION_KEY] = translation @ rotation.t()
    return motion, extrasOut


def canonicalizeMotionUpright(
    motion: torch.Tensor,
    extras: MutableMapping[str, object] | Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Rotate a clip so frame 0 stands upright (Y-up) and faces +Z.

    Root cause fix (2026-06-17): the AMASS→rot6d conversion applied the
    raw AMASS root orientation (Z-up world) to the Y-up rest skeleton, so
    preprocessed bodies were stored lying down in arbitrary directions.
    This supersedes the yaw-only :func:`canonicalizeMotionFacing` by
    building a full body frame from the hips + spine at frame 0 and
    rotating the global (pelvis) rotation + root translation so the body
    is canonical: up → world +Y, facing → world +Z.  All downstream
    joints inherit the new frame via forward kinematics.

    Parameters
    ----------
    motion : torch.Tensor
        Rotation6d tensor shaped ``(frames, 22, 6)``.
    extras : Mapping[str, object]
        Top-level extras (may include ``"trans"``).

    Returns
    -------
    tuple[torch.Tensor, dict[str, object]]
        Canonicalised ``(motion, extras)``.
    """
    if (
        motion.dim() != 3
        or motion.shape[1] != len(SMPL22_BONE_ORDER)
        or motion.shape[2] != 6
    ):
        raise ValueError(
            "canonicalizeMotionUpright expects motion of shape "
            f"(frames, 22, 6); got {tuple(motion.shape)}."
        )
    joints = rot6dToJointXYZ(motion.unsqueeze(0).float()).squeeze(0)
    rotation = _buildUprightFrame(joints).to(motion.dtype)
    return _applyRootRotation(motion, extras, rotation)


def _computeRootYaw(motion: torch.Tensor) -> torch.Tensor:
    """Approximate root heading from the pelvis local rotation."""
    pelvisRotation = motion[:, PELVIS_INDEX, :]
    pelvisMatrix = sixdToRotationMatrix(pelvisRotation)
    forward = pelvisMatrix[..., :, 2]
    yaw = torch.atan2(forward[..., 0], forward[..., 2] + EPSILON)
    return yaw.unsqueeze(-1)


def _buildContactFeatures(
    jointXyz: torch.Tensor,
    indices: Sequence[int],
    threshold: float,
) -> torch.Tensor:
    """Build simple MDM-style contact labels from joint speed."""
    positions = jointXyz[:, indices, :]
    if positions.shape[0] < 2:
        return torch.zeros(
            positions.shape[0],
            positions.shape[1],
            dtype=jointXyz.dtype,
            device=jointXyz.device,
        )
    velocity = temporalDifference(positions)[1:]
    speedSquared = (velocity ** 2).sum(dim=-1)
    contact = (speedSquared < threshold).to(jointXyz.dtype)
    firstFrame = contact[:1]
    return torch.cat([firstFrame, contact], dim=0)
