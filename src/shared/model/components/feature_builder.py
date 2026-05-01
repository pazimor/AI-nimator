"""Build component tensors from a base motion window."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from src.shared.constants.skeletons import SMPL22_BONE_ORDER
from src.shared.model.components.base import MotionComponent
from src.shared.model.components.ops import (
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
