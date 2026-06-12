"""Shared tensor operations for motion components and losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ainimator.core.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL22_DEFAULT_OFFSETS,
    SMPL22_HIERARCHY,
)

MIN_SIXD_CHANNELS = 6


def maskedMean(
    values: torch.Tensor,
    motionMask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute a masked mean over valid frames."""
    values = torch.nan_to_num(values)
    if motionMask is None:
        return values.mean()
    mask = motionMask.to(values.device).float()
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    masked = values * mask
    valid = mask.sum()
    if float(valid.item()) == 0.0:
        return torch.tensor(0.0, device=values.device)
    scale = values.numel() / mask.numel()
    return masked.sum() / (valid * scale)


def temporalDifference(
    values: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Return a same-length finite difference along the requested axis."""
    axis = dim if dim >= 0 else values.dim() + dim
    result = torch.zeros_like(values)
    if values.shape[axis] < 2:
        return result

    diff = torch.diff(values, dim=axis)
    tailSlices = [slice(None)] * values.dim()
    tailSlices[axis] = slice(1, None)
    result[tuple(tailSlices)] = diff

    firstSlices = [slice(None)] * values.dim()
    secondSlices = [slice(None)] * values.dim()
    firstSlices[axis] = 0
    secondSlices[axis] = 1
    result[tuple(firstSlices)] = result[tuple(secondSlices)]
    return result


def temporalAngleDifference(
    angles: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Return wrapped finite differences for angular signals."""
    axis = dim if dim >= 0 else angles.dim() + dim
    result = torch.zeros_like(angles)
    if angles.shape[axis] < 2:
        return result

    diff = torch.diff(angles, dim=axis)
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))

    tailSlices = [slice(None)] * angles.dim()
    tailSlices[axis] = slice(1, None)
    result[tuple(tailSlices)] = diff

    firstSlices = [slice(None)] * angles.dim()
    secondSlices = [slice(None)] * angles.dim()
    firstSlices[axis] = 0
    secondSlices[axis] = 1
    result[tuple(firstSlices)] = result[tuple(secondSlices)]
    return result


def sixdToRotationMatrix(sixd: torch.Tensor) -> torch.Tensor:
    """
    Convert a 6D rotation representation to a rotation matrix.

    Uses Gram-Schmidt orthogonalization.
    """
    a1 = sixd[..., :3]
    a2 = sixd[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rot6dToJointXYZ(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert local 6D rotations to global joint XYZ via forward kinematics.
    """
    if rot6d.dim() != 4 or rot6d.shape[-1] != MIN_SIXD_CHANNELS:
        raise ValueError(
            "Expected rot6d shape (batch, frames, bones, 6), got "
            f"{tuple(rot6d.shape)}"
        )

    batchSize, frameCount, boneCount, _ = rot6d.shape
    parentIndices, offsets = smpl22KinematicParams(
        boneCount,
        rot6d.device,
        rot6d.dtype,
    )
    localRotations = sixdToRotationMatrix(rot6d)

    globalRotations: list[torch.Tensor] = []
    globalPositions: list[torch.Tensor] = []

    for boneIndex in range(boneCount):
        localRotation = localRotations[:, :, boneIndex]
        if parentIndices[boneIndex] < 0:
            globalRotations.append(localRotation)
            rootOffset = offsets[boneIndex].view(1, 1, 3)
            rootOffset = rootOffset.expand(batchSize, frameCount, 3)
            globalPositions.append(rootOffset)
            continue

        parentIndex = parentIndices[boneIndex]
        parentRotation = globalRotations[parentIndex]
        parentPosition = globalPositions[parentIndex]
        globalRotation = torch.matmul(parentRotation, localRotation)
        childOffset = offsets[boneIndex].view(1, 1, 3, 1)
        childOffset = torch.matmul(parentRotation, childOffset).squeeze(-1)
        globalRotations.append(globalRotation)
        globalPositions.append(parentPosition + childOffset)

    return torch.stack(globalPositions, dim=2)


def smpl22KinematicParams(
    boneCount: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[int], torch.Tensor]:
    """Return parent indices and offsets for the first SMPL22 joints."""
    maxBones = len(SMPL22_BONE_ORDER)
    if boneCount > maxBones:
        raise ValueError(
            f"Unsupported boneCount={boneCount}, max supported is {maxBones}"
        )

    boneNames = SMPL22_BONE_ORDER[:boneCount]
    indexByName = {name: idx for idx, name in enumerate(boneNames)}
    parentIndices: list[int] = []
    offsetValues: list[list[float]] = []

    for boneName in boneNames:
        parentName = SMPL22_HIERARCHY[boneName]
        if parentName is None:
            parentIndices.append(-1)
        else:
            parentIndex = indexByName.get(parentName)
            if parentIndex is None:
                raise ValueError(
                    "Invalid skeleton order: parent "
                    f"{parentName} missing for {boneName}"
                )
            parentIndices.append(parentIndex)
        offsetValues.append(SMPL22_DEFAULT_OFFSETS[boneName])

    offsets = torch.tensor(
        offsetValues,
        device=device,
        dtype=dtype,
    )
    return parentIndices, offsets
