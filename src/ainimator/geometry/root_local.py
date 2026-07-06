"""Root-local motion conversion utilities (ROADMAP_DETERMINIST §2.2.a).

Converts absolute root trajectories (GT in AMASS) to / from the
4-channel root-local representation used by the autoregressive
controller state:

    (Δforward, Δlateral, Δheight, Δyaw)

where ``forward`` and ``lateral`` are expressed in the *character-local*
ground-plane frame at frame ``t`` (X-Z, Y-up), ``height`` is the
absolute vertical displacement ``Δy``, and ``yaw`` is the signed rotation
around the Y axis in radians.

The local frame is derived from the **yaw of the root (pelvis) joint**,
consistent with ``rot6d[0]`` (the first 6 channels of the lean state)
and the ``canonicalizeMotionUpright`` convention.  Using the pelvis yaw
decouples locomotion direction from facing direction — a character can
face east while moving north.

All operations are pure tensor math (no control flow data-dependent on
values, no ``.item()`` calls) so they are safe to call in the data
pipeline (outside the ONNX graph, as required by §2.10).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Pelvis is bone index 0 in SMPL-22 (``SMPL22_BONE_ORDER[0] == "pelvis"``).
_PELVIS_BONE_INDEX: int = 0

# Y-up convention: ground plane axes are X (0) and Z (2), vertical is Y (1).
_AXIS_X: int = 0
_AXIS_Y: int = 1
_AXIS_Z: int = 2


# -------------------------------------------------------------------------
# Yaw extraction from pelvis rot6d
# -------------------------------------------------------------------------

def pelvisYawFromRot6d(pelvisRot6d: torch.Tensor) -> torch.Tensor:
    """Extract the yaw angle of the pelvis from its 6D rotation.

    The 6D representation stores the first two columns of the rotation
    matrix (Zhou et al. 2019).  The yaw is the signed angle around the
    Y axis, derived by projecting the forward column (col 0) onto the
    ground plane and taking ``atan2``.

    Parameters
    ----------
    pelvisRot6d : torch.Tensor
        Pelvis 6D rotation, shape ``(..., 6)``.  The first three
        channels are column 0 of the rotation matrix (the local X axis
        in world space); the next three are column 1.

    Returns
    -------
    torch.Tensor
        Yaw angle in radians, shape ``(...,)``.
    """
    # Column 0 of the rotation matrix: the local forward/X axis in world.
    col0 = pelvisRot6d[..., :3]
    # Project onto the ground plane (drop Y) and recover the yaw.
    forward_x = col0[..., _AXIS_X]
    forward_z = col0[..., _AXIS_Z]
    return torch.atan2(forward_z, forward_x)


def yawToRotationMatrix2d(yaw: torch.Tensor) -> torch.Tensor:
    """Build a 2x2 rotation matrix from a yaw angle.

    Parameters
    ----------
    yaw : torch.Tensor
        Yaw angle in radians, shape ``(...,)``.

    Returns
    -------
    torch.Tensor
        2x2 rotation matrix, shape ``(..., 2, 2)``.
    """
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    # Row 0: [ cos, -sin ]   Row 1: [ sin,  cos ]
    row0 = torch.stack([cos, -sin], dim=-1)
    row1 = torch.stack([sin, cos], dim=-1)
    return torch.stack([row0, row1], dim=-2)


# -------------------------------------------------------------------------
# GT absolute trajectory → root-local delta sequence
# -------------------------------------------------------------------------

def absoluteToRootLocalDeltas(
    rootTranslation: torch.Tensor,
    rotation6d: torch.Tensor,
) -> torch.Tensor:
    """Convert an absolute root trajectory to root-local motion deltas.

    At each frame ``t`` the displacement ``rootTranslation[t+1] -
    rootTranslation[t]`` is expressed in the *local frame* defined by
    the pelvis yaw at frame ``t``.  This is the target representation for
    the controller state (ROADMAP_DETERMINIST §2.2.a).

    Parameters
    ----------
    rootTranslation : torch.Tensor
        Absolute root translation, shape ``(F, 3)`` — world-space XYZ.
    rotation6d : torch.Tensor
        Bone rotations, shape ``(F, numBones, 6)``.  Only the pelvis
        (bone 0) is used for yaw extraction.

    Returns
    -------
    torch.Tensor
        Root-local motion deltas, shape ``(F, 4)`` — channels are
        ``(Δforward, Δlateral, Δheight, Δyaw)``.  Frame 0 is zeroed
        (no predecessor).
    """
    yaw = pelvisYawFromRot6d(rotation6d[:, _PELVIS_BONE_INDEX, :])
    worldDelta = _worldSpaceDisplacement(rootTranslation)
    localPlanar, heightDelta = _planarDeltaInLocalFrame(worldDelta, yaw)
    yawDelta = _wrappedYawDelta(yaw)
    return torch.stack(
        [localPlanar[:, 0], localPlanar[:, 1], heightDelta, yawDelta],
        dim=-1,
    )


def _worldSpaceDisplacement(
    rootTranslation: torch.Tensor,
) -> torch.Tensor:
    """Compute per-frame world-space displacement; frame 0 is zeroed."""
    frames = rootTranslation.shape[0]
    device = rootTranslation.device
    dtype = rootTranslation.dtype
    worldDelta = torch.zeros(frames, 3, device=device, dtype=dtype)
    worldDelta[1:] = rootTranslation[1:] - rootTranslation[:-1]
    return worldDelta


def _planarDeltaInLocalFrame(
    worldDelta: torch.Tensor,
    yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate planar displacement into the local frame; return (local, height).

    Uses the *departure* yaw (shifted by one frame) so the integration
    inverse in :func:`rootLocalDeltasToAbsolute` stays consistent.
    """
    planarDelta = worldDelta[:, [_AXIS_X, _AXIS_Z]]       # (F, 2)
    heightDelta = worldDelta[:, _AXIS_Y]                   # (F,)
    departureYaw = torch.cat([yaw[:1], yaw[:-1]], dim=0)   # shift by 1
    rotMat = yawToRotationMatrix2d(departureYaw)            # (F, 2, 2)
    localPlanar = torch.matmul(
        rotMat.transpose(-1, -2), planarDelta.unsqueeze(-1)
    ).squeeze(-1)                                          # (F, 2)
    return localPlanar, heightDelta


def _wrappedYawDelta(yaw: torch.Tensor) -> torch.Tensor:
    """Compute wrapped yaw deltas ([-π, π]); frame 0 is zero."""
    frames = yaw.shape[0]
    device = yaw.device
    dtype = yaw.dtype
    yawDelta = torch.zeros(frames, device=device, dtype=dtype)
    rawDiff = yaw[1:] - yaw[:-1]
    yawDelta[1:] = torch.atan2(torch.sin(rawDiff), torch.cos(rawDiff))
    return yawDelta


# -------------------------------------------------------------------------
# Root-local delta sequence → absolute trajectory (rollout / export)
# -------------------------------------------------------------------------

def rootLocalDeltasToAbsolute(
    seedTranslation: torch.Tensor,
    seedYaw: float | torch.Tensor,
    localDeltas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate root-local motion deltas back to world-space trajectory.

    This is the **inverse of** :func:`absoluteToRootLocalDeltas`.
    It lives *outside* the ONNX graph (rollout / generation side).

    Parameters
    ----------
    seedTranslation : torch.Tensor
        World-space XYZ of the last seed frame, shape ``(3,)`` or
        ``(B, 3)``.
    seedYaw : float or torch.Tensor
        World-space yaw of the last seed frame (scalar or ``(B,)``).
    localDeltas : torch.Tensor
        Root-local motion deltas to integrate, shape ``(N, 4)`` or
        ``(B, N, 4)`` where channels are
        ``(Δforward, Δlateral, Δheight, Δyaw)``.

    Returns
    -------
    translations : torch.Tensor
        Integrated world-space XYZ, shape ``(N, 3)`` or ``(B, N, 3)``.
    yaws : torch.Tensor
        Cumulative world-space yaw, shape ``(N,)`` or ``(B, N)``.
    """
    batched = localDeltas.ndim == 3
    localDeltas, seedTranslation, seedYaw = _toBatched(
        localDeltas, seedTranslation, seedYaw
    )
    translations, yaws = _integrateDeltas(
        localDeltas, seedTranslation, seedYaw
    )
    if not batched:
        return translations.squeeze(0), yaws.squeeze(0)
    return translations, yaws


def _toBatched(
    localDeltas: torch.Tensor,
    seedTranslation: torch.Tensor,
    seedYaw: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Promote unbatched inputs to batch-dim-1 for uniform processing."""
    if localDeltas.ndim == 3:
        yawTensor = _toYawTensor(
            seedYaw, localDeltas.shape[0],
            localDeltas.dtype, localDeltas.device
        )
        return localDeltas, seedTranslation, yawTensor
    localDeltas = localDeltas.unsqueeze(0)
    seedTranslation = seedTranslation.unsqueeze(0)
    if isinstance(seedYaw, (int, float)):
        yawTensor = torch.tensor(
            seedYaw, dtype=localDeltas.dtype, device=localDeltas.device
        ).unsqueeze(0)
    else:
        yawTensor = seedYaw.unsqueeze(0)  # type: ignore[union-attr]
    return localDeltas, seedTranslation, yawTensor


def _toYawTensor(
    seedYaw: float | torch.Tensor,
    batchSize: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Materialise a scalar or tensor yaw to a ``(B,)`` tensor."""
    if isinstance(seedYaw, (int, float)):
        return torch.full(
            (batchSize,), float(seedYaw), dtype=dtype, device=device
        )
    return seedYaw.to(device=device, dtype=dtype)  # type: ignore[union-attr]


def _integrateDeltas(
    localDeltas: torch.Tensor,
    seedTranslation: torch.Tensor,
    seedYaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate ``(B, N, 4)`` deltas from a ``(B, 3)`` seed position."""
    device = localDeltas.device
    dtype = localDeltas.dtype
    currentYaw = seedYaw.to(device=device, dtype=dtype)
    currentPos = seedTranslation.to(device=device, dtype=dtype)
    numSteps = localDeltas.shape[1]
    translations: list[torch.Tensor] = []
    yaws: list[torch.Tensor] = []
    for step in range(numSteps):
        currentPos, currentYaw = _applyOneDelta(
            currentPos, currentYaw, localDeltas[:, step, :]
        )
        translations.append(currentPos.clone())
        yaws.append(currentYaw.clone())
    return torch.stack(translations, dim=1), torch.stack(yaws, dim=1)


def _applyOneDelta(
    currentPos: torch.Tensor,
    currentYaw: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one root-local delta step; return updated (pos, yaw)."""
    dFwd, dLat, dHeight, dYaw = (
        delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    )
    cos = torch.cos(currentYaw)
    sin = torch.sin(currentYaw)
    worldDx = cos * dFwd - sin * dLat
    worldDz = sin * dFwd + cos * dLat
    nextPos = currentPos + torch.stack([worldDx, dHeight, worldDz], dim=-1)
    return nextPos, currentYaw + dYaw


# -------------------------------------------------------------------------
# Aim direction: facing direction from pelvis yaw (decoupled from velocity)
# -------------------------------------------------------------------------

def aimDirectionFromPelvisYaw(
    rotation6d: torch.Tensor,
) -> torch.Tensor:
    """Derive the aim direction from the pelvis yaw (decoupled from velocity).

    Returns a unit 2-vector ``(cos θ, sin θ)`` in the world ground plane,
    representing where the character is *facing* regardless of which
    direction it is moving.  This decouples aim from locomotion, as
    required by ROADMAP_DETERMINIST §2.2.b.

    Parameters
    ----------
    rotation6d : torch.Tensor
        Bone rotations, shape ``(F, numBones, 6)``.

    Returns
    -------
    torch.Tensor
        Aim direction unit vector, shape ``(F, 2)`` — ``(cos θ, sin θ)``.
    """
    pelvisRot6d = rotation6d[:, _PELVIS_BONE_INDEX, :]  # (F, 6)
    yaw = pelvisYawFromRot6d(pelvisRot6d)               # (F,)
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    return torch.stack([cos, sin], dim=-1)
