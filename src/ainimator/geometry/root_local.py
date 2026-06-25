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
    frames = rootTranslation.shape[0]
    device = rootTranslation.device
    dtype = rootTranslation.dtype

    pelvisRot6d = rotation6d[:, _PELVIS_BONE_INDEX, :]  # (F, 6)
    yaw = pelvisYawFromRot6d(pelvisRot6d)               # (F,)

    # World-space displacement per frame; frame 0 has no predecessor → zero.
    worldDelta = torch.zeros(frames, 3, device=device, dtype=dtype)
    worldDelta[1:] = rootTranslation[1:] - rootTranslation[:-1]

    # Ground-plane displacement (x, z).
    planarDelta = worldDelta[:, [_AXIS_X, _AXIS_Z]]     # (F, 2)
    heightDelta = worldDelta[:, _AXIS_Y]                # (F,)

    # Rotate planar delta into the local frame using the yaw at the
    # **departure** frame (where the step originates), so that the
    # integration in :func:`rootLocalDeltasToAbsolute` uses the same
    # reference frame.  Delta at index t goes from frame t-1 → frame t;
    # the departure yaw is yaw[t-1].  Frame 0 has no predecessor, so it
    # uses yaw[0] (its delta is zero anyway).
    departureYaw = torch.cat([yaw[:1], yaw[:-1]], dim=0)  # shift by 1
    rotMat = yawToRotationMatrix2d(departureYaw)        # (F, 2, 2)
    # planarDelta: (F, 2) → (F, 2, 1)
    localPlanar = torch.matmul(
        rotMat.transpose(-1, -2), planarDelta.unsqueeze(-1)
    ).squeeze(-1)                                       # (F, 2)

    # Yaw delta (wrapped to [-π, π]).
    yawDelta = torch.zeros(frames, device=device, dtype=dtype)
    rawDiff = yaw[1:] - yaw[:-1]
    yawDelta[1:] = torch.atan2(torch.sin(rawDiff), torch.cos(rawDiff))

    # Stack: (Δforward, Δlateral, Δheight, Δyaw).
    return torch.stack(
        [localPlanar[:, 0], localPlanar[:, 1], heightDelta, yawDelta],
        dim=-1,
    )


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
        Integrated world-space XYZ trajectory, shape ``(N, 3)`` or
        ``(B, N, 3)``.  The seed frame itself is **not** included.
    yaws : torch.Tensor
        Cumulative world-space yaw, shape ``(N,)`` or ``(B, N)``.
    """
    batched = localDeltas.ndim == 3
    if not batched:
        # Add a batch dimension for uniform processing.
        localDeltas = localDeltas.unsqueeze(0)
        seedTranslation = seedTranslation.unsqueeze(0)
        if isinstance(seedYaw, (int, float)):
            seedYaw = torch.tensor(
                seedYaw,
                dtype=localDeltas.dtype,
                device=localDeltas.device,
            ).unsqueeze(0)
        else:
            seedYaw = seedYaw.unsqueeze(0)  # type: ignore[union-attr]

    batchSize, numSteps, _ = localDeltas.shape
    device = localDeltas.device
    dtype = localDeltas.dtype

    if isinstance(seedYaw, (int, float)):
        currentYaw = torch.full(
            (batchSize,), float(seedYaw), dtype=dtype, device=device
        )
    else:
        currentYaw = seedYaw.to(device=device, dtype=dtype)

    currentPos = seedTranslation.to(device=device, dtype=dtype)  # (B, 3)

    translations: list[torch.Tensor] = []
    yaws: list[torch.Tensor] = []

    for step in range(numSteps):
        dFwd = localDeltas[:, step, 0]
        dLat = localDeltas[:, step, 1]
        dHeight = localDeltas[:, step, 2]
        dYaw = localDeltas[:, step, 3]

        # Rotate local (fwd, lat) displacement into world frame.
        cos = torch.cos(currentYaw)
        sin = torch.sin(currentYaw)
        worldDx = cos * dFwd - sin * dLat
        worldDz = sin * dFwd + cos * dLat

        currentPos = currentPos + torch.stack(
            [worldDx, dHeight, worldDz], dim=-1
        )
        currentYaw = currentYaw + dYaw

        translations.append(currentPos.clone())
        yaws.append(currentYaw.clone())

    translationsTensor = torch.stack(translations, dim=1)  # (B, N, 3)
    yawsTensor = torch.stack(yaws, dim=1)                  # (B, N)

    if not batched:
        return translationsTensor.squeeze(0), yawsTensor.squeeze(0)
    return translationsTensor, yawsTensor


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
