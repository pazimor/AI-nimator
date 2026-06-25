"""Channel layout and enums for the Goal A deterministic controller.

This module is the deterministic counterpart of the lean diffusion
representation (ROADMAP §2 truth #2): it freezes — once, before any
controller logic — the **state vector** the autoregressive controller
regresses and the **control signal** that drives it (ROADMAP_DETERMINIST
§2.2).

State vector (136 channels — ROADMAP_DETERMINIST §2.2.a)
---------------------------------------------------------
The controller predicts ``Δstate`` over the *lean* representation:

* ``rotation6d``         — 22 bones × 6 channels (132)
* ``root_local_motion``  — 4 channels: ``(Δforward, Δlateral, Δheight, Δyaw)``

The root is expressed in the **character-local frame** (pelvis-yaw origin),
**not** as an absolute world-space translation.  This prevents the
translation from drifting to infinity during autoregression and keeps the
z-norm / mean ≈ 0 (ROADMAP_DETERMINIST §2.2.a, §2.3 truth #3).

Every FK-derivable signal (planar root velocity, yaw rate, foot
contacts, joint velocities) is **supervised at the loss** via forward
kinematics, never stacked as a redundant predicted channel.  The
velocity / contact constants below therefore describe *supervision and
conditioning* layouts, not extra outputs.
"""

from __future__ import annotations

from enum import Enum

from ainimator.core.constants.skeletons import SMPL22_BONE_ORDER

# ---------------------------------------------------------------------
# Lean state (regressed) — root-local representation (136 channels).
# ---------------------------------------------------------------------
#: Channels of a single 6D rotation (Zhou et al. 2019).
ROTATION6D_CHANNELS: int = 6

#: Root-local motion channels: (Δforward, Δlateral, Δheight, Δyaw).
#: Replaces the 3-channel absolute ``root_translation`` from the diffusion
#: lean representation (ROADMAP_DETERMINIST §2.2.a, 2026-06-24).
ROOT_LOCAL_MOTION_CHANNELS: int = 4

#: Index of the Δforward channel within the root-local motion vector.
ROOT_LOCAL_FORWARD_IDX: int = 0
#: Index of the Δlateral channel.
ROOT_LOCAL_LATERAL_IDX: int = 1
#: Index of the Δheight channel.
ROOT_LOCAL_HEIGHT_IDX: int = 2
#: Index of the Δyaw channel (radians, Y-up).
ROOT_LOCAL_YAW_IDX: int = 3

#: Number of SMPL-22 bones carrying a 6D rotation.
NUM_SMPL22_BONES: int = len(SMPL22_BONE_ORDER)

#: Flattened length of the lean regressed state for SMPL-22 (136).
#: = 22 × 6 (rotation6d) + 4 (root-local motion).
LEAN_STATE_CHANNELS: int = (
    NUM_SMPL22_BONES * ROTATION6D_CHANNELS + ROOT_LOCAL_MOTION_CHANNELS
)


# ---------------------------------------------------------------------
# Control signal (input conditioning, ROADMAP_DETERMINIST §2.2.b).
# Layout (4 channels, ordered):
#   (vx, vz, aim_x, aim_z)
#    ├── (vx, vz)        desired planar velocity, root-local frame, m/frame
#    └── (aim_x, aim_z)  facing direction unit 2-vector (cos θ, sin θ)
#
# vx/vz are z-normalized (asserted by post_norm_stats).
# aim_x/aim_z are unit-norm by construction; excluded from z-norm.
# ---------------------------------------------------------------------
#: Desired planar root velocity (forward, lateral) in the root-local
#: ground-plane frame (X-Z, Y-up), meters per frame.
CONTROL_PLANAR_VELOCITY_CHANNELS: int = 2

#: Desired aim/facing direction as a unit 2-vector ``(cos θ, sin θ)``
#: in the world ground plane, **decoupled** from locomotion direction.
#: Added in phase A2; absent in the A1 minimal gate.
CONTROL_AIM_DIRECTION_CHANNELS: int = 2


def controlSignalChannels(useAimDirection: bool) -> int:
    """Return the control-signal width for the given feature toggle.

    Parameters
    ----------
    useAimDirection : bool
        When ``True`` the aim-direction unit vector is appended to the
        desired planar velocity (the A2 rich control signal).

    Returns
    -------
    int
        Total number of control-signal channels.
    """
    channels = CONTROL_PLANAR_VELOCITY_CHANNELS
    if useAimDirection:
        channels += CONTROL_AIM_DIRECTION_CHANNELS
    return channels


# ---------------------------------------------------------------------
# Locomotor phase (ROADMAP_DETERMINIST §2.2 — non optional, §7).
# ---------------------------------------------------------------------
#: Phase encoded as a 2-D point on the unit circle ``(cos φ, sin φ)`` so
#: the conditioning is continuous and wrap-around safe.
PHASE_CHANNELS: int = 2


class PhaseMode(str, Enum):
    """How the locomotor phase is supplied to the controller.

    * ``NONE``     — no phase conditioning (debug / ablation only).
    * ``EXPLICIT`` — phase fed as an input signal (A1/A2 default).
    * ``LEARNED``  — phase inferred internally (Pazimor arbitrage).
    """

    NONE = "none"
    EXPLICIT = "explicit"
    LEARNED = "learned"


def phaseConditioningChannels(phaseMode: PhaseMode) -> int:
    """Return the phase conditioning width for a phase mode."""
    if phaseMode is PhaseMode.NONE:
        return 0
    return PHASE_CHANNELS


# ---------------------------------------------------------------------
# Engine selector (ROADMAP_DETERMINIST §2.3 / §3.2).
# ---------------------------------------------------------------------
class GenerationModelType(str, Enum):
    """Which generation engine a run uses.

    ``DIFFUSION`` is the canonical default (Goal A/B); ``CONTROLLER``
    is the Goal A deterministic engine.  The two coexist behind the
    ``model-type`` flag in ``network.yaml``.
    """

    DIFFUSION = "diffusion"
    CONTROLLER = "controller"
