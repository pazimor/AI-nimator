"""Channel layout and enums for the Goal C deterministic controller.

This module is the deterministic counterpart of the lean diffusion
representation (ROADMAP §2 truth #2): it freezes — once, before any
controller logic — the **state vector** the autoregressive controller
regresses and the **control signal** that drives it (ROADMAP_DETERMINIST
§2.2).

Design rule inherited from the diffusion stack
----------------------------------------------
The controller predicts ``Δstate`` over the *lean* representation only:

* ``rotation6d``        — 22 bones × 6 channels (132)
* ``root_translation``  — 3 channels

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
# Lean state (regressed) — shared with the diffusion representation.
# ---------------------------------------------------------------------
#: Channels of a single 6D rotation (Zhou et al. 2019).
ROTATION6D_CHANNELS: int = 6

#: Global root translation channels (x, y, z), in meters.
ROOT_TRANSLATION_CHANNELS: int = 3

#: Number of SMPL-22 bones carrying a 6D rotation.
NUM_SMPL22_BONES: int = len(SMPL22_BONE_ORDER)

#: Flattened length of the lean regressed state for SMPL-22 (135).
LEAN_STATE_CHANNELS: int = (
    NUM_SMPL22_BONES * ROTATION6D_CHANNELS + ROOT_TRANSLATION_CHANNELS
)


# ---------------------------------------------------------------------
# Control signal (input conditioning, ROADMAP_DETERMINIST §2.2 / C2).
# ---------------------------------------------------------------------
#: Desired planar root velocity on the ground plane (forward, lateral),
#: expressed in the root-local frame, meters per frame.
CONTROL_PLANAR_VELOCITY_CHANNELS: int = 2

#: Desired aim/heading direction as a unit 2-vector (cos θ, sin θ) in
#: the ground plane.  Added in phase C2; absent in the C1 minimal gate.
CONTROL_AIM_DIRECTION_CHANNELS: int = 2


def controlSignalChannels(useAimDirection: bool) -> int:
    """Return the control-signal width for the given feature toggle.

    Parameters
    ----------
    useAimDirection : bool
        When ``True`` the aim-direction unit vector is appended to the
        desired planar velocity (the C2 rich control signal).

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
    * ``EXPLICIT`` — phase fed as an input signal (C1/C2 default).
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
    is the Goal C deterministic engine.  The two coexist behind the
    ``model-type`` flag in ``network.yaml``.
    """

    DIFFUSION = "diffusion"
    CONTROLLER = "controller"
