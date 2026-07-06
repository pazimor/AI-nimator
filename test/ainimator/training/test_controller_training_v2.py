"""Phase A1 feasibility-gate tests — overfit one clip, roll it out.

ROADMAP_DETERMINIST A1 acceptance: overfit one sequence → a short
rollout that reproduces it (bounded ``rollout_drift``), ``post_norm_stats``
OK on state and deltas, and a round-tripping checkpoint.  Runs on
synthetic data (no dataset on disk), mirroring ``make smoke-test``.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from ainimator.core.constants.controller import (
    CONTROL_AIM_DIRECTION_CHANNELS,
    CONTROL_PLANAR_VELOCITY_CHANNELS,
    PhaseMode,
)
from ainimator.health.contract import Verdict
from ainimator.model.losses_controller_v2 import ControllerLossWeights
from ainimator.training.controller_training_v2 import (
    ControllerTrainingConfig,
    loadControllerCheckpoint,
    runControllerOverfit,
)


def _syntheticClip(frames: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """A smooth, full-variance periodic clip the controller can overfit."""
    phase = torch.linspace(0.0, 4.0 * math.pi, frames)
    rotation6d = torch.zeros(frames, 22, 6)
    for bone in range(22):
        for channel in range(6):
            rotation6d[:, bone, channel] = math.cos(bone + channel) + (
                0.3 * torch.sin(phase + 0.2 * bone + 0.5 * channel)
            )
    # Project onto valid 6D rotations: real (AMASS) states are always on
    # the rotation manifold, and the normative inference loop
    # (inference_contract.md §3.6) assumes it.
    from ainimator.geometry.components import orthonormalizeRot6d

    rotation6d = orthonormalizeRot6d(rotation6d)
    rootTranslation = torch.zeros(frames, 3)
    rootTranslation[:, 0] = 0.02 * torch.arange(frames)
    rootTranslation[:, 1] = 0.10 * torch.sin(phase)
    rootTranslation[:, 2] = 0.05 * torch.cos(phase)
    return rotation6d, rootTranslation


def _overfitConfig(outputDir: Path) -> ControllerTrainingConfig:
    return ControllerTrainingConfig(
        outputDir=outputDir,
        # 600 epochs: the manifold-projected synthetic clip (valid 6D,
        # lower variance) converges slower than the pre-§3.6 raw one —
        # reaches < 0.02 at 600 (probe 2026-07-05), vs 0.17 at 250.
        epochs=600,
        device="cpu",
        embedDim=128,
        numHeads=4,
        numLayers=3,
        logEvery=120,
        seed=0,
    )


def test_run_overfit_reproduces_clip(tmp_path: Path) -> None:
    rotation6d, rootTranslation = _syntheticClip()
    result = runControllerOverfit(
        rotation6d, rootTranslation, _overfitConfig(tmp_path)
    )
    # The autoregressive rollout under GT control reproduces the clip.
    assert result.verdicts["rollout_drift"] is Verdict.OK
    assert result.metrics["rollout_drift"] < 0.05
    # Z-normalization holds on state AND deltas.
    assert result.verdicts["post_norm_stats"] is Verdict.OK
    # The overfit actually drove the loss down.
    assert result.finalLoss < 0.05


def test_run_overfit_writes_artifacts(tmp_path: Path) -> None:
    rotation6d, rootTranslation = _syntheticClip()
    result = runControllerOverfit(
        rotation6d, rootTranslation, _overfitConfig(tmp_path)
    )
    assert result.checkpointPath.exists()
    assert result.rolloutPath.exists()
    assert (tmp_path / "resolved_config.yaml").exists()


def test_checkpoint_round_trips(tmp_path: Path) -> None:
    rotation6d, rootTranslation = _syntheticClip()
    result = runControllerOverfit(
        rotation6d, rootTranslation, _overfitConfig(tmp_path)
    )
    model, stateNorm, deltaNorm, controlMean, controlStd = (
        loadControllerCheckpoint(result.checkpointPath)
    )
    model.eval()
    bone = torch.randn(2, model.config.contextFrames, 22, 6)
    glob = torch.randn(2, model.config.contextFrames, model.config.globalChannels)
    control = torch.randn(2, model.config.controlChannels)
    out = model(bone, control, globalWindow=glob)
    assert out.boneDelta.shape == (2, 22, 6)
    assert out.globalDelta is not None
    assert out.globalDelta.shape == (2, model.config.globalChannels)
    # controlMean/controlStd cover only the velocity channels (vx, vz);
    # aim channels are unit-norm by construction and are not z-normalised
    # (ROADMAP_DETERMINIST §2.2.b / G-ZNORM aim invariant).
    assert controlMean.shape[-1] == CONTROL_PLANAR_VELOCITY_CHANNELS
    assert controlStd.shape[-1] == CONTROL_PLANAR_VELOCITY_CHANNELS


# ---------------------------------------------------------------------
# A2 — explicit phase + rich control + foot contact
# ---------------------------------------------------------------------
def _c2Config(outputDir: Path) -> ControllerTrainingConfig:
    return ControllerTrainingConfig(
        outputDir=outputDir,
        epochs=250,
        device="cpu",
        embedDim=128,
        numHeads=4,
        numLayers=3,
        logEvery=120,
        seed=0,
        phaseMode=PhaseMode.EXPLICIT,
        useAimDirection=True,
        lossWeights=ControllerLossWeights(
            velocity=1.0, geodesic=1.0, footContact=0.5
        ),
    )


def test_c2_phase_and_rich_control_trains_and_reproduces(
    tmp_path: Path,
) -> None:
    rotation6d, rootTranslation = _syntheticClip()
    result = runControllerOverfit(
        rotation6d, rootTranslation, _c2Config(tmp_path)
    )
    # Still overfits + reproduces with phase + aim + foot-contact wired.
    assert result.verdicts["rollout_drift"] is Verdict.OK
    assert result.verdicts["post_norm_stats"] is Verdict.OK
    # Rich control (velocity + aim) keeps the control signal informative:
    # control_sensitivity must stay strictly positive.
    assert result.metrics["control_sensitivity"] > 0.0
    # mean_collapse SAIN is a varied-control (multi-clip) gate, NOT a
    # single-overfit-clip property — see ROADMAP_DETERMINIST A2.


# ---------------------------------------------------------------------
# Z-norm aim invariant unit test
# ---------------------------------------------------------------------
def test_standardise_control_aim_channels_unchanged() -> None:
    """Aim channels must not be z-normalised (they are unit-norm by
    construction; ROADMAP_DETERMINIST §2.2.b, G-ZNORM aim invariant).

    When aim-direction is active the control tensor has 4 channels:
    (vx, vz, aim_x, aim_z).  After _standardiseControl the velocity
    channels must be z-normalised (mean≈0, std≈1) and the aim channels
    must be returned **unchanged** (still on the unit circle).
    """
    from ainimator.training.controller_training_v2 import _standardiseControl

    torch.manual_seed(0)
    totalChannels = (
        CONTROL_PLANAR_VELOCITY_CHANNELS + CONTROL_AIM_DIRECTION_CHANNELS
    )
    nSamples = 64
    # Velocity: arbitrary non-zero mean + scale.
    velocity = torch.randn(nSamples, CONTROL_PLANAR_VELOCITY_CHANNELS) * 3.0
    velocity += torch.tensor([2.0, -1.5])
    # Aim: unit vectors on the circle.
    angles = torch.linspace(0.0, 2 * math.pi, nSamples)
    aim = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    control = torch.cat([velocity, aim], dim=-1)

    normControl, mean, std = _standardiseControl(control)

    # mean/std cover only the velocity channels.
    assert mean.shape == (1, CONTROL_PLANAR_VELOCITY_CHANNELS)
    assert std.shape == (1, CONTROL_PLANAR_VELOCITY_CHANNELS)

    # Velocity channels are z-normalised.
    normVel = normControl[:, :CONTROL_PLANAR_VELOCITY_CHANNELS]
    assert abs(float(normVel.mean())) < 1e-5, (
        "Velocity mean should be ≈0 after z-norm"
    )
    assert abs(float(normVel.std()) - 1.0) < 0.05, (
        "Velocity std should be ≈1 after z-norm"
    )

    # Aim channels are unchanged.
    normAim = normControl[:, CONTROL_PLANAR_VELOCITY_CHANNELS:]
    assert normAim.shape[-1] == CONTROL_AIM_DIRECTION_CHANNELS
    assert torch.allclose(normAim, aim, atol=1e-6), (
        "Aim channels must pass through _standardiseControl unchanged"
    )
    # Still unit-norm (sanity check).
    norms = normAim.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(nSamples), atol=1e-5)
