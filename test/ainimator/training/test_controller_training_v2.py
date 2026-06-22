"""Phase C1 feasibility-gate tests — overfit one clip, roll it out.

ROADMAP_DETERMINIST C1 acceptance: overfit one sequence → a short
rollout that reproduces it (bounded ``rollout_drift``), ``post_norm_stats``
OK on state and deltas, and a round-tripping checkpoint.  Runs on
synthetic data (no dataset on disk), mirroring ``make smoke-test``.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from ainimator.core.constants.controller import PhaseMode
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
    rootTranslation = torch.zeros(frames, 3)
    rootTranslation[:, 0] = 0.02 * torch.arange(frames)
    rootTranslation[:, 1] = 0.10 * torch.sin(phase)
    rootTranslation[:, 2] = 0.05 * torch.cos(phase)
    return rotation6d, rootTranslation


def _overfitConfig(outputDir: Path) -> ControllerTrainingConfig:
    return ControllerTrainingConfig(
        outputDir=outputDir,
        epochs=250,
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
    glob = torch.randn(2, model.config.contextFrames, 3)
    control = torch.randn(2, model.config.controlChannels)
    out = model(bone, control, globalWindow=glob)
    assert out.boneDelta.shape == (2, 22, 6)
    assert controlMean.shape[-1] == model.config.controlChannels
    assert controlStd.shape[-1] == model.config.controlChannels


# ---------------------------------------------------------------------
# C2 — explicit phase + rich control + foot contact
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
    # single-overfit-clip property — see ROADMAP_DETERMINIST C2.
