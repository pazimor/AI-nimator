"""Health metrics for the Goal C controller (ROADMAP_DETERMINIST §4).

Reinterprets the existing collapse probes for the deterministic engine:
in diffusion, collapse = the prompt is ignored; in the controller,
collapse = the output is **insensitive to the control** or **regresses
to the mean**.  The probe internals ``effective_rank`` /
``intra_batch_sim`` are reused verbatim (only reinterpreted), as required
by §4.

The three C1 contracts are:

* ``control_sensitivity`` — output change when the control is shuffled
  (higher is better; ≈ 0 means the control is ignored).
* ``mean_collapse`` — ``effective_rank`` + ``intra_batch_sim`` of the
  outputs under varied control (rank up / sim down = healthy).
* ``rollout_drift`` — trajectory error vs the ground truth over the
  rollout horizon (lower / flat = healthy).

Plus the inherited ``post_norm_stats`` assertion on the normalized state
and deltas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import yaml

from ainimator.health.contract import Contract
from ainimator.health.probe import (
    _computeEffectiveRank,
    _computeIntraBatchSim,
)
from ainimator.model.controller_rollout import (
    RolloutResult,
    rolloutControllerClosedLoop,
)
from ainimator.model.controller_v2 import ControllerOutput, MotionController
from ainimator.model.motion_normalizer import MotionNormalizer

# Names of the contracts owned by the controller engine.  Loaded from
# the shared health.yaml; diffusion never produces these metrics.
CONTROLLER_CONTRACT_NAMES: tuple[str, ...] = (
    "control_sensitivity",
    "mean_collapse_rank",
    "mean_collapse_sim",
    "rollout_drift",
    "post_norm_stats",
)

_EPS = 1e-8


def flattenOutput(output: ControllerOutput) -> torch.Tensor:
    """Flatten a controller output into ``(B, totalOutputDim)``."""
    batchSize = output.boneDelta.shape[0]
    parts = [output.boneDelta.reshape(batchSize, -1)]
    if output.globalDelta is not None:
        parts.append(output.globalDelta.reshape(batchSize, -1))
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def controlSensitivity(
    model: MotionController,
    boneWindow: torch.Tensor,
    control: torch.Tensor,
    globalWindow: torch.Tensor | None = None,
    phase: torch.Tensor | None = None,
) -> float:
    """Relative output change when the control signal is shuffled.

    Mirrors the diffusion conditioning-sensitivity probe.  Requires a
    batch of at least two samples so the shuffle is a real permutation.
    The phase (when present) is held fixed so the measured change is
    attributable to the control alone.

    Returns
    -------
    float
        ``mean ||f(ctrl) - f(shuffle(ctrl))|| / ||f(ctrl)||``; near 0
        means the controller ignores the control.
    """
    if boneWindow.shape[0] < 2:
        raise ValueError("controlSensitivity needs batch size >= 2.")
    realOut = flattenOutput(
        model(boneWindow, control, globalWindow=globalWindow, phase=phase)
    )
    shuffled = control[torch.roll(torch.arange(control.shape[0]), 1)]
    shufOut = flattenOutput(
        model(boneWindow, shuffled, globalWindow=globalWindow, phase=phase)
    )
    delta = (realOut - shufOut).norm(dim=-1)
    base = realOut.norm(dim=-1).clamp(min=_EPS)
    return float((delta / base).mean().item())


def meanCollapse(output: ControllerOutput) -> tuple[float, float]:
    """Return ``(effective_rank, intra_batch_sim)`` of varied outputs.

    The outputs must come from a batch driven by *varied* control so a
    healthy controller produces a high-rank, low-similarity set.
    """
    flat = flattenOutput(output)
    rank = _computeEffectiveRank(flat)
    sim = _computeIntraBatchSim(flat) if flat.shape[0] >= 2 else 0.0
    return rank, sim


def rolloutDrift(
    rollout: RolloutResult,
    groundTruthRotation6d: torch.Tensor,
    groundTruthRootTranslation: torch.Tensor,
) -> float:
    """Mean per-frame trajectory error between rollout and ground truth.

    Parameters
    ----------
    rollout : RolloutResult
        The rolled-out trajectory (seed + predicted frames).
    groundTruthRotation6d : torch.Tensor
        ``(B, K + N, numBones, 6)`` reference rotations.
    groundTruthRootTranslation : torch.Tensor
        ``(B, K + N, 3)`` reference root translation.

    Returns
    -------
    float
        ``mean MSE(rot6d) + mean MSE(root)`` over all frames; bounded and
        flat for a healthy short-horizon rollout.
    """
    rotError = torch.mean(
        (rollout.rotation6d - groundTruthRotation6d) ** 2
    )
    rootError = torch.mean(
        (rollout.rootTranslation - groundTruthRootTranslation) ** 2
    )
    return float((rotError + rootError).item())


def rolloutDriftCurve(
    rollout: RolloutResult,
    groundTruthRotation6d: torch.Tensor,
    groundTruthRootTranslation: torch.Tensor,
    horizons: Iterable[int],
) -> dict[int, float]:
    """Cumulative rollout drift at increasing horizons (C4).

    Produces the "drift vs rollout length" curve recorded in the health
    report (ROADMAP_DETERMINIST C4): for each horizon ``h`` it measures
    the trajectory error over the first ``h`` frames.  A healthy
    long-horizon controller keeps this curve bounded (no freeze /
    explosion).

    Parameters
    ----------
    rollout : RolloutResult
        The rolled-out trajectory.
    groundTruthRotation6d, groundTruthRootTranslation : torch.Tensor
        Reference trajectory aligned with the rollout frames.
    horizons : Iterable[int]
        Frame counts at which to evaluate the drift.

    Returns
    -------
    dict[int, float]
        Maps each (clamped) horizon to its drift value.
    """
    totalFrames = rollout.rotation6d.shape[1]
    curve: dict[int, float] = {}
    for horizon in horizons:
        clamped = max(1, min(int(horizon), totalFrames))
        truncated = RolloutResult(
            rotation6d=rollout.rotation6d[:, :clamped],
            rootTranslation=rollout.rootTranslation[:, :clamped],
        )
        curve[clamped] = rolloutDrift(
            truncated,
            groundTruthRotation6d[:, :clamped],
            groundTruthRootTranslation[:, :clamped],
        )
    return curve


def closedLoopDriftByPeriod(
    model: MotionController,
    stateNormalizer: MotionNormalizer,
    deltaNormalizer: MotionNormalizer,
    groundTruthRotation6d: torch.Tensor,
    groundTruthRootTranslation: torch.Tensor,
    controlSequence: torch.Tensor,
    periods: Iterable[int],
    phaseSequence: torch.Tensor | None = None,
) -> dict[int, float]:
    """Rollout drift vs ground-truth re-injection period (deployment proxy).

    ``period <= 0`` is open-loop (the worst case); smaller positive
    periods model more frequent engine re-grounding.  A controller is
    deployment-viable if a realistic period (e.g. every 16-64 frames)
    keeps the drift small even when the open-loop drift is large.

    Returns
    -------
    dict[int, float]
        Maps each period to the trajectory drift under that re-injection
        rate (``0`` key = open-loop).
    """
    curve: dict[int, float] = {}
    for period in periods:
        rollout = rolloutControllerClosedLoop(
            model, stateNormalizer, deltaNormalizer, groundTruthRotation6d,
            groundTruthRootTranslation, controlSequence, max(int(period), 0),
            phaseSequence=phaseSequence,
        )
        curve[int(period)] = rolloutDrift(
            rollout, groundTruthRotation6d, groundTruthRootTranslation
        )
    return curve


def postNormStats(normalized: torch.Tensor) -> float:
    """Worst-case deviation of a normalized tensor from ``N(0, 1)``.

    Returns ``max(|mean|, |std - 1|)`` so the inherited
    ``post_norm_stats`` contract (lower is better) applies unchanged.
    """
    mean = float(normalized.float().mean().item())
    std = float(normalized.float().std(unbiased=False).item())
    return max(abs(mean), abs(std - 1.0))


def loadControllerContracts(
    healthPath: Path,
    names: Iterable[str] = CONTROLLER_CONTRACT_NAMES,
) -> dict[str, Contract]:
    """Load the controller contracts from the shared health.yaml.

    Parameters
    ----------
    healthPath : Path
        Path to ``src/configs/health.yaml``.
    names : Iterable[str]
        Contract names to extract.

    Returns
    -------
    dict[str, Contract]
        Keyed by contract name.
    """
    payload = yaml.safe_load(healthPath.read_text(encoding="utf-8")) or {}
    wanted = set(names)
    contracts: dict[str, Contract] = {}
    for entry in payload.get("contracts", []):
        if str(entry.get("name")) in wanted:
            contract = Contract.fromYamlDict(entry)
            contracts[contract.name] = contract
    return contracts
