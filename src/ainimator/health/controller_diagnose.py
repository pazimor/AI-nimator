"""Controller-specific offline diagnostics (Goal A contracts).

Extracted from :mod:`ainimator.health.hub` to keep ``hub.py`` under
G-FILELEN.  This module is **health layer (L4)**: it does not import
``ainimator.training``; the CLI (L5) loads checkpoints and passes
components in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def buildControlVector(
    controlSpec: str,
    controlChannels: int,
    seed: int = 0,
) -> torch.Tensor:
    """Build a 1-D control tensor from a spec string.

    Parameters
    ----------
    controlSpec : str
        Spec string, e.g. ``"forward:1.0"`` or ``"random:0.5"``.
    controlChannels : int
        Number of control channels expected by the model.
    seed : int
        RNG seed for ``"random"`` mode (ignored for ``"forward"``).

    Returns
    -------
    torch.Tensor
        Shape ``(controlChannels,)``.
    """
    keyword, _, scaleStr = controlSpec.partition(":")
    scale = float(scaleStr) if scaleStr else 1.0
    keyword = keyword.strip().lower()

    if keyword == "forward":
        ctrl = torch.zeros(controlChannels)
        ctrl[0] = scale  # forward velocity channel
        return ctrl

    rng = torch.Generator()
    rng.manual_seed(seed)
    ctrl = torch.randn(controlChannels, generator=rng)
    norm = ctrl.norm()
    if float(norm.item()) > 0.0:
        ctrl = ctrl / norm
    return ctrl * scale


def computeRolloutDrift(
    model: nn.Module,
    stateNormalizer: Any,
    deltaNormalizer: Any,
    boneBatch: torch.Tensor,
    globalBatch: torch.Tensor,
    controlBatch: torch.Tensor,
    rolloutFrames: int,
    device: Any,
) -> float:
    """Compute open-loop rollout drift relative to the seed state.

    Parameters
    ----------
    model : MotionController
        Loaded eval'd controller.
    stateNormalizer, deltaNormalizer : MotionNormalizer
        Fitted normalizers from the checkpoint.
    boneBatch : torch.Tensor
        ``(B, contextFrames, numBones, 6)`` seed rotations.
    globalBatch : torch.Tensor
        ``(B, contextFrames, 4)`` seed root-local motion.
    controlBatch : torch.Tensor
        ``(B, controlChannels)`` control signals.
    rolloutFrames : int
        Number of frames to roll out.
    device : torch.device
        Inference device.

    Returns
    -------
    float
        Mean per-frame MSE vs. static repetition of the seed last frame.
    """
    try:
        from ainimator.model.controller_rollout import rolloutController

        seedRot6d = boneBatch.detach().cpu()
        seedRoot = globalBatch.detach().cpu()
        controlSeq = controlBatch.detach().cpu().unsqueeze(1).expand(
            -1, rolloutFrames, -1
        )
        batchSize = seedRot6d.shape[0]
        seedRootAbs = torch.zeros(batchSize, seedRot6d.shape[1], 3)
        rollout = rolloutController(
            model,  # type: ignore[arg-type]
            stateNormalizer,
            deltaNormalizer,
            seedRotation6d=seedRot6d,
            seedRootTranslation=seedRootAbs,
            controlSequence=controlSeq,
        )
        lastFrame = seedRot6d[:, -1:, :, :]
        predictedRot = rollout.rotation6d[:, seedRot6d.shape[1]:, :, :]
        refRot = lastFrame.expand_as(predictedRot)
        return float(torch.mean((predictedRot - refRot) ** 2).item())
    except Exception as exc:
        LOGGER.warning(
            "rollout_drift computation failed: %s; returning NaN", exc
        )
        return float("nan")


def _buildSeedBatch(
    controlSpec: str,
    controlChannels: int,
    contextFrames: int,
    numBones: int,
    batchSize: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch of synthetic seed states for diagnose."""
    boneWindows: list[torch.Tensor] = []
    globalWindows: list[torch.Tensor] = []
    controls: list[torch.Tensor] = []
    for seed in range(batchSize):
        rng = torch.Generator()
        rng.manual_seed(seed)
        boneWindows.append(
            torch.randn(contextFrames, numBones, 6, generator=rng)
        )
        globalWindows.append(
            torch.randn(contextFrames, 4, generator=rng)
        )
        controls.append(
            buildControlVector(controlSpec, controlChannels, seed=seed)
        )
    return (
        torch.stack(boneWindows).to(device),
        torch.stack(globalWindows).to(device),
        torch.stack(controls).to(device),
    )


def _resolveModelConfig(
    model: nn.Module,
    controlChannels: int | None,
) -> tuple[int, int, int]:
    """Read numBones, contextFrames, controlChannels from model config."""
    cfg = getattr(model, "config", None)
    numBones = (
        cfg.numBones if cfg is not None and hasattr(cfg, "numBones") else 22
    )
    contextFrames = getattr(cfg, "contextFrames", 1)
    if controlChannels is None:
        controlChannels = getattr(cfg, "controlChannels", 4)
    return numBones, contextFrames, controlChannels


def _computeContracts(
    model: nn.Module,
    boneBatch: torch.Tensor,
    globalBatch: torch.Tensor,
    controlBatch: torch.Tensor,
    shuffleControl: bool,
) -> dict[str, Any]:
    """Evaluate control_sensitivity, mean_collapse, post_norm."""
    from ainimator.health.controller_metrics import (
        controlSensitivity,
        meanCollapse,
        postNormStats,
    )
    metrics: dict[str, Any] = {}
    numBones = boneBatch.shape[2]
    if shuffleControl:
        metrics["control_sensitivity"] = controlSensitivity(
            model,  # type: ignore[arg-type]
            boneBatch,
            controlBatch,
            globalWindow=globalBatch,
        )
    else:
        metrics["control_sensitivity"] = None
    with torch.no_grad():
        output = model(boneBatch, controlBatch, globalWindow=globalBatch)
    rank, sim = meanCollapse(output)  # type: ignore[arg-type]
    metrics["mean_collapse_rank"] = rank
    metrics["mean_collapse_sim"] = sim
    metrics["post_norm_stats"] = postNormStats(
        boneBatch.reshape(-1, numBones, 6)
    )
    return metrics


def diagnoseController(
    model: nn.Module,
    stateNormalizer: Any,
    deltaNormalizer: Any,
    device: Any,
    outputDir: Path,
    controlSpec: str = "forward:1.0",
    rolloutFrames: int = 120,
    shuffleControl: bool = False,
    seeds: tuple[int, ...] = (0, 42, 123),
    phaseMode: str | None = None,
    controlChannels: int | None = None,
) -> dict[str, Any]:
    """Offline diagnostics for a controller checkpoint.

    Computes the three Goal A contracts:

    * ``control_sensitivity`` — output change under shuffled control.
    * ``mean_collapse_rank`` / ``mean_collapse_sim`` — rank + cosine
      similarity of varied-control outputs.
    * ``rollout_drift`` — short-horizon trajectory error vs. the seed.

    Accepts pre-loaded components so ``health`` does not import
    ``ainimator.training`` (layer L4 constraint).  The CLI (L5) loads
    the checkpoint and passes components here.

    Parameters
    ----------
    model : MotionController
        Loaded and eval'd controller.
    stateNormalizer, deltaNormalizer : MotionNormalizer
        Fitted normalizers from the checkpoint.
    device : torch.device or str
        Inference device.
    outputDir : Path
        Directory where ``health/diagnose_controller.json`` is written.
    controlSpec : str
        Control specification string (e.g. ``"forward:1.0"``).
    rolloutFrames : int
        Number of frames to roll out after the seed window.
    shuffleControl : bool
        When ``True``, also measures ``control_sensitivity``.
    seeds : tuple of int
        RNG seeds for synthetic seed-state generation.
    phaseMode : str or None
        Force a phase mode; ``None`` reads it from the checkpoint.
    controlChannels : int or None
        Override control channel count; inferred from model when ``None``.

    Returns
    -------
    dict
        Diagnostic metrics including all Goal A contract values.
    """
    import json as _json

    dev = torch.device(device) if isinstance(device, str) else device
    numBones, contextFrames, controlChannels = _resolveModelConfig(
        model, controlChannels
    )
    batchSize = max(len(seeds), 2)

    metrics: dict[str, Any] = {
        "model_type": "controller",
        "control_spec": controlSpec,
        "rollout_frames": rolloutFrames,
        "shuffle_control": shuffleControl,
        "seeds": list(seeds),
    }

    boneBatch, globalBatch, controlBatch = _buildSeedBatch(
        controlSpec, controlChannels, contextFrames, numBones, batchSize, dev
    )
    model.eval()
    metrics.update(
        _computeContracts(model, boneBatch, globalBatch, controlBatch,
                          shuffleControl)
    )
    metrics["rollout_drift"] = computeRolloutDrift(
        model, stateNormalizer, deltaNormalizer,
        boneBatch, globalBatch, controlBatch, rolloutFrames, dev,
    )

    diagPath = outputDir / "health" / "diagnose_controller.json"
    diagPath.parent.mkdir(parents=True, exist_ok=True)
    with diagPath.open("w", encoding="utf-8") as fh:
        _json.dump(metrics, fh, indent=2, default=str)
    LOGGER.info("diagnoseController: results written to %s", diagPath)
    return metrics
