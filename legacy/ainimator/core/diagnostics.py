"""Lightweight training/inference diagnostics writer.

Enable by setting the environment variable ``AI_NIMATOR_DIAG_DIR`` to a
directory path before launching training or generation.  A JSONL file is
opened per run and every hooked event is appended as one line.

The writer is a singleton: when not initialized, all ``log_*`` calls are
no-ops so the diagnostics plumbing can stay live in the codebase with
zero runtime cost when the env var is absent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

LOGGER = logging.getLogger("generation.diag")
ENV_VAR = "AI_NIMATOR_DIAG_DIR"

_CURRENT: Optional["DiagLogger"] = None


@dataclass
class TensorStats:
    """Compact descriptor for a tensor (detached, cpu)."""

    mean: float
    std: float
    minV: float
    maxV: float
    absMean: float

    @classmethod
    def fromTensor(cls, tensor: Optional[torch.Tensor]) -> Optional["TensorStats"]:
        """Compute stats safely; return None for ``None`` input."""
        if tensor is None:
            return None
        detached = tensor.detach()
        if detached.numel() == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0)
        floatTensor = detached.float()
        return cls(
            mean=float(floatTensor.mean().item()),
            std=float(floatTensor.std(unbiased=False).item()),
            minV=float(floatTensor.min().item()),
            maxV=float(floatTensor.max().item()),
            absMean=float(floatTensor.abs().mean().item()),
        )

    def toDict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minV,
            "max": self.maxV,
            "abs_mean": self.absMean,
        }


class DiagLogger:
    """JSONL diagnostics writer."""

    def __init__(self, directory: Path, tag: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = directory / f"{stamp}_{tag}.jsonl"
        self._handle = open(self.path, "a", buffering=1)  # line-buffered
        self._runStart = time.time()
        LOGGER.info("Diagnostics writing to %s", self.path)

    def write(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("t_elapsed", time.time() - self._runStart)
        self._handle.write(json.dumps(payload, default=_jsonDefault) + "\n")

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # noqa: BLE001
            pass


def _jsonDefault(value: Any) -> Any:
    if isinstance(value, TensorStats):
        return value.toDict()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def init_from_env(tag: str = "run") -> Optional[DiagLogger]:
    """Create the global logger from ``AI_NIMATOR_DIAG_DIR`` if set."""
    global _CURRENT
    directory = os.environ.get(ENV_VAR)
    if not directory:
        return None
    if _CURRENT is not None:
        return _CURRENT
    _CURRENT = DiagLogger(Path(directory).expanduser(), tag=tag)
    return _CURRENT


def init(directory: Path, tag: str = "run") -> DiagLogger:
    """Explicit init used by tests/CLI overrides."""
    global _CURRENT
    if _CURRENT is not None:
        return _CURRENT
    _CURRENT = DiagLogger(directory, tag=tag)
    return _CURRENT


def get_logger() -> Optional[DiagLogger]:
    return _CURRENT


def close() -> None:
    global _CURRENT
    if _CURRENT is not None:
        _CURRENT.close()
        _CURRENT = None


def _write(record: Mapping[str, Any]) -> None:
    logger = _CURRENT
    if logger is None:
        return
    try:
        logger.write(record)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Diagnostics write failed; disabling.")
        close()


def computeGradientNorm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Compute the total L2 norm of available gradients."""
    total = 0.0
    hasAny = False
    for param in parameters:
        if param.grad is None:
            continue
        hasAny = True
        total += float(param.grad.detach().norm(2).item()) ** 2
    if not hasAny:
        return 0.0
    return total ** 0.5


def logTrainBatch(
    *,
    epoch: int,
    batchIdx: int,
    globalStep: int,
    timesteps: torch.Tensor,
    loss: float,
    components: Mapping[str, float],
    normalizedTarget: Optional[torch.Tensor],
    noisyInput: Optional[torch.Tensor],
    noise: Optional[torch.Tensor],
    predictedMotion: Optional[torch.Tensor],
    targetMotion: Optional[torch.Tensor],
    gradNorm: Optional[float],
) -> None:
    """Log a single training batch record."""
    if _CURRENT is None:
        return
    tCpu = timesteps.detach().cpu().tolist()
    record = {
        "phase": "train_batch",
        "epoch": int(epoch),
        "batch_idx": int(batchIdx),
        "global_step": int(globalStep),
        "timesteps": tCpu,
        "t_mean": float(sum(tCpu) / max(len(tCpu), 1)),
        "loss": float(loss),
        "components": {str(k): float(v) for k, v in components.items()},
        "grad_norm": float(gradNorm) if gradNorm is not None else None,
        "stats": {
            "normalized_target": _statDict(normalizedTarget),
            "noisy_input": _statDict(noisyInput),
            "noise": _statDict(noise),
            "predicted_motion": _statDict(predictedMotion),
            "target_motion": _statDict(targetMotion),
        },
    }
    _write(record)


def _statDict(tensor: Optional[torch.Tensor]) -> Optional[dict[str, float]]:
    stats = TensorStats.fromTensor(tensor)
    return stats.toDict() if stats is not None else None


def logEpochSummary(
    *,
    epoch: int,
    trainLoss: float,
    valLoss: Optional[float],
    components: Mapping[str, float],
    timestepBuckets: Mapping[int, dict[str, float]],
    learningRate: float,
) -> None:
    if _CURRENT is None:
        return
    record = {
        "phase": "epoch_summary",
        "epoch": int(epoch),
        "train_loss": float(trainLoss),
        "val_loss": None if valLoss is None else float(valLoss),
        "components": {str(k): float(v) for k, v in components.items()},
        "timestep_buckets": {
            str(bucket): {str(k): float(v) for k, v in values.items()}
            for bucket, values in timestepBuckets.items()
        },
        "learning_rate": float(learningRate),
    }
    _write(record)


def logValSummary(
    *,
    epoch: int,
    valLoss: float,
    components: Mapping[str, float],
) -> None:
    """
    Log a validation-phase summary as a separate structured event.

    The training-side ``epoch_summary`` is emitted inside ``trainOneEpoch``
    (before validation runs) and therefore cannot carry validation metrics.
    This helper is called from the CLI after ``evaluateValidation`` so the
    jsonl stream contains a structured val record — otherwise the only
    trace of validation was an ad-hoc console line.  Downstream analysis
    joins ``val_summary`` and ``epoch_summary`` by ``epoch``.
    """
    if _CURRENT is None:
        return
    record = {
        "phase": "val_summary",
        "epoch": int(epoch),
        "val_loss": float(valLoss),
        "components": {str(k): float(v) for k, v in components.items()},
    }
    _write(record)


def logDdimStep(
    *,
    generationId: str,
    stepIdx: int,
    timestep: int,
    xBone: Optional[torch.Tensor],
    condX0: Optional[torch.Tensor],
    uncondX0: Optional[torch.Tensor],
    guidedX0: Optional[torch.Tensor],
    cfgScale: float,
) -> None:
    if _CURRENT is None:
        return
    cfgDelta = None
    if condX0 is not None and uncondX0 is not None:
        delta = (condX0.detach() - uncondX0.detach())
        cfgDelta = {
            "norm_mean": float(delta.float().norm(dim=-1).mean().item()),
            "abs_max": float(delta.float().abs().max().item()),
        }
    record = {
        "phase": "ddim_step",
        "generation_id": str(generationId),
        "step_idx": int(stepIdx),
        "timestep": int(timestep),
        "cfg_scale": float(cfgScale),
        "x_bone": _statDict(xBone),
        "cond_x0": _statDict(condX0),
        "uncond_x0": _statDict(uncondX0),
        "guided_x0": _statDict(guidedX0),
        "cfg_delta": cfgDelta,
    }
    _write(record)


def logGenerationSummary(
    *,
    generationId: str,
    prompt: str,
    numFrames: int,
    ddimSteps: int,
    cfgScale: float,
    motionQuat: Optional[torch.Tensor],
) -> None:
    if _CURRENT is None:
        return
    perFrameStd: Optional[list[float]] = None
    frameDelta: Optional[dict[str, float]] = None
    if motionQuat is not None and motionQuat.dim() >= 3:
        # motionQuat: (batch, frames, bones, 4) — measure temporal variation.
        m = motionQuat.detach().float()
        if m.shape[1] >= 2:
            delta = m[:, 1:] - m[:, :-1]  # frame-to-frame delta
            frameDelta = {
                "mean_abs": float(delta.abs().mean().item()),
                "std": float(delta.std(unbiased=False).item()),
                "max_abs": float(delta.abs().max().item()),
            }
        # Std over frames per bone-channel, then averaged.
        stdPerFrameAxis = m.std(dim=1, unbiased=False)
        perFrameStd = [float(stdPerFrameAxis.mean().item())]
    record = {
        "phase": "generation_summary",
        "generation_id": str(generationId),
        "prompt": prompt,
        "num_frames": int(numFrames),
        "ddim_steps": int(ddimSteps),
        "cfg_scale": float(cfgScale),
        "motion_quat": _statDict(motionQuat),
        "frame_delta": frameDelta,
        "per_frame_std_mean": perFrameStd,
    }
    _write(record)
