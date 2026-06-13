"""Probe: forward/backward hooks that capture scalar stats only.

A Probe attaches to a named module path and records lightweight scalar
statistics at each call without storing any tensor.  This keeps the
memory cost bounded and avoids retaining computation graphs.

Captured stats:
  mean, std, norm      — activation summary stats
  effective_rank       — Shannon-entropy rank of singular values
  intra_batch_sim      — mean pairwise cosine (collapse detector)
  update_ratio         — ||Δw|| / ||w|| per step (gradient health)
  encoder_cond_uncond_sim — pooled cosine cond vs null embedding

All public values are plain Python floats; no tensor is kept alive
after the hook returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
_EPS = 1e-8

# Cap on the batch dimension used for intra_batch_sim to bound cost.
_SIM_BATCH_CAP = 8


# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------
@dataclass
class ProbeSnapshot:
    """One captured snapshot of scalar stats for a probe.

    All values are plain Python floats; ``None`` means the stat
    was not requested or could not be computed.

    Attributes
    ----------
    step : int
        Training step at which the snapshot was captured.
    mean : float or None
        Mean of the flattened activation tensor.
    std : float or None
        Standard deviation of the flattened activation tensor.
    norm : float or None
        L2 norm of the flattened activation tensor.
    effective_rank : float or None
        Stable-rank proxy: ||A||_F^2 / ||A||_2^2.
        Normalised to [0, 1] by dividing by min(B*F, D).
    intra_batch_sim : float or None
        Mean pairwise cosine similarity across the batch dimension
        (flattened per-sample).  Near 1.0 signals collapse.
    update_ratio : float or None
        ||Δw|| / ||w|| measured on the first weight parameter of the
        hooked module (requires backward hook).
    encoder_cond_uncond_sim : float or None
        Cosine between cond and uncond pooled embeddings (set
        externally by HealthHub, not by the hook itself).
    """

    step: int
    mean: float | None = None
    std: float | None = None
    norm: float | None = None
    effective_rank: float | None = None
    intra_batch_sim: float | None = None
    update_ratio: float | None = None
    encoder_cond_uncond_sim: float | None = None


# ------------------------------------------------------------------
# Scalar helpers — no tensors survive beyond the call
# ------------------------------------------------------------------
def _computeMeanStdNorm(
    tensor: torch.Tensor,
) -> tuple[float, float, float]:
    """Return (mean, std, L2-norm) from a detached CPU copy."""
    flat = tensor.detach().float().cpu().reshape(-1)
    return (
        float(flat.mean().item()),
        float(flat.std().item()),
        float(flat.norm().item()),
    )


_RANK_MAX_ROWS = 64  # cap rows fed to SVD (keeps cost O(64^3))
_RANK_MAX_COLS = 64  # cap cols fed to SVD


def _computeEffectiveRank(
    tensor: torch.Tensor,
    maxDim: int = _RANK_MAX_COLS,
    maxRows: int = _RANK_MAX_ROWS,
) -> float:
    """Stable-rank proxy: ||A||_F^2 / ||A||_2^2, normalised to [0,1].

    The SVD is run on a *capped submatrix* of shape
    (min(rows, maxRows), min(cols, maxDim)) to bound cost to
    O(maxRows * maxDim^2).  Capping rows uses evenly-spaced stride
    sampling to preserve the rank signal across the full batch.

    Parameters
    ----------
    tensor : torch.Tensor
        Activation of shape (B, T, D) or (B, D).
    maxDim : int
        Cap the column dimension (default 64).
    maxRows : int
        Cap the row dimension after reshape (default 64).

    Returns
    -------
    float
        Value in [0, 1]; near 0 means rank-1 (collapse), near 1 healthy.
    """
    with torch.no_grad():
        flat = tensor.detach().float().cpu()
        if flat.ndim == 3:
            # (B, T, D) → (B*T, D)
            flat = flat.reshape(-1, flat.shape[-1])
        elif flat.ndim != 2:
            flat = flat.reshape(flat.shape[0], -1)
        # Cap columns (feature dim)
        flat = flat[:, :maxDim]
        # Cap rows via uniform stride (preserves distributional range)
        if flat.shape[0] > maxRows:
            stride = flat.shape[0] // maxRows
            flat = flat[::stride][:maxRows]
        frobSq = float((flat * flat).sum().item())
        if frobSq < _EPS:
            return 0.0
        singVals = torch.linalg.svdvals(flat)
        spectralSq = float((singVals[0] ** 2).item())
        if spectralSq < _EPS:
            return 0.0
        rank = frobSq / spectralSq
        maxRank = float(min(flat.shape[0], flat.shape[1]))
        return min(1.0, rank / max(1.0, maxRank))


def _computeIntraBatchSim(tensor: torch.Tensor) -> float:
    """Mean pairwise cosine similarity across the batch dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (B, ...).  Batch capped at ``_SIM_BATCH_CAP``.

    Returns
    -------
    float
        Mean off-diagonal cosine; near 1.0 signals mode collapse.
    """
    with torch.no_grad():
        flat = tensor.detach().float().cpu()
        batchSize = min(flat.shape[0], _SIM_BATCH_CAP)
        flat = flat[:batchSize].reshape(batchSize, -1)
        norms = flat.norm(dim=1, keepdim=True).clamp(min=_EPS)
        normed = flat / norms
        sims = normed @ normed.T
        # Off-diagonal mean
        mask = ~torch.eye(batchSize, dtype=torch.bool)
        return float(sims[mask].mean().item())


# ------------------------------------------------------------------
# Probe
# ------------------------------------------------------------------
class Probe:
    """Hooks onto a module path and captures scalar stats per call.

    Parameters
    ----------
    name : str
        Human-readable identifier (used as JSONL key prefix).
    modulePath : str
        Dot-separated attribute path relative to the root model,
        e.g. ``"denoiser"`` or ``"denoiser.outputProj"``.
    capture : list[str]
        Which stats to capture.  Valid values: ``mean``, ``std``,
        ``norm``, ``effective_rank``, ``intra_batch_sim``,
        ``update_ratio``.
    hookType : str
        ``"forward"``, ``"backward"``, or ``"forward_and_backward"``.
    """

    def __init__(
        self,
        name: str,
        modulePath: str,
        capture: list[str],
        hookType: str = "forward",
        captureEvery: int = 1,
    ) -> None:
        self.name = name
        self.modulePath = modulePath
        self.capture = set(capture)
        self.hookType = hookType
        self._captureEvery = captureEvery

        self._snapshot: ProbeSnapshot | None = None
        self._handles: list[Any] = []
        self._prevWeightNorm: float | None = None
        self._currentStep: int = 0
        self._callCount: int = 0

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------
    def attach(self, rootModel: nn.Module) -> None:
        """Resolve module path and register hooks.

        Parameters
        ----------
        rootModel : nn.Module
            The top-level model that contains the target sub-module.

        Raises
        ------
        AttributeError
            If ``modulePath`` does not resolve to a sub-module.
        """
        module = self._resolveModule(rootModel)
        if self.hookType in ("forward", "forward_and_backward"):
            handle = module.register_forward_hook(self._forwardHook)
            self._handles.append(handle)
        if self.hookType in ("backward", "forward_and_backward"):
            handle = module.register_full_backward_hook(
                self._backwardHook
            )
            self._handles.append(handle)

    def detach(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def setStep(self, step: int) -> None:
        """Inform the probe of the current training step."""
        self._currentStep = step

    # ------------------------------------------------------------------
    # Snapshot access
    # ------------------------------------------------------------------
    @property
    def lastSnapshot(self) -> ProbeSnapshot | None:
        """Most recently captured snapshot (None before first call)."""
        return self._snapshot

    # ------------------------------------------------------------------
    # Hooks (never store tensors; only scalars survive)
    # ------------------------------------------------------------------
    def _forwardHook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        self._callCount += 1
        if self._callCount % self._captureEvery != 0:
            return
        tensor = self._extractTensor(output)
        if tensor is None:
            return

        snap = ProbeSnapshot(step=self._currentStep)

        if "mean" in self.capture or "std" in self.capture or \
                "norm" in self.capture:
            mean, std, norm = _computeMeanStdNorm(tensor)
            snap.mean = mean
            snap.std = std
            snap.norm = norm

        if "effective_rank" in self.capture:
            snap.effective_rank = _computeEffectiveRank(tensor)

        if "intra_batch_sim" in self.capture and tensor.shape[0] >= 2:
            snap.intra_batch_sim = _computeIntraBatchSim(tensor)

        if "update_ratio" in self.capture:
            snap.update_ratio = self._computeUpdateRatio(module)

        self._snapshot = snap

    def _backwardHook(
        self,
        module: nn.Module,
        gradInput: tuple[Any, ...],
        gradOutput: tuple[Any, ...],
    ) -> None:
        if "update_ratio" not in self.capture:
            return
        if self._snapshot is None:
            return
        # Update ratio from grad norms when backward hook fires
        for param in module.parameters():
            if param.grad is not None:
                wNorm = float(param.data.norm().item())
                gNorm = float(param.grad.norm().item())
                if wNorm > _EPS:
                    self._snapshot.update_ratio = gNorm / wNorm
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolveModule(self, rootModel: nn.Module) -> nn.Module:
        """Walk the dot-path to find the target sub-module."""
        module: nn.Module = rootModel
        for part in self.modulePath.split("."):
            module = getattr(module, part)
        return module

    @staticmethod
    def _extractTensor(output: Any) -> torch.Tensor | None:
        """Try to get a tensor from various output types."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
        return None

    def _computeUpdateRatio(self, module: nn.Module) -> float | None:
        """Compute ||Δw|| / ||w|| from the first param of the module."""
        for param in module.parameters():
            currentNorm = float(param.data.norm().item())
            if self._prevWeightNorm is None:
                self._prevWeightNorm = currentNorm
                return None
            delta = abs(currentNorm - self._prevWeightNorm)
            ratio = delta / max(self._prevWeightNorm, _EPS)
            self._prevWeightNorm = currentNorm
            return ratio
        return None
