"""
Exponential Moving Average (EMA) on model parameters.

Standard diffusion-model stabilisation technique: maintain a shadow copy of
the trainable parameters that is updated after each real optimizer step as

    shadow = decay * shadow + (1 - decay) * param

Validation and inference then use the shadow parameters, which filter out the
high-frequency oscillations of the online weights and yield a much more
stable val_loss and a better-behaved checkpoint.

The implementation follows the Karras / MDM convention of a warmup schedule
to avoid the shadow being stuck at random initialisation for the first
thousand steps:

    decay_effective(step) = min(decay, (1 + step) / (10 + step))

Shadow tensors live on the same device as the source parameters to avoid
host↔device transfers on every optimizer step.
"""

from __future__ import annotations

from typing import Iterable, List

import torch


class ExponentialMovingAverage:
    """Maintain an EMA shadow of an iterable of ``torch.nn.Parameter``.

    Parameters
    ----------
    parameters : Iterable[torch.nn.Parameter]
        Parameters whose values should be averaged. The shadow tensors are
        created once from this list; the *same* parameter objects (or any
        iterable yielding parameters in the same order with the same shapes)
        must be passed to every subsequent call of ``update``,
        ``storeAndSwap``, ``restore`` and ``copyTo``.
    decay : float
        Asymptotic decay rate (typical 0.9999).
    useWarmup : bool
        When True, the effective decay ramps up from 0 toward ``decay``
        according to ``(1 + step) / (10 + step)``. Strongly recommended
        unless the training run is very long.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        useWarmup: bool = True,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(
                f"EMA decay must be in (0, 1); got {decay!r}"
            )
        self._decay = float(decay)
        self._useWarmup = bool(useWarmup)
        self._numUpdates: int = 0
        # Detach + clone keeps the shadow outside the autograd graph.
        self._shadow: List[torch.Tensor] = [
            p.detach().clone() for p in parameters
        ]
        self._backup: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Decay schedule
    # ------------------------------------------------------------------
    def _effectiveDecay(self) -> float:
        """Return the decay rate to apply at the current step."""
        if not self._useWarmup:
            return self._decay
        step = self._numUpdates
        warmupDecay = (1.0 + step) / (10.0 + step)
        return min(self._decay, warmupDecay)

    # ------------------------------------------------------------------
    # Update / copy
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Advance the shadow using the latest online parameters."""
        decay = self._effectiveDecay()
        oneMinusDecay = 1.0 - decay
        paramList = list(parameters)
        if len(paramList) != len(self._shadow):
            raise RuntimeError(
                f"EMA.update: expected {len(self._shadow)} parameters, "
                f"got {len(paramList)}"
            )
        for shadow, param in zip(self._shadow, paramList):
            if shadow.device != param.device:
                shadow.data = shadow.data.to(param.device)
            shadow.mul_(decay).add_(param.detach(), alpha=oneMinusDecay)
        self._numUpdates += 1

    @torch.no_grad()
    def copyTo(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Overwrite ``parameters`` with the shadow values (no backup)."""
        paramList = list(parameters)
        if len(paramList) != len(self._shadow):
            raise RuntimeError(
                f"EMA.copyTo: expected {len(self._shadow)} parameters, "
                f"got {len(paramList)}"
            )
        for shadow, param in zip(self._shadow, paramList):
            param.data.copy_(shadow.data)

    # ------------------------------------------------------------------
    # Swap helpers (for evaluating validation on EMA weights)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def storeAndSwap(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Back up ``parameters`` and install the shadow values in-place."""
        if self._backup:
            raise RuntimeError(
                "EMA.storeAndSwap called while a previous swap is still "
                "active; call restore() before swapping again."
            )
        paramList = list(parameters)
        if len(paramList) != len(self._shadow):
            raise RuntimeError(
                f"EMA.storeAndSwap: expected {len(self._shadow)} parameters, "
                f"got {len(paramList)}"
            )
        self._backup = [p.detach().clone() for p in paramList]
        for shadow, param in zip(self._shadow, paramList):
            param.data.copy_(shadow.data)

    @torch.no_grad()
    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Reinstate the online parameters saved by the last swap."""
        if not self._backup:
            raise RuntimeError(
                "EMA.restore called with no active swap; did you forget "
                "to call storeAndSwap first?"
            )
        paramList = list(parameters)
        if len(paramList) != len(self._backup):
            raise RuntimeError(
                f"EMA.restore: expected {len(self._backup)} parameters, "
                f"got {len(paramList)}"
            )
        for backup, param in zip(self._backup, paramList):
            param.data.copy_(backup.data)
        self._backup = []

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def stateDict(self) -> dict:
        """Return a serialisable snapshot of the EMA state."""
        return {
            "shadow": [s.detach().cpu().clone() for s in self._shadow],
            "numUpdates": int(self._numUpdates),
            "decay": float(self._decay),
            "useWarmup": bool(self._useWarmup),
        }

    def loadStateDict(self, state: dict) -> None:
        """Restore EMA state produced by ``stateDict``.

        ``decay`` and ``useWarmup`` stay those of the *current* manager: the
        config is authoritative, we only restore the averaged values and the
        update counter.
        """
        if "shadow" not in state or "numUpdates" not in state:
            raise ValueError(
                "EMA state_dict must contain 'shadow' and 'numUpdates' keys."
            )
        loadedShadow = state["shadow"]
        if len(loadedShadow) != len(self._shadow):
            raise RuntimeError(
                "EMA.loadStateDict: parameter count mismatch "
                f"(checkpoint={len(loadedShadow)}, "
                f"current={len(self._shadow)})"
            )
        for dst, src in zip(self._shadow, loadedShadow):
            if dst.shape != src.shape:
                raise RuntimeError(
                    f"EMA.loadStateDict: shape mismatch "
                    f"(checkpoint={tuple(src.shape)}, "
                    f"current={tuple(dst.shape)})"
                )
            dst.data.copy_(src.to(dst.device))
        self._numUpdates = int(state["numUpdates"])

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def numUpdates(self) -> int:
        return self._numUpdates

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def useWarmup(self) -> bool:
        return self._useWarmup

    @property
    def shadow(self) -> List[torch.Tensor]:
        """Return the shadow tensors (for inspection / testing only)."""
        return self._shadow
