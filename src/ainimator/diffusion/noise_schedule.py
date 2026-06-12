"""Noise schedule utilities for AI-nimator v2 diffusion.

This module provides a unified :class:`NoiseSchedule` that supports both
the linear β schedule used by Étape 1 and the **cosine** schedule
introduced by Nichol & Dhariwal 2021, plus the conversion utilities
needed for **v-prediction** (Salimans & Ho 2022).

Why v-prediction?
-----------------
* Better conditioning at the extreme timesteps: ``ε`` is undefined at
  ``t=0`` (no noise to predict), ``x0`` is undefined at ``t=T`` (signal
  fully destroyed).  The v-target ``v = α·ε − σ·x0`` is well-behaved
  everywhere.
* Standard for motion diffusion since 2024 (MotionDiffuse, T2M-GPT,
  AnimateDiff, ...).
* No additional model parameters — only the loss target and the
  sampler conversion change.

Why cosine?
-----------
* Linear ``β=0.0001→0.02`` concentrates the noise at the start of the
  diffusion chain; the model spends most of its capacity on low-t
  samples and barely sees the high-t regime.
* Cosine ``ᾱ(t) = cos²((t/T + s) / (1+s) · π/2) / cos²(s/(1+s) · π/2)``
  spreads the noise evenly across the chain, so every timestep
  contributes meaningful gradient.

Mathematical conventions
------------------------
We use the canonical ``α_t² + σ_t² = 1`` parametrisation:

* ``α_t  = sqrt(ᾱ_t)``
* ``σ_t  = sqrt(1 − ᾱ_t)``
* ``x_t  = α_t · x_0 + σ_t · ε``           (forward diffusion)
* ``v    = α_t · ε   − σ_t · x_0``         (v-target)
* From v:
    ``x_0 = α_t · x_t − σ_t · v``
    ``ε   = σ_t · x_t + α_t · v``

These identities are unit-tested against round-trips so a future
refactor can't silently break them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# Canonical names for the supported prediction targets.
PREDICTION_X0 = "x0"
PREDICTION_V = "v"
PREDICTION_EPSILON = "epsilon"
SUPPORTED_PREDICTIONS: tuple[str, ...] = (
    PREDICTION_X0,
    PREDICTION_V,
    PREDICTION_EPSILON,
)

# Canonical names for the supported β schedules.
SCHEDULE_LINEAR = "linear"
SCHEDULE_COSINE = "cosine"
SUPPORTED_SCHEDULES: tuple[str, ...] = (SCHEDULE_LINEAR, SCHEDULE_COSINE)


@dataclass(frozen=True)
class NoiseScheduleConfig:
    """Configuration for :class:`NoiseSchedule`.

    Parameters
    ----------
    numSteps : int
        Number of diffusion timesteps used at training time.  ``1000``
        is the canonical value used by DDPM/DDIM/MDM and the v2 plan.
    scheduleType : str
        Either ``"linear"`` or ``"cosine"``.  ``"cosine"`` is the v2
        default; ``"linear"`` is kept for parity with Étape 1 and for
        ablation studies.
    linearBetaStart, linearBetaEnd : float
        Endpoints of the linear β schedule.  Ignored when
        ``scheduleType="cosine"``.
    cosineS : float
        Small offset preventing ``ᾱ_T`` from exactly reaching zero
        (which would make ``α=0`` and break v-pred at the extreme step).
        ``0.008`` is the value from Nichol & Dhariwal 2021.
    """

    numSteps: int = 1000
    scheduleType: str = SCHEDULE_COSINE
    linearBetaStart: float = 0.0001
    linearBetaEnd: float = 0.02
    cosineS: float = 0.008

    def __post_init__(self) -> None:
        if self.numSteps < 1:
            raise ValueError("numSteps must be >= 1.")
        if self.scheduleType not in SUPPORTED_SCHEDULES:
            raise ValueError(
                f"scheduleType must be one of {SUPPORTED_SCHEDULES}; "
                f"got {self.scheduleType!r}."
            )
        if self.linearBetaStart <= 0 or self.linearBetaEnd <= 0:
            raise ValueError("linear β endpoints must be > 0.")
        if self.linearBetaStart >= self.linearBetaEnd:
            raise ValueError("linearBetaStart must be < linearBetaEnd.")
        if not (0.0 < self.cosineS < 0.5):
            raise ValueError("cosineS must be in (0, 0.5).")


class NoiseSchedule(nn.Module):
    """Diffusion noise schedule with v-prediction utilities.

    All buffers are registered with ``register_buffer`` so they move
    with ``.to(device)`` and round-trip through ``state_dict``.

    The buffers most consumers need:

    * :attr:`alphasCumprod` — ``ᾱ_t`` of shape ``(numSteps,)``.
    * :attr:`alphas` — ``α_t = sqrt(ᾱ_t)`` (note: NOT ``1 − β`` here;
      ``alphas`` follows the v-pred convention used everywhere else in
      this module).
    * :attr:`sigmas` — ``σ_t = sqrt(1 − ᾱ_t)``.

    For tests and external introspection, the original DDPM ``betas``
    are also exposed.
    """

    def __init__(self, config: NoiseScheduleConfig) -> None:
        super().__init__()
        self._config = config

        # Compute the schedule in float64 for numerical stability — the
        # cumulative product over 1000 steps loses precision in float32.
        if config.scheduleType == SCHEDULE_LINEAR:
            betas = self._linearBetas(config)
        else:
            betas = self._cosineBetas(config)
        # Numerical safety: clamp betas into a sane range.  Without this
        # the cosine schedule can produce slightly-negative tail values
        # because of floating-point noise.
        betas = torch.clamp(betas, min=1e-8, max=0.999)

        ddpmAlphas = 1.0 - betas
        alphasCumprod = torch.cumprod(ddpmAlphas, dim=0)

        # v-pred parametrisation: α and σ such that α² + σ² = 1.
        alphas = torch.sqrt(alphasCumprod)
        sigmas = torch.sqrt(torch.clamp(1.0 - alphasCumprod, min=0.0))

        # Cast everything to float32 before exposing it.  Inference and
        # training both run in float32 (or mixed precision down the
        # road), and float64 buffers would force expensive promotions
        # at every operation that mixes them with float32 tensors.
        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer(
            "alphasCumprod", alphasCumprod.to(torch.float32)
        )
        self.register_buffer("alphas", alphas.to(torch.float32))
        self.register_buffer("sigmas", sigmas.to(torch.float32))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _linearBetas(config: NoiseScheduleConfig) -> torch.Tensor:
        return torch.linspace(
            config.linearBetaStart,
            config.linearBetaEnd,
            config.numSteps,
            dtype=torch.float64,
        )

    @staticmethod
    def _cosineBetas(config: NoiseScheduleConfig) -> torch.Tensor:
        """Cosine schedule from Nichol & Dhariwal 2021.

        Returns a length-``numSteps`` tensor of ``β`` values derived
        from the cumulative product, NOT a direct cosine on β.
        """
        steps = config.numSteps
        s = config.cosineS
        # Compute ᾱ at integer timesteps t=0..numSteps inclusive.
        timeFraction = torch.linspace(
            0.0, 1.0, steps + 1, dtype=torch.float64
        )
        alphaBar = torch.cos(
            (timeFraction + s) / (1.0 + s) * math.pi / 2.0
        ) ** 2
        alphaBar = alphaBar / alphaBar[0]
        # β_t = 1 − ᾱ_t / ᾱ_{t-1}; clip for numerical stability later.
        betas = 1.0 - (alphaBar[1:] / alphaBar[:-1])
        return betas

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> NoiseScheduleConfig:
        return self._config

    @property
    def numSteps(self) -> int:
        return self._config.numSteps

    # ------------------------------------------------------------------
    # Per-sample alpha/sigma extraction
    # ------------------------------------------------------------------
    def alphaSigma(
        self,
        timesteps: torch.Tensor,
        broadcastShape: torch.Size | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(α_t, σ_t)`` indexed at ``timesteps``.

        Parameters
        ----------
        timesteps : torch.Tensor
            Long tensor of shape ``(B,)`` with values in ``[0, numSteps)``.
        broadcastShape : tuple, optional
            When provided, the returned tensors are reshaped to
            ``(B, 1, 1, ..., 1)`` with enough singleton trailing dims to
            broadcast against this shape.  Pass ``x.shape`` from the
            caller to avoid manual reshaping.

        Returns
        -------
        (alpha, sigma) : tuple[torch.Tensor, torch.Tensor]
            Same dtype as the schedule buffers (typically float32 once
            the module has been moved to a device).
        """
        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must be 1-D; got shape "
                f"{tuple(timesteps.shape)}."
            )
        timesteps = timesteps.long()
        alpha = self.alphas.to(device=timesteps.device).gather(0, timesteps)
        sigma = self.sigmas.to(device=timesteps.device).gather(0, timesteps)
        if broadcastShape is not None:
            alpha = self._broadcast(alpha, broadcastShape)
            sigma = self._broadcast(sigma, broadcastShape)
        return alpha, sigma

    @staticmethod
    def _broadcast(
        values: torch.Tensor,
        shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        """Reshape a length-B vector to ``(B, 1, 1, ...)``."""
        if values.ndim != 1:
            raise ValueError("Internal: _broadcast expects a 1-D tensor.")
        trailing = (1,) * (len(shape) - 1)
        return values.reshape(values.shape[0], *trailing)

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------
    def qSample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-diffuse ``x_0`` to ``x_t`` using fresh Gaussian noise.

        Returns
        -------
        (xT, noise) : tuple[torch.Tensor, torch.Tensor]
            Both have the same shape as ``x0``.  When ``noise`` was
            provided, it is returned unchanged so the caller can keep
            the exact noise tensor used.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        elif noise.shape != x0.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} does not match x0 "
                f"shape {tuple(x0.shape)}."
            )
        alpha, sigma = self.alphaSigma(timesteps, broadcastShape=x0.shape)
        xT = alpha * x0 + sigma * noise
        return xT, noise

    # ------------------------------------------------------------------
    # Prediction-target conversions
    # ------------------------------------------------------------------
    def predictionTarget(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        predictionMode: str,
    ) -> torch.Tensor:
        """Compute the supervised target for a given prediction mode.

        * ``"x0"``      → returns ``x0``
        * ``"v"``       → returns ``v = α·ε − σ·x0``
        * ``"epsilon"`` → returns ``noise`` (passed through verbatim)
        """
        self._validateMode(predictionMode)
        if predictionMode == PREDICTION_X0:
            return x0
        if predictionMode == PREDICTION_EPSILON:
            return noise
        # v-prediction
        alpha, sigma = self.alphaSigma(timesteps, broadcastShape=x0.shape)
        return alpha * noise - sigma * x0

    def x0FromPrediction(
        self,
        prediction: torch.Tensor,
        xT: torch.Tensor,
        timesteps: torch.Tensor,
        predictionMode: str,
    ) -> torch.Tensor:
        """Recover ``x_0`` from any supported prediction target."""
        self._validateMode(predictionMode)
        if predictionMode == PREDICTION_X0:
            return prediction
        alpha, sigma = self.alphaSigma(timesteps, broadcastShape=xT.shape)
        if predictionMode == PREDICTION_V:
            return alpha * xT - sigma * prediction
        # ε-prediction: x0 = (xt − σ·ε) / α
        return (xT - sigma * prediction) / torch.clamp(alpha, min=1e-8)

    def epsilonFromPrediction(
        self,
        prediction: torch.Tensor,
        xT: torch.Tensor,
        timesteps: torch.Tensor,
        predictionMode: str,
    ) -> torch.Tensor:
        """Recover ``ε`` from any supported prediction target."""
        self._validateMode(predictionMode)
        if predictionMode == PREDICTION_EPSILON:
            return prediction
        alpha, sigma = self.alphaSigma(timesteps, broadcastShape=xT.shape)
        if predictionMode == PREDICTION_V:
            return sigma * xT + alpha * prediction
        # x0-prediction: ε = (xt − α·x0) / σ
        return (xT - alpha * prediction) / torch.clamp(sigma, min=1e-8)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validateMode(predictionMode: str) -> None:
        if predictionMode not in SUPPORTED_PREDICTIONS:
            raise ValueError(
                "predictionMode must be one of "
                f"{SUPPORTED_PREDICTIONS}; got {predictionMode!r}."
            )
