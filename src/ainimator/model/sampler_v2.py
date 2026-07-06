"""DDIM sampler for AI-nimator v2 (v-prediction + CFG).

This sampler drives :class:`MotionDenoiserV2` end-to-end during
inference.  Compared to the legacy ``motion_generator``:

* **Decoupled** from the model itself — :class:`DDIMSamplerV2` does not
  hold the denoiser, the schedule, or the text encoder.  Callers wire
  the components together explicitly, which makes the sampler trivial
  to unit-test in isolation and reusable for evaluation
  (CFG sweeps, multi-seed sampling, ablations).
* **v-prediction friendly** — the denoiser output mode is configurable
  (``"v"``, ``"x0"``, ``"epsilon"``) and converted internally via
  :class:`NoiseSchedule`.
* **CFG via dual pass** — when ``cfgScale > 1`` the denoiser is called
  twice per step (conditional + unconditional with empty/null text);
  the two predictions are linearly combined in the prediction space.
* **Deterministic by default** (``eta=0``).  Set ``eta>0`` to recover
  stochastic DDIM.

The sampler does **not** know about post-processing (FK enforcement,
foot-skating fix, etc.).  Those are pluggable downstream of
:meth:`sample`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ainimator.model.denoiser_v2 import (
    DenoiserOutput,
    MotionDenoiserV2,
)
from ainimator.model.motion_normalizer import (
    MotionNormalizer,
)
from ainimator.diffusion.noise_schedule import (
    NoiseSchedule,
    PREDICTION_V,
    SUPPORTED_PREDICTIONS,
)


def _applyGuidanceRescale(
    guided: torch.Tensor,
    conditional: torch.Tensor,
    rescale: float,
) -> torch.Tensor:
    """Rescale a CFG-guided prediction toward the conditional std.

    Implements the guidance-rescale trick (Lin et al. 2024): high CFG
    inflates the prediction variance and pushes samples off-manifold
    (the over-energetic "trembling" motion observed at high cfgScale).
    Each sample's guided prediction is rescaled so its standard
    deviation matches the conditional prediction, then blended back with
    strength ``rescale`` in ``[0, 1]`` (``0`` disables, ``1`` fully
    rescales).
    """
    if rescale <= 0.0:
        return guided
    flatGuided = guided.flatten(1)
    flatCond = conditional.flatten(1)
    stdGuided = flatGuided.std(dim=1, keepdim=True).clamp(min=1e-8)
    stdCond = flatCond.std(dim=1, keepdim=True)
    factor = (stdCond / stdGuided).view(-1, *([1] * (guided.dim() - 1)))
    rescaled = guided * factor
    return rescale * rescaled + (1.0 - rescale) * guided


# ---------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class SamplerOutput:
    """Result of :meth:`DDIMSamplerV2.sample`.

    Attributes
    ----------
    boneMotion : torch.Tensor
        Sampled bone-scoped features of shape
        ``(B, F, numBones, motionChannels)``.
    globalMotion : torch.Tensor or None
        Sampled global-scoped features of shape
        ``(B, F, globalChannels)`` or ``None`` when the denoiser has no
        global branch.
    """

    boneMotion: torch.Tensor
    globalMotion: torch.Tensor | None


# ---------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------
class DDIMSamplerV2:
    """DDIM sampler with v-prediction and classifier-free guidance.

    Usage
    -----
    .. code-block:: python

        schedule = NoiseSchedule(NoiseScheduleConfig())
        sampler = DDIMSamplerV2(schedule, predictionMode="v")
        result = sampler.sample(
            denoiser=denoiser,
            textHiddenStates=textOutput.hiddenStates,
            textKeyPaddingMask=textOutput.keyPaddingMask,
            unconditionalTextHiddenStates=nullText.hiddenStates,
            unconditionalTextKeyPaddingMask=nullText.keyPaddingMask,
            frames=120,
            numSteps=100,
            cfgScale=3.5,
            device=torch.device("mps"),
        )

    The unconditional path is mandatory whenever ``cfgScale != 1``.
    Callers typically pass an empty-string text encoded by the same
    text encoder.  Passing ``None`` is allowed only when ``cfgScale=1``
    (no guidance) — this is enforced.
    """

    def __init__(
        self,
        schedule: NoiseSchedule,
        predictionMode: str = PREDICTION_V,
    ) -> None:
        if predictionMode not in SUPPORTED_PREDICTIONS:
            raise ValueError(
                "predictionMode must be one of "
                f"{SUPPORTED_PREDICTIONS}; got {predictionMode!r}."
            )
        self._schedule = schedule
        self._predictionMode = predictionMode

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def schedule(self) -> NoiseSchedule:
        return self._schedule

    @property
    def predictionMode(self) -> str:
        return self._predictionMode

    # ------------------------------------------------------------------
    # Time-step planning
    # ------------------------------------------------------------------
    def planTimesteps(
        self,
        numSteps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the descending timestep schedule used by DDIM.

        Returns a long tensor of length ``numSteps`` with values in
        ``[0, schedule.numSteps − 1]`` in **descending** order.
        """
        if numSteps < 1:
            raise ValueError("numSteps must be >= 1.")
        if numSteps > self._schedule.numSteps:
            raise ValueError(
                f"Requested numSteps={numSteps} exceeds the training "
                f"schedule length {self._schedule.numSteps}."
            )
        # Equally-spaced descending timesteps.  Endpoint is the largest
        # valid index (numSteps_train − 1).
        plan = torch.linspace(
            self._schedule.numSteps - 1,
            0,
            numSteps,
            dtype=torch.float32,
            device=device,
        )
        return plan.round().long()

    # ------------------------------------------------------------------
    # Core sampling loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        denoiser: MotionDenoiserV2,
        textHiddenStates: torch.Tensor,
        frames: int,
        numSteps: int = 100,
        cfgScale: float = 3.5,
        eta: float = 0.0,
        device: torch.device | None = None,
        *,
        guidanceRescale: float = 0.0,
        textKeyPaddingMask: torch.Tensor | None = None,
        unconditionalTextHiddenStates: torch.Tensor | None = None,
        unconditionalTextKeyPaddingMask: torch.Tensor | None = None,
        motionKeyPaddingMask: torch.Tensor | None = None,
        seed: int | None = None,
        normalizer: MotionNormalizer | None = None,
    ) -> SamplerOutput:
        """Generate motion samples conditioned on ``textHiddenStates``.

        Parameters
        ----------
        denoiser : MotionDenoiserV2
            Trained v2 denoiser.  Must be in eval mode (the sampler
            switches it back to whichever mode it found).
        textHiddenStates : torch.Tensor
            Conditional text embedding of shape ``(B, T_text, D_text)``.
            ``D_text`` must match
            ``denoiser.config.effectiveTextEmbedDim``.
        frames : int
            Number of motion frames to generate.
        numSteps : int
            Number of DDIM sampling steps (typically 50–200).  100 is
            the v2 default.
        cfgScale : float
            Classifier-free guidance scale.  ``1.0`` disables CFG.
        eta : float
            DDIM stochasticity coefficient.  ``0`` = deterministic
            (default), ``1`` = full DDPM ancestral sampling.
        device : torch.device, optional
            Device for sampling; defaults to ``denoiser`` device.
        textKeyPaddingMask : torch.Tensor, optional
            Bool mask of padded text positions, shape ``(B, T_text)``.
        unconditionalTextHiddenStates, unconditionalTextKeyPaddingMask :
            Required when ``cfgScale != 1``.  Same shapes as the
            conditional inputs (typically the encoded empty string).
        motionKeyPaddingMask : torch.Tensor, optional
            Bool mask of padded motion frames, shape ``(B, frames)``.
        seed : int, optional
            Seed for the initial noise.  When ``None`` the global RNG
            is used.
        normalizer : MotionNormalizer, optional
            When provided, the final ``x_0`` is denormalised through
            this normaliser before being returned.  This is mandatory
            for any checkpoint that was trained on z-normalised motion
            (i.e. anything produced by the v3 training pipeline);
            without it the sampler returns motion in the unit-variance
            normalised space and downstream consumers see pure noise.

        Returns
        -------
        SamplerOutput
        """
        if cfgScale != 1.0 and unconditionalTextHiddenStates is None:
            raise ValueError(
                "unconditionalTextHiddenStates is required when "
                "cfgScale != 1."
            )
        device = device or self._inferDevice(denoiser)
        config = denoiser.config

        batchSize = textHiddenStates.shape[0]
        bonesShape = (
            batchSize,
            frames,
            config.numBones,
            config.motionChannels,
        )

        previousMode = denoiser.training
        denoiser.eval()
        try:
            xT, globalT = self._initialiseNoise(
                batchSize=batchSize,
                frames=frames,
                config=config,
                device=device,
                seed=seed,
            )

            timestepPlan = self.planTimesteps(numSteps, device=device)
            useSelfCond = denoiser.config.useSelfConditioning
            selfCondBone: torch.Tensor | None = None
            selfCondGlobal: torch.Tensor | None = None
            for step in range(numSteps):
                tNow = timestepPlan[step]
                tNext = (
                    timestepPlan[step + 1]
                    if step + 1 < numSteps
                    else None
                )
                xT, globalT, x0Bone, x0Global = self._stepDdim(
                    denoiser=denoiser,
                    xT=xT,
                    globalT=globalT,
                    timestepNow=tNow,
                    timestepNext=tNext,
                    textHiddenStates=textHiddenStates,
                    textKeyPaddingMask=textKeyPaddingMask,
                    unconditionalTextHiddenStates=
                        unconditionalTextHiddenStates,
                    unconditionalTextKeyPaddingMask=
                        unconditionalTextKeyPaddingMask,
                    motionKeyPaddingMask=motionKeyPaddingMask,
                    cfgScale=cfgScale,
                    eta=eta,
                    guidanceRescale=guidanceRescale,
                    selfCondBone=selfCondBone,
                    selfCondGlobal=selfCondGlobal,
                )
                if useSelfCond:
                    selfCondBone = x0Bone
                    selfCondGlobal = x0Global
        finally:
            denoiser.train(previousMode)

        # Denormalise the final x_0 back to raw motion space so downstream
        # consumers (Collada exporter, .npz dumps, etc.) see metres /
        # rotation6d in the same scale as the training samples.
        if normalizer is not None:
            xT = normalizer.denormalizeBone(xT)
            if globalT is not None and normalizer.hasGlobalBranch:
                globalT = normalizer.denormalizeGlobal(globalT)

        return SamplerOutput(boneMotion=xT, globalMotion=globalT)

    # ------------------------------------------------------------------
    # Per-step DDIM update
    # ------------------------------------------------------------------
    def _stepDdim(
        self,
        denoiser: MotionDenoiserV2,
        xT: torch.Tensor,
        globalT: torch.Tensor | None,
        timestepNow: torch.Tensor,
        timestepNext: torch.Tensor | None,
        textHiddenStates: torch.Tensor,
        textKeyPaddingMask: torch.Tensor | None,
        unconditionalTextHiddenStates: torch.Tensor | None,
        unconditionalTextKeyPaddingMask: torch.Tensor | None,
        motionKeyPaddingMask: torch.Tensor | None,
        cfgScale: float,
        eta: float,
        guidanceRescale: float = 0.0,
        selfCondBone: torch.Tensor | None = None,
        selfCondGlobal: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        batchSize = xT.shape[0]
        timestepBatch = timestepNow.expand(batchSize)

        # --- Predict (conditional + optional unconditional CFG pass) ---
        prediction = denoiser(
            noisyMotion=xT,
            timesteps=timestepBatch,
            textHiddenStates=textHiddenStates,
            textKeyPaddingMask=textKeyPaddingMask,
            noisyGlobalFeatures=globalT,
            motionKeyPaddingMask=motionKeyPaddingMask,
            selfCondBone=selfCondBone,
            selfCondGlobal=selfCondGlobal,
        )
        boneSignal = prediction.boneOutput
        globalSignal = prediction.globalOutput

        if cfgScale != 1.0:
            assert unconditionalTextHiddenStates is not None
            uncondPrediction = denoiser(
                noisyMotion=xT,
                timesteps=timestepBatch,
                textHiddenStates=unconditionalTextHiddenStates,
                textKeyPaddingMask=unconditionalTextKeyPaddingMask,
                noisyGlobalFeatures=globalT,
                motionKeyPaddingMask=motionKeyPaddingMask,
                selfCondBone=selfCondBone,
                selfCondGlobal=selfCondGlobal,
            )
            condBone = boneSignal
            boneSignal = uncondPrediction.boneOutput + cfgScale * (
                boneSignal - uncondPrediction.boneOutput
            )
            boneSignal = _applyGuidanceRescale(
                guided=boneSignal,
                conditional=condBone,
                rescale=guidanceRescale,
            )
            if globalSignal is not None and uncondPrediction.globalOutput is not None:
                condGlobal = globalSignal
                globalSignal = uncondPrediction.globalOutput + cfgScale * (
                    globalSignal - uncondPrediction.globalOutput
                )
                globalSignal = _applyGuidanceRescale(
                    guided=globalSignal,
                    conditional=condGlobal,
                    rescale=guidanceRescale,
                )

        # --- Convert prediction to (x0, ε) ----------------------------
        x0Bone, epsilonBone = self._toX0Epsilon(
            prediction=boneSignal,
            xT=xT,
            timestepBatch=timestepBatch,
        )
        x0Global = None
        epsilonGlobal = None
        if globalSignal is not None and globalT is not None:
            x0Global, epsilonGlobal = self._toX0Epsilon(
                prediction=globalSignal,
                xT=globalT,
                timestepBatch=timestepBatch,
            )

        # --- DDIM update ---------------------------------------------
        if timestepNext is None:
            # Last step: collapse onto the predicted x_0.
            return x0Bone, x0Global, x0Bone, x0Global

        timestepNextBatch = timestepNext.expand(batchSize)
        xTNextBone = self._ddimNextX(
            xT=xT,
            x0=x0Bone,
            epsilon=epsilonBone,
            timestepNow=timestepBatch,
            timestepNext=timestepNextBatch,
            eta=eta,
        )
        xTNextGlobal: torch.Tensor | None = None
        if x0Global is not None and epsilonGlobal is not None and globalT is not None:
            xTNextGlobal = self._ddimNextX(
                xT=globalT,
                x0=x0Global,
                epsilon=epsilonGlobal,
                timestepNow=timestepBatch,
                timestepNext=timestepNextBatch,
                eta=eta,
            )
        return xTNextBone, xTNextGlobal, x0Bone, x0Global

    def _toX0Epsilon(
        self,
        prediction: torch.Tensor,
        xT: torch.Tensor,
        timestepBatch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a denoiser prediction to (x0, ε) using the schedule."""
        x0 = self._schedule.x0FromPrediction(
            prediction=prediction,
            xT=xT,
            timesteps=timestepBatch,
            predictionMode=self._predictionMode,
        )
        epsilon = self._schedule.epsilonFromPrediction(
            prediction=prediction,
            xT=xT,
            timesteps=timestepBatch,
            predictionMode=self._predictionMode,
        )
        return x0, epsilon

    def _ddimNextX(
        self,
        xT: torch.Tensor,
        x0: torch.Tensor,
        epsilon: torch.Tensor,
        timestepNow: torch.Tensor,
        timestepNext: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        """Compute ``x_{t-1}`` from ``x_t`` and the (x0, ε) prediction."""
        alphaNow, sigmaNow = self._schedule.alphaSigma(
            timestepNow, broadcastShape=xT.shape
        )
        alphaNext, sigmaNext = self._schedule.alphaSigma(
            timestepNext, broadcastShape=xT.shape
        )

        # DDIM noise scaling (Song et al. 2020, eq 12 with our notation):
        #   σ_t² = η² · σ_next² · (1 − ᾱ_t / ᾱ_{t-1})
        # where ᾱ_t = α_now² (we go from a larger-noise step ``now`` to a
        # smaller-noise step ``next``, so α_now < α_next monotonically).
        # The ratio α_now² / α_next² therefore lives in (0, 1].
        if eta > 0.0:
            ddimSigma = eta * sigmaNext * torch.sqrt(
                torch.clamp(
                    1.0 - (alphaNow ** 2)
                    / torch.clamp(alphaNext ** 2, min=1e-8),
                    min=0.0,
                )
            )
        else:
            ddimSigma = torch.zeros_like(sigmaNext)
        directionTerm = torch.sqrt(
            torch.clamp(sigmaNext ** 2 - ddimSigma ** 2, min=0.0)
        ) * epsilon
        noiseTerm = ddimSigma * torch.randn_like(xT) if eta > 0.0 else 0.0
        return alphaNext * x0 + directionTerm + noiseTerm

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _inferDevice(denoiser: MotionDenoiserV2) -> torch.device:
        for parameter in denoiser.parameters():
            return parameter.device
        return torch.device("cpu")

    @staticmethod
    def _initialiseNoise(
        batchSize: int,
        frames: int,
        config: object,
        device: torch.device,
        seed: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample the initial Gaussian noise for x_T (and optionally globals)."""
        # mypy: config is a MotionDenoiserV2Config, but typing it here
        # would require a cyclic import.  We access only public fields.
        boneShape = (
            batchSize,
            frames,
            config.numBones,  # type: ignore[attr-defined]
            config.motionChannels,  # type: ignore[attr-defined]
        )
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            boneNoise = torch.randn(
                boneShape, generator=generator, device=device
            )
        else:
            boneNoise = torch.randn(boneShape, device=device)

        globalChannels = config.globalChannels  # type: ignore[attr-defined]
        if globalChannels == 0:
            return boneNoise, None
        globalShape = (batchSize, frames, globalChannels)
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + 1)
            globalNoise = torch.randn(
                globalShape, generator=generator, device=device
            )
        else:
            globalNoise = torch.randn(globalShape, device=device)
        return boneNoise, globalNoise
