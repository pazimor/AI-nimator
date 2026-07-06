"""Z-normalization helper for AI-nimator v2 motion features.

Why this module exists
----------------------
The diffusion process ``x_t = α·x_0 + σ·ε`` mixes the clean motion
``x_0`` with Gaussian noise ``ε ~ N(0, 1)``.  For the schedule to behave
the way DDPM/DDIM/v-prediction theory predicts, ``x_0`` and ``ε`` must
share the same scale — otherwise:

* At intermediate ``t`` the noisy input ``x_t`` is dominated by
  whichever has the larger variance, and the denoiser cannot learn a
  consistent mapping back to ``x_0``.
* At inference, the sampler initialises ``x_T = ε ~ N(0, 1)``; if the
  training distribution of ``x_0`` had std ≠ 1, the sampler is
  effectively starting from a noise level the network never saw.

Concretely in v2 motion features: rotation6d values typically sit in
``[-1, 1]`` while root_translation can extend to several meters.
Without normalisation, the same noise ``ε`` is "loud" relative to the
rotations and "quiet" relative to the trajectory, and the network fails
in both directions.

The legacy v1 stack solves this with z-normalization buffers stored on
the :class:`MotionGenerator` itself
(``motion_generator.py:174-181, 781-823``).  v2 keeps the same idea but
factors the stats out of the denoiser into a standalone
:class:`MotionNormalizer` so the model stays purely about denoising and
the data-pipeline concern (normalize / denormalize) lives elsewhere.

Usage flow
----------
.. code-block:: python

    normalizer = MotionNormalizer(numBones=22, motionChannels=6, globalChannels=3)
    normalizer.fitFromTensors(boneSamples=[rot6d], globalSamples=[rtrans])
    # Training: feed normalized motion to the diffusion process
    rotNorm = normalizer.normalizeBone(rotation6d)
    rtransNorm = normalizer.normalizeGlobal(rootTranslation)
    # Inference: denormalize after sampling
    rotation6d = normalizer.denormalizeBone(rotNorm)
    rootTranslation = normalizer.denormalizeGlobal(rtransNorm)

The module also round-trips cleanly through ``state_dict`` so the
trained statistics can be saved alongside the denoiser checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

# Tiny floor on ``std`` so we never divide by exactly zero.  ``1e-5`` is
# the same value used by :func:`computeMotionStatistics` in the v1
# training loop, so v2 mirrors v1's numerical behaviour at the edges.
EPSILON_STD = 1e-5


@dataclass(frozen=True)
class MotionNormalizerStats:
    """Plain container for the four normalization tensors.

    Useful for serialisation: a frozen dataclass converts trivially to
    a state-dict-like payload without having to instantiate a full
    :class:`nn.Module`.
    """

    boneMean: torch.Tensor
    boneStd: torch.Tensor
    globalMean: torch.Tensor | None
    globalStd: torch.Tensor | None


class MotionNormalizer(nn.Module):
    """Z-score normalization for the v2 lean motion representation.

    Two pairs of buffers are maintained:

    * ``boneMean`` / ``boneStd`` — shape
      ``(1, 1, numBones, motionChannels)``, applied to bone-scoped
      features (rotation6d).
    * ``globalMean`` / ``globalStd`` — shape ``(1, 1, globalChannels)``,
      applied to global per-frame features (root_translation).  Only
      registered when ``globalChannels > 0``.

    Initial values are mean=0 / std=1 (identity transform).  Call
    :meth:`fitFromTensors` once before training to populate them with
    the dataset statistics.  After fitting, :meth:`normalizeBone` /
    :meth:`denormalizeBone` (and the global counterparts) are inverse
    of each other up to ``EPSILON_STD`` precision.
    """

    def __init__(
        self,
        numBones: int,
        motionChannels: int,
        globalChannels: int = 0,
    ) -> None:
        super().__init__()
        if numBones < 1:
            raise ValueError("numBones must be >= 1.")
        if motionChannels < 1:
            raise ValueError("motionChannels must be >= 1.")
        if globalChannels < 0:
            raise ValueError("globalChannels must be >= 0.")
        self._numBones = int(numBones)
        self._motionChannels = int(motionChannels)
        self._globalChannels = int(globalChannels)

        self.register_buffer(
            "boneMean",
            torch.zeros(1, 1, numBones, motionChannels),
        )
        self.register_buffer(
            "boneStd",
            torch.ones(1, 1, numBones, motionChannels),
        )
        if globalChannels > 0:
            self.register_buffer(
                "globalMean",
                torch.zeros(1, 1, globalChannels),
            )
            self.register_buffer(
                "globalStd",
                torch.ones(1, 1, globalChannels),
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def numBones(self) -> int:
        return self._numBones

    @property
    def motionChannels(self) -> int:
        return self._motionChannels

    @property
    def globalChannels(self) -> int:
        return self._globalChannels

    @property
    def hasGlobalBranch(self) -> bool:
        return self._globalChannels > 0

    # ------------------------------------------------------------------
    # Statistics fitting
    # ------------------------------------------------------------------
    def fitFromTensors(
        self,
        boneSamples: Sequence[torch.Tensor],
        globalSamples: Sequence[torch.Tensor] | None = None,
    ) -> MotionNormalizerStats:
        """Compute ``mean`` / ``std`` from a list of training tensors.

        Each entry of ``boneSamples`` may be ``(F, B, C)`` (one sample)
        or ``(N, F, B, C)`` (a batch).  Frames are flattened across all
        batches and samples, then per-channel mean / std is computed
        along that combined axis.

        ``globalSamples`` mirrors the pattern with shape ``(F, C)`` or
        ``(N, F, C)``.

        Returns the freshly-computed :class:`MotionNormalizerStats` so
        callers can persist them outside of the module if they want.
        """
        if not boneSamples:
            raise ValueError("boneSamples must contain at least one tensor.")

        bone = self._stackPerFrame(
            boneSamples,
            expectedTrailing=(self._numBones, self._motionChannels),
        )
        boneMean = bone.mean(dim=0, keepdim=True).unsqueeze(0)
        boneStd = bone.std(dim=0, keepdim=True, unbiased=False).unsqueeze(0)
        boneStd = boneStd.clamp(min=EPSILON_STD)
        self.boneMean.copy_(boneMean.to(self.boneMean.dtype))
        self.boneStd.copy_(boneStd.to(self.boneStd.dtype))

        globalMean: torch.Tensor | None = None
        globalStd: torch.Tensor | None = None
        if self.hasGlobalBranch:
            if globalSamples is None or len(globalSamples) == 0:
                raise ValueError(
                    "globalSamples must be provided when "
                    "globalChannels > 0."
                )
            global_ = self._stackPerFrame(
                globalSamples,
                expectedTrailing=(self._globalChannels,),
            )
            globalMean = global_.mean(dim=0, keepdim=True).unsqueeze(0)
            globalStd = global_.std(
                dim=0, keepdim=True, unbiased=False
            ).unsqueeze(0)
            globalStd = globalStd.clamp(min=EPSILON_STD)
            self.globalMean.copy_(globalMean.to(self.globalMean.dtype))
            self.globalStd.copy_(globalStd.to(self.globalStd.dtype))

        return MotionNormalizerStats(
            boneMean=self.boneMean.detach().clone(),
            boneStd=self.boneStd.detach().clone(),
            globalMean=(
                self.globalMean.detach().clone()
                if self.hasGlobalBranch
                else None
            ),
            globalStd=(
                self.globalStd.detach().clone()
                if self.hasGlobalBranch
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Forward / inverse
    # ------------------------------------------------------------------
    def normalizeBone(self, bone: torch.Tensor) -> torch.Tensor:
        """Apply ``(bone − mean) / std`` to a bone-scoped tensor.

        Accepts shapes ``(F, B, C)`` or ``(N, F, B, C)``; the buffers
        broadcast over the leading dims.
        """
        self._validateBoneShape(bone)
        return (bone - self.boneMean) / self.boneStd.clamp(min=EPSILON_STD)

    def denormalizeBone(self, bone: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalizeBone`."""
        self._validateBoneShape(bone)
        return bone * self.boneStd + self.boneMean

    def normalizeGlobal(self, features: torch.Tensor) -> torch.Tensor:
        """Apply ``(features − mean) / std`` to a global-scoped tensor."""
        if not self.hasGlobalBranch:
            raise RuntimeError(
                "normalizeGlobal called on a normalizer without a "
                "global branch (globalChannels=0)."
            )
        self._validateGlobalShape(features)
        return (
            (features - self.globalMean)
            / self.globalStd.clamp(min=EPSILON_STD)
        )

    def denormalizeGlobal(self, features: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalizeGlobal`."""
        if not self.hasGlobalBranch:
            raise RuntimeError(
                "denormalizeGlobal called on a normalizer without a "
                "global branch (globalChannels=0)."
            )
        self._validateGlobalShape(features)
        return features * self.globalStd + self.globalMean

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _stackPerFrame(
        samples: Iterable[torch.Tensor],
        expectedTrailing: tuple[int, ...],
    ) -> torch.Tensor:
        """Flatten one or more samples to ``(total_frames, *trailing)``."""
        flattened: list[torch.Tensor] = []
        for sample in samples:
            tensor = sample.float()
            if tensor.ndim < len(expectedTrailing) + 1:
                raise ValueError(
                    "Sample tensor must have at least "
                    f"{len(expectedTrailing) + 1} dims; got shape "
                    f"{tuple(tensor.shape)}."
                )
            if tuple(tensor.shape[-len(expectedTrailing):]) != expectedTrailing:
                raise ValueError(
                    "Sample trailing dims do not match expected "
                    f"{expectedTrailing}; got shape "
                    f"{tuple(tensor.shape)}."
                )
            flattened.append(
                tensor.reshape(-1, *expectedTrailing)
            )
        return torch.cat(flattened, dim=0)

    def _validateBoneShape(self, bone: torch.Tensor) -> None:
        if bone.shape[-2:] != (self._numBones, self._motionChannels):
            raise ValueError(
                "Bone tensor trailing dims must be "
                f"({self._numBones}, {self._motionChannels}); got "
                f"shape {tuple(bone.shape)}."
            )

    def _validateGlobalShape(self, features: torch.Tensor) -> None:
        if features.shape[-1] != self._globalChannels:
            raise ValueError(
                "Global tensor last dim must be "
                f"{self._globalChannels}; got shape "
                f"{tuple(features.shape)}."
            )

    # ------------------------------------------------------------------
    # Serialization helpers (used by the training checkpoint format)
    # ------------------------------------------------------------------
    def configToDict(self) -> dict[str, int]:
        """Return the constructor arguments as a JSON-friendly dict."""
        return {
            "numBones": self._numBones,
            "motionChannels": self._motionChannels,
            "globalChannels": self._globalChannels,
        }

    @classmethod
    def fromConfigDict(
        cls,
        payload: dict[str, int],
    ) -> "MotionNormalizer":
        """Inverse of :meth:`configToDict`."""
        return cls(
            numBones=int(payload["numBones"]),
            motionChannels=int(payload["motionChannels"]),
            globalChannels=int(payload["globalChannels"]),
        )


# ---------------------------------------------------------------------
# Single-frame delta helpers (Goal A controller)
# ---------------------------------------------------------------------
# The controller works with per-step deltas of shape ``(B, numBones, C)``
# (bone) and ``(B, C)`` (global) — i.e. *frame-less*.  The normalizer
# buffers are 4-D / 3-D and treat a frame-less bone tensor as
# ``(F, B, C)``, broadcasting in a phantom leading axis.  These helpers
# add a singleton frame axis around the call so single-frame deltas
# round-trip with the same statistics as the state.


def normalizeStepDelta(
    normalizer: MotionNormalizer,
    boneDelta: torch.Tensor,
    globalDelta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Z-normalize frame-less bone/global deltas.

    Parameters
    ----------
    normalizer : MotionNormalizer
        A normalizer fitted on delta statistics.
    boneDelta : torch.Tensor
        Bone delta, shape ``(..., numBones, motionChannels)``.
    globalDelta : torch.Tensor or None
        Global delta, shape ``(..., globalChannels)``, or ``None``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        Normalized ``(boneDelta, globalDelta)``.
    """
    normBone = normalizer.normalizeBone(boneDelta.unsqueeze(1)).squeeze(1)
    normGlobal: torch.Tensor | None = None
    if globalDelta is not None and normalizer.hasGlobalBranch:
        normGlobal = normalizer.normalizeGlobal(
            globalDelta.unsqueeze(1)
        ).squeeze(1)
    return normBone, normGlobal


def denormalizeStepDelta(
    normalizer: MotionNormalizer,
    boneDelta: torch.Tensor,
    globalDelta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Inverse of :func:`normalizeStepDelta`."""
    rawBone = normalizer.denormalizeBone(boneDelta.unsqueeze(1)).squeeze(1)
    rawGlobal: torch.Tensor | None = None
    if globalDelta is not None and normalizer.hasGlobalBranch:
        rawGlobal = normalizer.denormalizeGlobal(
            globalDelta.unsqueeze(1)
        ).squeeze(1)
    return rawBone, rawGlobal
