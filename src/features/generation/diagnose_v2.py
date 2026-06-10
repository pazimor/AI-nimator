"""Diagnostic helpers for AI-nimator v2 generated motion.

This module computes simple per-sample sanity statistics on a sampler
output, the kind of numbers you want to look at when the trained model
gives "tremblant / mostly static" animations and you don't yet have the
full Phase C eval suite (FID, R-precision, …) to fall back on.

The metrics are deliberately cheap and do **not** require any external
encoder — they run on whatever rotation6d / root_translation tensors
the v2 sampler produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationStats:
    """Lightweight statistics for one generated motion sequence.

    Attributes
    ----------
    rotationMean, rotationStd : float
        Per-frame, per-bone, per-channel mean / std of the rotation6d
        output.  Should be close to the training distribution stats
        (computed by :class:`MotionNormalizer.fitFromTensors`).
    rootDisplacement : float
        Total Euclidean distance travelled by the pelvis across the
        sequence (root_translation deltas summed in metres).  A walking
        motion should travel a few metres; "in-place" tremblement leaves
        this near zero.
    rotationVelocityMean : float
        Average L2 norm of the per-frame rotation6d delta.  A natural
        motion has a moderate value; ``≈ 0`` means a static pose,
        ``>> training`` means high-frequency jitter.
    rotationAccelerationMean : float
        Average L2 norm of the second difference of rotation6d.  This
        is the most direct **tremblement** indicator — a tremor has
        high frame-to-frame acceleration changes; smooth motion has
        low ones.
    rangeBone : tuple[float, float]
        (min, max) of the rotation6d tensor.  Out-of-distribution range
        (e.g. saturated outside [-1.5, 1.5]) is a sign that the
        normalizer denormalisation went wrong somewhere.
    rangeGlobal : tuple[float, float] or None
        (min, max) of the root_translation tensor in metres, or None
        when the sampler ran without a global branch.
    """

    rotationMean: float
    rotationStd: float
    rootDisplacement: float
    rotationVelocityMean: float
    rotationAccelerationMean: float
    rangeBone: tuple[float, float]
    rangeGlobal: tuple[float, float] | None


def computeGenerationStats(
    boneMotion: torch.Tensor,
    globalMotion: torch.Tensor | None = None,
) -> GenerationStats:
    """Compute :class:`GenerationStats` for a single sampled sequence.

    Parameters
    ----------
    boneMotion : torch.Tensor
        rotation6d tensor of shape ``(F, numBones, 6)`` (single-sample
        — the leading batch dimension must already be peeled off).
    globalMotion : torch.Tensor, optional
        root_translation tensor of shape ``(F, 3)``.

    Returns
    -------
    GenerationStats
    """
    if boneMotion.ndim != 3:
        raise ValueError(
            "boneMotion must be 3-D (F, numBones, 6); got shape "
            f"{tuple(boneMotion.shape)}."
        )

    bone = boneMotion.detach().float().cpu()
    rotationMean = float(bone.mean().item())
    rotationStd = float(bone.std().item())
    rangeBone = (float(bone.min().item()), float(bone.max().item()))

    # Rotation velocity / acceleration in flattened 6D space.
    if bone.shape[0] >= 2:
        deltaBone = bone[1:] - bone[:-1]
        rotationVelocityMean = float(deltaBone.abs().mean().item())
    else:
        rotationVelocityMean = 0.0
    if bone.shape[0] >= 3:
        accelBone = (bone[2:] - 2 * bone[1:-1] + bone[:-2])
        rotationAccelerationMean = float(accelBone.abs().mean().item())
    else:
        rotationAccelerationMean = 0.0

    rootDisplacement = 0.0
    rangeGlobal: tuple[float, float] | None = None
    if globalMotion is not None:
        rt = globalMotion.detach().float().cpu()
        if rt.shape[0] >= 2:
            deltas = rt[1:] - rt[:-1]
            rootDisplacement = float(deltas.norm(dim=-1).sum().item())
        rangeGlobal = (float(rt.min().item()), float(rt.max().item()))

    return GenerationStats(
        rotationMean=rotationMean,
        rotationStd=rotationStd,
        rootDisplacement=rootDisplacement,
        rotationVelocityMean=rotationVelocityMean,
        rotationAccelerationMean=rotationAccelerationMean,
        rangeBone=rangeBone,
        rangeGlobal=rangeGlobal,
    )


def formatStatsLine(
    stats: GenerationStats,
    prefix: str = "",
) -> str:
    """One-line readable representation of :class:`GenerationStats`."""
    globalRange = (
        f"global=[{stats.rangeGlobal[0]:.3f}, {stats.rangeGlobal[1]:.3f}]"
        if stats.rangeGlobal is not None
        else "global=N/A"
    )
    return (
        f"{prefix}rot μ={stats.rotationMean:+.3f} σ={stats.rotationStd:.3f} "
        f"Δrot/frame={stats.rotationVelocityMean:.4f} "
        f"Δ²rot/frame={stats.rotationAccelerationMean:.4f}  "
        f"root_displ={stats.rootDisplacement:.3f}m  "
        f"bone=[{stats.rangeBone[0]:.3f}, {stats.rangeBone[1]:.3f}] "
        f"{globalRange}"
    )
