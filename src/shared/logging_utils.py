"""Shared logging helpers for training workflows."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import torch

LOGGER = logging.getLogger("clip.training")


def logClipBatchStats(
    batchIndex: int,
    outputs: Mapping[str, torch.Tensor | float],
    logger: logging.Logger | None = None,
) -> None:
    """
    Log tensor shapes and logits emitted by a CLIP batch.

    Parameters
    ----------
    batchIndex : int
        Index of the processed batch.
    outputs : Mapping[str, torch.Tensor | float]
        Model forward outputs to summarize.
    logger : logging.Logger | None, optional
        Custom logger; defaults to clip.training.
    """
    resolvedLogger = logger or LOGGER
    textShape = _tensorShape(outputs.get("text_embeds"))
    motionShape = _tensorShape(outputs.get("motion_embeds"))
    logitsMean = _tensorMean(outputs.get("logits_per_text"))
    logitScale = _tensorMean(outputs.get("logit_scale"))
    resolvedLogger.info(
        "[batch %s] text=%s motion=%s logits=%0.4f logit_scale=%0.4f",
        batchIndex,
        textShape,
        motionShape,
        logitsMean,
        logitScale,
    )


def _tensorShape(tensor: torch.Tensor | float | None) -> Sequence[int]:
    """
    Return the tensor shape or an empty tuple for scalars.

    Parameters
    ----------
    tensor : torch.Tensor | float | None
        Tensor to inspect.

    Returns
    -------
    Sequence[int]
        Tensor shape description.
    """
    if isinstance(tensor, torch.Tensor):
        return tuple(int(dimension) for dimension in tensor.shape)
    return ()


def _tensorMean(tensor: torch.Tensor | float | None) -> float:
    """
    Return the scalar mean extracted from a tensor or float.

    Parameters
    ----------
    tensor : torch.Tensor | float | None
        Value to average.

    Returns
    -------
    float
        Mean value cast to float.
    """
    if isinstance(tensor, torch.Tensor):
        return float(tensor.detach().mean().item())
    if tensor is not None:
        return float(tensor)
    return 0.0
