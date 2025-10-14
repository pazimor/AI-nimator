"""Data-oriented dataclasses shared between features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import torch
from torch import Tensor


@dataclass
class AnimationPromptSample:
    """Dataset sample pairing rotation, text and optional context."""

    rotation6d: Tensor
    textPrompt: str
    tagLabel: str
    boneNames: List[str]
    contextSequence: Optional[Tensor]


@dataclass
class ClipRecord:
    """Metadata gathered for a single animation clip."""

    animationPath: Path
    rotations: Dict[str, List[str]]
    prompts: List[Dict[str, str]]


@dataclass
class DatasetCache:
    """Pre-computed tensors for fast sampling during training."""

    rotationSequences: List[Tensor]
    textEmbeddings: List[Tensor]
    tagEmbeddings: List[Tensor]


@dataclass
class AnimationClip:
    """In-memory representation of an animation clip.

    Parameters
    ----------
    positions : np.ndarray
        Array shaped ``(frameCount, jointCount, 3)`` describing root-relative
        joint positions for every frame.
    rotations : np.ndarray
        Array shaped ``(frameCount, jointCount, 4)`` storing quaternions in
        ``(x, y, z, w)`` order for each joint and frame.
    boneNames : List[str]
        Sequence of joint identifiers matching the second dimension of
        ``positions`` and ``rotations``.
    frameRate : float
        Sampling rate expressed in frames per second.
    """

    positions: np.ndarray
    rotations: np.ndarray
    boneNames: List[str]
    frameRate: float


@dataclass
class AnimationExportPayload:
    """Container describing an exported animation artefact.

    Parameters
    ----------
    content : bytes
        Raw serialised payload ready to be written to disk.
    mediaType : str
        MIME-like string describing the payload type (for example
        ``"application/octet-stream"`` or ``"text/plain"``).
    fileExtension : str
        Extension (without dot) recommended when persisting ``content``.
    """

    content: bytes
    mediaType: str
    fileExtension: str
