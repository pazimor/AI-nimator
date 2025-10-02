"""Data-oriented dataclasses shared between features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
