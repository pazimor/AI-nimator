"""Helpers to parse converted dataset assets for downstream tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from src.shared.constants.clip import ROTATION_CHANNELS
from src.shared.skeleton import SkeletonNormalizer
from src.shared.types import PromptSegment

MotionPayload = Tuple[torch.Tensor, Dict[str, object]]


def loadPromptSegments(
    path: str | Path,
) -> tuple[str, Dict[str, object], List[PromptSegment]]:
    """
    Return prompt segments plus file-level metadata.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the prompt JSON file.

    Returns
    -------
    tuple[str, Dict[str, object], List[PromptSegment]]
        File tag, metadata, and parsed prompt segments.
    """
    payload = _readJsonLike(path)
    tag = str(payload.get("tag", "") or "")
    metadata = payload.get("meta", {}) or {}
    segments = [
        _segmentFromDict(rawSegment)
        for rawSegment in payload.get("segments", [])
    ]
    return tag, metadata, segments


def loadAnimationPayload(
    path: str | Path,
    skeletonNormalizer: SkeletonNormalizer | None = None,
) -> MotionPayload:
    """
    Return the motion tensor and metadata stored in an animation payload.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the animation.js or animation.json file.
    skeletonNormalizer : SkeletonNormalizer | None, optional
        Normalizer applied to incoming bones before tensorization.

    Returns
    -------
    MotionPayload
        Motion tensor shaped (frames, bones, 6) and associated metadata.
    """
    payload = _readJsonLike(path)
    bones = payload.get("bones", [])
    if skeletonNormalizer is not None:
        bones = skeletonNormalizer.normalizeBones(bones)
    metadata = _extractAnimationMeta(payload)
    frameCount = _inferFrameCount(metadata, bones)
    motion = _bonesToTensor(bones, frameCount)
    return motion, metadata


def _segmentFromDict(rawSegment: Dict[str, object]) -> PromptSegment:
    return PromptSegment(
        startFrame=int(rawSegment.get("startFrame", 0)),
        endFrame=int(rawSegment.get("endFrame", 0)),
        text=str(rawSegment.get("text", "")),
        sourceFile=str(rawSegment.get("sourceFile", "")),
    )


def _readJsonLike(path: str | Path) -> Dict[str, object]:
    normalizedPath = Path(path)
    content = normalizedPath.read_text(encoding="utf-8").strip()
    sanitized = _sanitizeJsonText(content)
    return json.loads(sanitized)


def _sanitizeJsonText(rawText: str) -> str:
    cleaned = rawText.strip()
    prefix = "export default"
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix):].strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    return cleaned


def _extractAnimationMeta(payload: Dict[str, object]) -> Dict[str, object]:
    meta = payload.get("meta", {}) or {}
    return dict(meta)


def _inferFrameCount(
    meta: Dict[str, object],
    bones: Sequence[Dict[str, object]],
) -> int:
    frameCount = int(meta.get("frames", 0) or 0)
    for bone in bones:
        for frame in bone.get("frames", []):
            frameIndex = int(frame.get("frameIndex", 0))
            frameCount = max(frameCount, frameIndex + 1)
    return frameCount


def _bonesToTensor(
    bones: Iterable[Dict[str, object]],
    frameCount: int,
) -> torch.Tensor:
    bonesList = list(bones)
    motion = torch.zeros(
        (frameCount, len(bonesList), ROTATION_CHANNELS),
        dtype=torch.float32,
    )
    for boneIndex, bone in enumerate(bonesList):
        for frame in bone.get("frames", []):
            frameIndex = int(frame.get("frameIndex", 0))
            rotation = frame.get("rotation", [0.0] * ROTATION_CHANNELS)
            motion[frameIndex, boneIndex] = torch.tensor(
                rotation[:ROTATION_CHANNELS],
                dtype=torch.float32,
            )
    return motion
