"""Tests for skeleton normalization utilities."""

from __future__ import annotations

from typing import List

from src.shared.skeleton import SkeletonNormalizer


def test_skeletonNormalizerDownConvertsSmpl24() -> None:
    """
    Ensure SMPL-24 payloads are down-converted to SMPL-22.

    Returns
    -------
    None
        Pytest manages assertions.
    """
    bones: List[dict] = []
    for index, name in enumerate(
        [
            "pelvis",
            "leftHip",
            "rightHip",
            "spine1",
            "leftKnee",
            "rightKnee",
            "spine2",
            "leftAnkle",
            "rightAnkle",
            "spine3",
            "leftFoot",
            "rightFoot",
            "neck",
            "leftCollar",
            "rightCollar",
            "head",
            "leftShoulder",
            "rightShoulder",
            "leftElbow",
            "rightElbow",
            "leftWrist",
            "rightWrist",
            "leftHand",
            "rightHand",
        ]
    ):
        bones.append({"name": name, "length": float(index), "frames": []})
    normalizer = SkeletonNormalizer()
    normalized = normalizer.normalizeBones(bones)
    assert len(normalized) == 22
    assert normalized[0]["name"] == "pelvis"
    assert normalized[-1]["name"] == "rightWrist"
