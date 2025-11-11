"""Skeleton normalization utilities shared across the project."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from src.shared.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL24_BONE_ORDER,
    SMPL_BONE_ALIASES,
    SkeletonDefinition,
)


class BoneNameNormalizer:
    """Normalize skeleton bone names across datasets."""

    _sanitizer = re.compile(r"[^a-z]")

    @staticmethod
    def normalizeName(rawName: str) -> str:
        """
        Return the canonical SMPL bone name.

        Parameters
        ----------
        rawName : str
            Bone name coming from a dataset.

        Returns
        -------
        str
            Canonical SMPL bone name.
        """
        normalized = BoneNameNormalizer._sanitize(rawName)
        for canonical, aliases in SMPL_BONE_ALIASES.items():
            if normalized == BoneNameNormalizer._sanitize(canonical):
                return canonical
            if normalized in {
                BoneNameNormalizer._sanitize(alias) for alias in aliases
            }:
                return canonical
        return rawName

    @staticmethod
    def _sanitize(value: str) -> str:
        """
        Normalize a bone name for comparison.

        Parameters
        ----------
        value : str
            Raw bone name.

        Returns
        -------
        str
            Lower-case string stripped from non-alphabetic characters.
        """
        return BoneNameNormalizer._sanitizer.sub("", value.lower())


@dataclass
class SkeletonNormalizer:
    """Convert any SMPL-like skeleton into the canonical SMPL-22 order."""

    targetSkeleton: SkeletonDefinition = SkeletonDefinition(
        name="SMPL-22",
        bones=SMPL22_BONE_ORDER,
    )

    def normalizeBones(
        self,
        bones: Iterable[Mapping[str, object]],
    ) -> List[Dict[str, object]]:
        """
        Return bones sorted according to the canonical order.

        Bones belonging to SMPL-24 are down-converted to SMPL-22 by removing
        hand tips. Missing bones are kept as placeholders to retain indexes.

        Parameters
        ----------
        bones : Iterable[Mapping[str, object]]
            Bone payloads that need to be reordered.

        Returns
        -------
        List[Dict[str, object]]
            Bones sorted according to the target skeleton.
        """
        normalized = self._indexBones(bones)
        ordered: List[Dict[str, object]] = []
        for boneName in self.targetSkeleton.bones:
            ordered.append(
                normalized.get(boneName, self._placeholder(boneName)),
            )
        return ordered

    def detectSourceSkeleton(
        self,
        bones: Sequence[Mapping[str, object]],
    ) -> str:
        """
        Return the detected skeleton variant name.

        Parameters
        ----------
        bones : Sequence[Mapping[str, object]]
            Bone payloads gathered from the dataset.

        Returns
        -------
        str
            One of "SMPL-22", "SMPL-24" or "Unknown".
        """
        names = {
            BoneNameNormalizer.normalizeName(str(bone.get("name", "")))
            for bone in bones
        }
        if all(name in set(SMPL24_BONE_ORDER) for name in names):
            return "SMPL-24"
        if all(name in set(SMPL22_BONE_ORDER) for name in names):
            return "SMPL-22"
        return "Unknown"

    def _indexBones(
        self,
        bones: Iterable[Mapping[str, object]],
    ) -> Dict[str, Dict[str, object]]:
        """
        Index bones by their canonical name.

        Parameters
        ----------
        bones : Iterable[Mapping[str, object]]
            Raw bone payloads.

        Returns
        -------
        Dict[str, Dict[str, object]]
            Mapping between canonical names and bone payloads.
        """
        indexed: Dict[str, Dict[str, object]] = {}
        for bone in bones:
            boneName = BoneNameNormalizer.normalizeName(
                str(bone.get("name", "")),
            )
            if boneName in {"leftHand", "rightHand"}:
                continue
            indexed[boneName] = dict(bone)
            indexed[boneName]["name"] = boneName
        return indexed

    @staticmethod
    def _placeholder(boneName: str) -> Dict[str, object]:
        """
        Return a placeholder bone preserving the canonical index.

        Parameters
        ----------
        boneName : str
            Canonical bone name.

        Returns
        -------
        Dict[str, object]
            Placeholder bone payload.
        """
        return {"name": boneName, "frames": [], "length": 0.0}
