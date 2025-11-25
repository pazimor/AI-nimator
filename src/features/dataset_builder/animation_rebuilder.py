
from src.shared.quaternion import Rotation
from src.shared.types import DatasetBuilderConfig, AnimationSample
from src.shared.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL24_BONE_ORDER,
)

from pathlib import Path
from typing import Dict, List

import logging
import numpy as np

LOGGER = logging.getLogger("converted Prompt Repository")

class AnimationRebuilder:
    """Convert AMASS pose parameters into the canonical JSON schema."""

    def __init__(self, config: DatasetBuilderConfig) -> None:
        self.config = config
        self.sourceBoneOrder = SMPL24_BONE_ORDER
        self.targetBoneOrder = SMPL22_BONE_ORDER
        self.boneIndex = {
            boneName: index
            for index, boneName in enumerate(self.sourceBoneOrder)
        }
        self.animationRoots = [config.paths.animationRoot.resolve()] ##TODO: might simplify array
        LOGGER.info("animations roots: %s", self.animationRoots)

    def loadSample(self, relativePath: Path) -> AnimationSample:
        """Load the NPZ file referenced by `relativePath`."""
        resolved = self._resolveAnimationPath(relativePath)
        with np.load(resolved) as raw:
            axisAngles = raw["poses"].astype(np.float32)
            extras = {
                key: self._ensureJsonSerializable(raw[key])
                for key in raw.files
                if key != "poses"
            }
        fps = self._resolveFps(extras)
        return AnimationSample(
            relativePath=relativePath,
            resolvedPath=resolved,
            axisAngles=axisAngles,
            fps=fps,
            extras=extras,
        )

    def buildPayload(self, sample: AnimationSample) -> Dict[str, object]:
        """Return the canonical payload with metadata, bones, and extras."""
        bones = self._buildBones(sample.axisAngles)
        meta = {
            "fps": sample.fps,
            "frames": int(sample.axisAngles.shape[0]),
            "joints": "SMPL-22",
            "source": sample.relativePath.as_posix(),
        }
        return {"meta": meta, "bones": bones, "extras": sample.extras}

    def _resolveAnimationPath(self, relativePath: Path) -> Path:
        for candidate in self._animationCandidates(relativePath):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Animation file not found: {relativePath} in {self._animationCandidates(relativePath)}")

    def _animationCandidates(self, relativePath: Path) -> List[Path]:
        bases: List[Path] = []
        parts = list(relativePath.parts)
        for root in self.animationRoots:
            bases.append(root / relativePath)
            if parts:
                bases.append(root / parts[0] / relativePath)
            if len(parts) >= 2:
                bases.append(root / parts[0] / parts[1] / relativePath)
        suffixPreferences = [
            "",
            self.config.processing.animationExtension,
            ".npz",
            ".npy",
        ]
        candidates: List[Path] = []
        for base in bases:
            suffixes: List[str] = []
            if base.suffix:
                suffixes.append(base.suffix)
            for suffix in suffixPreferences:
                if suffix and suffix not in suffixes:
                    suffixes.append(suffix)
            candidates.append(base)
            for suffix in suffixes:
                candidates.append(base.with_suffix(suffix))
        unique: List[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = str(candidate.resolve(strict=False))
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(candidate)
        return unique

    def _resolveFps(self, extras: Dict[str, object]) -> int:
        value = extras.get("mocap_framerate")
        if isinstance(value, (float, int)):
            return int(round(value))
        if isinstance(value, list) and value:
            return int(round(float(value[0])))
        return self.config.processing.fallbackFps

    def _buildBones(self, axisAngles: np.ndarray) -> List[Dict[str, object]]:
        frames = axisAngles.shape[0]
        reshaped = axisAngles.reshape(frames, -1, 3)
        bones: List[Dict[str, object]] = []

        for boneName in self.targetBoneOrder:
            sourceIndex = self.boneIndex.get(boneName)
            if sourceIndex is None or sourceIndex >= reshaped.shape[1]:
                bones.append(self._emptyBone(boneName, frames))
                continue
            angles = reshaped[:, sourceIndex, :]
            rotations6d = Rotation(angles, kind="euler").rot6d

            bones.append(self._boneWithFrames(boneName, rotations6d.tolist()))

        return bones

    def _boneWithFrames(
        self,
        boneName: str,
        rotations: List[List[float]],
    ) -> Dict[str, object]:
        frames = [
            {"frameIndex": index, "rotation": rotation} ##TODO: delete extra
            for index, rotation in enumerate(rotations)
        ]
        return {"name": boneName, "length": 0.0, "frames": frames}

    def _emptyBone(self, boneName: str, frameCount: int) -> Dict[str, object]:
        frameEntries = []
        for index in range(frameCount):
            frameEntries.append(
                {
                    "frameIndex": index,
                    "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                }
            )
        return {"name": boneName, "length": 0.0, "frames": frameEntries}

    def _serialize(self, value: object) -> object:
        return

    def _ensureJsonSerializable(self, value: object) -> object:
        if isinstance(value, np.ndarray):
            return self._ensureJsonSerializable(value.tolist())
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, (list, tuple, set)):
            return [self._ensureJsonSerializable(entry) for entry in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

