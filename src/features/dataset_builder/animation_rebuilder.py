
from src.shared.quaternion import Rotation
from src.shared.types import DatasetBuilderConfig, AnimationSample
from src.shared.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL24_BONE_ORDER,
    SMPL22_HIERARCHY,
    SMPL22_DEFAULT_OFFSETS,
)

from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

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
        self.animationRoots = [
            config.paths.animationRoot.resolve(),
        ]  # TODO: might simplify array
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
        raise FileNotFoundError(
            f"Animation file not found: {relativePath} "
            f"in {self._animationCandidates(relativePath)}"
        )

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
            if np.allclose(angles, 0.0):
                LOGGER.warning(
                    "Bone %s has zero rotation across %s frames.",
                    boneName,
                    frames,
                )
            rotations6d = Rotation(angles, kind="axis_angle").rot6d
            # Warn if animation is static between consecutive frames
            if frames > 1:
                identical = np.all(
                    np.isclose(angles[1:], angles[:-1]),
                    axis=1,
                )
                if identical.any():
                    idxs = np.nonzero(identical)[0][:5].tolist()
                    LOGGER.warning(
                        "Bone %s: %s identical consecutive frames; first %s.",
                        boneName,
                        int(identical.sum()),
                        idxs,
                    )
            magnitudes = np.linalg.norm(angles, axis=1)
            LOGGER.debug(
                "Bone %s norm min=%.4f max=%.4f mean=%.4f",
                boneName,
                float(np.min(magnitudes)),
                float(np.max(magnitudes)),
                float(np.mean(magnitudes)),
            )

            bones.append(self._boneWithFrames(boneName, rotations6d.tolist()))

        return bones

    def _boneWithFrames(
        self,
        boneName: str,
        rotations: List[List[float]],
    ) -> Dict[str, object]:
        frames = [
            {"frameIndex": index, "rotation": rotation}
            for index, rotation in enumerate(rotations)
        ]
        offset = SMPL22_DEFAULT_OFFSETS.get(boneName, [0.0, 0.0, 0.0])
        length = float(np.linalg.norm(offset))
        return {
            "name": boneName,
            "length": length,
            "frames": frames,
        }

    def _emptyBone(self, boneName: str, frameCount: int) -> Dict[str, object]:
        frameEntries = []
        for index in range(frameCount):
            frameEntries.append(
                {
                    "frameIndex": index,
                    "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                }
            )
        offset = SMPL22_DEFAULT_OFFSETS.get(boneName, [0.0, 0.0, 0.0])
        length = float(np.linalg.norm(offset))
        return {
            "name": boneName,
            "length": length,
            "frames": frameEntries,
        }

    def exportCollada(
        self,
        sample: AnimationSample,
        outputPath: Path,
        zeroRootTranslation: bool = False,
        anchorRootTranslation: bool = False,
    ) -> None:
        """
        Export the animation sample to a Collada (.dae) file.

        Parameters
        ----------
        sample : AnimationSample
            Animation data with SMPL axis-angles and extras.
        outputPath : Path
            Destination path for the .dae file.
        zeroRootTranslation : bool, optional
            When True, forces pelvis translation to zero for all frames.
        anchorRootTranslation : bool, optional
            When True, subtracts the first frame translation from the rest.
        """
        root = ET.Element(
            "COLLADA",
            xmlns="http://www.collada.org/2005/11/COLLADASchema",
            version="1.4.1",
        )
        
        # Asset metadata
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "created").text = "2023-01-01T00:00:00"
        ET.SubElement(asset, "modified").text = "2023-01-01T00:00:00"
        unit = ET.SubElement(asset, "unit", name="meter", meter="1")
        ET.SubElement(asset, "up_axis").text = "Z_UP"

        # Library Visual Scenes
        lib_scenes = ET.SubElement(root, "library_visual_scenes")
        visual_scene = ET.SubElement(
            lib_scenes,
            "visual_scene",
            id="Scene",
            name="Scene",
        )
        
        # Build hierarchy
        # We need to map bone names to their XML elements to nest them
        # correctly.
        bone_elements: Dict[str, ET.Element] = {}
        
        # Create nodes. Since dictionary is unordered, we must ensure parents
        # exist. SMPL22_BONE_ORDER is topologically sorted (root first), so
        # iterating it is safe.
        
        # Root node (Armature)
        armature = ET.SubElement(
            visual_scene,
            "node",
            id="Armature",
            name="Armature",
        )
        ET.SubElement(
            armature,
            "matrix",
            sid="transform",
        ).text = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
        
        for bone_name in SMPL22_BONE_ORDER:
            parent_name = SMPL22_HIERARCHY.get(bone_name)
            parent_node = (
                bone_elements.get(parent_name) if parent_name else armature
            )
            
            node = ET.SubElement(
                parent_node,
                "node",
                id=bone_name,
                name=bone_name,
                sid=bone_name,
                type="JOINT",
            )
            bone_elements[bone_name] = node
            
            # Use a single matrix transform that combines translation and rotation
            # Initial matrix is identity with offset translation
            offset = SMPL22_DEFAULT_OFFSETS.get(bone_name, [0.0, 0.0, 0.0])
            # Column-major order for Collada: m00 m10 m20 m30 m01 m11 m21 m31 ...
            # But Collada uses row-major string format:
            # m00 m01 m02 m03 m10 m11 m12 m13 m20 m21 m22 m23 m30 m31 m32 m33
            init_matrix = (
                f"1 0 0 {offset[0]} "
                f"0 1 0 {offset[1]} "
                f"0 0 1 {offset[2]} "
                f"0 0 0 1"
            )
            ET.SubElement(
                node,
                "matrix",
                sid="transform",
            ).text = init_matrix

        # Library Animations
        lib_anims = ET.SubElement(root, "library_animations")
        
        # Prepare data
        fps = sample.fps
        frames = sample.axisAngles.shape[0]
        times = np.arange(frames) / fps
        times_str = " ".join(f"{t:.4f}" for t in times)
        time_count = frames
        
        reshaped_poses = sample.axisAngles.reshape(frames, -1, 3)
        
        # Handle root translation if available
        trans_data = self._prepareRootTranslation(
            sample.extras.get("trans"),
            frames,
            zeroRootTranslation,
            anchorRootTranslation,
        )

        for bone_name in SMPL22_BONE_ORDER:
            source_index = self.boneIndex.get(bone_name)
            if source_index is None or source_index >= reshaped_poses.shape[1]:
                continue
                
            bone_rotations = reshaped_poses[:, source_index, :]
            offset = SMPL22_DEFAULT_OFFSETS.get(bone_name, [0.0, 0.0, 0.0])
            
            # Convert axis-angle to rotation matrices
            # Shape: (frames, 3, 3)
            rot_matrices = (
                Rotation(bone_rotations, kind="axis_angle")
                .matrix
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            
            # Build 4x4 transformation matrices for each frame
            # Include translation offset and optionally root translation
            matrices_4x4 = np.zeros((frames, 4, 4), dtype=np.float32)
            matrices_4x4[:, :3, :3] = rot_matrices
            matrices_4x4[:, 0, 3] = offset[0]
            matrices_4x4[:, 1, 3] = offset[1]
            matrices_4x4[:, 2, 3] = offset[2]
            matrices_4x4[:, 3, 3] = 1.0
            
            # For pelvis, add root translation if available
            if bone_name == "pelvis" and trans_data is not None:
                matrices_4x4[:, 0, 3] += trans_data[:, 0]
                matrices_4x4[:, 1, 3] += trans_data[:, 1]
                matrices_4x4[:, 2, 3] += trans_data[:, 2]
            
            # Add matrix animation channel
            self._add_matrix_animation_channel(
                lib_anims,
                bone_name,
                "transform",
                times_str,
                time_count,
                matrices_4x4,
            )

        # Scene instance
        scene = ET.SubElement(root, "scene")
        instance_visual_scene = ET.SubElement(
            scene,
            "instance_visual_scene",
            url="#Scene",
        )
        
        # Write to file
        xml_str = minidom.parseString(
            ET.tostring(root)
        ).toprettyxml(indent="  ")
        with open(outputPath, "w") as f:
            f.write(xml_str)

    def _add_animation_channel(
        self,
        lib_anims: ET.Element,
        bone_name: str,
        target_sid: str,
        times_str: str,
        time_count: int,
        data: np.ndarray,
        params: List[str],
    ) -> None:
        """Create a Collada animation channel bound to a node sid."""
        anim_id = f"{bone_name}_{target_sid}"
        anim_node = ET.SubElement(lib_anims, "animation", id=anim_id)
        
        # Source: Input (Time)
        source_input = ET.SubElement(anim_node, "source", id=f"{anim_id}_input")
        float_array = ET.SubElement(
            source_input,
            "float_array",
            id=f"{anim_id}_input_array",
            count=str(time_count),
        )
        float_array.text = times_str
        technique = ET.SubElement(source_input, "technique_common")
        accessor = ET.SubElement(
            technique,
            "accessor",
            source=f"#{anim_id}_input_array",
            count=str(time_count),
            stride="1",
        )
        ET.SubElement(accessor, "param", name="TIME", type="float")
        
        # Source: Output (Values)
        # Flatten data
        flat_data = data.flatten()
        data_str = " ".join(f"{v:.4f}" for v in flat_data)
        count = len(flat_data)
        
        source_output = ET.SubElement(
            anim_node,
            "source",
            id=f"{anim_id}_output",
        )
        float_array = ET.SubElement(
            source_output,
            "float_array",
            id=f"{anim_id}_output_array",
            count=str(count),
        )
        float_array.text = data_str
        technique = ET.SubElement(source_output, "technique_common")
        accessor = ET.SubElement(
            technique,
            "accessor",
            source=f"#{anim_id}_output_array",
            count=str(len(data)),
            stride=str(len(params)),
        )
        for param in params:
            ET.SubElement(accessor, "param", name=param, type="float")
            
        # Sampler
        sampler = ET.SubElement(
            anim_node,
            "sampler",
            id=f"{anim_id}_sampler",
        )
        ET.SubElement(
            sampler,
            "input",
            semantic="INPUT",
            source=f"#{anim_id}_input",
        )
        ET.SubElement(
            sampler,
            "input",
            semantic="OUTPUT",
            source=f"#{anim_id}_output",
        )
        ET.SubElement(
            sampler,
            "input",
            semantic="INTERPOLATION",
            source=f"#{anim_id}_interpolation",
        )
        
        # Add interpolation source (LINEAR)
        source_interp = ET.SubElement(
            anim_node,
            "source",
            id=f"{anim_id}_interpolation",
        )
        name_array = ET.SubElement(
            source_interp,
            "Name_array",
            id=f"{anim_id}_interpolation_array",
            count=str(time_count),
        )
        name_array.text = " ".join(["LINEAR"] * time_count)
        technique = ET.SubElement(source_interp, "technique_common")
        accessor = ET.SubElement(
            technique,
            "accessor",
            source=f"#{anim_id}_interpolation_array",
            count=str(time_count),
            stride="1",
        )
        ET.SubElement(accessor, "param", name="INTERPOLATION", type="name")
        
        # Channel
        # Target is node_id/sid
        # For rotation: bone_name/rotationX.ANGLE
        # For translation: bone_name/location.X (or just location)
        
        target = f"{bone_name}/{target_sid}"
        if target_sid.startswith("rotation"):
            target += ".ANGLE"
            
        ET.SubElement(
            anim_node,
            "channel",
            source=f"#{anim_id}_sampler",
            target=target,
        )

    def _axisAngleToQuat(self, axisAngle: np.ndarray) -> np.ndarray:
        # axisAngle: (N, 3)
        angles = np.linalg.norm(axisAngle, axis=1, keepdims=True)
        # Avoid division by zero
        mask = angles > 1e-8
        axes = np.zeros_like(axisAngle)
        axes[mask[:, 0]] = axisAngle[mask[:, 0]] / angles[mask[:, 0]]
        # If angle is 0, axis doesn't matter, (0,0,0) is fine.
        
        sin_half = np.sin(angles / 2)
        cos_half = np.cos(angles / 2)
        # (x, y, z, w)
        quats = np.concatenate([axes * sin_half, cos_half], axis=1)
        return quats

    def _add_matrix_animation_channel(
        self,
        lib_anims: ET.Element,
        bone_name: str,
        target_sid: str,
        times_str: str,
        time_count: int,
        matrices: np.ndarray,
    ) -> None:
        """
        Create a Collada animation channel for 4x4 matrix transforms.
        
        Parameters
        ----------
        matrices : np.ndarray
            Shape (frames, 4, 4) transformation matrices.
        """
        anim_id = f"{bone_name}_{target_sid}"
        anim_node = ET.SubElement(lib_anims, "animation", id=anim_id)
        
        # Source: Input (Time)
        source_input = ET.SubElement(anim_node, "source", id=f"{anim_id}_input")
        float_array = ET.SubElement(
            source_input,
            "float_array",
            id=f"{anim_id}_input_array",
            count=str(time_count),
        )
        float_array.text = times_str
        technique = ET.SubElement(source_input, "technique_common")
        accessor = ET.SubElement(
            technique,
            "accessor",
            source=f"#{anim_id}_input_array",
            count=str(time_count),
            stride="1",
        )
        ET.SubElement(accessor, "param", name="TIME", type="float")
        
        # Source: Output (Matrix values)
        # Flatten matrices to row-major format for Collada
        # Each 4x4 matrix becomes 16 floats
        flat_matrices = matrices.reshape(time_count, 16)
        data_str = " ".join(f"{v:.6f}" for v in flat_matrices.flatten())
        count = time_count * 16
        
        source_output = ET.SubElement(
            anim_node,
            "source",
            id=f"{anim_id}_output",
        )
        float_array = ET.SubElement(
            source_output,
            "float_array",
            id=f"{anim_id}_output_array",
            count=str(count),
        )
        float_array.text = data_str
        technique = ET.SubElement(source_output, "technique_common")
        accessor = ET.SubElement(
            technique,
            "accessor",
            source=f"#{anim_id}_output_array",
            count=str(time_count),
            stride="16",
        )
        ET.SubElement(accessor, "param", name="TRANSFORM", type="float4x4")
            
        # Sampler
        sampler = ET.SubElement(
            anim_node,
            "sampler",
            id=f"{anim_id}_sampler",
        )
        ET.SubElement(
            sampler,
            "input",
            semantic="INPUT",
            source=f"#{anim_id}_input",
        )
        ET.SubElement(
            sampler,
            "input",
            semantic="OUTPUT",
            source=f"#{anim_id}_output",
        )
        ET.SubElement(
            sampler,
            "input",
            semantic="INTERPOLATION",
            source=f"#{anim_id}_interpolation",
        )
        
        # Add interpolation source (LINEAR)
        source_interp = ET.SubElement(
            anim_node,
            "source",
            id=f"{anim_id}_interpolation",
        )
        name_array = ET.SubElement(
            source_interp,
            "Name_array",
            id=f"{anim_id}_interpolation_array",
            count=str(time_count),
        )
        name_array.text = " ".join(["LINEAR"] * time_count)
        technique = ET.SubElement(source_interp, "technique_common")
        accessor = ET.SubElement(
            technique,
            "accessor",
            source=f"#{anim_id}_interpolation_array",
            count=str(time_count),
            stride="1",
        )
        ET.SubElement(accessor, "param", name="INTERPOLATION", type="name")
        
        # Channel - target the matrix transform
        target = f"{bone_name}/{target_sid}"
        ET.SubElement(
            anim_node,
            "channel",
            source=f"#{anim_id}_sampler",
            target=target,
        )

    def _prepareRootTranslation(
        self,
        rawTrans: object,
        frameCount: int,
        zeroRootTranslation: bool,
        anchorRootTranslation: bool,
    ) -> Optional[np.ndarray]:
        if rawTrans is None or not isinstance(rawTrans, (np.ndarray, list)):
            return None
        trans_data = np.array(rawTrans)
        if trans_data.shape != (frameCount, 3):
            LOGGER.warning(
                "Ignoring root translation with unexpected shape %s, "
                "expected (%s, 3).",
                trans_data.shape,
                frameCount,
            )
            return None
        if zeroRootTranslation:
            LOGGER.info(
                "Zeroing root translation across %s frames.",
                frameCount,
            )
            return np.zeros_like(trans_data)
        if anchorRootTranslation:
            LOGGER.info("Anchoring root translation to first frame.")
            return trans_data - trans_data[0]
        return trans_data

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
