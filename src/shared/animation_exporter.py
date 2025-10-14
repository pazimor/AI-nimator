"""Utilities to export animation clips into standard file formats."""

from __future__ import annotations

import io
import math
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
from xml.dom import minidom
from xml.etree import ElementTree as ET

from src.shared.animation_io import load_animation_json
from src.shared.quaternion import Quaternion
from src.shared.types.data import AnimationClip, AnimationExportPayload
from src.shared.constants.skeleton import resolve_skeleton


class AnimationExportError(RuntimeError):
    """Raised when a requested export operation cannot be satisfied."""


class AnimationExportFormat(Enum):
    """Enumeration of export formats supported by the project."""

    FRAME_BINARY = "fb"
    BVH = "bvh"
    COLLADA = "collada"

    @classmethod
    def fromName(cls, formatName: str) -> "AnimationExportFormat":
        """Return the enum entry matching a user-provided format name."""

        normalised = formatName.strip().lower()
        aliases = {"fbx": "fb", "dae": "collada"}
        normalised = aliases.get(normalised, normalised)
        for entry in cls:
            if entry.value == normalised:
                return entry
        raise AnimationExportError(f"Unsupported export format: '{formatName}'")

    def fileExtension(self) -> str:
        """Return the recommended filename extension for the format."""

        if self is AnimationExportFormat.FRAME_BINARY:
            return "fb"
        if self is AnimationExportFormat.BVH:
            return "bvh"
        return "dae"

    def mediaType(self) -> str:
        """Return a MIME-like identifier for the exported payload."""

        if self is AnimationExportFormat.FRAME_BINARY:
            return "application/octet-stream"
        if self is AnimationExportFormat.BVH:
            return "text/plain"
        return "model/vnd.collada+xml"


class AnimationFormatConverter:
    """Convert ``AnimationClip`` instances to external serialised formats."""

    def __init__(self, clip: AnimationClip) -> None:
        """Initialise the converter with an in-memory animation clip."""

        self.clip = clip
        self._boneParents, self._boneOffsets = resolve_skeleton(tuple(clip.boneNames))
        self._boneIndex = {name: index for index, name in enumerate(clip.boneNames)}
        if clip.boneNames:
            self._boneChildren = self._buildChildrenMapping(clip.boneNames)
            self._bvhTraversal = self._buildTraversalSequence(clip.boneNames)
        else:
            self._boneChildren = {}
            self._bvhTraversal = []

    @classmethod
    def fromJson(cls, sourcePath: Path) -> "AnimationFormatConverter":
        """Instantiate a converter by loading a rotroot JSON payload."""

        positions, rotations, boneNames, frameRate = load_animation_json(sourcePath)
        clip = AnimationClip(
            positions=positions,
            rotations=rotations,
            boneNames=boneNames,
            frameRate=frameRate,
        )
        return cls(clip)

    def render(self, exportFormat: AnimationExportFormat) -> AnimationExportPayload:
        """Render the animation clip into the requested format."""

        handlers: Dict[
            AnimationExportFormat, Callable[[], AnimationExportPayload]
        ] = {
            AnimationExportFormat.FRAME_BINARY: self._renderFrameBinary,
            AnimationExportFormat.BVH: self._renderBvh,
            AnimationExportFormat.COLLADA: self._renderCollada,
        }
        try:
            handler = handlers[exportFormat]
        except KeyError as error:
            raise AnimationExportError(
                f"No renderer registered for format '{exportFormat.name}'"
            ) from error
        return handler()

    def writeToPath(
        self,
        exportFormat: AnimationExportFormat,
        outputPath: Path,
    ) -> Path:
        """Render the clip and persist it to the requested path."""

        payload = self.render(exportFormat)
        resolved = self._resolveOutputPath(outputPath, payload.fileExtension)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "wb") as handle:
            handle.write(payload.content)
        return resolved

    def _resolveOutputPath(self, target: Path, extension: str) -> Path:
        """Ensure the output path carries the expected extension."""

        if target.suffix.lower() == f".{extension.lower()}":
            return target
        return target.with_suffix(f".{extension}")

    def _renderFrameBinary(self) -> AnimationExportPayload:
        """Serialise the clip as a compressed NumPy archive."""

        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            positions=self.clip.positions.astype(np.float32),
            rotations=self.clip.rotations.astype(np.float32),
            bone_names=np.array(self.clip.boneNames, dtype=object),
            fps=float(self.clip.frameRate),
        )
        return AnimationExportPayload(
            content=buffer.getvalue(),
            mediaType=AnimationExportFormat.FRAME_BINARY.mediaType(),
            fileExtension=AnimationExportFormat.FRAME_BINARY.fileExtension(),
        )

    def _renderBvh(self) -> AnimationExportPayload:
        """Serialise the clip into the BVH text format."""

        if not self.clip.boneNames:
            raise AnimationExportError("Cannot export BVH without bone names")
        frameCount = self.clip.positions.shape[0]
        lines = ["HIERARCHY"]
        lines.extend(self._buildBvhHierarchy())
        lines.append("MOTION")
        lines.append(f"Frames: {frameCount}")
        frameTime = 0.0 if self.clip.frameRate == 0 else 1.0 / self.clip.frameRate
        lines.append(f"Frame Time: {frameTime:.6f}")
        for frameIndex in range(frameCount):
            frameValues: list[str] = []
            for jointIndex, hasTranslation in self._bvhTraversal:
                if hasTranslation:
                    rootPosition = self.clip.positions[frameIndex, jointIndex]
                    frameValues.extend(f"{component:.6f}" for component in rootPosition)
                frameValues.extend(
                    self._formatEulerAngles(
                        self.clip.rotations[frameIndex, jointIndex]
                    )
                )
            lines.append(" ".join(frameValues))
        payload = "\n".join(lines)
        return AnimationExportPayload(
            content=payload.encode("utf-8"),
            mediaType=AnimationExportFormat.BVH.mediaType(),
            fileExtension=AnimationExportFormat.BVH.fileExtension(),
        )

    def _formatEulerAngles(self, quaternionValues: Iterable[float]) -> list[str]:
        """Return BVH-compatible Euler angles extracted from a quaternion."""

        quaternion = Quaternion.from_iterable(quaternionValues)
        roll, pitch, yaw = quaternion.to_euler_xyz()
        zRotation = math.degrees(yaw)
        xRotation = math.degrees(roll)
        yRotation = math.degrees(pitch)
        return [
            f"{zRotation:.6f}",
            f"{xRotation:.6f}",
            f"{yRotation:.6f}",
        ]

    def _renderCollada(self) -> AnimationExportPayload:
        """Serialise the clip into a Collada 1.4.1 document."""

        if not self.clip.boneNames:
            raise AnimationExportError("Cannot export Collada without bone names")
        frameCount = self.clip.positions.shape[0]
        times = self._buildTimeStamps(frameCount, self.clip.frameRate)
        colladaRoot = self._createColladaRoot()
        skeletonRoot = self._appendVisualScene(colladaRoot)
        animationsNode = ET.SubElement(colladaRoot, "library_animations")
        for jointIndex in range(len(self.clip.boneNames)):
            self._appendAnimationEntry(animationsNode, jointIndex, times)
        self._linkScene(colladaRoot)
        pretty = minidom.parseString(
            ET.tostring(colladaRoot, encoding="utf-8")
        ).toprettyxml(indent="  ", encoding="utf-8")
        return AnimationExportPayload(
            content=pretty,
            mediaType=AnimationExportFormat.COLLADA.mediaType(),
            fileExtension=AnimationExportFormat.COLLADA.fileExtension(),
        )

    def _createColladaRoot(self) -> ET.Element:
        """Create the Collada document root with metadata."""

        colladaRoot = ET.Element(
            "COLLADA",
            {"xmlns": "http://www.collada.org/2005/11/COLLADASchema", "version": "1.4.1"},
        )
        asset = ET.SubElement(colladaRoot, "asset")
        contributor = ET.SubElement(asset, "contributor")
        tool = ET.SubElement(contributor, "authoring_tool")
        tool.text = "AI-nimator AnimationFormatConverter"
        now = datetime.now(timezone.utc).isoformat()
        created = ET.SubElement(asset, "created")
        created.text = now
        modified = ET.SubElement(asset, "modified")
        modified.text = now
        ET.SubElement(asset, "unit", {"name": "meter", "meter": "1.0"})
        ET.SubElement(asset, "up_axis").text = "Y_UP"
        return colladaRoot

    def _appendVisualScene(self, colladaRoot: ET.Element) -> ET.Element:
        """Create the visual scene and return the root joint node."""

        visualScenes = ET.SubElement(colladaRoot, "library_visual_scenes")
        visualScene = ET.SubElement(
            visualScenes,
            "visual_scene",
            {"id": "Scene", "name": "Scene"},
        )
        rootName = self.clip.boneNames[0]
        rootNode = ET.SubElement(
            visualScene,
            "node",
            {"id": rootName, "name": rootName, "type": "JOINT"},
        )
        rootMatrix = ET.SubElement(rootNode, "matrix", {"sid": "transform"})
        rootMatrix.text = self._formatFloatList(
            self._buildTransformMatrix(
                self.clip.positions[0, 0],
                self.clip.rotations[0, 0],
            ).reshape(-1)
        )
        for childName in self._boneChildren.get(rootName, []):
            self._appendColladaNode(rootNode, childName)
        return rootNode

    def _appendColladaNode(self, parentNode: ET.Element, jointName: str) -> None:
        jointNode = ET.SubElement(
            parentNode,
            "node",
            {"id": jointName, "name": jointName, "type": "JOINT"},
        )
        jointMatrix = ET.SubElement(jointNode, "matrix", {"sid": "transform"})
        jointMatrix.text = self._formatFloatList(
            self._buildTransformMatrix(
                translation=self._boneOffsets.get(jointName, np.zeros(3, dtype=np.float32)),
                quaternion=self.clip.rotations[0, self._boneIndex[jointName]],
            ).reshape(-1)
        )
        for childName in self._boneChildren.get(jointName, []):
            self._appendColladaNode(jointNode, childName)

    def _appendAnimationEntry(
        self,
        animationsNode: ET.Element,
        jointIndex: int,
        times: np.ndarray,
    ) -> None:
        """Append a single animation entry for a joint."""

        jointName = self.clip.boneNames[jointIndex]
        animation = ET.SubElement(animationsNode, "animation", {
            "id": f"{jointName}_animation",
        })
        timeSourceId = f"{jointName}_times"
        transformSourceId = f"{jointName}_transforms"
        self._appendTimeSource(animation, timeSourceId, times)
        self._appendTransformSource(
            animation,
            transformSourceId,
            jointIndex,
            times,
        )
        sampler = ET.SubElement(animation, "sampler", {
            "id": f"{jointName}_sampler",
        })
        ET.SubElement(sampler, "input", {
            "semantic": "INPUT",
            "source": f"#{timeSourceId}",
        })
        ET.SubElement(sampler, "input", {
            "semantic": "OUTPUT",
            "source": f"#{transformSourceId}",
        })
        ET.SubElement(animation, "channel", {
            "source": f"#{jointName}_sampler",
            "target": f"{jointName}/transform",
        })

    def _linkScene(self, colladaRoot: ET.Element) -> None:
        """Add the main scene instantiation node."""

        scene = ET.SubElement(colladaRoot, "scene")
        ET.SubElement(scene, "instance_visual_scene", {"url": "#Scene"})

    def _buildTimeStamps(
        self,
        frameCount: int,
        frameRate: float,
    ) -> np.ndarray:
        """Return an array of timestamps in seconds for each frame."""

        if frameCount == 0:
            return np.zeros(0, dtype=np.float32)
        if frameRate == 0:
            return np.arange(frameCount, dtype=np.float32)
        return np.arange(frameCount, dtype=np.float32) / float(frameRate)

    def _appendTimeSource(
        self,
        animation: ET.Element,
        sourceId: str,
        times: np.ndarray,
    ) -> None:
        """Append a Collada time source element to an animation node."""

        source = ET.SubElement(animation, "source", {"id": sourceId})
        floatArray = ET.SubElement(
            source,
            "float_array",
            {"id": f"{sourceId}_array", "count": str(len(times))},
        )
        floatArray.text = " ".join(f"{timeValue:.6f}" for timeValue in times)
        technique = ET.SubElement(source, "technique_common")
        accessor = ET.SubElement(
            technique,
            "accessor",
            {
                "source": f"#{sourceId}_array",
                "count": str(len(times)),
                "stride": "1",
            },
        )
        ET.SubElement(accessor, "param", {"name": "TIME", "type": "float"})

    def _appendTransformSource(
        self,
        animation: ET.Element,
        sourceId: str,
        jointIndex: int,
        times: np.ndarray,
    ) -> None:
        """Append a transform matrix source for a bone animation."""

        frameCount = len(times)
        source = ET.SubElement(animation, "source", {"id": sourceId})
        floatArray = ET.SubElement(
            source,
            "float_array",
            {
                "id": f"{sourceId}_array",
                "count": str(frameCount * 16),
            },
        )
        matrices: list[str] = []
        jointName = self.clip.boneNames[jointIndex]
        offset = self._boneOffsets.get(jointName, np.zeros(3, dtype=np.float32))
        for frameIndex in range(frameCount):
            translation = (
                self.clip.positions[frameIndex, 0]
                if jointIndex == 0
                else offset
            )
            matrix = self._buildTransformMatrix(
                translation=translation,
                quaternion=self.clip.rotations[frameIndex, jointIndex],
            )
            matrices.append(self._formatFloatList(matrix.reshape(-1)))
        floatArray.text = " ".join(matrices)
        technique = ET.SubElement(source, "technique_common")
        accessor = ET.SubElement(
            technique,
            "accessor",
            {
                "source": f"#{sourceId}_array",
                "count": str(frameCount),
                "stride": "16",
            },
        )
        ET.SubElement(
            accessor,
            "param",
            {"name": "TRANSFORM", "type": "float4x4"},
        )

    def _buildTransformMatrix(
        self,
        translation: np.ndarray,
        quaternion: np.ndarray,
    ) -> np.ndarray:
        """Return a homogeneous transform matrix from translation and rotation."""

        quat = np.asarray(quaternion, dtype=np.float32)
        quat = quat / max(np.linalg.norm(quat), 1e-8)
        xValue, yValue, zValue, wValue = quat
        xx = xValue * xValue
        yy = yValue * yValue
        zz = zValue * zValue
        xy = xValue * yValue
        xz = xValue * zValue
        yz = yValue * zValue
        wx = wValue * xValue
        wy = wValue * yValue
        wz = wValue * zValue
        rotation = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        translationVector = np.asarray(translation, dtype=np.float32)
        matrix[:3, 3] = translationVector
        return matrix

    def _formatFloatList(self, values: Iterable[float]) -> str:
        """Return floats joined with spaces using six decimal places."""

        return " ".join(f"{float(value):.6f}" for value in values)

    def _buildChildrenMapping(self, boneNames: List[str]) -> Dict[str, List[str]]:
        """Create a mapping of parent to ordered children for the skeleton."""

        children: Dict[str, List[str]] = {name: [] for name in boneNames}
        for name in boneNames:
            parent = self._boneParents.get(name)
            if parent is None or parent not in children:
                continue
            children[parent].append(name)
        # Ensure ordering matches the clip order for deterministic traversal.
        for parent, entries in children.items():
            entries.sort(key=lambda value: self._boneIndex[value])
        return children

    def _buildTraversalSequence(self, boneNames: List[str]) -> List[tuple[int, bool]]:
        """Return BVH traversal order with translation channel flags."""

        order: List[tuple[int, bool]] = []

        def visit(name: str) -> None:
            index = self._boneIndex[name]
            order.append((index, index == 0))
            for child in self._boneChildren.get(name, []):
                visit(child)

        visit(boneNames[0])
        return order

    def _buildBvhHierarchy(self) -> List[str]:
        """Construct the BVH hierarchy description lines."""

        lines: List[str] = []

        def format_joint(name: str, level: int) -> None:
            indent = "  " * level
            offset = self._boneOffsets.get(name, np.zeros(3, dtype=np.float32))
            if level == 0:
                lines.append(f"ROOT {name}")
            else:
                lines.append(f"{indent}JOINT {name}")
            lines.append(f"{indent}" + "{")
            lines.append(
                f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}"
            )
            if level == 0:
                lines.append(
                    f"{indent}  CHANNELS 6 Xposition Yposition Zposition "
                    "Zrotation Xrotation Yrotation"
                )
            else:
                lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")
            children = self._boneChildren.get(name, [])
            if not children:
                end_offset = self._default_end_site(offset)
                lines.append(f"{indent}  End Site")
                lines.append(f"{indent}  " + "{")
                lines.append(
                    f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}"
                )
                lines.append(f"{indent}  " + "}")
            else:
                for child in children:
                    format_joint(child, level + 1)
            lines.append(f"{indent}" + "}")

        format_joint(self.clip.boneNames[0], 0)
        return lines

    def _default_end_site(self, offset: np.ndarray) -> np.ndarray:
        """Return a small vector to place BVH end sites sensibly."""

        if np.linalg.norm(offset) < 1e-5:
            return np.array([0.0, 0.0, 0.1], dtype=np.float32)
        direction = offset / max(np.linalg.norm(offset), 1e-5)
        return direction * 0.2


__all__ = [
    "AnimationExportError",
    "AnimationExportFormat",
    "AnimationFormatConverter",
]
