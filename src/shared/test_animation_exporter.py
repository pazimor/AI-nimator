"""Tests for the animation format exporter."""

from __future__ import annotations

import io
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from src.shared.animation_exporter import (
    AnimationExportError,
    AnimationExportFormat,
    AnimationFormatConverter,
)
from src.shared.constants.motion import SMPL22_BONES
from src.shared.types.data import AnimationClip


COLLADA_NAMESPACE = "{http://www.collada.org/2005/11/COLLADASchema}"


def _build_sample_clip() -> AnimationClip:
    positions = np.zeros((2, 2, 3), dtype=np.float32)
    positions[1, 0, 0] = 1.0
    rotations = np.zeros((2, 2, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    boneNames = ["root", "spine"]
    return AnimationClip(
        positions=positions,
        rotations=rotations,
        boneNames=boneNames,
        frameRate=30.0,
    )


def test_frame_binary_roundtrip():
    clip = _build_sample_clip()
    converter = AnimationFormatConverter(clip)
    payload = converter.render(AnimationExportFormat.FRAME_BINARY)
    assert payload.fileExtension == "fb"
    archive = np.load(io.BytesIO(payload.content))
    assert archive["positions"].shape == (2, 2, 3)
    assert archive["rotations"].shape == (2, 2, 4)
    assert archive["fps"] == pytest.approx(30.0)


def _build_smpl22_clip() -> AnimationClip:
    joint_count = len(SMPL22_BONES)
    positions = np.zeros((1, joint_count, 3), dtype=np.float32)
    rotations = np.zeros((1, joint_count, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    return AnimationClip(
        positions=positions,
        rotations=rotations,
        boneNames=list(SMPL22_BONES),
        frameRate=30.0,
    )


def test_bvh_serialisation_contains_expected_lines():
    clip = _build_sample_clip()
    converter = AnimationFormatConverter(clip)
    payload = converter.render(AnimationExportFormat.BVH)
    text = payload.content.decode("utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert lines[0] == "HIERARCHY"
    assert lines[1] == "ROOT root"
    assert any(line.startswith("Frames: 2") for line in lines)
    frameLines = [line for line in lines if line and line[0].isdigit()]
    assert frameLines[0].startswith("0.000000 0.000000 0.000000")
    assert frameLines[1].startswith("1.000000 0.000000 0.000000")


def test_bvh_skeleton_uses_smpl_offsets():
    clip = _build_smpl22_clip()
    converter = AnimationFormatConverter(clip)
    payload = converter.render(AnimationExportFormat.BVH)
    text = payload.content.decode("utf-8")
    assert "JOINT thigh_l" in text
    assert "OFFSET 0.090000 -0.080000 0.010000" in text


def test_collada_serialisation_produces_valid_xml():
    clip = _build_sample_clip()
    converter = AnimationFormatConverter(clip)
    payload = converter.render(AnimationExportFormat.COLLADA)
    root = ET.fromstring(payload.content)
    assert root.tag.endswith("COLLADA")
    visualScenes = root.find(f"{COLLADA_NAMESPACE}library_visual_scenes")
    assert visualScenes is not None
    nodes = visualScenes.findall(f".//{COLLADA_NAMESPACE}node")
    assert len(nodes) == 2
    animations = root.find(f"{COLLADA_NAMESPACE}library_animations")
    assert animations is not None
    channels = animations.findall(f".//{COLLADA_NAMESPACE}channel")
    assert len(channels) == 2


def test_collada_includes_child_offsets():
    clip = _build_smpl22_clip()
    converter = AnimationFormatConverter(clip)
    payload = converter.render(AnimationExportFormat.COLLADA)
    root = ET.fromstring(payload.content)
    thighNode = root.find(
        f".//{COLLADA_NAMESPACE}node[@id='thigh_l']/{COLLADA_NAMESPACE}matrix"
    )
    assert thighNode is not None
    values = [float(item) for item in thighNode.text.split()]
    matrix = np.array(values, dtype=np.float32).reshape(4, 4)
    translation = matrix[:3, 3]
    expected = np.array([0.09, -0.08, 0.01], dtype=np.float32)
    np.testing.assert_allclose(translation, expected, atol=1e-4)


def test_missing_bones_raise_error():
    clip = AnimationClip(
        positions=np.zeros((1, 0, 3), dtype=np.float32),
        rotations=np.zeros((1, 0, 4), dtype=np.float32),
        boneNames=[],
        frameRate=24.0,
    )
    converter = AnimationFormatConverter(clip)
    with pytest.raises(AnimationExportError):
        converter.render(AnimationExportFormat.BVH)
    with pytest.raises(AnimationExportError):
        converter.render(AnimationExportFormat.COLLADA)


def test_format_aliases_are_supported():
    assert AnimationExportFormat.fromName("fbx") is AnimationExportFormat.FRAME_BINARY
    assert AnimationExportFormat.fromName("dae") is AnimationExportFormat.COLLADA
    with pytest.raises(AnimationExportError):
        AnimationExportFormat.fromName("unknown")


def test_write_to_path_applies_extension(tmp_path: Path):
    clip = _build_sample_clip()
    converter = AnimationFormatConverter(clip)
    destination = tmp_path / "clip"
    writtenPath = converter.writeToPath(
        AnimationExportFormat.BVH,
        destination,
    )
    assert writtenPath.suffix == ".bvh"
    assert writtenPath.exists()
