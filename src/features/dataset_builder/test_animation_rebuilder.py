"""Tests for AnimationRebuilder export pipeline."""

from pathlib import Path
import re

import numpy as np
import pytest

from src.features.dataset_builder.animation_rebuilder import AnimationRebuilder
from src.shared.constants.skeletons import SMPL24_BONE_ORDER
from src.shared.quaternion import Rotation
from src.shared.types import (
    AnimationSample,
    DatasetBuilderConfig,
    DatasetBuilderPaths,
)


def _makeConfig(tmpPath: Path) -> DatasetBuilderConfig:
    """Return a minimal builder config rooted in tmpPath."""
    paths = DatasetBuilderPaths(
        animationRoot=tmpPath,
        promptRoot=tmpPath,
        promptSources=[],
        indexCsv=tmpPath / "index.csv",
        outputRoot=tmpPath,
    )
    return DatasetBuilderConfig(paths=paths)


def test_axis_angle_preserved_in_rot6d(tmp_path: Path) -> None:
    """axis-angle rotations must survive the 6D conversion in _buildBones."""
    config = _makeConfig(tmp_path)
    builder = AnimationRebuilder(config)

    frames = 1
    bones = len(SMPL24_BONE_ORDER)
    axisAngles = np.zeros((frames, bones * 3), dtype=np.float32)
    shoulderIndex = SMPL24_BONE_ORDER.index("rightShoulder")
    axisAngles[:, shoulderIndex * 3 : shoulderIndex * 3 + 3] = np.array(
        [[np.pi / 2, 0.0, 0.0]],
        dtype=np.float32,
    )

    bonesPayload = builder._buildBones(axisAngles)
    shoulder = next(b for b in bonesPayload if b["name"] == "rightShoulder")
    rot6d = np.array(shoulder["frames"][0]["rotation"], dtype=np.float32)
    recovered = Rotation(rot6d, kind="rot6d").axis_angle.numpy()[0]

    assert np.allclose(recovered, [np.pi / 2, 0.0, 0.0], atol=1e-3)


def test_root_translation_zeroing(tmp_path: Path) -> None:
    """Root translation must be zeroed when requested."""
    config = _makeConfig(tmp_path)
    builder = AnimationRebuilder(config)

    frames = 2
    bones = len(SMPL24_BONE_ORDER)
    axisAngles = np.zeros((frames, bones * 3), dtype=np.float32)
    trans = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    sample = AnimationSample(
        relativePath=Path("dummy.npz"),
        resolvedPath=tmp_path / "dummy.npz",
        axisAngles=axisAngles,
        fps=30,
        extras={"trans": trans},
    )

    output = tmp_path / "out.dae"
    builder.exportCollada(sample, output, zeroRootTranslation=True)

    content = output.read_text()
    match = re.search(
        r'id="pelvis_location_output_array"[^>]*>([^<]+)<',
        content,
        flags=re.MULTILINE,
    )
    assert match, "pelvis translation channel missing"
    values = [float(v) for v in match.group(1).split()]
    assert all(abs(v) < 1e-4 for v in values)
