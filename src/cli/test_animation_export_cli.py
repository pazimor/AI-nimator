"""Tests for the animation export command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.cli import animation_export_cli


@pytest.fixture()
def sample_json_path(tmp_path: Path) -> Path:
    """Return the JSON clip created next to its NPZ counterpart."""

    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    positions = np.zeros((3, 2, 3), dtype=np.float32)
    positions[1, 0, :] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    positions[2, 0, :] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rotations = np.zeros((3, 2, 4), dtype=np.float32)
    rotations[..., -1] = 1.0
    rotations[1, 1, :] = np.array([0.0, 0.38268343, 0.0, 0.9238795], dtype=np.float32)
    rotations[2, 1, :] = np.array([0.0, 0.70710678, 0.0, 0.70710678], dtype=np.float32)
    bone_names = np.array(["root", "spine"], dtype=object)
    frame_rate = 30.0
    np.savez_compressed(
        asset_dir / "sample_clip.npz",
        positions=positions,
        rotations=rotations,
        bone_names=bone_names,
        fps=frame_rate,
    )
    frames = []
    for frame_index in range(positions.shape[0]):
        frame_payload = {
            "root_pos": positions[frame_index, 0, :].round(6).tolist(),
            "root": rotations[frame_index, 0, :].round(6).tolist(),
            "spine": rotations[frame_index, 1, :].round(6).tolist(),
        }
        frames.append(frame_payload)
    payload = {
        "meta": {
            "format": "rotroot",
            "fps": frame_rate,
            "source": "sample_clip.npz",
        },
        "bones": bone_names.tolist(),
        "frames": frames,
    }
    json_path = asset_dir / "sample_clip.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def test_animation_export_cli_generates_all_formats(sample_json_path: Path) -> None:
    """Ensure the CLI exports each supported format next to the JSON clip."""

    output_requests = {
        "fbx": sample_json_path.with_name("sample_clip.fb"),
        "bvh": sample_json_path.with_name("sample_clip.bvh"),
        "collada": sample_json_path.with_name("sample_clip.dae"),
    }
    for format_name, output_path in output_requests.items():
        arguments = [
            str(sample_json_path),
            str(output_path),
            "--format",
            format_name,
        ]
        exit_code = animation_export_cli.main(arguments)
        assert exit_code == 0
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert output_path.parent == sample_json_path.parent
