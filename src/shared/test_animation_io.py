"""Tests for shared animation I/O helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.shared.animation_io import (
    AnimationIOError,
    convertRotationsToTensor,
    extractBoneNames,
    loadAnimationFile,
    loadPromptsFile,
    load_animation_json,
    load_animation_npz,
    rotationsTo6d,
)


def test_load_animation_npz_native(tmp_path):
    positions = np.zeros((2, 1, 3), dtype=np.float32)
    rotations = np.zeros((2, 1, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    archive_path = tmp_path / "native.npz"
    np.savez(archive_path, positions=positions, rotations=rotations, bone_names=np.array(["root"], dtype=object), fps=60)

    pos, rot, bones, fps = load_animation_npz(archive_path)

    assert pos.shape == (2, 1, 3)
    assert rot.shape == (2, 1, 4)
    assert bones == ["root"]
    assert pytest.approx(fps) == 60.0


def test_load_animation_json_rotroot(tmp_path):
    frames = [
        {"root_pos": [0.0, 0.0, 0.0], "root": [0.0, 0.0, 0.0, 1.0]},
        {"root_pos": [1.0, 0.0, 0.0], "root": [0.0, 0.0, 0.0, 1.0]},
    ]
    payload = {"bones": ["root"], "frames": frames, "meta": {"fps": 30}}
    json_path = tmp_path / "clip.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    pos, rot, bones, fps = load_animation_json(json_path)

    assert pos[1, 0, 0] == pytest.approx(1.0)
    assert rot[0, 0, 3] == pytest.approx(1.0)
    assert bones == ["root"]
    assert pytest.approx(fps) == 30.0


def test_load_animation_and_prompts(tmp_path):
    rotation_payload = {
        "rotations": {
            "hip": ["1|0|0|0", "0.9239|0|0.3827|0"],
        }
    }
    prompt_payload = {
        "entry": {
            "Simple": "walk",
            "advanced": "slow pace",
            "expert": "",
            "tag": "test",
        }
    }
    rotation_file = tmp_path / "animation.json"
    prompt_file = tmp_path / "prompt.json"
    rotation_file.write_text(json.dumps(rotation_payload), encoding="utf-8")
    prompt_file.write_text(json.dumps(prompt_payload), encoding="utf-8")

    rotations = loadAnimationFile(rotation_file)
    prompts = loadPromptsFile(prompt_file)

    assert list(rotations.keys()) == ["hip"]
    assert prompts[0]["Simple"] == "walk"


def test_load_animation_file_uniformizer_payload(tmp_path):
    frames = [
        {
            "root_pos": [0.0, 0.0, 0.0],
            "pelvis": [0.0, 0.0, 0.0, 1.0],
            "spine": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "root_pos": [0.1, 0.0, 0.0],
            "pelvis": [0.0, 0.0, 0.1, 0.995],
            "spine": [0.0, 0.05, 0.0, 0.9987],
        },
    ]
    payload = {"bones": ["pelvis", "spine"], "frames": frames, "meta": {"fps": 30.0}}
    rotation_file = tmp_path / "animation.json"
    rotation_file.write_text(json.dumps(payload), encoding="utf-8")

    rotations = loadAnimationFile(rotation_file)
    assert list(rotations.keys()) == ["pelvis", "spine"]

    pelvis_components = [float(value) for value in rotations["pelvis"][1].split("|")]
    assert pytest.approx(pelvis_components[2], rel=1e-3) == 0.1

    tensor, bones = convertRotationsToTensor(rotations)
    assert tensor.shape == (2, 2, 4)
    assert bones == ["pelvis", "spine"]

def test_convert_rotations_to_tensor_and_to_6d():
    data = {
        "hip": ["1|0|0|0", "0.9239|0|0.3827|0"],
    }
    tensor, bones = convertRotationsToTensor(data)
    assert tensor.shape == (2, 1, 4)
    assert bones == ["hip"]
    rot6d = rotationsTo6d({"hip": data["hip"]})
    assert rot6d.shape == (2, 1, 6)
    assert torch.isfinite(rot6d).all()


def test_load_prompts_list(tmp_path):
    payload = [
        {
            "Simple": "jump",
            "advanced": "",
            "expert": "",
            "tag": "",
        }
    ]
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(payload), encoding="utf-8")
    loaded = loadPromptsFile(prompt_file)
    assert loaded[0]["Simple"] == "jump"


def test_load_animation_npz_missing(tmp_path):
    missing = tmp_path / "missing.npz"
    with pytest.raises(AnimationIOError):
        load_animation_npz(missing)


def test_extract_bone_names_and_invalid_prompts(tmp_path):
    data = {"hip": ["1|0|0|0"]}
    assert extractBoneNames(data) == ["hip"]

    prompt_file = tmp_path / "prompt_invalid.json"
    prompt_file.write_text(json.dumps({"unexpected": 42}), encoding="utf-8")
    loaded = loadPromptsFile(prompt_file)
    assert loaded[0]["unexpected"] == 42
