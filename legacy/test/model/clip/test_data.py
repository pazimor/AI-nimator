"""Tests for CLIP dataset utilities."""

from __future__ import annotations

import json

import torch

from ainimator.training.clip_data import MotionTextClipDataset


IDENTITY_ROTATION_6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class _DummyTokenizer:
    def __call__(
        self,
        text: str,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del text, padding, truncation, return_tensors
        return {
            "input_ids": torch.arange(max_length, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }


def test_motion_text_clip_dataset_returns_optional_motion_features(
    tmp_path,
) -> None:
    promptsRoot = tmp_path / "prompts"
    animationsRoot = tmp_path / "animations"
    promptDir = promptsRoot / "set_a"
    animationDir = animationsRoot / "set_a"
    promptDir.mkdir(parents=True)
    animationDir.mkdir(parents=True)

    (promptDir / "prompt.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "startFrame": 0,
                        "endFrame": 2,
                        "text": "walk forward",
                        "sourceFile": "set_a/example",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    bones = []
    for boneIndex in range(22):
        bones.append(
            {
                "name": f"bone_{boneIndex}",
                "frames": [
                    {"frameIndex": 0, "rotation": IDENTITY_ROTATION_6D},
                    {"frameIndex": 1, "rotation": IDENTITY_ROTATION_6D},
                ],
            }
        )

    (animationDir / "animation.json").write_text(
        json.dumps(
            {
                "meta": {"frames": 2},
                "bones": bones,
                "extras": {
                    "trans": [
                        [0.0, 1.0, 0.0],
                        [0.1, 1.1, 0.0],
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = MotionTextClipDataset(
        rootPrompts=promptsRoot,
        rootAnimations=animationsRoot,
        tokenizer=_DummyTokenizer(),
        maxLength=8,
    )

    sample = dataset[0]

    assert tuple(sample["motion"].shape) == (2, 22, 6)
    assert tuple(sample["joint_xyz"].shape) == (2, 22, 3)
    assert tuple(sample["root_translation"].shape) == (2, 3)
    assert tuple(sample["root_velocity"].shape) == (2, 3)
