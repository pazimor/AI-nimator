"""End-to-end tests for the dataset builder pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from src.features.dataset_builder.config_loader import loadBuilderConfig
from src.features.dataset_builder.dataset_builder import (
    AnimationRebuilder,
    DatasetBuilder,
)
from src.shared.types import (
    DatasetBuildOptions,
    DatasetBuilderConfig,
    DatasetBuilderPaths,
    DatasetBuilderProcessing,
)


def test_datasetBuilder_rebuilds_animation_and_prompts(tmp_path: Path) -> None:
    """
    Ensure the builder rebuilds animation/prompt files from the sample assets.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.
    """

    dataDir = tmp_path / "data"
    dataDir.mkdir()
    sampleRoot = Path("test")
    poseDataRoot = dataDir / "ACCAD" / "Female1General_c3d"
    poseDataRoot.mkdir(parents=True)
    shutil.copy(
        sampleRoot / "A1 - Stand_poses.npz",
        poseDataRoot / "A1 - Stand_poses.npz",
    )
    shutil.copy(sampleRoot / "004501.txt", dataDir / "004501.txt")
    indexPath = tmp_path / "index.csv"
    indexPath.write_text(
        "./pose_data/ACCAD/Female1General_c3d/"
        "A1 - Stand_poses.npy,0,60,004501.npy\n",
        encoding="utf-8",
    )
    outputDir = tmp_path / "output"
    configPath = tmp_path / "dataset.yaml"
    customPromptPath = dataDir / "custom_prompts.jsonl"
    convertedRoot = tmp_path / "converted"
    convertedSampleDir = convertedRoot / "sample_001"
    convertedSampleDir.mkdir(parents=True, exist_ok=True)
    convertedAnimation = {
        "meta": {
            "fps": 60,
            "frames": 360,
            "joints": 22,
            "source": "ACCAD/Female1General_c3d/A1 - Stand_poses.npz",
        },
        "bones": [],
        "frames": [],
    }
    (convertedSampleDir / "animation.json").write_text(
        json.dumps(convertedAnimation),
        encoding="utf-8",
    )
    convertedPrompts = {
        "Simple": "Converted simple description",
        "advanced": "Converted advanced description",
        "tag": "Converted tag",
        "ignored": "should not be read",
    }
    (convertedSampleDir / "prompts.json").write_text(
        json.dumps(convertedPrompts),
        encoding="utf-8",
    )
    customRecord = {
        "custom_id": "ACCAD/Female1General_c3d/A1 - Stand_poses",
        "response": {
            "body": {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "Simple": "Custom simple",
                                        "advanced": "Custom advanced",
                                        "tag": "Dance",
                                    }
                                ),
                            }
                        ],
                    }
                ]
            }
        },
    }
    customPromptPath.write_text(
        json.dumps(customRecord) + "\n",
        encoding="utf-8",
    )
    configPath.write_text(
        "paths:\n"
        f"  animation_root: {dataDir.as_posix()}\n"
        f"  prompt_root: {dataDir.as_posix()}\n"
        f"  index_csv: {indexPath.as_posix()}\n"
        f"  converted_root: {convertedRoot.as_posix()}\n"
        f"  custom_prompt_files:\n"
        f"    - {customPromptPath.as_posix()}\n"
        f"  output_root: {outputDir.as_posix()}\n"
        "processing:\n"
        "  animation_extension: .npz\n"
        "  prompt_text_extension: .txt\n"
        "  fallback_fps: 60\n",
        encoding="utf-8",
    )
    config = loadBuilderConfig(configPath)
    options = DatasetBuildOptions(debugMode=True, progressStyle="none")
    builder = DatasetBuilder(config=config, options=options)
    report = builder.buildDataset()
    assert report.processedSamples == 1
    sampleDir = outputDir / "ACCAD" / "Female1General_c3d" / "A1 - Stand_poses"
    animationPayload = json.loads(
        (sampleDir / "animation.json").read_text(encoding="utf-8")
    )
    promptPayload = json.loads(
        (sampleDir / "prompt.json").read_text(encoding="utf-8")
    )
    assert animationPayload["meta"]["frames"] == 360
    assert len(animationPayload["bones"]) == 22
    assert promptPayload["segments"][0]["startFrame"] == 0
    assert promptPayload["segments"][0]["endFrame"] == 360
    assert promptPayload["segments"][0]["text"] == "Converted simple description"
    assert promptPayload["segments"][1]["text"] == "Converted advanced description"
    assert promptPayload["segments"][2]["text"].startswith("a person is facing")
    assert promptPayload["tag"] == "Converted tag"
    assert promptPayload["customPrompts"]["simple"] == "Custom simple"


def test_animationRebuilder_serializes_bytes(tmp_path: Path) -> None:
    """Ensure metadata bytes payloads are converted to JSON-safe strings."""

    dummyPaths = DatasetBuilderPaths(
        animationRoot=tmp_path,
        animationRoots=[tmp_path],
        promptRoot=tmp_path,
        promptSources=[tmp_path],
        customPromptFiles=[],
        indexCsv=tmp_path / "index.csv",
        outputRoot=tmp_path / "out",
    )
    config = DatasetBuilderConfig(
        paths=dummyPaths,
        processing=DatasetBuilderProcessing(),
    )
    rebuilder = AnimationRebuilder(config=config)
    raw = {"binary": b"\xff\xfehello", "nested": [np.array([b"a", b"b"])]}
    serialized = rebuilder._ensureJsonSerializable(raw)  # type: ignore[attr-defined]
    assert isinstance(serialized["binary"], str)
    assert serialized["nested"][0][0] == "a"
