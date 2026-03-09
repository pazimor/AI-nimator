"""Tests for the V2 preprocessed dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.shared.constants.preprocessed import (
    PREPROCESSED_GENERATION_TEXT_CACHE_DIRNAME,
    PREPROCESSED_GENERATION_TEXT_CACHE_MANIFEST_FILENAME,
    PREPROCESSED_GENERATION_TEXT_CACHE_VERSION,
    PREPROCESSED_GENERATION_TEXT_EMBED_SHARDS_DIRNAME,
)
from src.shared.preprocessed_dataset import (
    PreprocessedLinkDataset,
    computeClipCheckpointFingerprint,
)


def test_preprocessed_link_dataset_loads_sample_and_text(tmp_path) -> None:
    datasetRoot = _writeDataset(tmp_path)

    dataset = PreprocessedLinkDataset(datasetRoot)
    item = dataset[0]

    assert len(dataset) == 1
    assert int(item["sample_id"]) == 0
    assert int(item["text_id"]) == 0
    assert tuple(item["motion"].shape) == (2, 22, 6)
    assert tuple(item["pooled_text"].shape) == (3,)


def test_preprocessed_link_dataset_loads_generation_cache(tmp_path) -> None:
    datasetRoot = _writeDataset(tmp_path)
    checkpointPath = datasetRoot / "clip_checkpoint.pt"
    checkpointPath.write_bytes(b"checkpoint")
    clipFingerprint = computeClipCheckpointFingerprint(checkpointPath)
    cacheRoot = (
        datasetRoot
        / PREPROCESSED_GENERATION_TEXT_CACHE_DIRNAME
        / clipFingerprint
    )
    shardsDir = cacheRoot / PREPROCESSED_GENERATION_TEXT_EMBED_SHARDS_DIRNAME
    shardsDir.mkdir(parents=True)
    torch.save(
        [
            {
                "text_id": 0,
                "text_embedding": torch.tensor([0.4, 0.5], dtype=torch.float32),
            }
        ],
        shardsDir / "text_embed_shard_00000.pt",
    )
    cacheManifest = {
        "version": PREPROCESSED_GENERATION_TEXT_CACHE_VERSION,
        "clipFingerprint": clipFingerprint,
        "embedDim": 2,
        "totalTexts": 1,
        "shardSize": 1,
        "shards": [
            {
                "path": (
                    PREPROCESSED_GENERATION_TEXT_EMBED_SHARDS_DIRNAME
                    + "/text_embed_shard_00000.pt"
                ),
                "sampleCount": 1,
            }
        ],
    }
    (cacheRoot / PREPROCESSED_GENERATION_TEXT_CACHE_MANIFEST_FILENAME).write_text(
        json.dumps(cacheManifest),
        encoding="utf-8",
    )

    dataset = PreprocessedLinkDataset(
        datasetRoot,
        generationCacheCheckpoint=checkpointPath,
    )
    item = dataset[0]

    assert tuple(item["generation_text_embedding"].shape) == (2,)
    dataset.validateCompatibility(
        modelName="xlm-roberta-base",
        maxPromptLength=8,
        expectedPooledTextDim=3,
        expectedGenerationEmbedDim=2,
        requireGenerationTextEmbedding=True,
    )


def test_preprocessed_link_dataset_rejects_legacy_manifest(tmp_path) -> None:
    datasetRoot = tmp_path / "dataset"
    datasetRoot.mkdir()
    (datasetRoot / "manifest.json").write_text(
        json.dumps({"version": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="V2"):
        PreprocessedLinkDataset(datasetRoot)


def test_preprocessed_link_dataset_preload_indices_uses_shard_locality(
    tmp_path,
) -> None:
    datasetRoot = _writeMultiShardDataset(tmp_path)
    dataset = PreprocessedLinkDataset(datasetRoot)

    items = dataset.preloadIndices([0, 1, 2, 3])
    orderedIds = [int(item["sample_id"]) for item in items]

    assert orderedIds == [0, 2, 1, 3]


def _writeDataset(tmp_path) -> Path:
    datasetRoot = tmp_path / "dataset"
    sampleShardsDir = datasetRoot / "sample_shards"
    textShardsDir = datasetRoot / "text_shards"
    sampleShardsDir.mkdir(parents=True)
    textShardsDir.mkdir()

    torch.save(
        [
            {
                "motion": torch.zeros(2, 22, 6, dtype=torch.float32),
                "time": torch.tensor([0, 2], dtype=torch.long),
                "meta": {"source": "demo"},
                "root_translation": torch.zeros(2, 3, dtype=torch.float32),
            }
        ],
        sampleShardsDir / "sample_shard_00000.pt",
    )
    torch.save(
        [
            {
                "text_id": 0,
                "raw_text": "walk",
                "input_ids": torch.tensor([1, 2, 0, 0], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 0, 0], dtype=torch.long),
                "pooled_text": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            }
        ],
        textShardsDir / "text_shard_00000.pt",
    )
    (datasetRoot / "sample_index.json").write_text(
        json.dumps(
            [
                {
                    "sampleId": 0,
                    "shardIndex": 0,
                    "shardOffset": 0,
                    "frames": 2,
                    "sampleBytes": 42,
                    "datasetFolder": "KIT",
                    "sourceFile": "demo",
                    "startFrame": 0,
                    "endFrame": 2,
                }
            ]
        ),
        encoding="utf-8",
    )
    (datasetRoot / "text_index.json").write_text(
        json.dumps(
            [
                {
                    "textId": 0,
                    "shardIndex": 0,
                    "shardOffset": 0,
                    "usageCount": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    (datasetRoot / "link_index.json").write_text(
        json.dumps(
            [
                {
                    "linkId": 0,
                    "sampleId": 0,
                    "textId": 0,
                    "datasetFolder": "KIT",
                    "sourceFile": "demo",
                    "frames": 2,
                    "pairBytes": 55,
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "version": 2,
        "modelName": "xlm-roberta-base",
        "maxPromptLength": 8,
        "splitFrames": 2,
        "downsampleTargetFrames": None,
        "maxSegmentFrames": 2,
        "sampleShardSize": 1,
        "textShardSize": 1,
        "totalSamples": 1,
        "totalTexts": 1,
        "totalLinks": 1,
        "averageSampleBytes": 42.0,
        "maxSampleBytes": 42,
        "averagePairBytes": 55.0,
        "averageFrames": 2.0,
        "maxFrames": 2,
        "sampleShards": [{"path": "sample_shards/sample_shard_00000.pt", "sampleCount": 1}],
        "textShards": [{"path": "text_shards/text_shard_00000.pt", "sampleCount": 1}],
        "sampleIndexPath": "sample_index.json",
        "textIndexPath": "text_index.json",
        "linkIndexPath": "link_index.json",
        "enabledComponents": ["rotation6d", "root_translation"],
        "pooledTextDim": 3,
    }
    (datasetRoot / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return datasetRoot


def _writeMultiShardDataset(tmp_path) -> Path:
    datasetRoot = tmp_path / "multi_dataset"
    sampleShardsDir = datasetRoot / "sample_shards"
    textShardsDir = datasetRoot / "text_shards"
    sampleShardsDir.mkdir(parents=True)
    textShardsDir.mkdir()

    torch.save(
        [
            {
                "motion": torch.full((2, 22, 6), 0.0, dtype=torch.float32),
                "time": torch.tensor([0, 2], dtype=torch.long),
                "meta": {"source": "s0"},
            },
            {
                "motion": torch.full((2, 22, 6), 2.0, dtype=torch.float32),
                "time": torch.tensor([0, 2], dtype=torch.long),
                "meta": {"source": "s2"},
            },
        ],
        sampleShardsDir / "sample_shard_00000.pt",
    )
    torch.save(
        [
            {
                "motion": torch.full((2, 22, 6), 1.0, dtype=torch.float32),
                "time": torch.tensor([0, 2], dtype=torch.long),
                "meta": {"source": "s1"},
            },
            {
                "motion": torch.full((2, 22, 6), 3.0, dtype=torch.float32),
                "time": torch.tensor([0, 2], dtype=torch.long),
                "meta": {"source": "s3"},
            },
        ],
        sampleShardsDir / "sample_shard_00001.pt",
    )
    torch.save(
        [
            {
                "text_id": 0,
                "raw_text": "t0",
                "input_ids": torch.tensor([1, 0], dtype=torch.long),
                "attention_mask": torch.tensor([1, 0], dtype=torch.long),
                "pooled_text": torch.tensor([0.0, 0.1], dtype=torch.float32),
            },
            {
                "text_id": 2,
                "raw_text": "t2",
                "input_ids": torch.tensor([2, 0], dtype=torch.long),
                "attention_mask": torch.tensor([1, 0], dtype=torch.long),
                "pooled_text": torch.tensor([0.2, 0.3], dtype=torch.float32),
            },
        ],
        textShardsDir / "text_shard_00000.pt",
    )
    torch.save(
        [
            {
                "text_id": 1,
                "raw_text": "t1",
                "input_ids": torch.tensor([3, 0], dtype=torch.long),
                "attention_mask": torch.tensor([1, 0], dtype=torch.long),
                "pooled_text": torch.tensor([0.4, 0.5], dtype=torch.float32),
            },
            {
                "text_id": 3,
                "raw_text": "t3",
                "input_ids": torch.tensor([4, 0], dtype=torch.long),
                "attention_mask": torch.tensor([1, 0], dtype=torch.long),
                "pooled_text": torch.tensor([0.6, 0.7], dtype=torch.float32),
            },
        ],
        textShardsDir / "text_shard_00001.pt",
    )
    (datasetRoot / "sample_index.json").write_text(
        json.dumps(
            [
                {
                    "sampleId": 0,
                    "shardIndex": 0,
                    "shardOffset": 0,
                    "frames": 2,
                    "sampleBytes": 10,
                    "datasetFolder": "KIT",
                    "sourceFile": "s0",
                    "startFrame": 0,
                    "endFrame": 2,
                },
                {
                    "sampleId": 1,
                    "shardIndex": 1,
                    "shardOffset": 0,
                    "frames": 2,
                    "sampleBytes": 10,
                    "datasetFolder": "KIT",
                    "sourceFile": "s1",
                    "startFrame": 0,
                    "endFrame": 2,
                },
                {
                    "sampleId": 2,
                    "shardIndex": 0,
                    "shardOffset": 1,
                    "frames": 2,
                    "sampleBytes": 10,
                    "datasetFolder": "KIT",
                    "sourceFile": "s2",
                    "startFrame": 0,
                    "endFrame": 2,
                },
                {
                    "sampleId": 3,
                    "shardIndex": 1,
                    "shardOffset": 1,
                    "frames": 2,
                    "sampleBytes": 10,
                    "datasetFolder": "KIT",
                    "sourceFile": "s3",
                    "startFrame": 0,
                    "endFrame": 2,
                },
            ]
        ),
        encoding="utf-8",
    )
    (datasetRoot / "text_index.json").write_text(
        json.dumps(
            [
                {"textId": 0, "shardIndex": 0, "shardOffset": 0, "usageCount": 1},
                {"textId": 1, "shardIndex": 1, "shardOffset": 0, "usageCount": 1},
                {"textId": 2, "shardIndex": 0, "shardOffset": 1, "usageCount": 1},
                {"textId": 3, "shardIndex": 1, "shardOffset": 1, "usageCount": 1},
            ]
        ),
        encoding="utf-8",
    )
    (datasetRoot / "link_index.json").write_text(
        json.dumps(
            [
                {
                    "linkId": 0,
                    "sampleId": 0,
                    "textId": 0,
                    "datasetFolder": "KIT",
                    "sourceFile": "demo0",
                    "frames": 2,
                    "pairBytes": 20,
                },
                {
                    "linkId": 1,
                    "sampleId": 1,
                    "textId": 1,
                    "datasetFolder": "KIT",
                    "sourceFile": "demo1",
                    "frames": 2,
                    "pairBytes": 20,
                },
                {
                    "linkId": 2,
                    "sampleId": 2,
                    "textId": 0,
                    "datasetFolder": "KIT",
                    "sourceFile": "demo2",
                    "frames": 2,
                    "pairBytes": 20,
                },
                {
                    "linkId": 3,
                    "sampleId": 3,
                    "textId": 1,
                    "datasetFolder": "KIT",
                    "sourceFile": "demo3",
                    "frames": 2,
                    "pairBytes": 20,
                },
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "version": 2,
        "modelName": "xlm-roberta-base",
        "maxPromptLength": 8,
        "splitFrames": 2,
        "downsampleTargetFrames": None,
        "maxSegmentFrames": 2,
        "sampleShardSize": 2,
        "textShardSize": 2,
        "totalSamples": 4,
        "totalTexts": 4,
        "totalLinks": 4,
        "averageSampleBytes": 10.0,
        "maxSampleBytes": 10,
        "averagePairBytes": 20.0,
        "averageFrames": 2.0,
        "maxFrames": 2,
        "sampleShards": [
            {"path": "sample_shards/sample_shard_00000.pt", "sampleCount": 2},
            {"path": "sample_shards/sample_shard_00001.pt", "sampleCount": 2},
        ],
        "textShards": [
            {"path": "text_shards/text_shard_00000.pt", "sampleCount": 2},
            {"path": "text_shards/text_shard_00001.pt", "sampleCount": 2},
        ],
        "sampleIndexPath": "sample_index.json",
        "textIndexPath": "text_index.json",
        "linkIndexPath": "link_index.json",
        "enabledComponents": ["rotation6d"],
        "pooledTextDim": 2,
    }
    (datasetRoot / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return datasetRoot
