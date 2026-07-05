"""Tests for the ``locomotion`` profile's caption-based clip filter.

Covers ``_loadCaptionsByTextId`` / ``_filterLinksByCaption`` /
``_selectFullIndices(..., captionFilter=...)`` in
:mod:`ainimator.cli.train_controller_v2`, using a synthetic on-disk
dataset root (manifest + link/text index + one text shard) instead of
the real preprocessed dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ainimator.cli.train_controller_v2 import (
    _filterLinksByCaption,
    _loadCaptionsByTextId,
    _selectFullIndices,
)

_CAPTIONS = [
    "a person walks forward",
    "a person waves hello",
    "a person jogs slowly",
    "a person sits down",
]


def _buildDatasetRoot(tmp_path: Path) -> Path:
    """Write a minimal manifest + link/text index + text shard."""
    textShardPath = tmp_path / "text_shards" / "shard_0.pt"
    textShardPath.parent.mkdir(parents=True)
    torch.save(
        [{"raw_text": caption} for caption in _CAPTIONS], textShardPath
    )

    manifest = {
        "textIndexPath": "text_index.json",
        "linkIndexPath": "link_index.json",
        "textShards": [{"path": "text_shards/shard_0.pt"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    textIndex = [
        {"textId": textId, "shardIndex": 0, "shardOffset": textId}
        for textId in range(len(_CAPTIONS))
    ]
    (tmp_path / "text_index.json").write_text(json.dumps(textIndex))

    # 4 links, one per caption, distinct linkId/frame counts so the
    # frame-count sort / numClips truncation stays observable.
    linkIndex = [
        {"linkId": linkId, "textId": linkId, "frames": 200 + 10 * linkId}
        for linkId in range(len(_CAPTIONS))
    ]
    (tmp_path / "link_index.json").write_text(json.dumps(linkIndex))
    return tmp_path


def test_loadCaptionsByTextId_returns_only_requested_ids(
    tmp_path: Path,
) -> None:
    """Only the requested textIds are resolved (not the whole shard)."""
    datasetRoot = _buildDatasetRoot(tmp_path)
    captions = _loadCaptionsByTextId(datasetRoot, {0, 2})
    assert captions == {0: _CAPTIONS[0], 2: _CAPTIONS[2]}


def test_filterLinksByCaption_keeps_only_matching_locomotion_clips(
    tmp_path: Path,
) -> None:
    """Locomotion regex keeps 'walks'/'jogs', drops 'waves'/'sits'."""
    datasetRoot = _buildDatasetRoot(tmp_path)
    links = json.loads((datasetRoot / "link_index.json").read_text())
    filtered = _filterLinksByCaption(
        datasetRoot,
        links,
        r"\b(walk\w*|run\w*|jog\w*|strid\w*|march\w*|sprint\w*)\b",
    )
    assert {link["linkId"] for link in filtered} == {0, 2}


def test_selectFullIndices_without_filter_is_unchanged(
    tmp_path: Path,
) -> None:
    """captionFilter=None keeps all eligible clips (default behaviour)."""
    datasetRoot = _buildDatasetRoot(tmp_path)
    train, heldOut = _selectFullIndices(
        datasetRoot, numClips=4, heldOutClips=1, minFrames=0, seed=0
    )
    assert len(train) + len(heldOut) == 4


def test_selectFullIndices_with_caption_filter_narrows_selection(
    tmp_path: Path,
) -> None:
    """captionFilter restricts the pool before numClips truncation."""
    datasetRoot = _buildDatasetRoot(tmp_path)
    train, heldOut = _selectFullIndices(
        datasetRoot,
        numClips=10,
        heldOutClips=0,
        minFrames=0,
        seed=0,
        captionFilter=r"\b(walk\w*|jog\w*)\b",
    )
    assert set(train) | set(heldOut) == {0, 2}
