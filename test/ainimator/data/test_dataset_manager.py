"""Tests for dataset chunk selection helpers."""

from __future__ import annotations

from ainimator.training.dataset_manager import (
    _selectShuffledChunkIndices,
    _validationMetadataMatches,
)


def test_select_shuffled_chunk_indices_covers_dataset_once_per_cycle() -> None:
    """A full cycle must cover every sample exactly once."""
    totalSize = 10
    chunkSize = 4

    seen: list[int] = []
    for chunkIndex in range(3):
        seen.extend(
            _selectShuffledChunkIndices(
                totalSize=totalSize,
                chunkSize=chunkSize,
                chunkIndex=chunkIndex,
                cycleIndex=0,
            )
        )

    assert sorted(seen) == list(range(totalSize))


def test_select_shuffled_chunk_indices_changes_order_between_cycles() -> None:
    """Successive full-coverage cycles should not reuse the same chunk order."""
    firstCycle = _selectShuffledChunkIndices(
        totalSize=12,
        chunkSize=4,
        chunkIndex=0,
        cycleIndex=0,
    )
    secondCycle = _selectShuffledChunkIndices(
        totalSize=12,
        chunkSize=4,
        chunkIndex=0,
        cycleIndex=1,
    )

    assert firstCycle != secondCycle


def test_validation_metadata_mismatch_invalidates_stale_cache() -> None:
    """Changing split or folders must regenerate fixed validation indices."""
    metadata = {
        "total_size": 1486,
        "validation_split": 1.0,
        "dataset_folders": ["ACCAD"],
    }

    assert not _validationMetadataMatches(
        metadata=metadata,
        totalSize=1486,
        validationSplit=0.05,
        datasetFolders=["ACCAD"],
    )


def test_validation_metadata_matches_active_folder_filter() -> None:
    """The cache stays valid only for the same normalized folder subset."""
    metadata = {
        "total_size": 1486,
        "validation_split": 0.05,
        "dataset_folders": ["accad"],
    }

    assert _validationMetadataMatches(
        metadata=metadata,
        totalSize=1486,
        validationSplit=0.05,
        datasetFolders=["ACCAD"],
    )
