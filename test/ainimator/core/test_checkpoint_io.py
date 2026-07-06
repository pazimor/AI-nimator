"""Tests for atomic checkpoint persistence."""

from __future__ import annotations

from pathlib import Path

import torch

from ainimator.core.checkpoint_io import saveTorchObjectAtomically


def test_save_torch_object_atomically_round_trips(tmp_path: Path) -> None:
    """Atomic save should produce a readable torch checkpoint."""
    targetPath = tmp_path / "best_model.pt"
    payload = {
        "epoch": 3,
        "tensor": torch.arange(4),
    }

    saveTorchObjectAtomically(payload, targetPath)

    loaded = torch.load(targetPath, map_location="cpu", weights_only=False)

    assert loaded["epoch"] == 3
    assert torch.equal(loaded["tensor"], payload["tensor"])


def test_save_torch_object_atomically_replaces_existing_file(
    tmp_path: Path,
) -> None:
    """Existing checkpoints should be replaced by a complete new file."""
    targetPath = tmp_path / "best_model.pt"
    targetPath.write_bytes(b"corrupted")

    saveTorchObjectAtomically({"epoch": 7}, targetPath)

    loaded = torch.load(targetPath, map_location="cpu", weights_only=False)

    assert loaded["epoch"] == 7
