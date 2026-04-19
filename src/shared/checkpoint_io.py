"""Helpers for robust checkpoint persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def saveTorchObjectAtomically(payload: Any, targetPath: Path) -> Path:
    """Persist a torch-serializable object without exposing partial files."""
    targetPath.parent.mkdir(parents=True, exist_ok=True)
    fileDescriptor, tempPathRaw = tempfile.mkstemp(
        prefix=f".{targetPath.name}.",
        suffix=".tmp",
        dir=targetPath.parent,
    )
    tempPath = Path(tempPathRaw)
    try:
        with os.fdopen(fileDescriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tempPath, targetPath)
    except Exception:
        try:
            tempPath.unlink()
        except FileNotFoundError:
            pass
        raise
    return targetPath
