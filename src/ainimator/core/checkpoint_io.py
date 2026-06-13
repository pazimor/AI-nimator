"""Helpers for robust checkpoint persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

# Subdirectory under outputDir where checkpoint files live.
# All training loops write here so the run layout is:
#   outputDir/
#     checkpoints/   ← this constant names the subdirectory
#     health/
#     resolved_config.yaml
#     log.txt
CHECKPOINT_SUBDIR = "checkpoints"


def checkpointDir(outputDir: Path) -> Path:
    """Return and create the ``checkpoints/`` subdirectory under *outputDir*.

    Parameters
    ----------
    outputDir : Path
        Root output directory of the training run.

    Returns
    -------
    Path
        ``outputDir / CHECKPOINT_SUBDIR``, created if missing.
    """
    directory = outputDir / CHECKPOINT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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
