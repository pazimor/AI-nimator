"""Dataclasses describing uniformizer tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class UniformizerJob:
    """User-facing options passed to the uniformizer service."""

    input_path: Path
    output_path: Path
    target_skeleton: Optional[str] = None
    target_map: Optional[Path] = None
    resample_fps: Optional[float] = None
    flatten_output: bool = False
    include_prompts: bool = False


@dataclass
class UniformizerDirectoryJob:
    """Batch conversion options for directory processing."""

    input_dir: Path
    output_dir: Path
    target_skeleton: Optional[str] = None
    target_map: Optional[Path] = None
    resample_fps: Optional[float] = None
    flatten_output: bool = False
    include_prompts: bool = False
