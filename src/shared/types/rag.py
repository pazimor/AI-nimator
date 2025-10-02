"""Dataclasses used by the RAG OpenAI feature."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RagTestJob:
    file_path: Path
    model: str


@dataclass
class RagBatchJob:
    input_dir: Path
    output_root: Path
    prompt_template: str
    glob_pattern: str
    model: str
    max_tokens_per_jsonl: Optional[int] = None
    max_items_per_jsonl: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_items: Optional[int] = None


@dataclass
class RagFetchJob:
    output_root: Path


@dataclass
class RagLocalFetchJob:
    input_dir: Path
    target_root: Path
    glob_pattern: str = "**/*.jsonl"
