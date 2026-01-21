"""Constants for preprocessed dataset tooling."""

from __future__ import annotations

from typing import Final

PREPROCESSED_PROMPT_FILENAME: Final[str] = "prompt.json"
PREPROCESSED_MANIFEST_FILENAME: Final[str] = "manifest.json"
PREPROCESSED_INDEX_FILENAME: Final[str] = "index.json"
PREPROCESSED_SHARDS_DIRNAME: Final[str] = "shards"
PREPROCESSED_MANIFEST_VERSION: Final[int] = 1
PREPROCESSED_TAG_PREFIX_TEMPLATE: Final[str] = "[Tag: {tag}] "
PREPROCESSED_MIN_FRAME_COUNT: Final[int] = 1
