"""Constants for preprocessed dataset tooling."""

from __future__ import annotations

from typing import Final

PREPROCESSED_PROMPT_FILENAME: Final[str] = "prompt.json"
PREPROCESSED_MANIFEST_FILENAME: Final[str] = "manifest.json"
PREPROCESSED_SAMPLE_INDEX_FILENAME: Final[str] = "sample_index.json"
PREPROCESSED_TEXT_INDEX_FILENAME: Final[str] = "text_index.json"
PREPROCESSED_LINK_INDEX_FILENAME: Final[str] = "link_index.json"
PREPROCESSED_SAMPLE_SHARDS_DIRNAME: Final[str] = "sample_shards"
PREPROCESSED_TEXT_SHARDS_DIRNAME: Final[str] = "text_shards"
PREPROCESSED_GENERATION_TEXT_CACHE_DIRNAME: Final[str] = "generation_text_cache"
PREPROCESSED_GENERATION_TEXT_EMBED_SHARDS_DIRNAME: Final[str] = "text_embed_shards"
PREPROCESSED_GENERATION_TEXT_CACHE_MANIFEST_FILENAME: Final[str] = "manifest.json"
PREPROCESSED_GENERATION_TEXT_CACHE_VERSION: Final[int] = 1
PREPROCESSED_MANIFEST_VERSION: Final[int] = 2
PREPROCESSED_MIN_FRAME_COUNT: Final[int] = 1

# Legacy aliases kept to reduce churn in callers while V2 rolls out.
PREPROCESSED_INDEX_FILENAME: Final[str] = PREPROCESSED_SAMPLE_INDEX_FILENAME
PREPROCESSED_SHARDS_DIRNAME: Final[str] = PREPROCESSED_SAMPLE_SHARDS_DIRNAME
