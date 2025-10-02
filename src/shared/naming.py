"""Naming helpers shared between CLI features."""

from __future__ import annotations

import hashlib
import re


def safe_slug(text: str) -> str:
    """Return a filesystem-safe slug derived from ``text``."""

    sanitized = re.sub(r"[\s/\\]+", "_", text.strip())
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "", sanitized)
    return sanitized or "clip"


def short_hash(text: str, length: int = 8) -> str:
    """Compact string fingerprint useful for deterministic folder names."""

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return digest[:length]

