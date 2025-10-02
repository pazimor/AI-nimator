"""Tests covering shared naming helpers."""

from __future__ import annotations

from src.shared.naming import safe_slug, short_hash


def test_safe_slug_sanitizes_characters():
    assert safe_slug("Hello World/Clip") == "Hello_World_Clip"
    assert safe_slug("   ") == "clip"


def test_short_hash_length_can_be_truncated():
    digest = short_hash("value", length=6)
    assert len(digest) == 6
    assert digest == short_hash("value", length=6)

