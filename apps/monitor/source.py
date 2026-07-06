"""RecordSource protocol and concrete implementations.

Defines how raw health records are acquired (disk, in-memory, …).
The protocol enables dependency injection so tests can swap in
fixed data without touching the file system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RecordSource(Protocol):
    """Protocol for acquiring raw health records."""

    def read(self) -> list[dict[str, Any]]:
        """Return all available records as raw dicts.

        Returns
        -------
        list[dict[str, Any]]
            Raw parsed JSON objects from the JSONL stream.
        """
        ...  # pragma: no cover


class JsonlRecordSource:
    """Read records from a ``health.jsonl`` file on disk.

    Parameters
    ----------
    path : Path
        Absolute path to the JSONL file.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self) -> list[dict[str, Any]]:
        """Return all records parsed from the JSONL file.

        Tolerant of truncated lines and JSON decode errors —
        invalid lines are silently skipped.

        Returns
        -------
        list[dict[str, Any]]
            Parsed records; empty list if the file is absent.
        """
        if not self._path.exists():
            return []
        return _read_jsonl(self._path)


class InMemoryRecordSource:
    """In-memory record source intended for tests.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Pre-loaded records returned verbatim by ``read()``.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = list(records)

    def read(self) -> list[dict[str, Any]]:
        """Return the pre-loaded records.

        Returns
        -------
        list[dict[str, Any]]
            Records provided at construction time.
        """
        return self._records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Tolerant JSONL reader.

    Parameters
    ----------
    path : Path
        Path to the JSONL file.

    Returns
    -------
    list[dict[str, Any]]
        Parsed records; invalid lines are skipped.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                pass
    return records
