"""JSONL writer for HealthHub metric streams.

Writes one JSON object per line to ``outputDir/health/<name>.jsonl``.
JSONL is the source-of-truth sink; TensorBoard is a secondary sink.

Each record includes a ``step`` key and all metric values.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Subdirectory name inside the run's outputDir.
HEALTH_SUBDIR = "health"


class JsonlWriter:
    """Append-mode JSONL writer for one metric stream.

    Parameters
    ----------
    outputDir : Path
        Root output directory of the run.
    name : str
        Stream name; file will be ``health/<name>.jsonl``.
    """

    def __init__(self, outputDir: Path, name: str) -> None:
        self._dir = outputDir / HEALTH_SUBDIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{name}.jsonl"

    @property
    def path(self) -> Path:
        """Absolute path to the JSONL file."""
        return self._path

    def write(self, record: dict[str, Any]) -> None:
        """Append one record to the stream.

        Parameters
        ----------
        record : dict
            Must contain a ``step`` key; all other keys are metric
            names with float values.
        """
        try:
            line = json.dumps(record, default=_jsonDefault)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            LOGGER.warning("JsonlWriter: could not write to %s: %s",
                           self._path, exc)


def _jsonDefault(obj: Any) -> Any:
    """JSON fallback for non-serializable types."""
    if hasattr(obj, "item"):  # tensor scalar
        return obj.item()
    return str(obj)
