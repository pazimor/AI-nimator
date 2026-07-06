"""Parse health.jsonl records into a pandas DataFrame.

One row per record; columns mirror the flat JSONL keys.  An extra
``output_idx`` column (0-based row index) is prepended so callers
always have a monotonic X axis independent of the ``step`` counter.

The ``step`` field in the JSONL is an *intra-epoch* counter, not a
global step — it resets between epochs.  Never use it as an X axis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from ainimator.health.record_schema import STEP_KEY

OUTPUT_IDX_COL: str = "output_idx"


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a ``health.jsonl`` file into a tidy DataFrame.

    Parameters
    ----------
    path : Path
        Absolute path to the JSONL file.

    Returns
    -------
    pd.DataFrame
        One row per valid JSON line.  Columns: ``output_idx``,
        ``step``, then all metric / verdict keys.  Returns an
        empty DataFrame when the file is missing or has no valid
        lines.
    """
    rows = _parse_file(path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.insert(0, OUTPUT_IDX_COL, range(len(df)))
    return df


def records_to_df(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert a list of raw record dicts to a tidy DataFrame.

    Intended for tests and in-memory pipelines.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Raw parsed JSON dicts.

    Returns
    -------
    pd.DataFrame
        Same layout as ``load_jsonl``.
    """
    if not records:
        return pd.DataFrame()
    rows = [_flatten(r) for r in records]
    df = pd.DataFrame(rows)
    df.insert(0, OUTPUT_IDX_COL, range(len(df)))
    return df


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _parse_file(path: Path) -> list[dict[str, Any]]:
    """Read and tolerantly parse JSONL lines from disk.

    Parameters
    ----------
    path : Path
        Path to the JSONL file.

    Returns
    -------
    list[dict[str, Any]]
        Flat row dicts; truncated or invalid lines are skipped.
    """
    import json  # local to avoid top-level cost when unused

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            rows.append(_flatten(raw))
    return rows


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw record dict to a flat row suitable for DataFrame.

    Non-finite floats (NaN, Inf) are replaced with ``float("nan")``
    so pandas handles them uniformly.

    Parameters
    ----------
    raw : dict[str, Any]
        One parsed JSON record.

    Returns
    -------
    dict[str, Any]
        Flat dict with all original keys preserved.
    """
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = float("nan")
        else:
            out[key] = value
    if STEP_KEY not in out:
        out[STEP_KEY] = 0
    return out
