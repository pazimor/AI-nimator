"""Verdict aggregation from a health DataFrame.

All functions are pure (no Streamlit, no torch) and work on a
DataFrame produced by ``ingest.load_jsonl`` or ``ingest.records_to_df``.
Verdicts are *read* from the JSONL — never recalculated here.
"""

from __future__ import annotations

import pandas as pd

from ainimator.health.record_schema import VERDICT_PREFIX, Verdict

# Severity rank used when comparing verdicts.
_SEVERITY: dict[str, int] = {
    Verdict.UNKNOWN.value: 0,
    Verdict.OK.value: 1,
    Verdict.WARNING.value: 2,
    Verdict.CRITICAL.value: 3,
}


def verdict_columns(df: pd.DataFrame) -> list[str]:
    """Return all verdict column names present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``ingest.load_jsonl``.

    Returns
    -------
    list[str]
        Column names starting with ``VERDICT_PREFIX``.
    """
    return [c for c in df.columns if c.startswith(VERDICT_PREFIX)]


def current_verdicts(df: pd.DataFrame) -> dict[str, str]:
    """Return the most-recent verdict per axis.

    Axes that are constantly UNKNOWN are excluded.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``ingest.load_jsonl``.

    Returns
    -------
    dict[str, str]
        Axis name → verdict string (from the last non-null row).
    """
    out: dict[str, str] = {}
    for col in verdict_columns(df):
        series = df[col].dropna()
        if series.empty:
            continue
        axis = _axis_name(col)
        if _is_always_unknown(series):
            continue
        out[axis] = str(series.iloc[-1])
    return out


def worst_now(df: pd.DataFrame) -> str:
    """Return the worst current verdict across all visible axes.

    UNKNOWN axes are excluded.  Returns ``"OK"`` when nothing is worse.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``ingest.load_jsonl``.

    Returns
    -------
    str
        Verdict string of the worst non-UNKNOWN axis.
    """
    current = current_verdicts(df)
    worst_rank = 0
    worst = Verdict.OK.value
    for verdict in current.values():
        rank = _SEVERITY.get(verdict, 0)
        if rank > worst_rank:
            worst_rank = rank
            worst = verdict
    return worst


def verdict_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a verdict matrix (output_idx × axes) for the heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``ingest.load_jsonl``.

    Returns
    -------
    pd.DataFrame
        Columns: ``output_idx`` (when present) + one column per
        non-always-UNKNOWN axis.  Values are verdict strings.
    """
    cols = verdict_columns(df)
    if not cols:
        return pd.DataFrame()

    keep_cols = [
        c for c in cols
        if not _is_always_unknown(df[c].dropna())
    ]
    if not keep_cols:
        return pd.DataFrame()

    x_cols = (
        ["output_idx"] if "output_idx" in df.columns else []
    )
    matrix = df[x_cols + keep_cols].copy()
    return matrix.rename(
        columns={c: _axis_name(c) for c in keep_cols}
    )


def transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Detect verdict-level transitions for each axis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``ingest.load_jsonl``.

    Returns
    -------
    pd.DataFrame
        Columns: ``output_idx``, ``axis``, ``from_verdict``,
        ``to_verdict``.  One row per transition event.
    """
    rows: list[dict[str, object]] = []
    idx_col = "output_idx" if "output_idx" in df.columns else None
    for col in verdict_columns(df):
        axis = _axis_name(col)
        sub = df[[idx_col, col]].dropna() if idx_col else df[[col]].dropna()
        prev: str | None = None
        for _, row in sub.iterrows():
            val = str(row[col])
            if prev is not None and val != prev:
                entry: dict[str, object] = {
                    "axis": axis,
                    "from_verdict": prev,
                    "to_verdict": val,
                }
                if idx_col:
                    entry["output_idx"] = row[idx_col]
                rows.append(entry)
            prev = val
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _axis_name(col: str) -> str:
    """Strip the verdict prefix from a column name."""
    return col[len(VERDICT_PREFIX):]


def _is_always_unknown(series: pd.Series) -> bool:
    """Return True when every non-null value equals UNKNOWN."""
    non_null = series.dropna()
    if non_null.empty:
        return True
    return bool((non_null == Verdict.UNKNOWN.value).all())
