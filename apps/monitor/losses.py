"""Multi-loss analysis from a health DataFrame.

All functions are pure (no Streamlit, no torch) and operate on a
DataFrame from ``ingest.load_jsonl`` or ``ingest.records_to_df``.
"""

from __future__ import annotations

import pandas as pd

from ainimator.health.record_schema import (
    LOSS_PREFIX,
    LOSS_SHARE_PREFIX,
    is_active_loss,
)

# Recent-window size for trend / limiting-loss detection.
_RECENT_WINDOW: int = 5

# Bare summary key that is not a component column.
_LOSS_SHARE_BARE: str = "loss_share"


def active_losses(df: pd.DataFrame) -> list[str]:
    """Return loss-component column names with at least one non-zero value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``ingest.load_jsonl``.

    Returns
    -------
    list[str]
        Column names like ``["loss_bone", "loss_global"]``.
    """
    candidates = [
        c for c in df.columns
        if c.startswith(LOSS_PREFIX)
        and not c.startswith(LOSS_SHARE_PREFIX)
        and c != _LOSS_SHARE_BARE
    ]
    return [
        c for c in candidates
        if is_active_loss(df[c].dropna().tolist())
    ]


def contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Build a loss-share contribution DataFrame for a stacked-area chart.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``ingest.load_jsonl``.

    Returns
    -------
    pd.DataFrame
        Columns: ``output_idx`` (when present) + one column per
        ``loss_share.*`` key renamed to ``share_<component>``.
        Empty DataFrame when no share columns exist.
    """
    share_cols = [
        c for c in df.columns
        if c.startswith(LOSS_SHARE_PREFIX)
    ]
    if not share_cols:
        return pd.DataFrame()
    x_cols = ["output_idx"] if "output_idx" in df.columns else []
    result = df[x_cols + share_cols].copy()
    return result.rename(
        columns={
            c: c.replace(LOSS_SHARE_PREFIX, "share_")
            for c in share_cols
        }
    )


def limiting_loss(df: pd.DataFrame) -> str | None:
    """Identify the limiting loss component.

    The *limiting* loss has the largest recent mean share **and** a
    non-declining trend (flat or positive slope) while ``loss_total``
    is decelerating.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``ingest.load_jsonl``.

    Returns
    -------
    str or None
        Loss column name (e.g. ``"loss_clip_guidance"``), or None
        when there are fewer than 2 records or no share columns.
    """
    share_cols = [
        c for c in df.columns
        if c.startswith(LOSS_SHARE_PREFIX)
    ]
    if len(df) < 2 or not share_cols:
        return None

    recent = df.tail(_RECENT_WINDOW)
    shares = _mean_shares(recent, share_cols)
    if not shares:
        return None

    dominant_axis = max(shares, key=lambda k: shares[k])
    loss_col = f"{LOSS_PREFIX}{dominant_axis}"
    if loss_col not in df.columns:
        return None
    if not _is_stalling(recent, loss_col):
        return None
    return loss_col


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _mean_shares(
    df: pd.DataFrame,
    share_cols: list[str],
) -> dict[str, float]:
    """Compute mean share value per component over recent rows.

    Parameters
    ----------
    df : pd.DataFrame
        Recent slice of the full DataFrame.
    share_cols : list[str]
        Columns with ``LOSS_SHARE_PREFIX``.

    Returns
    -------
    dict[str, float]
        Component name → mean share (only positive entries).
    """
    shares: dict[str, float] = {}
    for col in share_cols:
        axis = col[len(LOSS_SHARE_PREFIX):]
        mean_val = df[col].dropna().mean()
        if pd.notna(mean_val) and float(mean_val) > 0:
            shares[axis] = float(mean_val)
    return shares


def _is_stalling(df: pd.DataFrame, col: str) -> bool:
    """Return True when a column's last value >= its first value.

    Parameters
    ----------
    df : pd.DataFrame
        Recent slice.
    col : str
        Column to inspect.

    Returns
    -------
    bool
        True when the slope is non-negative (flat or worsening).
    """
    series = df[col].dropna()
    if len(series) < 2:
        return False
    return float(series.iloc[-1]) >= float(series.iloc[0])
