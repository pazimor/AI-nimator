"""Tests for apps.monitor.verdicts — isolated, no Streamlit."""

from __future__ import annotations

import pandas as pd
import pytest

from apps.monitor.ingest import OUTPUT_IDX_COL, records_to_df
from apps.monitor.verdicts import (
    current_verdicts,
    transitions,
    verdict_columns,
    verdict_matrix,
    worst_now,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
_EARLY = [
    {
        "step": 50,
        "loss_total": 6.0,
        "verdict.conditioning_sensitivity": "CRITICAL",
        "verdict.update_ratio": "WARNING",
        "verdict.post_norm_stats": "UNKNOWN",
    },
    {
        "step": 100,
        "loss_total": 5.5,
        "verdict.conditioning_sensitivity": "CRITICAL",
        "verdict.update_ratio": "WARNING",
        "verdict.post_norm_stats": "UNKNOWN",
    },
]

_LATE = [
    {
        "step": 50,
        "loss_total": 1.2,
        "verdict.conditioning_sensitivity": "OK",
        "verdict.update_ratio": "OK",
        "verdict.post_norm_stats": "UNKNOWN",
    },
    {
        "step": 100,
        "loss_total": 0.9,
        "verdict.conditioning_sensitivity": "OK",
        "verdict.update_ratio": "OK",
        "verdict.post_norm_stats": "UNKNOWN",
    },
]

_TRANSITION = _EARLY + _LATE


def _df(records: list[dict]) -> pd.DataFrame:
    return records_to_df(records)


# ------------------------------------------------------------------
# verdict_columns
# ------------------------------------------------------------------
def test_verdict_columns_found() -> None:
    """All verdict.* columns are detected."""
    df = _df(_EARLY)
    cols = verdict_columns(df)
    assert "verdict.conditioning_sensitivity" in cols
    assert "verdict.update_ratio" in cols


# ------------------------------------------------------------------
# current_verdicts
# ------------------------------------------------------------------
def test_current_verdicts_late_window() -> None:
    """current_verdicts returns the last-row verdict, not the first."""
    df = _df(_TRANSITION)
    current = current_verdicts(df)
    assert current["conditioning_sensitivity"] == "OK"
    assert current["update_ratio"] == "OK"


def test_current_verdicts_excludes_always_unknown() -> None:
    """Axes that are always UNKNOWN are not returned."""
    df = _df(_EARLY)
    current = current_verdicts(df)
    assert "post_norm_stats" not in current


# ------------------------------------------------------------------
# worst_now
# ------------------------------------------------------------------
def test_worst_now_returns_critical_in_early() -> None:
    """CRITICAL is returned when it is the worst current verdict."""
    df = _df(_EARLY)
    assert worst_now(df) == "CRITICAL"


def test_worst_now_returns_ok_in_late() -> None:
    """OK is returned when all axes have recovered."""
    df = _df(_LATE)
    assert worst_now(df) == "OK"


def test_worst_now_empty() -> None:
    """Empty DataFrame returns OK (no data = no crisis)."""
    assert worst_now(pd.DataFrame()) == "OK"


# ------------------------------------------------------------------
# verdict_matrix
# ------------------------------------------------------------------
def test_verdict_matrix_shape() -> None:
    """Matrix has the right number of rows and visible axes."""
    df = _df(_TRANSITION)
    matrix = verdict_matrix(df)
    assert len(matrix) == len(df)
    # post_norm_stats is always UNKNOWN → excluded
    assert "post_norm_stats" not in matrix.columns


def test_verdict_matrix_output_idx_column() -> None:
    """output_idx column is preserved in the matrix."""
    df = _df(_TRANSITION)
    matrix = verdict_matrix(df)
    assert OUTPUT_IDX_COL in matrix.columns


def test_verdict_matrix_empty_when_no_verdicts() -> None:
    """No verdict columns → empty matrix."""
    df = records_to_df([{"step": 1, "loss_total": 1.0}])
    assert verdict_matrix(df).empty


# ------------------------------------------------------------------
# transitions
# ------------------------------------------------------------------
def test_transitions_detected() -> None:
    """A CRITICAL→OK transition is detected for each axis."""
    df = _df(_TRANSITION)
    trans = transitions(df)
    assert not trans.empty
    assert "conditioning_sensitivity" in trans["axis"].values


def test_transitions_no_false_positives() -> None:
    """No transitions when all records have the same verdict."""
    df = _df(_EARLY)
    trans = transitions(df)
    assert trans.empty
