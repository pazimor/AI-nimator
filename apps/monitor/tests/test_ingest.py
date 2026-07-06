"""Tests for apps.monitor.ingest — isolated, no Streamlit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from apps.monitor.ingest import OUTPUT_IDX_COL, load_jsonl, records_to_df

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
_SAMPLE_RECORDS = [
    {
        "step": 50,
        "loss_total": 6.048,
        "loss_bone": 0.403,
        "loss_share.bone": 0.066,
        "verdict.update_ratio": "WARNING",
        "verdict.conditioning_sensitivity": "CRITICAL",
    },
    {
        "step": 100,
        "loss_total": 5.1,
        "loss_bone": 0.31,
        "loss_share.bone": 0.06,
        "verdict.update_ratio": "OK",
        "verdict.conditioning_sensitivity": "OK",
    },
]


# ------------------------------------------------------------------
# records_to_df
# ------------------------------------------------------------------
def test_records_to_df_shape() -> None:
    """DataFrame has one row per record plus output_idx column."""
    df = records_to_df(_SAMPLE_RECORDS)
    assert len(df) == 2
    assert OUTPUT_IDX_COL in df.columns


def test_records_to_df_output_idx_monotonic() -> None:
    """output_idx is 0-based and monotonically increasing."""
    df = records_to_df(_SAMPLE_RECORDS)
    assert list(df[OUTPUT_IDX_COL]) == [0, 1]


def test_records_to_df_preserves_keys() -> None:
    """All keys from the source records appear as columns."""
    df = records_to_df(_SAMPLE_RECORDS)
    for key in _SAMPLE_RECORDS[0]:
        assert key in df.columns


def test_records_to_df_empty() -> None:
    """Empty input returns an empty DataFrame."""
    df = records_to_df([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_non_finite_replaced_with_nan() -> None:
    """Inf and -Inf values are replaced with NaN."""
    record = {"step": 1, "loss_total": float("inf"), "x": float("-inf")}
    df = records_to_df([record])
    assert math.isnan(df["loss_total"].iloc[0])
    assert math.isnan(df["x"].iloc[0])


# ------------------------------------------------------------------
# load_jsonl
# ------------------------------------------------------------------
def test_load_jsonl_missing_file(tmp_path: Path) -> None:
    """Missing file returns an empty DataFrame (no crash)."""
    df = load_jsonl(tmp_path / "nonexistent.jsonl")
    assert df.empty


def test_load_jsonl_round_trip(tmp_path: Path) -> None:
    """Records written to disk are recovered with correct shape."""
    path = tmp_path / "health.jsonl"
    with path.open("w") as fh:
        for rec in _SAMPLE_RECORDS:
            fh.write(json.dumps(rec) + "\n")
    df = load_jsonl(path)
    assert len(df) == 2
    assert OUTPUT_IDX_COL in df.columns


def test_load_jsonl_tolerates_truncated_line(tmp_path: Path) -> None:
    """A truncated (invalid) JSON line is skipped, rest is loaded."""
    path = tmp_path / "health.jsonl"
    with path.open("w") as fh:
        fh.write(json.dumps(_SAMPLE_RECORDS[0]) + "\n")
        fh.write("{INVALID\n")
        fh.write(json.dumps(_SAMPLE_RECORDS[1]) + "\n")
    df = load_jsonl(path)
    assert len(df) == 2


def test_load_jsonl_empty_file(tmp_path: Path) -> None:
    """Empty JSONL file returns an empty DataFrame."""
    path = tmp_path / "health.jsonl"
    path.write_text("")
    df = load_jsonl(path)
    assert df.empty
