"""Tests for apps.monitor.losses — isolated, no Streamlit."""

from __future__ import annotations

import pandas as pd
import pytest

from apps.monitor.ingest import records_to_df
from apps.monitor.losses import active_losses, contribution, limiting_loss

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
_RECORDS = [
    {
        "step": 50,
        "loss_total": 6.0,
        "loss_bone": 0.40,
        "loss_global": 0.23,
        "loss_clip_guidance": 4.55,
        "loss_clip_aux_pool": 1.97,
        "loss_vel_xyz": 0.0,
        "loss_foot_contact": 0.0,
        "loss_share.bone": 0.066,
        "loss_share.global": 0.038,
        "loss_share.clip_guidance": 0.732,
        "loss_share.clip_aux_pool": 0.162,
    },
    {
        "step": 100,
        "loss_total": 5.1,
        "loss_bone": 0.35,
        "loss_global": 0.20,
        "loss_clip_guidance": 3.90,
        "loss_clip_aux_pool": 1.60,
        "loss_vel_xyz": 0.0,
        "loss_foot_contact": 0.0,
        "loss_share.bone": 0.07,
        "loss_share.global": 0.04,
        "loss_share.clip_guidance": 0.74,
        "loss_share.clip_aux_pool": 0.16,
    },
]


def _df(records: list[dict] | None = None) -> pd.DataFrame:
    return records_to_df(records or _RECORDS)


# ------------------------------------------------------------------
# active_losses
# ------------------------------------------------------------------
def test_active_losses_excludes_zero_components() -> None:
    """Inactive losses (always 0.0) are not returned."""
    active = active_losses(_df())
    assert "loss_vel_xyz" not in active
    assert "loss_foot_contact" not in active


def test_active_losses_includes_nonzero() -> None:
    """Components with non-zero values are returned."""
    active = active_losses(_df())
    assert "loss_bone" in active
    assert "loss_clip_guidance" in active


def test_active_losses_excludes_share_columns() -> None:
    """loss_share.* columns are never treated as loss components."""
    active = active_losses(_df())
    assert not any(c.startswith("loss_share.") for c in active)


def test_active_losses_empty_df() -> None:
    """Empty DataFrame returns an empty list."""
    assert active_losses(pd.DataFrame()) == []


# ------------------------------------------------------------------
# contribution
# ------------------------------------------------------------------
def test_contribution_columns() -> None:
    """Contribution DataFrame has share_* columns."""
    contrib = contribution(_df())
    assert "share_bone" in contrib.columns
    assert "share_clip_guidance" in contrib.columns


def test_contribution_no_share_columns() -> None:
    """DataFrame without loss_share.* returns an empty DataFrame."""
    df = records_to_df([{"step": 1, "loss_total": 1.0}])
    assert contribution(df).empty


def test_contribution_values_positive() -> None:
    """All share values in the fixture are > 0."""
    contrib = contribution(_df())
    share_cols = [c for c in contrib.columns if c != "output_idx"]
    assert all(contrib[col].dropna().gt(0).all() for col in share_cols)


# ------------------------------------------------------------------
# limiting_loss
# ------------------------------------------------------------------
def test_limiting_loss_detected() -> None:
    """Dominant stalling component is identified."""
    # Build records where clip_guidance is dominant and stalling.
    stalling = [
        {
            "step": i * 50,
            "loss_total": 5.0,
            "loss_clip_guidance": 4.0,
            "loss_share.clip_guidance": 0.80,
        }
        for i in range(1, 6)
    ]
    df = records_to_df(stalling)
    lim = limiting_loss(df)
    assert lim == "loss_clip_guidance"


def test_limiting_loss_none_when_declining() -> None:
    """No limiting loss when the dominant component is declining."""
    declining = [
        {
            "step": i * 50,
            "loss_total": 5.0 - i * 0.3,
            "loss_bone": 4.0 - i * 0.5,
            "loss_share.bone": 0.80,
        }
        for i in range(1, 6)
    ]
    df = records_to_df(declining)
    assert limiting_loss(df) is None


def test_limiting_loss_insufficient_data() -> None:
    """Returns None with fewer than 2 records."""
    df = records_to_df([_RECORDS[0]])
    assert limiting_loss(df) is None


def test_limiting_loss_no_share_data() -> None:
    """Returns None when no loss_share.* columns exist."""
    df = records_to_df([{"step": 1, "loss_bone": 1.0}])
    assert limiting_loss(df) is None
