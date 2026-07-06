"""AI-nimator health monitor — Streamlit entry point.

Launch:
    streamlit run apps/monitor/app.py

Set the JSONL path in the sidebar (default: output/health/health.jsonl),
e.g. ``output/<run>/health/health.jsonl`` for a specific run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so both ``ainimator.*`` and
# ``apps.monitor.*`` are importable when launched via Streamlit.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402 (import after sys.path fix)
import streamlit as st  # noqa: E402

from apps.monitor.ingest import load_jsonl  # noqa: E402
from apps.monitor.panels import (  # noqa: E402
    HeaderPanel,
    KpiPanel,
    LossPanel,
    Panel,
    RawTablePanel,
    RenderContext,
    VerdictsPanel,
)
from apps.monitor.verdicts import (  # noqa: E402
    current_verdicts,
    worst_now,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
_DEFAULT_JSONL: str = "output/health/health.jsonl"
_AUTO_REFRESH_S: float = 3.0
_WINDOW_OPTIONS: list[int] = [0, 50, 100, 200]

_PANELS: list[Panel] = [
    HeaderPanel(),
    KpiPanel(),
    VerdictsPanel(),
    LossPanel(),
    RawTablePanel(),
]


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
def _sidebar() -> tuple[Path, int, bool]:
    """Render sidebar controls.

    Returns
    -------
    tuple[Path, int, bool]
        (jsonl_path, window_size, auto_refresh).
        ``window_size`` of 0 means show all records.
    """
    st.sidebar.title("Health Monitor")
    path_str = st.sidebar.text_input(
        "JSONL path", value=_DEFAULT_JSONL
    )
    window = int(st.sidebar.selectbox(
        "Window",
        options=_WINDOW_OPTIONS,
        format_func=lambda v: "All" if v == 0 else f"Last {v}",
    ))
    auto = st.sidebar.checkbox("Auto-refresh", value=True)
    return Path(path_str), window, auto


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _apply_window(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Slice the DataFrame to the last N rows.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame.
    window : int
        Number of rows to keep; 0 means all.

    Returns
    -------
    pd.DataFrame
        Sliced (and re-indexed) DataFrame.
    """
    if window > 0 and len(df) > window:
        return df.tail(window).reset_index(drop=True)
    return df


def _apply_step_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Add a step-value multiselect to the sidebar and filter the df.

    The ``step`` field is an intra-epoch counter (e.g. 50, 100).
    Keeping only one value removes the within-epoch duplication and
    produces cleaner curves.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``load_jsonl`` (before windowing).

    Returns
    -------
    pd.DataFrame
        Rows whose ``step`` value is in the selected set.
        Re-indexed so ``output_idx`` remains monotonic.
    """
    from ainimator.health.record_schema import STEP_KEY

    if STEP_KEY not in df.columns:
        return df

    available = sorted(df[STEP_KEY].dropna().unique().tolist())
    if len(available) <= 1:
        return df

    selected = st.sidebar.multiselect(
        "Filter by step",
        options=available,
        default=available,
        format_func=lambda v: f"step = {int(v)}",
        help=(
            "Intra-epoch step counter. "
            "Keep one value to remove within-epoch duplicates."
        ),
    )
    if not selected:
        return df

    filtered = df[df[STEP_KEY].isin(selected)].copy()
    filtered["output_idx"] = range(len(filtered))
    return filtered.reset_index(drop=True)


def _render_all(df: pd.DataFrame) -> None:
    """Render every panel for one refresh cycle.

    Parameters
    ----------
    df : pd.DataFrame
        Windowed DataFrame to display.
    """
    current = current_verdicts(df)
    ctx = RenderContext(df=df, current=current, worst=worst_now(df))
    for panel in _PANELS:
        panel.render(ctx)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="AI-nimator Health Monitor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    jsonl_path, window, auto_refresh = _sidebar()
    placeholder = st.empty()

    if not jsonl_path.exists():
        placeholder.warning(f"File not found: {jsonl_path}")
    else:
        df = load_jsonl(jsonl_path)
        if df.empty:
            placeholder.info("No records in the JSONL yet.")
        else:
            df = _apply_step_filter(df)
            df = _apply_window(df, window)
            with placeholder.container():
                _render_all(df)

    if auto_refresh:
        time.sleep(_AUTO_REFRESH_S)
        st.rerun()


if __name__ == "__main__":
    main()
