"""Panel protocol and concrete Streamlit panel implementations.

Each panel renders one visual section of the monitor.  Panels
depend only on ``RenderContext`` (pre-computed views), not on raw
health internals — this keeps the rendering layer independent of
business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ainimator.health.record_schema import Verdict
from apps.monitor.losses import active_losses, contribution, limiting_loss
from apps.monitor.palette import (
    AXIS_HELP,
    VERDICT_BADGE,
    VERDICT_CODE,
    VERDICT_COLORSCALE,
)
from apps.monitor.verdicts import (
    verdict_matrix as build_verdict_matrix,
)


# ------------------------------------------------------------------
# Context
# ------------------------------------------------------------------
@dataclass
class RenderContext:
    """Data bundle passed to every Panel.render() call.

    Attributes
    ----------
    df : pd.DataFrame
        Full (windowed) DataFrame from ``ingest.load_jsonl``.
    current : dict[str, str]
        Latest verdict per axis (from ``verdicts.current_verdicts``).
    worst : str
        Worst current verdict string.
    """

    df: pd.DataFrame
    current: dict[str, str]
    worst: str


# ------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------
class Panel(Protocol):
    """Protocol that every monitor panel must satisfy."""

    def render(self, ctx: RenderContext) -> None:
        """Render the panel into the Streamlit layout.

        Parameters
        ----------
        ctx : RenderContext
            Pre-computed view data for this refresh cycle.
        """
        ...  # pragma: no cover


# ------------------------------------------------------------------
# Panels
# ------------------------------------------------------------------
class HeaderPanel:
    """Displays record count + global-worst verdict badge."""

    def render(self, ctx: RenderContext) -> None:
        """Render the header row.

        Parameters
        ----------
        ctx : RenderContext
            Current render context.
        """
        col_info, col_badge = st.columns([3, 1])
        badge = VERDICT_BADGE.get(ctx.worst, "⚪")
        col_info.markdown(f"**{len(ctx.df)} records loaded**")
        col_badge.markdown(
            f"<span style='font-size:2rem'>{badge} {ctx.worst}</span>",
            unsafe_allow_html=True,
        )


class KpiPanel:
    """Metric deltas for the four key health indicators."""

    _COLS: list[str] = [
        "loss_total",
        "update_ratio",
        "effective_rank",
        "conditioning_sensitivity",
    ]

    def render(self, ctx: RenderContext) -> None:
        """Render KPI metric tiles.

        Parameters
        ----------
        ctx : RenderContext
            Current render context.
        """
        present = [c for c in self._COLS if c in ctx.df.columns]
        if not present:
            return
        st.subheader("KPIs")
        cols = st.columns(len(present))
        for col, metric in zip(cols, present):
            self._render_kpi(col, ctx.df, metric)

    def _render_kpi(
        self,
        col: "st.delta_generator.DeltaGenerator",
        df: pd.DataFrame,
        metric: str,
    ) -> None:
        """Render one KPI tile.

        Parameters
        ----------
        col : DeltaGenerator
            Streamlit column object.
        df : pd.DataFrame
            Full windowed DataFrame.
        metric : str
            Column name to display.
        """
        series = df[metric].dropna()
        if series.empty:
            return
        last = series.iloc[-1]
        delta = (
            f"{last - series.iloc[-2]:.3g}"
            if len(series) >= 2
            else None
        )
        col.metric(label=metric, value=f"{last:.4g}", delta=delta)


class VerdictsPanel:
    """Status pills, heatmap, and drill-down for verdict axes."""

    def render(self, ctx: RenderContext) -> None:
        """Render the full verdicts section.

        Parameters
        ----------
        ctx : RenderContext
            Current render context.
        """
        st.subheader("Verdicts")
        self._render_pills(ctx.current)
        self._render_heatmap(ctx.df)
        self._render_drilldown(ctx.df, ctx.current)

    def _render_pills(self, current: dict[str, str]) -> None:
        """One status pill per visible verdict axis."""
        if not current:
            st.info("No verdict data yet.")
            return
        cols = st.columns(max(1, len(current)))
        for col, (axis, verdict) in zip(cols, current.items()):
            badge = VERDICT_BADGE.get(verdict, "⚪")
            col.metric(
                label=f"{badge} {axis}",
                value=verdict,
                help=AXIS_HELP.get(axis, axis),
            )

    def _render_heatmap(self, df: pd.DataFrame) -> None:
        """Verdict heatmap: axes × output_idx."""
        matrix = build_verdict_matrix(df)
        if matrix.empty:
            return
        axes = [c for c in matrix.columns if c != "output_idx"]
        if not axes:
            return
        x = (
            matrix["output_idx"].tolist()
            if "output_idx" in matrix.columns
            else list(range(len(matrix)))
        )
        z = _build_z(matrix, axes)
        fig = go.Figure(data=go.Heatmap(
            z=z, x=x, y=axes,
            colorscale=VERDICT_COLORSCALE,
            showscale=False,
            zmin=0, zmax=3,
        ))
        fig.update_layout(
            title="Verdict heatmap",
            xaxis_title="output_idx",
            template="plotly_dark",
            height=max(200, 120 + len(axes) * 40),
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_drilldown(
        self, df: pd.DataFrame, current: dict[str, str]
    ) -> None:
        """Metric curve for a user-selected verdict axis."""
        if not current:
            return
        axis = st.selectbox("Drill-down axis", list(current.keys()))
        if axis and axis in df.columns:
            fig = px.line(
                df, x="output_idx", y=axis,
                title=f"{axis} over time",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)


class LossPanel:
    """Multi-loss overlay, contribution stack, and limiting badge."""

    def render(self, ctx: RenderContext) -> None:
        """Render the full loss section.

        Parameters
        ----------
        ctx : RenderContext
            Current render context.
        """
        st.subheader("Losses")
        active = active_losses(ctx.df)
        if not active:
            st.info("No loss data yet.")
            return
        self._render_overlay(ctx.df, active)
        self._render_contribution(ctx.df)
        self._render_limiting(ctx.df)

    def _render_overlay(
        self, df: pd.DataFrame, active: list[str]
    ) -> None:
        """Overlay all active loss curves on one chart."""
        log_scale = st.checkbox("Log scale", key="loss_log")
        fig = go.Figure()
        for col in active:
            fig.add_trace(go.Scatter(
                x=df["output_idx"], y=df[col],
                mode="lines", name=col,
            ))
        if log_scale:
            fig.update_yaxes(type="log")
        fig.update_layout(
            title="Active losses",
            xaxis_title="output_idx",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_contribution(self, df: pd.DataFrame) -> None:
        """Stacked-area chart from loss_share.* values."""
        contrib = contribution(df)
        if contrib.empty:
            return
        share_cols = [c for c in contrib.columns if c != "output_idx"]
        x = (
            contrib["output_idx"].tolist()
            if "output_idx" in contrib.columns
            else list(range(len(contrib)))
        )
        fig = go.Figure()
        for col in share_cols:
            fig.add_trace(go.Scatter(
                x=x, y=contrib[col],
                stackgroup="one", name=col, mode="lines",
            ))
        fig.update_layout(
            title="Loss contribution (share)",
            xaxis_title="output_idx",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_limiting(self, df: pd.DataFrame) -> None:
        """Show limiting-loss badge when detectable."""
        lim = limiting_loss(df)
        if lim:
            st.info(f"🔒 Limiting loss: **{lim}**")


class RawTablePanel:
    """Collapsible raw JSONL table for ad-hoc inspection."""

    def render(self, ctx: RenderContext) -> None:
        """Render the raw data expander.

        Parameters
        ----------
        ctx : RenderContext
            Current render context.
        """
        with st.expander("Raw data", expanded=False):
            st.dataframe(ctx.df, use_container_width=True)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _build_z(
    matrix: pd.DataFrame,
    axes: list[str],
) -> list[list[int]]:
    """Build the Z matrix (list-of-lists) for the heatmap.

    Parameters
    ----------
    matrix : pd.DataFrame
        Output of ``verdicts.verdict_matrix``.
    axes : list[str]
        Axis column names (rows of the heatmap).

    Returns
    -------
    list[list[int]]
        Outer list = axes, inner list = output_idx values.
    """
    return [
        [
            VERDICT_CODE.get(str(matrix[axis].iloc[i]), 0)
            for i in range(len(matrix))
        ]
        for axis in axes
    ]
