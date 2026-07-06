"""Color palette and display constants for the health monitor."""

from __future__ import annotations

from ainimator.health.record_schema import Verdict

# ------------------------------------------------------------------
# Status colors (hex, dark-theme compatible)
# ------------------------------------------------------------------
STATUS_COLORS: dict[str, str] = {
    Verdict.CRITICAL.value: "#EF5350",
    Verdict.WARNING.value: "#FFA726",
    Verdict.OK.value: "#66BB6A",
    Verdict.UNKNOWN.value: "#9E9E9E",
}

# ------------------------------------------------------------------
# Numeric codes for heatmap color-mapping
# ------------------------------------------------------------------
VERDICT_CODE: dict[str, int] = {
    Verdict.UNKNOWN.value: 0,
    Verdict.OK.value: 1,
    Verdict.WARNING.value: 2,
    Verdict.CRITICAL.value: 3,
}

# Plotly colorscale: [[fraction, color], ...]
VERDICT_COLORSCALE: list[list[object]] = [
    [0.00, STATUS_COLORS[Verdict.UNKNOWN.value]],
    [0.34, STATUS_COLORS[Verdict.OK.value]],
    [0.67, STATUS_COLORS[Verdict.WARNING.value]],
    [1.00, STATUS_COLORS[Verdict.CRITICAL.value]],
]

# ------------------------------------------------------------------
# Badge emojis
# ------------------------------------------------------------------
VERDICT_BADGE: dict[str, str] = {
    Verdict.CRITICAL.value: "🔴",
    Verdict.WARNING.value: "🟠",
    Verdict.OK.value: "🟢",
    Verdict.UNKNOWN.value: "⚪",
}

# ------------------------------------------------------------------
# Tooltip help text per verdict axis
# ------------------------------------------------------------------
AXIS_HELP: dict[str, str] = {
    "conditioning_sensitivity": (
        "Δloss real vs shuffled text — should be > 0"
    ),
    "update_ratio": (
        "||Δw|| / ||w|| per step — healthy range 1e-4 to 1e-2"
    ),
    "loss_decomposition": (
        "share of each loss component — 1–70 % is healthy"
    ),
    "intra_batch_sim": (
        "cosine mean hidden states intra-batch — < 0.5 healthy"
    ),
    "effective_rank": (
        "normalised rank of hidden states — > 0.5 healthy"
    ),
}
