"""Published JSONL contract for HealthHub records.

Single source of truth for field-name conventions shared between
the health producer (``hub.py``) and any consumer (e.g.
``tools/monitor``).  Pure Python — no torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Re-export so consumers import only this module.
from ainimator.health.contract import Verdict

__all__ = [
    "STEP_KEY",
    "VERDICT_PREFIX",
    "LOSS_PREFIX",
    "LOSS_SHARE_PREFIX",
    "EVERY_STEPS_DEFAULT",
    "Verdict",
    "ParsedRecord",
    "verdict_axis",
    "is_active_loss",
    "group_record",
]

# ------------------------------------------------------------------
# Key / prefix constants
# ------------------------------------------------------------------
STEP_KEY: str = "step"
VERDICT_PREFIX: str = "verdict."
LOSS_PREFIX: str = "loss_"
LOSS_SHARE_PREFIX: str = "loss_share."
EVERY_STEPS_DEFAULT: int = 50


# ------------------------------------------------------------------
# Structured record
# ------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedRecord:
    """Structured view of one JSONL line.

    Attributes
    ----------
    step : int
        Value of the ``step`` key.
    verdicts : dict[str, str]
        Axis name → verdict string,
        e.g. ``{"update_ratio": "WARNING"}``.
    losses : dict[str, float]
        Loss column name → value,
        e.g. ``{"loss_bone": 0.403}``.
    loss_shares : dict[str, float]
        Component name → share fraction,
        e.g. ``{"bone": 0.066}``.
    metrics : dict[str, float]
        All remaining numeric keys.
    raw : dict[str, Any]
        Original record dict, unmodified.
    """

    step: int
    verdicts: dict[str, str]
    losses: dict[str, float]
    loss_shares: dict[str, float]
    metrics: dict[str, float]
    raw: dict[str, Any]


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------
def verdict_axis(key: str) -> str | None:
    """Extract the axis name from a verdict key, or None.

    Parameters
    ----------
    key : str
        A JSONL key, e.g. ``"verdict.update_ratio"``.

    Returns
    -------
    str or None
        ``"update_ratio"`` for a verdict key, None otherwise.

    Examples
    --------
    >>> verdict_axis("verdict.update_ratio")
    'update_ratio'
    >>> verdict_axis("loss_bone") is None
    True
    """
    if key.startswith(VERDICT_PREFIX):
        return key[len(VERDICT_PREFIX):]
    return None


def is_active_loss(values: list[float]) -> bool:
    """Return True when a loss component has at least one non-zero value.

    Parameters
    ----------
    values : list[float]
        Observed values across time steps.

    Returns
    -------
    bool
        True when at least one value differs from zero.

    Examples
    --------
    >>> is_active_loss([0.0, 0.0])
    False
    >>> is_active_loss([0.0, 0.4])
    True
    """
    return any(v != 0.0 for v in values)


def group_record(raw: dict[str, Any]) -> ParsedRecord:
    """Parse a raw JSONL dict into a structured ParsedRecord.

    Classifies each key by prefix into one of: verdicts,
    loss_shares, losses, or plain metrics.

    Parameters
    ----------
    raw : dict[str, Any]
        One parsed JSON line from ``health.jsonl``.

    Returns
    -------
    ParsedRecord
        Grouped, structured view of the record.
    """
    step = int(raw.get(STEP_KEY, 0))
    verdicts: dict[str, str] = {}
    losses: dict[str, float] = {}
    loss_shares: dict[str, float] = {}
    metrics: dict[str, float] = {}

    for key, value in raw.items():
        if key == STEP_KEY:
            continue
        axis = verdict_axis(key)
        if axis is not None:
            verdicts[axis] = str(value)
        elif key.startswith(LOSS_SHARE_PREFIX):
            suffix = key[len(LOSS_SHARE_PREFIX):]
            if isinstance(value, (int, float)):
                loss_shares[suffix] = float(value)
        elif key == "loss_share":
            # Bare loss_share summary key — treat as a plain metric.
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        elif key.startswith(LOSS_PREFIX):
            if isinstance(value, (int, float)):
                losses[key] = float(value)
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)

    return ParsedRecord(
        step=step,
        verdicts=verdicts,
        losses=losses,
        loss_shares=loss_shares,
        metrics=metrics,
        raw=raw,
    )
