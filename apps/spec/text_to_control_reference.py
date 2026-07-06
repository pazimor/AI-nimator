"""Reference implementation of the B6 text-to-control mapper.

Normative counterpart of ``text_to_control.md`` §2, driven by the
canonical table ``text_to_control.json``. The engine plugins mirror
this algorithm value-for-value; the shared test cases live in
``test/apps/test_text_to_control.py`` and are duplicated in the
EditMode / Automation suites.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_TABLE_PATH = Path(__file__).parent / "text_to_control.json"
_TOKEN_PATTERN = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class ResolvedControl:
    """A successfully resolved raw control vector.

    Attributes
    ----------
    vx, vz : float
        Raw desired velocity, meters/frame, root-local frame.
    aimX, aimZ : float
        Unit aim direction (movement direction; forward when idle).
    """

    vx: float
    vz: float
    aimX: float
    aimZ: float


def _normalizeText(text: str) -> list[str]:
    """Lowercase, strip accents, split into alphabetic tokens."""
    lowered = unicodedata.normalize("NFKD", text.lower())
    ascii_ = lowered.encode("ascii", "ignore").decode("ascii")
    return _TOKEN_PATTERN.findall(ascii_)


def resolveTextCommand(
    text: str,
    tablePath: Path = _TABLE_PATH,
) -> ResolvedControl | None:
    """Resolve a free-text command into a raw control vector.

    Parameters
    ----------
    text : str
        Free-text command, French or English (e.g. "cours vers la
        gauche").
    tablePath : Path
        Canonical keyword table (defaults to the spec copy).

    Returns
    -------
    ResolvedControl or None
        ``None`` when nothing was recognized or the directions cancel
        out (ambiguous) — the caller keeps its current control.
    """
    table = json.loads(tablePath.read_text(encoding="utf-8"))
    tokens = _normalizeText(text)
    direction = _resolveDirection(tokens, table)
    speed = _resolveSpeed(tokens, table)
    if direction is None and speed is None:
        return None
    if direction == (0.0, 0.0):
        return None  # direction words present but cancelling: ambiguous
    if speed is None:
        speed = float(table["defaults"]["speed_when_direction_only"])
    if direction is None:
        default = table["defaults"]["direction_when_speed_only"]
        direction = (float(default[0]), float(default[1]))
    if speed == 0.0:
        return ResolvedControl(vx=0.0, vz=0.0, aimX=0.0, aimZ=1.0)
    return ResolvedControl(
        vx=direction[0] * speed,
        vz=direction[1] * speed,
        aimX=direction[0],
        aimZ=direction[1],
    )


def _resolveDirection(
    tokens: list[str], table: dict
) -> tuple[float, float] | None:
    """Sum direction keywords; unit-normalize; None when absent."""
    directions = table["directions"]
    sumX = sumZ = 0.0
    found = False
    for token in tokens:
        if token in directions:
            found = True
            sumX += float(directions[token][0])
            sumZ += float(directions[token][1])
    if not found:
        return None
    norm = math.hypot(sumX, sumZ)
    if norm < 1e-9:
        return (0.0, 0.0)
    return (sumX / norm, sumZ / norm)


def _resolveSpeed(tokens: list[str], table: dict) -> float | None:
    """Max speed keyword; a zero-speed (stop) keyword always wins."""
    speeds = table["speeds"]
    best: float | None = None
    for token in tokens:
        if token in speeds:
            value = float(speeds[token])
            if value == 0.0:
                return 0.0
            best = value if best is None else max(best, value)
    return best
