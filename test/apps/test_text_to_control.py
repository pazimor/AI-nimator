"""Shared test cases for the B6 text-to-control mapper (reference).

These cases are the cross-engine parity set: the Unity EditMode and
Unreal Automation suites duplicate them verbatim — same phrases, same
expected vectors (text_to_control.md §5).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "apps" / "spec"))
from text_to_control_reference import resolveTextCommand  # noqa: E402

_WALK = 0.033
_RUN = 0.1
_DIAG = 1.0 / math.sqrt(2.0)


@pytest.mark.parametrize("text,vx,vz", [
    ("cours vers la gauche", -_RUN, 0.0),
    ("run left", -_RUN, 0.0),
    ("walk forward", 0.0, _WALK),
    ("avance", 0.0, _WALK),
    ("recule lentement", 0.0, -0.015),
    ("sprint", 0.0, 0.15),
    ("marche a droite", _WALK, 0.0),
    ("run forward and left", -_RUN * _DIAG, _RUN * _DIAG),
    ("cours lentement", 0.0, _RUN),
])
def test_resolves_expected_vector(text: str, vx: float, vz: float) -> None:
    """Phrase → exact raw control vector (parity reference values)."""
    result = resolveTextCommand(text)
    assert result is not None, text
    assert result.vx == pytest.approx(vx, abs=1e-9)
    assert result.vz == pytest.approx(vz, abs=1e-9)


def test_stop_always_wins_and_zeroes_control() -> None:
    """A stop-family word forces (0,0) whatever else is present."""
    result = resolveTextCommand("cours vite et arrete a gauche")
    assert result is not None
    assert (result.vx, result.vz) == (0.0, 0.0)
    assert (result.aimX, result.aimZ) == (0.0, 1.0)


def test_aim_follows_movement_direction() -> None:
    """aim = unit movement direction on a resolved command."""
    result = resolveTextCommand("run left")
    assert result is not None
    assert (result.aimX, result.aimZ) == (-1.0, 0.0)


@pytest.mark.parametrize("text", [
    "bonjour tout le monde",
    "",
    "gauche droite",
])
def test_unresolvable_returns_none(text: str) -> None:
    """Unknown or cancelling phrases resolve to None (keep control)."""
    assert resolveTextCommand(text) is None


def test_accents_are_stripped() -> None:
    """'arrête' resolves like 'arrete' (accent-insensitive)."""
    result = resolveTextCommand("arrête-toi")
    assert result is not None
    assert (result.vx, result.vz) == (0.0, 0.0)
