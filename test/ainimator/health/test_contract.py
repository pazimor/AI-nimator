"""Unit tests for health/contract.py.

Tests:
- Contract.fromYamlDict parses correctly.
- lower_is_better produces correct verdicts.
- higher_is_better produces correct verdicts.
- range direction produces correct verdicts.
- Missing metric → UNKNOWN verdict.
"""

from __future__ import annotations

import pytest

from ainimator.health.contract import Contract, Verdict


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _lowerContract(
    ok: float = 0.95, warn: float = 0.96, crit: float = 0.99
) -> Contract:
    return Contract(
        name="test_lower",
        metric="cfg_sim",
        direction="lower_is_better",
        thresholds={
            "ok_threshold": ok,
            "warning_threshold": warn,
            "critical_above": crit,
        },
    )


def _higherContract() -> Contract:
    return Contract(
        name="test_higher",
        metric="cond_sens",
        direction="higher_is_better",
        thresholds={
            "ok_threshold": 0.01,
            "warning_threshold": 0.001,
            "critical_below": 0.001,
        },
    )


def _rangeContract() -> Contract:
    return Contract(
        name="test_range",
        metric="update_ratio",
        direction="range",
        thresholds={
            "critical_below": 1e-7,
            "critical_above": 1.0,
            "warning_min": 1e-5,
            "warning_max": 0.1,
        },
    )


# ------------------------------------------------------------------
# lower_is_better
# ------------------------------------------------------------------
def test_lower_ok() -> None:
    c = _lowerContract()
    result = c.evaluate({"cfg_sim": 0.50})
    assert result.verdict == Verdict.OK


def test_lower_warning() -> None:
    c = _lowerContract()
    result = c.evaluate({"cfg_sim": 0.97})
    assert result.verdict == Verdict.WARNING


def test_lower_critical() -> None:
    c = _lowerContract()
    result = c.evaluate({"cfg_sim": 0.9998})
    assert result.verdict == Verdict.CRITICAL


# ------------------------------------------------------------------
# higher_is_better
# ------------------------------------------------------------------
def test_higher_ok() -> None:
    c = _higherContract()
    result = c.evaluate({"cond_sens": 0.05})
    assert result.verdict == Verdict.OK


def test_higher_warning() -> None:
    c = _higherContract()
    result = c.evaluate({"cond_sens": 0.005})
    assert result.verdict == Verdict.WARNING


def test_higher_critical() -> None:
    c = _higherContract()
    result = c.evaluate({"cond_sens": 0.0001})
    assert result.verdict == Verdict.CRITICAL


# ------------------------------------------------------------------
# range
# ------------------------------------------------------------------
def test_range_ok() -> None:
    c = _rangeContract()
    result = c.evaluate({"update_ratio": 1e-3})
    assert result.verdict == Verdict.OK


def test_range_warning_too_low() -> None:
    c = _rangeContract()
    result = c.evaluate({"update_ratio": 1e-6})
    assert result.verdict == Verdict.WARNING


def test_range_critical_below() -> None:
    c = _rangeContract()
    result = c.evaluate({"update_ratio": 1e-9})
    assert result.verdict == Verdict.CRITICAL


def test_range_critical_above() -> None:
    c = _rangeContract()
    result = c.evaluate({"update_ratio": 5.0})
    assert result.verdict == Verdict.CRITICAL


# ------------------------------------------------------------------
# Missing metric
# ------------------------------------------------------------------
def test_unknown_when_metric_missing() -> None:
    c = _lowerContract()
    result = c.evaluate({"other_metric": 0.5})
    assert result.verdict == Verdict.UNKNOWN


# ------------------------------------------------------------------
# fromYamlDict
# ------------------------------------------------------------------
def test_from_yaml_dict() -> None:
    data = {
        "name": "cond_sim",
        "metric": "cfg_sim",
        "direction": "lower_is_better",
        "ok_threshold": 0.95,
        "warning_threshold": 0.99,
        "critical_above": 0.99,
    }
    c = Contract.fromYamlDict(data)
    assert c.name == "cond_sim"
    assert c.metric == "cfg_sim"
    assert c.direction == "lower_is_better"
    result = c.evaluate({"cfg_sim": 0.999})
    assert result.verdict == Verdict.CRITICAL
