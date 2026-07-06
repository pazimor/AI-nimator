"""Tests for ainimator.health.record_schema.

Covers: prefix constants, group_record parsing, verdict_axis,
is_active_loss, and a byte-level non-regression check that
hub.py still writes ``"verdict."`` (not a renamed prefix).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ainimator.health.record_schema import (
    EVERY_STEPS_DEFAULT,
    LOSS_PREFIX,
    LOSS_SHARE_PREFIX,
    STEP_KEY,
    VERDICT_PREFIX,
    Verdict,
    group_record,
    is_active_loss,
    verdict_axis,
)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
def test_verdict_prefix_value() -> None:
    """VERDICT_PREFIX must equal the literal used in existing JSONLs."""
    assert VERDICT_PREFIX == "verdict."


def test_loss_prefix_value() -> None:
    assert LOSS_PREFIX == "loss_"


def test_loss_share_prefix_value() -> None:
    assert LOSS_SHARE_PREFIX == "loss_share."


def test_step_key_value() -> None:
    assert STEP_KEY == "step"


def test_every_steps_default() -> None:
    assert EVERY_STEPS_DEFAULT == 50


# ------------------------------------------------------------------
# verdict_axis
# ------------------------------------------------------------------
def test_verdict_axis_strips_prefix() -> None:
    assert verdict_axis("verdict.update_ratio") == "update_ratio"


def test_verdict_axis_none_for_non_verdict() -> None:
    assert verdict_axis("loss_bone") is None
    assert verdict_axis("step") is None


# ------------------------------------------------------------------
# is_active_loss
# ------------------------------------------------------------------
def test_is_active_loss_all_zeros() -> None:
    assert not is_active_loss([0.0, 0.0, 0.0])


def test_is_active_loss_has_nonzero() -> None:
    assert is_active_loss([0.0, 0.4, 0.0])


def test_is_active_loss_empty() -> None:
    assert not is_active_loss([])


# ------------------------------------------------------------------
# group_record
# ------------------------------------------------------------------
_SAMPLE: dict = {
    "step": 50,
    "loss_total": 6.048,
    "loss_bone": 0.403,
    "loss_share.bone": 0.066,
    "loss_share": 0.038,
    "verdict.update_ratio": "WARNING",
    "verdict.post_norm_stats": "UNKNOWN",
    "conditioning_sensitivity": 0.00017,
}


def test_group_record_step() -> None:
    parsed = group_record(_SAMPLE)
    assert parsed.step == 50


def test_group_record_verdicts() -> None:
    parsed = group_record(_SAMPLE)
    assert parsed.verdicts == {
        "update_ratio": "WARNING",
        "post_norm_stats": "UNKNOWN",
    }


def test_group_record_losses() -> None:
    parsed = group_record(_SAMPLE)
    assert "loss_total" in parsed.losses
    assert "loss_bone" in parsed.losses
    # loss_share.* and loss_share bare must NOT be in losses
    assert "loss_share.bone" not in parsed.losses
    assert "loss_share" not in parsed.losses


def test_group_record_loss_shares() -> None:
    parsed = group_record(_SAMPLE)
    assert parsed.loss_shares == {"bone": 0.066}


def test_group_record_bare_loss_share_in_metrics() -> None:
    """Bare 'loss_share' key goes to metrics, not losses."""
    parsed = group_record(_SAMPLE)
    assert "loss_share" in parsed.metrics


def test_group_record_metrics() -> None:
    parsed = group_record(_SAMPLE)
    assert "conditioning_sensitivity" in parsed.metrics


def test_group_record_raw_preserved() -> None:
    parsed = group_record(_SAMPLE)
    assert parsed.raw is _SAMPLE


# ------------------------------------------------------------------
# Non-regression: hub.py JSONL output is byte-identical before/after
# using VERDICT_PREFIX (P0 refactor).
# ------------------------------------------------------------------
def test_hub_verdict_key_matches_prefix(tmp_path: Path) -> None:
    """hub.step() writes 'verdict.<name>' — not a renamed prefix."""
    from unittest.mock import MagicMock, patch

    import torch

    from ainimator.health.contract import Contract, ContractResult, Verdict
    from ainimator.health.hub import HealthHub
    from ainimator.health.probe import Probe

    contract = MagicMock(spec=Contract)
    contract.evaluate.return_value = ContractResult(
        name="update_ratio",
        metric="update_ratio",
        value=1e-3,
        verdict=Verdict.OK,
        message="ok",
    )

    hub = HealthHub(
        outputDir=tmp_path,
        contracts=[contract],
        probes=[],
        everySteps=1,
    )
    hub.step(globalStep=1, metrics={"update_ratio": 1e-3})

    jsonl_path = tmp_path / "health" / "health.jsonl"
    line = json.loads(jsonl_path.read_text().strip())
    assert "verdict.update_ratio" in line, (
        "hub.step() must write 'verdict.update_ratio', got: "
        + str(list(line.keys()))
    )
