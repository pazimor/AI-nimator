"""Phase A1 tests — controller health metrics + contract loading."""

from __future__ import annotations

from pathlib import Path

import torch

from ainimator.core.constants.controller import PhaseMode
from ainimator.core.types.controller import ControllerV2Config
from ainimator.health.contract import Verdict
from ainimator.health.controller_metrics import (
    CONTROLLER_CONTRACT_NAMES,
    controlSensitivity,
    loadControllerContracts,
    meanCollapse,
    postNormStats,
)
from ainimator.model.controller_v2 import MotionController

_HEALTH_YAML = Path("src/configs/health.yaml")


def _model() -> MotionController:
    config = ControllerV2Config(
        embedDim=64, numHeads=4, numLayers=2, contextFrames=1,
        phaseMode=PhaseMode.NONE,
    )
    return MotionController(config).eval()


def test_control_sensitivity_positive_for_varied_control() -> None:
    model = _model()
    bone = torch.randn(8, 1, 22, 6)
    glob = torch.randn(8, 1, 4)
    control = torch.randn(8, model.config.controlChannels)
    value = controlSensitivity(model, bone, control, globalWindow=glob)
    assert value > 0.0


def test_mean_collapse_returns_rank_and_sim() -> None:
    model = _model()
    bone = torch.randn(8, 1, 22, 6)
    glob = torch.randn(8, 1, 4)
    control = torch.randn(8, model.config.controlChannels)
    out = model(bone, control, globalWindow=glob)
    rank, sim = meanCollapse(out)
    assert 0.0 <= rank <= 1.0
    assert -1.0 <= sim <= 1.0


def test_post_norm_stats_zero_for_standard_normal() -> None:
    torch.manual_seed(0)
    standard = torch.randn(2000, 22, 6)
    assert postNormStats(standard) < 0.1


def test_post_norm_stats_flags_unnormalized() -> None:
    shifted = torch.randn(2000, 22, 6) * 5.0 + 3.0
    assert postNormStats(shifted) > 0.3


def test_load_controller_contracts_present() -> None:
    contracts = loadControllerContracts(_HEALTH_YAML)
    assert set(contracts) == set(CONTROLLER_CONTRACT_NAMES)


def test_mean_collapse_rank_calibrated_for_motion_deltas() -> None:
    """Recalibrated rank contract: healthy motion-delta rank ~0.05-0.15.

    A converged controller emits low-rank 135-dim deltas; the contract
    must accept ~0.077 (healthy, evidenced by control_sensitivity 0.54)
    yet still flag a genuine near-zero-rank collapse.
    """
    contracts = loadControllerContracts(_HEALTH_YAML)
    rankContract = contracts["mean_collapse_rank"]
    healthy = rankContract.evaluate({"mean_collapse_rank": 0.077})
    assert healthy.verdict is Verdict.OK
    collapsed = rankContract.evaluate({"mean_collapse_rank": 0.005})
    assert collapsed.verdict is Verdict.CRITICAL


def test_contracts_evaluate_healthy_metrics() -> None:
    contracts = loadControllerContracts(_HEALTH_YAML)
    healthy = {
        "control_sensitivity": 0.2,
        "mean_collapse_rank": 0.7,
        "mean_collapse_sim": 0.2,
        "rollout_drift": 0.001,
        "post_norm_stats": 0.02,
        "prompt_sensitivity": 0.05,
    }
    for name, contract in contracts.items():
        assert contract.evaluate(healthy).verdict is Verdict.OK, name
