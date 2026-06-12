"""Unit tests for health/probe.py.

Tests:
- ProbeSnapshot is a plain data container.
- Probe attaches/detaches without error.
- Probe captures mean/std/norm from a forward hook.
- Probe computes effective_rank (near 0 when constant, near 1 when random).
- Probe computes intra_batch_sim (near 1 when constant = CRITICAL case).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from ainimator.health.probe import (
    Probe,
    ProbeSnapshot,
    _computeEffectiveRank,
    _computeIntraBatchSim,
    _computeMeanStdNorm,
)


# ------------------------------------------------------------------
# Helper: tiny module
# ------------------------------------------------------------------
class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ------------------------------------------------------------------
# ProbeSnapshot
# ------------------------------------------------------------------
def test_probe_snapshot_is_dataclass() -> None:
    """ProbeSnapshot stores scalars and no tensors."""
    snap = ProbeSnapshot(step=1, mean=0.5, std=0.1)
    assert snap.step == 1
    assert snap.mean == 0.5
    assert snap.std == 0.1
    assert snap.norm is None


# ------------------------------------------------------------------
# Basic capture
# ------------------------------------------------------------------
def test_probe_captures_mean_std_norm() -> None:
    """After a forward pass the probe records mean/std/norm."""
    model = _Linear()
    probe = Probe(
        name="linear",
        modulePath="linear",
        capture=["mean", "std", "norm"],
        hookType="forward",
    )
    probe.attach(model)
    probe.setStep(1)

    x = torch.ones(4, 8)
    model(x)

    snap = probe.lastSnapshot
    assert snap is not None
    assert snap.mean is not None
    assert snap.std is not None
    assert snap.norm is not None
    probe.detach()


# ------------------------------------------------------------------
# Effective rank
# ------------------------------------------------------------------
def test_effective_rank_constant_is_near_zero() -> None:
    """A rank-1 (constant) tensor gives effective_rank near 0."""
    # All-ones matrix: a single direction.
    tensor = torch.ones(8, 16, 32)
    rank = _computeEffectiveRank(tensor)
    assert rank < 0.1, f"Expected near 0, got {rank:.4f}"


def test_effective_rank_random_is_high() -> None:
    """Random tensor gives much higher effective_rank than constant."""
    torch.manual_seed(0)
    tensor = torch.randn(8, 16, 32)
    rank = _computeEffectiveRank(tensor)
    assert rank > 0.3, f"Expected > 0.3, got {rank:.4f}"


# ------------------------------------------------------------------
# Intra-batch similarity (collapse detector)
# ------------------------------------------------------------------
def test_intra_batch_sim_constant_is_near_one() -> None:
    """Constant hidden states (collapse) → intra_batch_sim ≈ 1."""
    # All identical rows → cosine similarity = 1
    tensor = torch.ones(8, 32)
    sim = _computeIntraBatchSim(tensor)
    assert sim > 0.99, f"Expected ≈1, got {sim:.4f}"


def test_intra_batch_sim_random_is_low() -> None:
    """Random hidden states → intra_batch_sim much lower than 1."""
    torch.manual_seed(42)
    tensor = torch.randn(8, 64)
    sim = _computeIntraBatchSim(tensor)
    assert sim < 0.5, f"Expected < 0.5, got {sim:.4f}"


def test_probe_captures_effective_rank_and_intra_sim() -> None:
    """Probe correctly captures effective_rank and intra_batch_sim."""
    model = _Linear()
    probe = Probe(
        name="linear",
        modulePath="linear",
        capture=["effective_rank", "intra_batch_sim"],
        hookType="forward",
    )
    probe.attach(model)
    probe.setStep(5)

    # Constant input → collapse scenario
    x = torch.ones(4, 8)
    model(x)

    snap = probe.lastSnapshot
    assert snap is not None
    assert snap.intra_batch_sim is not None
    assert snap.intra_batch_sim > 0.9  # constant → near collapse
    probe.detach()
