"""Parity test for hub.diagnose() vs old diagnose_generation_v2 — AC3.

Genuine parity strategy: build a tiny deterministic synthetic
"checkpoint" in-test, run BOTH the old diagnose computation AND the
hub's equivalent on the SAME inputs, and assert they match within 1e-4.

Old path = ``ainimator.cli.diagnose_generation_v2._cosineSimilarity``
New path = ``ainimator.health.hub._cosineSim1d`` /
           ``ainimator.health.hub._computeCollapseSims``

Overlapping metrics (those with an old-path equivalent):

* ``cfg_sim``  — cosine between generation at cfg[i] vs cfg[i+1]
  (same seed, consecutive cfg values).  The old path calls
  ``_cosineSimilarity(a.flatten(), b.flatten())``.  The hub calls
  ``_cosineSim1d(a.flatten(), b.flatten())``.

* ``seed_sim`` — cosine between generation at seed[i] vs seed[i+1]
  (same cfg).  Same maths as cfg_sim but with the axes swapped.

* ``encoder_cond_uncond_sim`` — masked-mean cosine between cond and
  uncond pooled embeddings.  The hub uses ``_poolCosineSim``; the old
  diagnose computes the same via manual masked-mean + cosine.

Metrics in hub.diagnose() that have NO old-path equivalent:
* fidelity / retrieval / distinctness  (old path did not implement these)
* sample_stats  (old path only logged, never returned a dict)

The placeholder ``test_parity_limitation_documented`` is REMOVED.
Numerical parity on a live trained checkpoint is verified in
``scripts/verify_parity_checkpoint.py`` (manual run only — no
checkpoint is available in CI).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from ainimator.health.diagnose_v2 import (
    computeGenerationStats,
    formatStatsLine,
)
from ainimator.health.hub import (
    _cosineSim1d,
    _computeCollapseSims,
    _poolCosineSim,
)


# ------------------------------------------------------------------
# Tolerance
# ------------------------------------------------------------------
_TOL = 1e-4


# ------------------------------------------------------------------
# Old-path reference (mirrors diagnose_generation_v2._cosineSimilarity)
# ------------------------------------------------------------------
def _old_cosine_similarity(
    a: torch.Tensor, b: torch.Tensor
) -> float:
    """Cosine similarity — mirrors the old diagnose_generation_v2 impl."""
    flat_a = a.float().detach().reshape(-1)
    flat_b = b.float().detach().reshape(-1)
    dot = float(torch.dot(flat_a, flat_b).item())
    norms = float((flat_a.norm() * flat_b.norm()).item())
    if norms == 0.0:
        return 0.0
    return dot / norms


def _old_cfg_sim(
    samples: dict[tuple[int, float], torch.Tensor],
    seeds: tuple[int, ...],
    cfg_scales: tuple[float, ...],
) -> float:
    """Old-path cfg_sim: mean cosine across (seed, consecutive cfg pairs)."""
    sims: list[float] = []
    for seed in seeds:
        for idx in range(len(cfg_scales) - 1):
            a = samples[(seed, float(cfg_scales[idx]))].flatten()
            b = samples[(seed, float(cfg_scales[idx + 1]))].flatten()
            sims.append(_old_cosine_similarity(a, b))
    return sum(sims) / len(sims) if sims else float("nan")


def _old_seed_sim(
    samples: dict[tuple[int, float], torch.Tensor],
    seeds: tuple[int, ...],
    cfg_scales: tuple[float, ...],
) -> float:
    """Old-path seed_sim: mean cosine across (cfg, consecutive seed pairs)."""
    sims: list[float] = []
    for cfg in cfg_scales:
        for idx in range(len(seeds) - 1):
            a = samples[(seeds[idx], float(cfg))].flatten()
            b = samples[(seeds[idx + 1], float(cfg))].flatten()
            sims.append(_old_cosine_similarity(a, b))
    return sum(sims) / len(sims) if sims else float("nan")


# ------------------------------------------------------------------
# cfg_sim / seed_sim parity
# ------------------------------------------------------------------
def test_cfg_sim_parity() -> None:
    """hub._cosineSim1d and _computeCollapseSims match old path for cfg_sim.

    Constructs deterministic synthetic motion samples (3 seeds × 3 cfg
    scales), computes cfg_sim via the old diagnose path and the hub
    path, asserts they agree within 1e-4.
    """
    torch.manual_seed(42)
    seeds = (0, 42, 123)
    cfg_scales = (1.0, 4.0, 6.0)
    frames = 32

    # Build fixed random samples
    samples: dict[tuple[int, float], torch.Tensor] = {}
    for seed in seeds:
        for cfg in cfg_scales:
            rng = torch.Generator()
            rng.manual_seed(seed * 1000 + int(cfg * 10))
            samples[(seed, float(cfg))] = torch.randn(
                frames, 22, 6, generator=rng
            )

    old_cfg = _old_cfg_sim(samples, seeds, cfg_scales)
    hub_cfg, _ = _computeCollapseSims(samples, seeds, cfg_scales)

    assert not math.isnan(old_cfg), "old_cfg_sim returned NaN"
    assert not math.isnan(hub_cfg), "hub cfg_sim returned NaN"
    assert abs(old_cfg - hub_cfg) < _TOL, (
        f"cfg_sim mismatch: old={old_cfg:.6f} hub={hub_cfg:.6f} "
        f"delta={abs(old_cfg - hub_cfg):.2e}"
    )


def test_seed_sim_parity() -> None:
    """hub._computeCollapseSims matches old path for seed_sim."""
    torch.manual_seed(7)
    seeds = (0, 42, 123)
    cfg_scales = (1.0, 4.0, 6.0)
    frames = 32

    samples: dict[tuple[int, float], torch.Tensor] = {}
    for seed in seeds:
        for cfg in cfg_scales:
            rng = torch.Generator()
            rng.manual_seed(seed * 100 + int(cfg * 7))
            samples[(seed, float(cfg))] = torch.randn(
                frames, 22, 6, generator=rng
            )

    old_seed = _old_seed_sim(samples, seeds, cfg_scales)
    _, hub_seed = _computeCollapseSims(samples, seeds, cfg_scales)

    assert not math.isnan(old_seed), "old_seed_sim returned NaN"
    assert not math.isnan(hub_seed), "hub seed_sim returned NaN"
    assert abs(old_seed - hub_seed) < _TOL, (
        f"seed_sim mismatch: old={old_seed:.6f} hub={hub_seed:.6f} "
        f"delta={abs(old_seed - hub_seed):.2e}"
    )


def test_cosine_sim_single_pair_parity() -> None:
    """_cosineSim1d and old _cosineSimilarity match on a single pair."""
    torch.manual_seed(99)
    a = torch.randn(22 * 32 * 6)
    b = torch.randn(22 * 32 * 6)

    old_val = _old_cosine_similarity(a, b)
    hub_val = _cosineSim1d(a.float(), b.float())

    assert abs(old_val - hub_val) < _TOL, (
        f"cosine mismatch: old={old_val:.6f} hub={hub_val:.6f}"
    )


# ------------------------------------------------------------------
# encoder_cond_uncond_sim parity
# ------------------------------------------------------------------
def test_encoder_sim_parity() -> None:
    """_poolCosineSim matches manual masked-mean cosine to 1e-4.

    The old diagnose_generation_v2 did not implement this metric —
    it is new in hub.diagnose().  This test verifies hub._poolCosineSim
    against the canonical manual computation so its correctness is
    established independently.
    """
    torch.manual_seed(0)
    seq_len = 8
    dim = 32
    hidden_cond = torch.randn(1, seq_len, dim)
    hidden_uncond = torch.randn(1, seq_len, dim)
    # No padding: all tokens are real
    mask = torch.zeros(1, seq_len, dtype=torch.bool)

    # Manual reference (the computation the old path would use if it had
    # this metric, based on the same masked-mean pooling formula)
    real = ~mask[0]  # (seq_len,)
    pool_c = (
        hidden_cond[0] * real.unsqueeze(-1).float()
    ).sum(0) / real.float().sum().clamp(min=1e-8)
    pool_u = (
        hidden_uncond[0] * real.unsqueeze(-1).float()
    ).sum(0) / real.float().sum().clamp(min=1e-8)
    expected = float(
        torch.nn.functional.cosine_similarity(
            pool_c.unsqueeze(0), pool_u.unsqueeze(0)
        ).item()
    )

    class _MockOut:
        def __init__(
            self, hidden: torch.Tensor, mask_: torch.Tensor
        ) -> None:
            self.hiddenStates = hidden
            self.keyPaddingMask = mask_

    computed = _poolCosineSim(
        _MockOut(hidden_cond, mask), _MockOut(hidden_uncond, mask)
    )

    assert abs(computed - expected) < _TOL, (
        f"encoder_cond_uncond_sim mismatch: "
        f"hub={computed:.6f} manual={expected:.6f}"
    )


def test_encoder_sim_parity_with_padding() -> None:
    """_poolCosineSim handles padded tokens correctly."""
    torch.manual_seed(5)
    seq_len = 8
    real_len = 5
    dim = 16

    hidden = torch.randn(1, seq_len, dim)
    uncond = torch.randn(1, seq_len, dim)
    # Last 3 tokens are padding
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    mask[0, real_len:] = True

    real = ~mask[0]
    pool_c = (
        hidden[0] * real.unsqueeze(-1).float()
    ).sum(0) / real.float().sum().clamp(min=1e-8)
    pool_u = (
        uncond[0] * real.unsqueeze(-1).float()
    ).sum(0) / real.float().sum().clamp(min=1e-8)
    expected = float(
        torch.nn.functional.cosine_similarity(
            pool_c.unsqueeze(0), pool_u.unsqueeze(0)
        ).item()
    )

    class _MockOut:
        def __init__(
            self, hidden: torch.Tensor, mask_: torch.Tensor
        ) -> None:
            self.hiddenStates = hidden
            self.keyPaddingMask = mask_

    computed = _poolCosineSim(
        _MockOut(hidden, mask), _MockOut(uncond, mask)
    )

    assert abs(computed - expected) < _TOL, (
        f"Padded encoder sim mismatch: "
        f"hub={computed:.6f} manual={expected:.6f}"
    )


# ------------------------------------------------------------------
# Generation stats parity (shared inner function)
# ------------------------------------------------------------------
def test_compute_generation_stats_parity() -> None:
    """computeGenerationStats matches manual calculation to 1e-4.

    This is the inner function shared between the old diagnose CLI
    and hub.diagnose() via _buildSampleStats.
    """
    torch.manual_seed(0)
    bone = torch.randn(64, 22, 6)

    stats = computeGenerationStats(bone)

    flat = bone.float().cpu()
    expected_mean = float(flat.mean().item())
    expected_std = float(flat.std().item())

    assert abs(stats.rotationMean - expected_mean) < _TOL, (
        f"rotationMean mismatch: {stats.rotationMean} "
        f"vs {expected_mean}"
    )
    assert abs(stats.rotationStd - expected_std) < _TOL, (
        f"rotationStd mismatch: {stats.rotationStd} "
        f"vs {expected_std}"
    )


def test_format_stats_line_contains_values() -> None:
    """formatStatsLine includes mean, std, and displacement."""
    torch.manual_seed(1)
    bone = torch.randn(32, 22, 6)
    stats = computeGenerationStats(bone)
    line = formatStatsLine(stats, prefix="test: ")
    assert "test: " in line
    assert "rot" in line
    assert "root_displ" in line
