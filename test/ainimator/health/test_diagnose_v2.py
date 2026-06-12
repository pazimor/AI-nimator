"""Unit tests for :mod:`src.features.generation.diagnose_v2`."""

from __future__ import annotations

import math

import pytest
import torch

from ainimator.health.diagnose_v2 import (
    GenerationStats,
    computeGenerationStats,
    formatStatsLine,
)


# ---------------------------------------------------------------------
# Shape & contract
# ---------------------------------------------------------------------
def test_compute_stats_returns_dataclass() -> None:
    bone = torch.randn(16, 22, 6)
    rt = torch.randn(16, 3)
    stats = computeGenerationStats(bone, rt)
    assert isinstance(stats, GenerationStats)


def test_compute_stats_rejects_non_3d_bone() -> None:
    with pytest.raises(ValueError, match="3-D"):
        computeGenerationStats(torch.randn(22, 6))


def test_compute_stats_handles_missing_global() -> None:
    bone = torch.randn(8, 22, 6)
    stats = computeGenerationStats(bone, globalMotion=None)
    assert stats.rangeGlobal is None
    assert stats.rootDisplacement == 0.0


# ---------------------------------------------------------------------
# Tremblement signal
# ---------------------------------------------------------------------
def test_static_pose_has_zero_velocity_and_acceleration() -> None:
    """A frozen pose should produce zero velocity / acceleration."""
    static = torch.zeros(16, 22, 6)
    static[..., 0] = 1.0
    static[..., 4] = 1.0  # constant identity rotation6d
    stats = computeGenerationStats(static)
    assert stats.rotationVelocityMean == pytest.approx(0.0)
    assert stats.rotationAccelerationMean == pytest.approx(0.0)


def test_jitter_has_high_acceleration_vs_smooth_motion() -> None:
    """A high-frequency jitter has much higher Δ²rot than a smooth ramp."""
    frames = 32
    smooth = torch.linspace(0.0, 1.0, frames).reshape(frames, 1, 1).expand(
        frames, 22, 6
    ).clone()
    # Jitter: alternating ±0.5 every frame.
    jitter = torch.zeros(frames, 22, 6)
    jitter[::2] = 0.5
    jitter[1::2] = -0.5

    smoothStats = computeGenerationStats(smooth)
    jitterStats = computeGenerationStats(jitter)
    # Smooth ramp has constant velocity → zero acceleration.
    assert smoothStats.rotationAccelerationMean < 1e-5
    # Jitter has strong acceleration.
    assert jitterStats.rotationAccelerationMean > 0.5
    assert (
        jitterStats.rotationAccelerationMean
        > 100 * smoothStats.rotationAccelerationMean + 0.1
    )


# ---------------------------------------------------------------------
# Root displacement
# ---------------------------------------------------------------------
def test_root_displacement_matches_known_path() -> None:
    """A root that walks 1m forward over 10 frames must report ~1m."""
    frames = 10
    rt = torch.zeros(frames, 3)
    rt[:, 0] = torch.linspace(0.0, 1.0, frames)
    bone = torch.zeros(frames, 22, 6)
    stats = computeGenerationStats(bone, rt)
    # Sum of ‖step‖ across 9 deltas, each step = 1/9 m.
    assert stats.rootDisplacement == pytest.approx(1.0, rel=1e-5)


def test_root_displacement_zero_for_static_pelvis() -> None:
    rt = torch.zeros(16, 3)
    bone = torch.zeros(16, 22, 6)
    stats = computeGenerationStats(bone, rt)
    assert stats.rootDisplacement == 0.0


# ---------------------------------------------------------------------
# Range & format
# ---------------------------------------------------------------------
def test_compute_stats_reports_correct_ranges() -> None:
    bone = torch.full((4, 22, 6), 0.7)
    bone[0, 0, 0] = -2.5
    bone[3, 5, 5] = 3.5
    rt = torch.full((4, 3), 1.0)
    rt[2, 1] = -4.0
    stats = computeGenerationStats(bone, rt)
    assert stats.rangeBone == (-2.5, 3.5)
    assert stats.rangeGlobal == (-4.0, 1.0)


def test_format_stats_line_contains_key_fields() -> None:
    bone = torch.randn(8, 22, 6)
    rt = torch.randn(8, 3)
    stats = computeGenerationStats(bone, rt)
    line = formatStatsLine(stats, prefix="[test] ")
    assert line.startswith("[test] ")
    assert "rot μ=" in line
    assert "Δ²rot/frame=" in line
    assert "root_displ=" in line


def test_format_stats_line_handles_missing_global() -> None:
    bone = torch.randn(8, 22, 6)
    stats = computeGenerationStats(bone, globalMotion=None)
    line = formatStatsLine(stats)
    assert "global=N/A" in line


# ---------------------------------------------------------------------
# Single-frame edge case
# ---------------------------------------------------------------------
def test_single_frame_has_no_velocity_or_displacement() -> None:
    bone = torch.randn(1, 22, 6)
    rt = torch.randn(1, 3)
    stats = computeGenerationStats(bone, rt)
    assert stats.rotationVelocityMean == 0.0
    assert stats.rotationAccelerationMean == 0.0
    assert stats.rootDisplacement == 0.0
