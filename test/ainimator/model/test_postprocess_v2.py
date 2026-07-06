

def test_smooth_rotation6d_reduces_jitter_and_preserves_motion() -> None:
    """Temporal smoothing cuts frame jitter while keeping the motion."""
    import torch
    from ainimator.model.postprocess_v2 import (
        smoothRotation6dTemporal,
    )

    torch.manual_seed(0)
    base = torch.randn(60, 22, 6).cumsum(dim=0) * 0.05
    jittery = base + torch.randn(60, 22, 6) * 0.2

    def accel(m: torch.Tensor) -> float:
        return float((m[2:] - 2 * m[1:-1] + m[:-2]).abs().mean())

    smoothed = smoothRotation6dTemporal(jittery, sigma=2.0)
    assert smoothed.shape == jittery.shape
    assert accel(smoothed) < 0.5 * accel(jittery)
    # global amplitude is preserved (no flattening)
    assert abs(float(smoothed.std()) - float(jittery.std())) < 0.15


def test_smooth_rotation6d_zero_sigma_is_identity() -> None:
    import torch
    from ainimator.model.postprocess_v2 import (
        smoothRotation6dTemporal,
    )

    motion = torch.randn(40, 22, 6)
    out = smoothRotation6dTemporal(motion, sigma=0.0)
    assert torch.equal(out, motion)
