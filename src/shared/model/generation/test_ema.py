"""Tests for the ExponentialMovingAverage manager."""

from __future__ import annotations

import torch

from src.shared.model.generation.ema import ExponentialMovingAverage


def _makeParams(values: list[list[float]]) -> list[torch.nn.Parameter]:
    """Build a list of ``torch.nn.Parameter`` from plain Python floats."""
    return [
        torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
        for v in values
    ]


def test_update_without_warmup_applies_exact_decay() -> None:
    """decay=0.5, no warmup: shadow = 0.5 * old + 0.5 * new."""
    params = _makeParams([[1.0, 2.0]])
    ema = ExponentialMovingAverage(params, decay=0.5, useWarmup=False)

    # Mutate the params as an optimizer.step() would have done.
    params[0].data.copy_(torch.tensor([2.0, 3.0]))
    ema.update(params)

    # shadow_0 = [1.0, 2.0] (init), update: 0.5*[1,2] + 0.5*[2,3] = [1.5, 2.5]
    assert torch.allclose(ema.shadow[0], torch.tensor([1.5, 2.5]))
    assert ema.numUpdates == 1


def test_warmup_effective_decay_at_first_step() -> None:
    """With warmup: decay_eff(step=0) = 1/10 = 0.1, far from target 0.9999."""
    params = _makeParams([[0.0]])
    ema = ExponentialMovingAverage(params, decay=0.9999, useWarmup=True)

    params[0].data.copy_(torch.tensor([10.0]))
    ema.update(params)

    # step counter is 0 when update() starts, so decay_eff = 1/10 = 0.1
    # shadow = 0.1 * 0.0 + 0.9 * 10.0 = 9.0
    assert torch.allclose(ema.shadow[0], torch.tensor([9.0]), atol=1e-6)


def test_warmup_converges_toward_target_decay() -> None:
    """After many updates, effective decay approaches the configured target."""
    params = _makeParams([[0.0]])
    ema = ExponentialMovingAverage(params, decay=0.99, useWarmup=True)

    # 1000 identical updates -> warmup decay ~= 1001/1010 ~= 0.9911 > 0.99
    # so effective decay is capped at 0.99.
    for _ in range(1000):
        ema.update(params)
    # _effectiveDecay is private, but we can verify by one more observable step:
    params[0].data.copy_(torch.tensor([1.0]))
    before = ema.shadow[0].clone()
    ema.update(params)
    # shadow_new = 0.99 * before + 0.01 * 1.0
    expected = 0.99 * before + 0.01
    assert torch.allclose(ema.shadow[0], expected, atol=1e-6)


def test_store_and_swap_then_restore_is_bit_exact() -> None:
    """storeAndSwap installs shadow; restore reverts to the backed-up values."""
    params = _makeParams([[1.0, 2.0, 3.0]])
    ema = ExponentialMovingAverage(params, decay=0.5, useWarmup=False)

    # Drift the shadow away from the params.
    params[0].data.copy_(torch.tensor([5.0, 6.0, 7.0]))
    ema.update(params)
    # shadow = 0.5*[1,2,3] + 0.5*[5,6,7] = [3,4,5]
    assert torch.allclose(ema.shadow[0], torch.tensor([3.0, 4.0, 5.0]))

    originalBeforeSwap = params[0].data.clone()
    ema.storeAndSwap(params)
    # Params now hold the shadow values.
    assert torch.allclose(params[0].data, torch.tensor([3.0, 4.0, 5.0]))

    ema.restore(params)
    # Params bit-exactly restored.
    assert torch.equal(params[0].data, originalBeforeSwap)


def test_restore_without_swap_raises() -> None:
    params = _makeParams([[0.0]])
    ema = ExponentialMovingAverage(params, decay=0.5, useWarmup=False)
    try:
        ema.restore(params)
    except RuntimeError:
        return
    raise AssertionError("restore without prior swap should raise")


def test_double_swap_without_restore_raises() -> None:
    params = _makeParams([[0.0]])
    ema = ExponentialMovingAverage(params, decay=0.5, useWarmup=False)
    ema.storeAndSwap(params)
    try:
        ema.storeAndSwap(params)
    except RuntimeError:
        ema.restore(params)
        return
    raise AssertionError("double swap without restore should raise")


def test_state_dict_round_trip_restores_shadow_and_counter() -> None:
    params1 = _makeParams([[1.0, 2.0], [3.0]])
    ema1 = ExponentialMovingAverage(params1, decay=0.9, useWarmup=False)

    # Drive the shadow with 5 updates.
    for step in range(5):
        params1[0].data.add_(1.0)
        params1[1].data.add_(0.5)
        ema1.update(params1)

    state = ema1.stateDict()
    assert state["numUpdates"] == 5
    assert state["decay"] == 0.9
    assert state["useWarmup"] is False

    # Fresh EMA on freshly-built params; load_state_dict should realign.
    params2 = _makeParams([[0.0, 0.0], [0.0]])
    ema2 = ExponentialMovingAverage(params2, decay=0.9, useWarmup=False)
    ema2.loadStateDict(state)

    assert ema2.numUpdates == 5
    for s1, s2 in zip(ema1.shadow, ema2.shadow):
        assert torch.allclose(s1, s2)


def test_copy_to_writes_shadow_without_backup() -> None:
    params = _makeParams([[0.0, 0.0]])
    ema = ExponentialMovingAverage(params, decay=0.5, useWarmup=False)
    params[0].data.copy_(torch.tensor([4.0, 8.0]))
    ema.update(params)
    # shadow = 0.5*[0,0] + 0.5*[4,8] = [2, 4]

    # Overwrite params with shadow values.
    ema.copyTo(params)
    assert torch.allclose(params[0].data, torch.tensor([2.0, 4.0]))

    # No backup was registered, restoring should raise.
    try:
        ema.restore(params)
    except RuntimeError:
        return
    raise AssertionError("copyTo must not register a backup")


def test_parameter_count_mismatch_raises() -> None:
    params = _makeParams([[1.0]])
    ema = ExponentialMovingAverage(params, decay=0.5, useWarmup=False)
    tooMany = _makeParams([[1.0], [2.0]])
    try:
        ema.update(tooMany)
    except RuntimeError:
        return
    raise AssertionError("update with wrong param count should raise")


def test_invalid_decay_raises() -> None:
    params = _makeParams([[0.0]])
    for badDecay in (0.0, 1.0, -0.1, 1.5):
        try:
            ExponentialMovingAverage(params, decay=badDecay)
        except ValueError:
            continue
        raise AssertionError(f"decay={badDecay} should raise")
