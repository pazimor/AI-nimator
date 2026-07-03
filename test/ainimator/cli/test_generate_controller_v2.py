"""Tests for generate_controller_v2 prompt resolution (A6-bis).

Locks the fix for the "walking" diagnostic (LOG 2026-07-01): when a
prompt is given, ``_resolvePromptEmb`` must NEVER silently fall back to
the null embedding. It either encodes the prompt or raises an explicit
error — a checkpoint trained without text conditioning
(``promptEmbChannels == 0``) or a missing encoder artifact are errors,
not silent no-ops. No prompt at all stays a legitimate ``None``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ainimator.cli import generate_controller_v2 as gen


def _model(promptEmbChannels: int) -> SimpleNamespace:
    """Fake controller exposing only ``config.promptEmbChannels``."""
    return SimpleNamespace(
        config=SimpleNamespace(promptEmbChannels=promptEmbChannels)
    )


@pytest.mark.parametrize("prompt", [None, "", "   "])
def test_no_prompt_returns_none(prompt: str | None) -> None:
    """No prompt requested → unconditioned rollout (None), no error."""
    result = gen._resolvePromptEmb(
        _model(16), prompt, None, torch.device("cpu")
    )
    assert result is None


def test_prompt_on_unconditioned_checkpoint_raises() -> None:
    """Prompt + promptEmbChannels==0 → explicit error, not null fallback."""
    with pytest.raises(SystemExit, match="promptEmbChannels=0"):
        gen._resolvePromptEmb(
            _model(0), "a person dances", None, torch.device("cpu")
        )


def test_prompt_without_encoder_artifact_raises() -> None:
    """Prompt + text-capable checkpoint but no encoder → explicit error."""
    with pytest.raises(SystemExit, match="encoder-artifact"):
        gen._resolvePromptEmb(
            _model(16), "a person dances", None, torch.device("cpu")
        )


def test_prompt_with_encoder_encodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: prompt encoded once, pooled embedding returned."""
    pooled = torch.ones(1, 16)
    monkeypatch.setattr(
        gen, "loadFrozenTextEncoder", lambda *_: ("enc", "tok")
    )
    monkeypatch.setattr(
        gen, "encodeTextToPooled", lambda *_: pooled
    )
    result = gen._resolvePromptEmb(
        _model(16), "a person dances", Path("artifact"), torch.device("cpu")
    )
    assert result is pooled


def _rollout() -> "RolloutResult":
    """Minimal 1-frame rollout with all three required fields set."""
    from ainimator.model.controller_rollout import RolloutResult

    return RolloutResult(
        rotation6d=torch.zeros(1, 4, 22, 6),
        rootTranslation=torch.zeros(1, 4, 3),
        rootLocalMotion=torch.zeros(1, 4, 4),
    )


def test_smooth_rollout_preserves_root_local_motion() -> None:
    """--smooth>0 must rebuild a valid RolloutResult (rootLocalMotion kept)."""
    smoothed = gen._smoothRollout(_rollout(), sigma=2.0)
    assert smoothed.rootLocalMotion.shape == (1, 4, 4)
    assert smoothed.rotation6d.shape == (1, 4, 22, 6)


def test_smooth_rollout_noop_when_sigma_zero() -> None:
    """sigma<=0 returns the rollout untouched."""
    rollout = _rollout()
    assert gen._smoothRollout(rollout, sigma=0.0) is rollout
