"""Tests for controller generalization training (phase A6).

Synthetic clips only — no dataset on disk (mirrors the smoke-test
contract).  The held-out clips are disjoint from the train clips, so the
reported held-out metrics are a genuine generalization signal.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ainimator.training.controller_generalization_v2 import (
    runControllerGeneralization,
)
from ainimator.training.controller_training_v2 import (
    ControllerTrainingConfig,
    loadControllerCheckpoint,
)


def _clip(seed: int, speed: float, frames: int = 56) -> tuple[
    torch.Tensor, torch.Tensor
]:
    phase = torch.linspace(0.0, 4.0 * math.pi, frames)
    rotation6d = torch.zeros(frames, 22, 6)
    for bone in range(22):
        for channel in range(6):
            rotation6d[:, bone, channel] = math.cos(
                bone + channel + seed
            ) + 0.3 * torch.sin(phase + 0.2 * bone + 0.5 * channel + seed)
    rootTranslation = torch.zeros(frames, 3)
    rootTranslation[:, 0] = speed * torch.arange(frames)
    rootTranslation[:, 2] = 0.05 * torch.cos(phase)
    return rotation6d, rootTranslation


def _trainClips() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [_clip(seed, 0.01 + 0.005 * seed) for seed in range(6)]


def _heldOutClips() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [_clip(100 + seed, 0.02 + 0.005 * seed) for seed in range(3)]


def _config(outputDir: Path) -> ControllerTrainingConfig:
    return ControllerTrainingConfig(
        outputDir=outputDir,
        epochs=60,
        device="cpu",
        embedDim=128,
        numHeads=4,
        numLayers=3,
        contextFrames=4,
        logEvery=999,
        seed=0,
    )


def test_generalization_requires_two_train_clips(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2 train"):
        runControllerGeneralization(
            [_clip(0, 0.01)], _heldOutClips(), _config(tmp_path)
        )


def test_generalization_requires_two_held_out_clips(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2 held-out"):
        runControllerGeneralization(
            _trainClips(), [_clip(0, 0.01)], _config(tmp_path)
        )


def test_generalization_reports_held_out_metrics(tmp_path: Path) -> None:
    result = runControllerGeneralization(
        _trainClips(), _heldOutClips(), _config(tmp_path),
        clipBatchSize=2, evalSampleClips=4,
    )
    assert result.numTrainClips == 6
    assert result.numHeldOutClips == 3
    for key in (
        "control_sensitivity",
        "mean_collapse_rank",
        "rollout_drift",
        "reconstruction_geodesic",
        "post_norm_stats",
    ):
        assert key in result.heldOutMetrics
        assert math.isfinite(result.heldOutMetrics[key])
        assert key in result.trainMetrics
    assert result.driftCurve  # non-empty held-out drift-vs-horizon curve


def test_generalization_checkpoint_round_trips(tmp_path: Path) -> None:
    result = runControllerGeneralization(
        _trainClips(), _heldOutClips(), _config(tmp_path), clipBatchSize=2
    )
    model, _s, _d, _m, _st = loadControllerCheckpoint(result.checkpointPath)
    assert model.config.contextFrames == 4


def test_prompt_sensitivity_computed_for_text_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A text-conditioned run must report prompt_sensitivity, not UNKNOWN.

    Locks the A6-bis wiring: evaluation feeds per-clip captions (train
    AND held-out) through the encoder and computes the real-vs-null
    output delta.  The encoder is mocked — no artifact on disk.
    """
    import ainimator.training.controller_generalization_v2 as gen
    from ainimator.health.contract import Verdict

    def _fakeEncode(texts, encoder, tokenizer, device):
        rng = torch.Generator().manual_seed(sum(map(ord, texts[0])))
        return torch.randn(len(texts), 8, generator=rng).to(device)

    monkeypatch.setattr(
        gen, "_maybeLoadEncoder", lambda config, device: ("enc", "tok")
    )
    monkeypatch.setattr(gen, "encodeTextToPooled", _fakeEncode)

    from dataclasses import replace

    config = replace(_config(tmp_path), epochs=10, promptEmbChannels=8)
    result = runControllerGeneralization(
        _trainClips(), _heldOutClips(), config,
        clipBatchSize=2, evalSampleClips=4,
        trainClipTexts=[f"train clip {i}" for i in range(6)],
        heldOutClipTexts=[f"held out clip {i}" for i in range(3)],
    )
    assert "prompt_sensitivity" in result.trainMetrics
    assert "prompt_sensitivity" in result.heldOutMetrics
    assert result.heldOutMetrics["prompt_sensitivity"] >= 0.0
    assert result.verdicts["prompt_sensitivity"] is not Verdict.UNKNOWN


def test_prompt_sensitivity_unknown_without_text(tmp_path: Path) -> None:
    """A run without text conditioning keeps the UNKNOWN verdict."""
    from ainimator.health.contract import Verdict

    result = runControllerGeneralization(
        _trainClips(), _heldOutClips(), _config(tmp_path),
        clipBatchSize=2, evalSampleClips=4,
    )
    assert "prompt_sensitivity" not in result.heldOutMetrics
    assert result.verdicts["prompt_sensitivity"] is Verdict.UNKNOWN
