"""Deterministic loss-parity regression test for the v2 training step.

This test pins the exact loss value produced by one forward+backward pass
on a fixed synthetic input with a seeded model.  Its purpose is to guard
against silent behavior regressions in the model, losses, or noise
schedule — a change that breaks this pin is a change that altered math.

The pinned value was captured on the A2 tree (commit 027b56f) with
``torch.manual_seed(0)`` and fixed (non-random) motion tensors, so it
is independent of dataset I/O and can run entirely on CPU in < 1 second.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from ainimator.text import CustomTokenizer, CustomTokenizerConfig
from ainimator.training.training_v2 import (
    LoadedSample,
    TrainingRandomState,
    V2TrainingConfig,
    buildTrainingComponents,
    trainStep,
)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
_FRAMES = 8
_BONES = 22
_PINNED_LOSS_TOTAL: float = 0.02388053
_PINNED_TOLERANCE: float = 1e-5


def _fixedMotion() -> torch.Tensor:
    """Return a deterministic (F, B, 6) rotation6d tensor.

    All channels are zero except the first channel of each bone which is
    set to 1.0 — a simple non-trivial fixed point.
    """
    motion = torch.zeros(_FRAMES, _BONES, 6)
    motion[..., 0] = 1.0
    return motion


def _fixedRootTrans() -> torch.Tensor:
    """Return a deterministic (F, 3) root-translation tensor (all zeros)."""
    return torch.zeros(_FRAMES, 3)


def _buildTokenizer(tokenizer_dir: Path) -> CustomTokenizer:
    """Train and save a minimal deterministic tokenizer."""
    texts = ["a person walks."] * 8
    tokenizer = CustomTokenizer.train(
        texts,
        config=CustomTokenizerConfig(
            vocabSize=64,
            maxLength=8,
            minFrequency=1,
        ),
    )
    tokenizer.save(tokenizer_dir)
    return tokenizer


def _buildConfig(
    dataset_root: Path,
    tokenizer_dir: Path,
    output_dir: Path,
) -> V2TrainingConfig:
    """Return a tiny V2TrainingConfig suitable for CPU regression testing."""
    return V2TrainingConfig(
        datasetRoot=dataset_root,
        tokenizerDir=tokenizer_dir,
        outputDir=output_dir,
        sampleLinkIndex=0,
        epochs=1,
        learningRate=1e-3,
        weightDecay=0.0,
        minSnrGamma=5.0,
        velocityXyzWeight=0.0,
        diffusionStepsTraining=50,
        scheduleType="cosine",
        predictionMode="v",
        encoderHiddenDim=32,
        encoderNumLayers=1,
        encoderNumHeads=4,
        denoiserEmbedDim=32,
        denoiserNumLayers=1,
        denoiserNumHeads=4,
        maxFrames=8,
        framesPerStep=0,
        seed=0,
        logEvery=10,
        device="cpu",
    )


def test_deterministic_loss_parity() -> None:
    """One seeded forward+backward must produce the pinned loss value.

    This is the A2 no-behavior-change guard: if model logic, the noise
    schedule, or the loss function changes silently, this test will fail
    before any long training run reveals the regression.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpDir = Path(tmp)
        tokenizerDir = tmpDir / "tokenizer"
        _buildTokenizer(tokenizerDir)

        config = _buildConfig(
            dataset_root=tmpDir,
            tokenizer_dir=tokenizerDir,
            output_dir=tmpDir / "out",
        )

        torch.manual_seed(0)
        components = buildTrainingComponents(
            config,
            normalizationSamples=None,
        )

        sample = LoadedSample(
            rotation6d=_fixedMotion(),
            rootTranslation=_fixedRootTrans(),
            rawText="a person walks.",
            metadata={},
            sampleId=0,
            textId=0,
        )

        generators = TrainingRandomState.fromSeed(
            seed=0,
            device=components.device,
        )
        metrics = trainStep(components, sample, config, generators)

        assert abs(metrics["loss_total"] - _PINNED_LOSS_TOTAL) < _PINNED_TOLERANCE, (
            f"Loss parity broken: got {metrics['loss_total']:.8f}, "
            f"expected {_PINNED_LOSS_TOTAL:.8f} "
            f"(tolerance {_PINNED_TOLERANCE})."
        )
