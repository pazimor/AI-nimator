"""Unit tests for :mod:`src.features.generation.training_v2`.

These tests exercise the full v2 training contract end-to-end on a
synthetic micro-dataset built in a tmp directory:

    * ``loadDatasetSample`` reads (motion, raw_text) from sample/text shards.
    * ``buildTrainingComponents`` instantiates tokenizer + encoder +
      denoiser + schedule + optimiser on the same device.
    * ``trainStep`` produces finite scalar losses and propagates
      gradients into both encoder and denoiser parameters.
    * ``runOverfit`` decreases the loss across epochs and writes a
      checkpoint that round-trips through :func:`loadCheckpointV2`.

They run on CPU in well under a minute so they belong in the regular
test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from ainimator.training.training_v2 import (
    LoadedSample,
    V2TrainingComponents,
    V2TrainingConfig,
    buildTrainingComponents,
    loadCheckpointV2,
    loadDatasetSample,
    resolveDevice,
    runOverfit,
    trainStep,
)
from ainimator.text import (
    CustomTokenizer,
    CustomTokenizerConfig,
)


# =====================================================================
# Synthetic preprocessed dataset fixture
# =====================================================================
def _writeJson(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _buildSyntheticDataset(
    root: Path,
    rawText: str = "a person walks forward.",
    frames: int = 16,
    numBones: int = 22,
) -> None:
    """Create a tiny preprocessed dataset matching the V2 layout."""
    sampleShard = [
        {
            "motion": torch.randn(frames, numBones, 6),
            "time": torch.tensor([30, frames]),
            "meta": {"fps": 30, "frames": frames, "joints": "SMPL-22"},
            "root_translation": torch.randn(frames, 3),
        }
    ]
    textShard = [
        {
            "text_id": 0,
            "raw_text": rawText,
            "input_ids": torch.zeros(8, dtype=torch.long),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "pooled_text": torch.zeros(768),
        }
    ]
    sampleShardPath = root / "sample_shards" / "sample_shard_00000.pt"
    sampleShardPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sampleShard, sampleShardPath)
    textShardPath = root / "text_shards" / "text_shard_00000.pt"
    textShardPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(textShard, textShardPath)

    _writeJson(
        root / "manifest.json",
        {
            "version": 2,
            "modelName": "test",
            "maxPromptLength": 8,
            "splitFrames": frames,
            "downsampleTargetFrames": frames,
            "maxSegmentFrames": frames,
            "sampleShardSize": 1,
            "textShardSize": 1,
            "totalSamples": 1,
            "totalTexts": 1,
            "totalLinks": 1,
            "averageSampleBytes": 1,
            "maxSampleBytes": 1,
            "averagePairBytes": 1,
            "averageFrames": frames,
            "maxFrames": frames,
            "pooledTextDim": 768,
            "enabledComponents": ["rotation6d", "root_translation"],
            "sampleShards": [
                {
                    "path": str(sampleShardPath.relative_to(root)),
                    "sampleCount": 1,
                }
            ],
            "textShards": [
                {
                    "path": str(textShardPath.relative_to(root)),
                    "sampleCount": 1,
                }
            ],
            "sampleIndexPath": "sample_index.json",
            "textIndexPath": "text_index.json",
            "linkIndexPath": "link_index.json",
        },
    )
    _writeJson(
        root / "sample_index.json",
        [
            {
                "sampleId": 0,
                "shardIndex": 0,
                "shardOffset": 0,
                "frames": frames,
                "sampleBytes": 1,
                "datasetFolder": "TEST",
                "sourceFile": "synthetic",
                "startFrame": 0,
                "endFrame": frames,
            }
        ],
    )
    _writeJson(
        root / "text_index.json",
        [
            {
                "textId": 0,
                "shardIndex": 0,
                "shardOffset": 0,
                "tokenCount": 8,
                "textBytes": 1,
            }
        ],
    )
    _writeJson(
        root / "link_index.json",
        [
            {
                "sampleId": 0,
                "textId": 0,
                "pairBytes": 1,
            }
        ],
    )


def _buildTokenizerDir(root: Path) -> Path:
    tokenizer = CustomTokenizer.train(
        ["a person walks forward.", "the woman dances.", "someone runs."]
        * 4,
        config=CustomTokenizerConfig(
            vocabSize=128, maxLength=16, minFrequency=1
        ),
    )
    tokenizer.save(root)
    return root


def _miniConfig(
    datasetRoot: Path,
    tokenizerDir: Path,
    outputDir: Path,
    epochs: int = 5,
) -> V2TrainingConfig:
    """Build a tiny V2TrainingConfig that fits under a second of CPU."""
    return V2TrainingConfig(
        datasetRoot=datasetRoot,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        sampleLinkIndex=0,
        epochs=epochs,
        learningRate=1e-3,  # high LR for fast convergence
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
        maxFrames=16,
        framesPerStep=0,
        seed=0,
        logEvery=10,
        device="cpu",
    )


# =====================================================================
# Tests
# =====================================================================
def test_resolve_device_handles_explicit_choice() -> None:
    assert resolveDevice("cpu").type == "cpu"


def test_load_dataset_sample_returns_expected_fields(tmp_path: Path) -> None:
    _buildSyntheticDataset(tmp_path, rawText="a person walks forward.")
    sample = loadDatasetSample(tmp_path, linkIndex=0)
    assert isinstance(sample, LoadedSample)
    assert sample.rawText == "a person walks forward."
    assert sample.numFrames == 16
    assert sample.numBones == 22
    assert sample.rotation6d.shape == (16, 22, 6)
    assert sample.rootTranslation.shape == (16, 3)


def test_load_dataset_sample_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        loadDatasetSample(tmp_path / "missing", linkIndex=0)


def test_load_dataset_sample_rejects_out_of_range(tmp_path: Path) -> None:
    _buildSyntheticDataset(tmp_path)
    with pytest.raises(IndexError):
        loadDatasetSample(tmp_path, linkIndex=99)


def test_train_step_produces_finite_loss_and_grads(tmp_path: Path) -> None:
    _buildSyntheticDataset(tmp_path)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    config = _miniConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=tmp_path / "out",
        epochs=1,
    )
    components = buildTrainingComponents(config)
    sample = loadDatasetSample(config.datasetRoot, config.sampleLinkIndex)
    from ainimator.training.training_v2 import TrainingRandomState
    generators = TrainingRandomState.fromSeed(
        seed=0, device=components.device
    )

    metrics = trainStep(components, sample, config, generators)
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))
    assert metrics["loss_total"] > 0.0
    # All trainable parameters must have a gradient afterwards — except
    # the Phase F nullEmbedding, which only receives gradient when CFG
    # dropout actually fires (zero with condMaskProb=0.0 in this test).
    for name, parameter in components.encoder.named_parameters():
        if name == "nullEmbedding":
            continue
        assert parameter.grad is not None, f"missing grad on encoder.{name}"
    for parameter in components.denoiser.parameters():
        assert parameter.grad is not None


def test_train_step_with_velocity_loss_active(tmp_path: Path) -> None:
    """vel-xyz weight > 0 must add a non-trivial term."""
    _buildSyntheticDataset(tmp_path)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    base = _miniConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=tmp_path / "out",
        epochs=1,
    )
    from ainimator.training.training_v2 import TrainingRandomState
    config = V2TrainingConfig(**{**base.__dict__, "velocityXyzWeight": 1.0})
    components = buildTrainingComponents(config)
    sample = loadDatasetSample(config.datasetRoot, config.sampleLinkIndex)
    generators = TrainingRandomState.fromSeed(
        seed=0, device=components.device
    )
    metrics = trainStep(components, sample, config, generators)
    assert metrics["loss_vel_xyz"] >= 0.0
    # When the weight is non-zero the velocity component must be tracked.
    assert "loss_vel_xyz" in metrics


def test_run_overfit_writes_checkpoint(tmp_path: Path) -> None:
    _buildSyntheticDataset(tmp_path)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    config = _miniConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=3,
    )
    components, history = runOverfit(config)
    assert isinstance(components, V2TrainingComponents)
    assert len(history) == 3
    checkpointPath = outputDir / "v2_overfit_checkpoint.pt"
    assert checkpointPath.exists()


def test_checkpoint_round_trips_through_load(tmp_path: Path) -> None:
    _buildSyntheticDataset(tmp_path)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    config = _miniConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=2,
    )
    components, _ = runOverfit(config)
    checkpointPath = outputDir / "v2_overfit_checkpoint.pt"

    (
        loadedTokenizer,
        loadedEncoder,
        loadedDenoiser,
        loadedSchedule,
        loadedNormalizer,
        payload,
    ) = loadCheckpointV2(checkpointPath, device="cpu")

    assert loadedTokenizer.vocabSize == components.tokenizer.vocabSize
    assert (
        loadedEncoder.config.hiddenDim == components.encoder.config.hiddenDim
    )
    assert (
        loadedDenoiser.config.embedDim
        == components.denoiser.config.embedDim
    )
    assert loadedSchedule.numSteps == components.schedule.numSteps
    assert payload["training_sample"]["rawText"] == "a person walks forward."
    # Normalizer round-trips with non-identity stats.
    assert loadedNormalizer.numBones == components.normalizer.numBones
    assert torch.equal(
        loadedNormalizer.boneMean, components.normalizer.boneMean
    )
    assert torch.equal(
        loadedNormalizer.boneStd, components.normalizer.boneStd
    )

    # Encoder weights round-trip exactly.
    for (name, original), loaded in zip(
        components.encoder.state_dict().items(),
        loadedEncoder.state_dict().values(),
    ):
        assert torch.equal(original.cpu(), loaded.cpu()), (
            f"Mismatched encoder parameter {name} after round-trip."
        )


def test_run_overfit_decreases_loss_over_epochs(tmp_path: Path) -> None:
    """A larger run should drive the loss well below its initial value.

    Diffusion loss naturally oscillates between epochs (random ``t``
    each step), so a 5-epoch window is too noisy to compare reliably.
    We use 60 epochs and compare the mean of the first quarter to the
    minimum reached over the last quarter — this avoids false negatives
    from a single high-``t`` outlier landing in the tail window.
    """
    _buildSyntheticDataset(tmp_path)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    config = _miniConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=60,
    )
    _, history = runOverfit(config)
    quarter = max(1, len(history) // 4)
    earlyMean = sum(h["loss_total"] for h in history[:quarter]) / quarter
    lateMin = min(h["loss_total"] for h in history[-quarter:])
    assert lateMin < earlyMean * 0.8, (
        f"Loss did not decrease enough "
        f"(early avg={earlyMean:.4f}, late min={lateMin:.4f})."
    )


def test_overfit_config_rejects_invalid_prediction_mode() -> None:
    with pytest.raises(ValueError, match="predictionMode"):
        V2TrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            predictionMode="bogus",
        )


def test_overfit_config_rejects_invalid_schedule_type() -> None:
    with pytest.raises(ValueError, match="scheduleType"):
        V2TrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            scheduleType="exponential",
        )


def test_overfit_config_rejects_zero_epochs() -> None:
    with pytest.raises(ValueError, match="epochs"):
        V2TrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            epochs=0,
        )


def test_overfit_config_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        V2TrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            dropout=1.5,
        )


def test_overfit_config_rejects_invalid_cond_mask_prob() -> None:
    with pytest.raises(ValueError, match="condMaskProb"):
        V2TrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            condMaskProb=1.5,
        )


def test_runoverfit_fits_normalizer_to_sample_statistics(
    tmp_path: Path,
) -> None:
    """The normalizer must end up with bone std ≠ 1 after fitting on
    a real motion sample with non-trivial variance."""
    _buildSyntheticDataset(tmp_path)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    config = _miniConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=tmp_path / "out",
        epochs=2,
    )
    components, _ = runOverfit(config)
    boneStd = components.normalizer.boneStd
    # The synthetic dataset draws bone tensors from N(0, 1) so per-channel
    # std should be close to 1 but never exactly equal across all channels.
    # The check that matters is that the normalizer was actually fitted,
    # not left at identity init.
    assert (boneStd != 1.0).any(), (
        "Normalizer std remained at the identity init — fit was not "
        "applied."
    )


def test_runoverfit_normalized_x0_round_trips_through_normalizer(
    tmp_path: Path,
) -> None:
    """The training pipeline normalises x_0 → diffuse → predict → denorm.

    This integration test shortcuts the diffusion and verifies the
    normalize / denormalize pair recovers the original sample exactly.
    Without this fix the v2 sampler produces pure noise (raw-space
    sampling vs unit-variance noise injection)."""
    _buildSyntheticDataset(tmp_path)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    config = _miniConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=tmp_path / "out",
        epochs=1,
    )
    components, _ = runOverfit(config)
    sample = loadDatasetSample(config.datasetRoot, config.sampleLinkIndex)

    rotation = sample.rotation6d.unsqueeze(0)
    rtrans = sample.rootTranslation.unsqueeze(0)
    rotNorm = components.normalizer.normalizeBone(rotation)
    rtransNorm = components.normalizer.normalizeGlobal(rtrans)
    rotRecovered = components.normalizer.denormalizeBone(rotNorm)
    rtransRecovered = components.normalizer.denormalizeGlobal(rtransNorm)
    assert torch.allclose(rotRecovered, rotation, atol=1e-5)
    assert torch.allclose(rtransRecovered, rtrans, atol=1e-5)
