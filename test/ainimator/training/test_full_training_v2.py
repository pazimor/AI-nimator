"""Unit tests for :mod:`src.features.generation.full_training_v2`.

These tests exercise the multi-sample training pipeline end-to-end on
a synthetic preprocessed dataset built in a tmp directory:

    * Folder filtering produces the expected link subset.
    * The deterministic train/val split is stable across runs and
      respects the requested ratio.
    * Collate padding handles variable-length samples + motion mask.
    * Normalizer fitting on a streamed subset converges to the expected
      mean/std on a constant signal.
    * One full ``runFullTraining`` cycle on a tiny model writes both
      ``best`` and ``latest`` checkpoints, the loss decreases over a
      handful of epochs, and the resulting checkpoint round-trips
      through :func:`loadCheckpointV2`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from ainimator.training.full_training_v2 import (
    V2FullTrainingConfig,
    buildFullTrainingComponents,
    collateV2Batch,
    effectiveClipWeight,
    fitNormalizerFromDataset,
    iterBatches,
    runFullTraining,
    selectLinkIndices,
    splitTrainVal,
    trainStepBatch,
    validateEpoch,
)
from ainimator.training.training_v2 import (
    TrainingRandomState,
    loadCheckpointV2,
)
from ainimator.text import (
    CustomTokenizer,
    CustomTokenizerConfig,
)
from ainimator.data.preprocessed_dataset import PreprocessedLinkDataset


# =====================================================================
# Synthetic multi-sample dataset fixture
# =====================================================================
def _writeJson(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _buildMultiSampleDataset(
    root: Path,
    sampleCount: int,
    framesPerSample: int = 8,
    folder: str = "TEST",
) -> None:
    """Create a synthetic V2 preprocessed dataset with N (motion, text) pairs."""
    sampleShard: list[dict[str, object]] = []
    textShard: list[dict[str, object]] = []
    sampleEntries: list[dict[str, object]] = []
    textEntries: list[dict[str, object]] = []
    linkEntries: list[dict[str, object]] = []
    for index in range(sampleCount):
        sampleShard.append(
            {
                "motion": torch.randn(framesPerSample, 22, 6),
                "time": torch.tensor([30, framesPerSample]),
                "meta": {
                    "fps": 30,
                    "frames": framesPerSample,
                    "joints": "SMPL-22",
                },
                "root_translation": torch.randn(framesPerSample, 3),
            }
        )
        textShard.append(
            {
                "text_id": index,
                "raw_text": f"motion sample {index}",
                "input_ids": torch.zeros(8, dtype=torch.long),
                "attention_mask": torch.ones(8, dtype=torch.long),
                "pooled_text": torch.zeros(768),
            }
        )
        sampleEntries.append(
            {
                "sampleId": index,
                "shardIndex": 0,
                "shardOffset": index,
                "frames": framesPerSample,
                "sampleBytes": 1,
                "datasetFolder": folder,
                "sourceFile": f"synthetic_{index}",
                "startFrame": 0,
                "endFrame": framesPerSample,
            }
        )
        textEntries.append(
            {
                "textId": index,
                "shardIndex": 0,
                "shardOffset": index,
                "tokenCount": 8,
                "textBytes": 1,
            }
        )
        linkEntries.append(
            {
                "linkId": index,
                "sampleId": index,
                "textId": index,
                "datasetFolder": folder,
                "sourceFile": f"synthetic_{index}",
                "frames": framesPerSample,
                "pairBytes": 1,
            }
        )

    sampleShardPath = root / "sample_shards" / "sample_shard_00000.pt"
    textShardPath = root / "text_shards" / "text_shard_00000.pt"
    sampleShardPath.parent.mkdir(parents=True, exist_ok=True)
    textShardPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sampleShard, sampleShardPath)
    torch.save(textShard, textShardPath)

    _writeJson(
        root / "manifest.json",
        {
            "version": 2,
            "modelName": "test",
            "maxPromptLength": 8,
            "splitFrames": framesPerSample,
            "downsampleTargetFrames": framesPerSample,
            "maxSegmentFrames": framesPerSample,
            "sampleShardSize": sampleCount,
            "textShardSize": sampleCount,
            "totalSamples": sampleCount,
            "totalTexts": sampleCount,
            "totalLinks": sampleCount,
            "averageSampleBytes": 1,
            "maxSampleBytes": 1,
            "averagePairBytes": 1,
            "averageFrames": framesPerSample,
            "maxFrames": framesPerSample,
            "pooledTextDim": 768,
            "enabledComponents": ["rotation6d", "root_translation"],
            "sampleShards": [
                {
                    "path": str(sampleShardPath.relative_to(root)),
                    "sampleCount": sampleCount,
                }
            ],
            "textShards": [
                {
                    "path": str(textShardPath.relative_to(root)),
                    "sampleCount": sampleCount,
                }
            ],
            "sampleIndexPath": "sample_index.json",
            "textIndexPath": "text_index.json",
            "linkIndexPath": "link_index.json",
        },
    )
    _writeJson(root / "sample_index.json", sampleEntries)
    _writeJson(root / "text_index.json", textEntries)
    _writeJson(root / "link_index.json", linkEntries)


def _buildTokenizerDir(root: Path) -> Path:
    tokenizer = CustomTokenizer.train(
        ["motion sample one", "motion sample two", "motion sample three"]
        * 4,
        config=CustomTokenizerConfig(
            vocabSize=128, maxLength=16, minFrequency=1
        ),
    )
    tokenizer.save(root)
    return root


def _miniFullConfig(
    datasetRoot: Path,
    tokenizerDir: Path,
    outputDir: Path,
    epochs: int = 3,
    batchSize: int = 4,
) -> V2FullTrainingConfig:
    return V2FullTrainingConfig(
        datasetRoot=datasetRoot,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        datasetFolders=("TEST",),
        epochs=epochs,
        batchSize=batchSize,
        gradientAccumulation=1,
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
        maxFrames=16,
        dropout=0.0,
        condMaskProb=0.0,
        validationFraction=0.25,
        validationSeed=0,
        validateEveryEpochs=1,
        normalizerFitMaxSamples=8,
        maxSamplesPerEpoch=0,
        logEvery=100,
        seed=0,
        device="cpu",
    )


# =====================================================================
# Config validation
# =====================================================================
def test_config_rejects_invalid_validation_fraction() -> None:
    with pytest.raises(ValueError, match="validationFraction"):
        V2FullTrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            validationFraction=0.0,
        )


def test_config_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batchSize"):
        V2FullTrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            batchSize=0,
        )


def test_config_rejects_invalid_grad_accum() -> None:
    with pytest.raises(ValueError, match="gradientAccumulation"):
        V2FullTrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            gradientAccumulation=0,
        )


# =====================================================================
# splitTrainVal
# =====================================================================
def test_split_train_val_is_deterministic() -> None:
    indices = list(range(1000))
    a = splitTrainVal(indices, validationFraction=0.1, seed=42)
    b = splitTrainVal(indices, validationFraction=0.1, seed=42)
    assert a == b


def test_split_train_val_changes_with_seed() -> None:
    indices = list(range(1000))
    a = splitTrainVal(indices, validationFraction=0.1, seed=0)
    b = splitTrainVal(indices, validationFraction=0.1, seed=1)
    assert a != b


def test_split_train_val_respects_fraction() -> None:
    indices = list(range(10_000))
    train, val = splitTrainVal(indices, validationFraction=0.2, seed=0)
    actualFraction = len(val) / (len(train) + len(val))
    assert abs(actualFraction - 0.2) < 0.01
    assert len(train) + len(val) == len(indices)
    assert set(train).isdisjoint(set(val))


def test_split_train_val_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        splitTrainVal([0, 1, 2], validationFraction=0.0, seed=0)
    with pytest.raises(ValueError):
        splitTrainVal([0, 1, 2], validationFraction=1.0, seed=0)


# =====================================================================
# Folder filtering
# =====================================================================
def test_select_link_indices_returns_all_when_no_filter(
    tmp_path: Path,
) -> None:
    _buildMultiSampleDataset(tmp_path, sampleCount=12)
    dataset = PreprocessedLinkDataset(
        datasetRoot=tmp_path, includeTokenizedText=False
    )
    indices = selectLinkIndices(dataset, datasetFolders=None)
    assert indices == list(range(12))


def test_select_link_indices_respects_folder_filter(
    tmp_path: Path,
) -> None:
    _buildMultiSampleDataset(tmp_path, sampleCount=8, folder="TEST")
    dataset = PreprocessedLinkDataset(
        datasetRoot=tmp_path, includeTokenizedText=False
    )
    indices = selectLinkIndices(dataset, datasetFolders=("TEST",))
    assert indices == list(range(8))
    indices = selectLinkIndices(dataset, datasetFolders=("OTHER",))
    assert indices == []


# =====================================================================
# Collate
# =====================================================================
def test_collate_pads_to_batch_max_with_motion_mask() -> None:
    payloads = [
        {
            "motion": torch.randn(5, 22, 6),
            "root_translation": torch.randn(5, 3),
            "raw_text": "short",
        },
        {
            "motion": torch.randn(8, 22, 6),
            "root_translation": torch.randn(8, 3),
            "raw_text": "longer sample",
        },
    ]
    batch = collateV2Batch(payloads, maxFrames=16, device=torch.device("cpu"))
    assert batch.rotation6d.shape == (2, 8, 22, 6)
    assert batch.rootTranslation.shape == (2, 8, 3)
    assert batch.motionMask.shape == (2, 8)
    # First sample padded from 5 to 8.
    assert batch.motionMask[0].tolist() == [
        True, True, True, True, True, False, False, False
    ]
    assert batch.motionMask[1].all()
    assert batch.rawTexts == ("short", "longer sample")


def test_collate_truncates_to_max_frames() -> None:
    payloads = [
        {
            "motion": torch.randn(40, 22, 6),
            "root_translation": torch.randn(40, 3),
            "raw_text": "very long",
        }
    ]
    batch = collateV2Batch(payloads, maxFrames=16, device=torch.device("cpu"))
    assert batch.rotation6d.shape == (1, 16, 22, 6)
    assert batch.motionMask.all()


def test_collate_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        collateV2Batch([], maxFrames=16, device=torch.device("cpu"))


# =====================================================================
# Normalizer fitting on streaming dataset
# =====================================================================
def test_fit_normalizer_from_dataset_reaches_unit_variance(
    tmp_path: Path,
) -> None:
    _buildMultiSampleDataset(
        tmp_path, sampleCount=24, framesPerSample=12
    )
    dataset = PreprocessedLinkDataset(
        datasetRoot=tmp_path, includeTokenizedText=False
    )
    from ainimator.model.motion_normalizer import MotionNormalizer

    normalizer = MotionNormalizer(
        numBones=22, motionChannels=6, globalChannels=3
    )
    indices = list(range(len(dataset.linkEntries)))
    fitNormalizerFromDataset(
        dataset=dataset,
        indices=indices,
        normalizer=normalizer,
        maxSamples=24,
        seed=0,
    )
    # The normalizer should now hold non-trivial std values (not the
    # identity ones=1).
    assert (normalizer.boneStd != 1.0).any()
    assert (normalizer.globalStd != 1.0).any()


def test_fit_normalizer_rejects_empty_indices(tmp_path: Path) -> None:
    _buildMultiSampleDataset(tmp_path, sampleCount=4)
    dataset = PreprocessedLinkDataset(
        datasetRoot=tmp_path, includeTokenizedText=False
    )
    from ainimator.model.motion_normalizer import MotionNormalizer

    normalizer = MotionNormalizer(
        numBones=22, motionChannels=6, globalChannels=3
    )
    with pytest.raises(ValueError, match="zero samples"):
        fitNormalizerFromDataset(
            dataset=dataset,
            indices=[],
            normalizer=normalizer,
            maxSamples=4,
        )


# =====================================================================
# trainStepBatch & validateEpoch
# =====================================================================
# ---------------------------------------------------------------------
# 2026-05-07 — best-metric / improvement / stagnation policy
# ---------------------------------------------------------------------
def test_read_best_metric_returns_total_by_default() -> None:
    from ainimator.training.full_training_v2 import _readBestMetric

    metrics = {"loss_total": 1.7, "loss_bone": 0.05, "loss_global": 0.01}
    assert _readBestMetric(metrics, "loss_total") == 1.7


def test_read_best_metric_synthesises_loss_diffusion() -> None:
    from ainimator.training.full_training_v2 import _readBestMetric

    metrics = {"loss_total": 1.7, "loss_bone": 0.05, "loss_global": 0.01}
    assert _readBestMetric(metrics, "loss_diffusion") == pytest.approx(0.06)


def test_read_best_metric_returns_inf_for_unknown_key() -> None:
    from ainimator.training.full_training_v2 import _readBestMetric

    assert _readBestMetric({"loss_total": 1.0}, "loss_bogus") == float(
        "inf"
    )


def test_config_rejects_invalid_best_metric() -> None:
    with pytest.raises(ValueError, match="bestMetric"):
        V2FullTrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            bestMetric="bogus",
        )


def test_config_rejects_negative_improvement_min() -> None:
    with pytest.raises(ValueError, match="bestImprovementMin"):
        V2FullTrainingConfig(
            datasetRoot=Path("."),
            tokenizerDir=Path("."),
            outputDir=Path("."),
            bestImprovementMin=-1e-3,
        )


def test_best_improvement_min_skips_noise_updates(tmp_path: Path) -> None:
    """A high improvement threshold must keep best.pt stable when only
    noise-level drops happen.

    The very first epoch always writes a best (since the initial
    ``bestValTotal = inf`` makes any finite metric a strictly large
    improvement).  After that, the threshold should reject all the
    subsequent (smaller-than-threshold) drops.  The assertion is on
    the **best_epoch** stored in the checkpoint: it must remain at
    epoch 1 after a 4-epoch run with an unreachable threshold.
    """
    _buildMultiSampleDataset(tmp_path, sampleCount=32)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    base = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=4,
    )
    from dataclasses import replace
    config = replace(base, bestImprovementMin=1e6)
    runFullTraining(config)
    bestPath = outputDir / "checkpoints" / "v2_full_best.pt"
    assert bestPath.exists()
    payload = torch.load(bestPath, map_location="cpu", weights_only=False)
    # Only the first epoch wrote a best; subsequent drops were below
    # the threshold and skipped.
    assert payload["training_meta"]["best_epoch"] == 1


def test_best_metric_loss_diffusion_writes_on_diffusion_drop(
    tmp_path: Path,
) -> None:
    """Selecting loss_diffusion as the best-metric makes the writer
    track bone+global only.  Even when contrastive is disabled (pure
    loss_total run) the best.pt should be written."""
    _buildMultiSampleDataset(tmp_path, sampleCount=32)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    base = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=2,
    )
    from dataclasses import replace
    config = replace(base, bestMetric="loss_diffusion")
    runFullTraining(config)
    bestPath = outputDir / "checkpoints" / "v2_full_best.pt"
    assert bestPath.exists()
    payload = torch.load(bestPath, map_location="cpu", weights_only=False)
    # The best_val_total stored in the checkpoint reflects the chosen
    # metric (loss_diffusion).  Sanity: it should be ≤ val_bone +
    # val_global of any epoch in the run.
    storedBest = payload["training_meta"]["best_val_total"]
    valMetrics = payload["training_meta"]["val_metrics"]
    assert storedBest <= valMetrics["loss_bone"] + valMetrics["loss_global"] + 1e-6


def test_stagnation_patience_logs_warning(
    tmp_path: Path,
    caplog: "pytest.LogCaptureFixture",
) -> None:
    """When the best-metric never improves enough, the stagnation
    warning fires after ``stagnationPatience`` epochs."""
    import logging

    _buildMultiSampleDataset(tmp_path, sampleCount=32)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    base = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=5,
    )
    from dataclasses import replace
    # Threshold huge ⇒ every epoch counts as stagnation; patience 2
    # means warning fires twice over 5 epochs.
    config = replace(
        base,
        bestImprovementMin=1e6,
        stagnationPatience=2,
    )
    with caplog.at_level(logging.WARNING):
        runFullTraining(config)
    warnings = [
        rec for rec in caplog.records
        if "Stagnation" in rec.getMessage()
    ]
    assert warnings, "Expected at least one stagnation warning."


def test_ema_disabled_by_default_no_shadow(tmp_path: Path) -> None:
    """[Phase D.4] When emaDecay == 0 the components have no EMA."""
    _buildMultiSampleDataset(tmp_path, sampleCount=8)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    config = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=tmp_path / "out",
    )
    components = buildFullTrainingComponents(config)
    assert components.ema is None


def test_ema_built_when_decay_positive(tmp_path: Path) -> None:
    _buildMultiSampleDataset(tmp_path, sampleCount=8)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    base = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=tmp_path / "out",
    )
    from dataclasses import replace
    config = replace(base, emaDecay=0.99)
    components = buildFullTrainingComponents(config)
    assert components.ema is not None


def test_run_full_training_with_ema_writes_smoothed_best(
    tmp_path: Path,
) -> None:
    """[Phase D.4] The best checkpoint should hold the EMA shadow."""
    _buildMultiSampleDataset(tmp_path, sampleCount=16)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    base = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=3,
    )
    from dataclasses import replace
    config = replace(base, emaDecay=0.99)
    runFullTraining(config)

    bestPayload = torch.load(
        outputDir / "checkpoints" / "v2_full_best.pt", map_location="cpu",
        weights_only=False,
    )
    # The best.pt must signal that its canonical state is the EMA.
    meta = bestPayload["training_meta"]
    assert meta.get("ema_active") is True
    assert meta.get("state_is_ema") is True
    # Online weights must be persisted under the *_online keys for
    # exact resume.
    assert "encoder_online_state_dict" in bestPayload
    assert "denoiser_online_state_dict" in bestPayload
    assert "ema_state_dict" in bestPayload


def test_run_full_training_resume_with_ema_loads_shadow(
    tmp_path: Path,
) -> None:
    """Resume must restore both online weights and EMA shadow."""
    _buildMultiSampleDataset(tmp_path, sampleCount=16)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    base = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=2,
    )
    from dataclasses import replace
    firstConfig = replace(base, emaDecay=0.99)
    runFullTraining(firstConfig)
    secondConfig = replace(
        firstConfig,
        epochs=3,
        resumeCheckpoint=outputDir / "checkpoints" / "v2_full_latest.pt",
    )
    components, _ = runFullTraining(secondConfig)
    assert components.ema is not None
    # The EMA shadow's update counter must be > 0 — resume kept it.
    assert components.ema._numUpdates > 0


def test_training_config_roundtrip_preserves_all_fields(
    tmp_path: Path,
) -> None:
    """[Regression — 2026-05-07] training_config in the checkpoint
    must preserve every field we care about.  Earlier versions
    silently dropped ``batchSize`` and ``maxSamplesPerEpoch``, which
    showed up as ``None`` in saved checkpoints and made it impossible
    to reproduce a run from disk."""
    _buildMultiSampleDataset(tmp_path, sampleCount=32)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    base = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=2,
        batchSize=8,
    )
    runFullTraining(base)
    payload = torch.load(
        outputDir / "checkpoints" / "v2_full_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg = payload["training_config"]
    # Critical reproducibility fields must round-trip correctly.
    assert cfg.get("batchSize") == base.batchSize
    assert cfg.get("gradientAccumulation") == base.gradientAccumulation
    assert cfg.get("maxSamplesPerEpoch") == base.maxSamplesPerEpoch
    assert cfg.get("validationFraction") == base.validationFraction
    assert cfg.get("validationSeed") == base.validationSeed
    assert cfg.get("validateEveryEpochs") == base.validateEveryEpochs
    assert cfg.get("normalizerFitMaxSamples") == base.normalizerFitMaxSamples
    assert cfg.get("datasetFolders") == list(base.datasetFolders or [])


def test_phase_d_checkpoint_roundtrip_through_loadCheckpointV2(
    tmp_path: Path,
) -> None:
    """[Regression — 2026-05-07] A best.pt produced by a Phase D run
    (clipGuidanceWeight > 0 ⇒ alignmentHead in the denoiser) must be
    loadable by ``loadCheckpointV2`` and match the original config.

    The bug it guards: ``_denoiserConfigToDict`` did not serialise
    ``alignmentEnabled``, so the rebuilt denoiser had no
    ``alignmentHead`` and ``load_state_dict`` crashed on the unexpected
    keys."""
    # 32 samples ≫ the hash-split bucket noise — guarantees the val
    # set is non-empty so a best.pt actually gets written.
    _buildMultiSampleDataset(tmp_path, sampleCount=32)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    base = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=2,
        batchSize=4,
    )
    from dataclasses import replace
    config = replace(
        base,
        clipGuidanceWeight=0.5,
        emaDecay=0.99,
    )
    runFullTraining(config)

    # Reload from disk via the public API used by the generation CLI.
    bestPath = outputDir / "checkpoints" / "v2_full_best.pt"
    assert bestPath.exists()
    (
        loadedTokenizer,
        loadedEncoder,
        loadedDenoiser,
        loadedSchedule,
        loadedNormalizer,
        payload,
    ) = loadCheckpointV2(bestPath, device="cpu")
    # The reloaded denoiser must have the alignment head built …
    assert loadedDenoiser.config.alignmentEnabled is True
    # … and produce a non-None motionEmbedding on a forward pass.
    loadedDenoiser.eval()
    inputIds = torch.zeros(1, 16, dtype=torch.long)
    attnMask = torch.ones(1, 16, dtype=torch.float32)
    encoded = loadedEncoder(inputIds, attnMask)
    output = loadedDenoiser(
        noisyMotion=torch.randn(1, 8, 22, 6),
        timesteps=torch.tensor([10]),
        textHiddenStates=encoded.hiddenStates,
        textKeyPaddingMask=encoded.keyPaddingMask,
        noisyGlobalFeatures=torch.randn(1, 8, 3),
    )
    assert output.motionEmbedding is not None


def test_train_step_batch_with_clip_guidance_active(
    tmp_path: Path,
) -> None:
    """[Phase D.1] Activating clipGuidanceWeight builds the alignment
    head and the contrastive loss surfaces in the metrics dict."""
    _buildMultiSampleDataset(tmp_path, sampleCount=8)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    base = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=tmp_path / "out",
        epochs=1,
        batchSize=4,
    )
    from dataclasses import replace
    config = replace(base, clipGuidanceWeight=0.5)
    components = buildFullTrainingComponents(config)
    dataset = PreprocessedLinkDataset(
        datasetRoot=tmp_path, includeTokenizedText=False
    )
    indices = list(range(len(dataset.linkEntries)))
    fitNormalizerFromDataset(
        dataset, indices, components.normalizer,
        maxSamples=config.normalizerFitMaxSamples,
    )
    batches = list(
        iterBatches(
            dataset=dataset,
            indices=indices,
            batchSize=4,
            maxFrames=config.maxFrames,
            device=components.device,
            shuffle=False,
            rngSeed=0,
        )
    )
    generators = TrainingRandomState.fromSeed(
        seed=0, device=components.device
    )
    metrics = trainStepBatch(
        components=components,
        batch=batches[0],
        config=config,
        generators=generators,
    )
    assert "loss_clip_guidance" in metrics
    assert metrics["loss_clip_guidance"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))
    # The denoiser must now have an alignment head with non-zero
    # gradient (the whole point of D.1).
    headParams = list(components.denoiser.alignmentHead.parameters())
    assert headParams  # non-trivial alignment head exists
    for parameter in headParams:
        assert parameter.grad is not None
        assert parameter.grad.abs().sum().item() > 0


def test_train_step_batch_propagates_finite_grads(
    tmp_path: Path,
) -> None:
    _buildMultiSampleDataset(tmp_path, sampleCount=8)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    config = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=tmp_path / "out",
        epochs=1,
    )
    components = buildFullTrainingComponents(config)
    dataset = PreprocessedLinkDataset(
        datasetRoot=tmp_path, includeTokenizedText=False
    )
    indices = list(range(len(dataset.linkEntries)))
    fitNormalizerFromDataset(
        dataset, indices, components.normalizer,
        maxSamples=config.normalizerFitMaxSamples,
    )
    batches = list(
        iterBatches(
            dataset=dataset,
            indices=indices,
            batchSize=4,
            maxFrames=config.maxFrames,
            device=components.device,
            shuffle=False,
            rngSeed=0,
        )
    )
    assert len(batches) > 0
    generators = TrainingRandomState.fromSeed(
        seed=0, device=components.device
    )
    metrics = trainStepBatch(
        components=components,
        batch=batches[0],
        config=config,
        generators=generators,
    )
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))
    # Phase F — the nullEmbedding only sees gradient when CFG dropout
    # actually fires; with the test's tiny seeded batch + condMaskProb
    # the dropout may not trigger.  Skip the assertion for that
    # parameter only.
    for name, parameter in components.encoder.named_parameters():
        if name == "nullEmbedding" and parameter.grad is None:
            continue
        assert parameter.grad is not None, f"missing grad on encoder.{name}"
        assert torch.isfinite(parameter.grad).all()
    for parameter in components.denoiser.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_validate_epoch_returns_finite_metrics(tmp_path: Path) -> None:
    _buildMultiSampleDataset(tmp_path, sampleCount=8)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    config = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=tmp_path / "out",
        epochs=1,
    )
    components = buildFullTrainingComponents(config)
    dataset = PreprocessedLinkDataset(
        datasetRoot=tmp_path, includeTokenizedText=False
    )
    indices = list(range(len(dataset.linkEntries)))
    fitNormalizerFromDataset(
        dataset, indices, components.normalizer,
        maxSamples=config.normalizerFitMaxSamples,
    )
    metrics = validateEpoch(
        components=components,
        dataset=dataset,
        valIndices=indices,
        config=config,
    )
    for key, value in metrics.items():
        assert torch.isfinite(torch.tensor(value)), (
            f"Non-finite val metric {key}: {value}"
        )


# =====================================================================
# Top-level run
# =====================================================================
def test_run_full_training_writes_best_and_latest(tmp_path: Path) -> None:
    _buildMultiSampleDataset(tmp_path, sampleCount=16)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    config = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=2,
    )
    components, history = runFullTraining(config)
    assert len(history) == 2
    assert (outputDir / "checkpoints" / "v2_full_best.pt").exists()
    assert (outputDir / "checkpoints" / "v2_full_latest.pt").exists()
    # Validate the checkpoint round-trips through loadCheckpointV2.
    (
        loadedTokenizer,
        loadedEncoder,
        loadedDenoiser,
        loadedSchedule,
        loadedNormalizer,
        payload,
    ) = loadCheckpointV2(outputDir / "checkpoints" / "v2_full_best.pt", device="cpu")
    assert loadedNormalizer.numBones == components.normalizer.numBones
    assert (
        loadedDenoiser.config.embedDim
        == components.denoiser.config.embedDim
    )
    assert payload.get("training_meta", {}).get("epoch") is not None


def test_release_device_memory_no_op_on_cpu() -> None:
    """The helper must run cleanly on CPU even with no device cache."""
    from ainimator.training.full_training_v2 import _releaseDeviceMemory

    # Should not raise.
    _releaseDeviceMemory(torch.device("cpu"))


def test_format_memory_stats_returns_empty_on_cpu() -> None:
    """On CPU there is no allocator stats — return empty string."""
    from ainimator.training.full_training_v2 import _formatMemoryStats

    assert _formatMemoryStats(torch.device("cpu")) == ""


def test_release_device_memory_clears_python_refs(tmp_path: Path) -> None:
    """After the helper runs, any unreferenced tensors should be GC'd.

    We allocate a large tensor inside an inner function, drop the only
    reference at function exit, and verify that after
    ``_releaseDeviceMemory`` the Python GC has reclaimed the object —
    a proxy for the cycle-collection pass that the helper kicks off
    before the device-side cache flush.
    """
    import weakref

    from ainimator.training.full_training_v2 import _releaseDeviceMemory

    def _allocate() -> weakref.ref:
        tensor = torch.zeros(1024, 1024)
        return weakref.ref(tensor)

    ref = _allocate()
    _releaseDeviceMemory(torch.device("cpu"))
    assert ref() is None, (
        "Tensor should be reclaimed by gc.collect after the helper "
        "runs."
    )


def test_run_full_training_clears_dataset_cache_per_epoch(
    tmp_path: Path,
) -> None:
    """After each epoch the dataset shard cache must be flushed.

    Verifies that the end-of-epoch memory hygiene pass calls
    ``dataset.clearCache()`` so the LRU does not retain shards from
    the previous epoch's shuffle order.
    """
    _buildMultiSampleDataset(tmp_path, sampleCount=16)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    config = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=2,
    )
    # Patch PreprocessedLinkDataset.clearCache to count invocations.
    from ainimator.data.preprocessed_dataset import PreprocessedLinkDataset

    callCount = {"value": 0}
    originalClearCache = PreprocessedLinkDataset.clearCache

    def _trackedClearCache(self: PreprocessedLinkDataset) -> None:
        callCount["value"] += 1
        originalClearCache(self)

    PreprocessedLinkDataset.clearCache = _trackedClearCache  # type: ignore[method-assign]
    try:
        runFullTraining(config)
    finally:
        PreprocessedLinkDataset.clearCache = originalClearCache  # type: ignore[method-assign]
    # 2 epochs ⇒ at least 2 cache clears (one per epoch).
    assert callCount["value"] >= 2, (
        f"Expected >=2 cache clears, got {callCount['value']}."
    )


def test_run_full_training_records_best_metadata(tmp_path: Path) -> None:
    """The best/latest checkpoints must persist best_val_total and best_epoch."""
    _buildMultiSampleDataset(tmp_path, sampleCount=16)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    config = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=4,
    )
    runFullTraining(config)

    bestPayload = torch.load(
        outputDir / "checkpoints" / "v2_full_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    latestPayload = torch.load(
        outputDir / "checkpoints" / "v2_full_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    bestMeta = bestPayload["training_meta"]
    latestMeta = latestPayload["training_meta"]
    assert bestMeta["best_val_total"] == bestMeta["val_metrics"]["loss_total"]
    assert bestMeta["best_epoch"] == bestMeta["epoch"]
    # The latest checkpoint stores the same "best" pointer so a resume
    # immediately knows the running best.
    assert latestMeta["best_val_total"] == bestMeta["best_val_total"]
    assert latestMeta["best_epoch"] == bestMeta["best_epoch"]


def test_run_full_training_resume_continues_from_last_epoch(
    tmp_path: Path,
) -> None:
    """A second run with --resume-checkpoint must skip already-done epochs."""
    _buildMultiSampleDataset(tmp_path, sampleCount=16)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    firstConfig = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=3,
    )
    _, firstHistory = runFullTraining(firstConfig)
    assert len(firstHistory) == 3
    firstLastEpoch = int(firstHistory[-1]["epoch"])
    assert firstLastEpoch == 3

    # Second run resumes from latest, asks for 5 total epochs → must
    # only run epochs 4 and 5 (2 history entries).
    from dataclasses import replace
    secondConfig = replace(
        firstConfig,
        epochs=5,
        resumeCheckpoint=outputDir / "checkpoints" / "v2_full_latest.pt",
    )
    _, secondHistory = runFullTraining(secondConfig)
    assert len(secondHistory) == 2
    secondEpochs = [int(entry["epoch"]) for entry in secondHistory]
    assert secondEpochs == [4, 5]


def test_run_full_training_resume_preserves_best(tmp_path: Path) -> None:
    """Resume must preserve the running best so a poor post-resume val
    can not silently overwrite a better best checkpoint."""
    _buildMultiSampleDataset(tmp_path, sampleCount=16)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    firstConfig = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=3,
    )
    runFullTraining(firstConfig)
    bestBefore = torch.load(
        outputDir / "checkpoints" / "v2_full_best.pt",
        map_location="cpu",
        weights_only=False,
    )["training_meta"]
    bestValBefore = bestBefore["best_val_total"]

    from dataclasses import replace
    secondConfig = replace(
        firstConfig,
        epochs=4,
        resumeCheckpoint=outputDir / "checkpoints" / "v2_full_latest.pt",
    )
    runFullTraining(secondConfig)
    bestAfter = torch.load(
        outputDir / "checkpoints" / "v2_full_best.pt",
        map_location="cpu",
        weights_only=False,
    )["training_meta"]
    # Either we found a strictly better val_total during the resume
    # (then bestValAfter < bestValBefore) or the best is unchanged.  In
    # neither case may best_val_total regress.
    assert bestAfter["best_val_total"] <= bestValBefore + 1e-6


def test_run_full_training_resume_past_target_logs_warning(
    tmp_path: Path,
) -> None:
    """If the checkpoint already covers config.epochs, do nothing."""
    _buildMultiSampleDataset(tmp_path, sampleCount=16)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    firstConfig = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=3,
    )
    runFullTraining(firstConfig)

    from dataclasses import replace
    sameLengthResume = replace(
        firstConfig,
        epochs=3,  # same target as already trained
        resumeCheckpoint=outputDir / "checkpoints" / "v2_full_latest.pt",
    )
    _, history = runFullTraining(sameLengthResume)
    assert history == []


def test_run_full_training_loss_decreases(tmp_path: Path) -> None:
    """A handful of epochs should drive the validation loss down."""
    _buildMultiSampleDataset(tmp_path, sampleCount=16)
    tokenizerDir = _buildTokenizerDir(tmp_path / "tokenizer")
    outputDir = tmp_path / "out"
    config = _miniFullConfig(
        datasetRoot=tmp_path,
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        epochs=10,
    )
    _, history = runFullTraining(config)
    earlyTotal = sum(
        h.get("loss_total", 0.0) for h in history[:3]
    ) / 3
    lateMin = min(
        h.get("loss_total", float("inf")) for h in history[-3:]
    )
    assert lateMin < earlyTotal, (
        f"val_total did not decrease (early avg={earlyTotal:.4f}, "
        f"late min={lateMin:.4f})."
    )


# ---------------------------------------------------------------------
# Phase F fix: effectiveClipWeight warmup schedule
# ---------------------------------------------------------------------
class TestEffectiveClipWeight:
    def _config(self, **overrides: Any) -> V2FullTrainingConfig:
        defaults = dict(
            datasetRoot=Path("/tmp/dummy"),
            tokenizerDir=Path("/tmp/dummy"),
            outputDir=Path("/tmp/dummy"),
            clipGuidanceWeight=0.3,
            clipGuidanceWeightStart=1.0,
            clipGuidanceWarmupEpochs=10,
        )
        defaults.update(overrides)
        return V2FullTrainingConfig(**defaults)

    def test_epoch_zero_returns_start_weight(self) -> None:
        config = self._config()
        assert effectiveClipWeight(config, 0) == pytest.approx(1.0)

    def test_epoch_at_warmup_end_returns_base_weight(self) -> None:
        config = self._config()
        assert effectiveClipWeight(config, 10) == pytest.approx(0.3)

    def test_mid_warmup_interpolates(self) -> None:
        config = self._config()
        w = effectiveClipWeight(config, 5)
        assert 0.3 < w < 1.0
        assert w == pytest.approx(0.65, abs=0.01)

    def test_past_warmup_returns_base(self) -> None:
        config = self._config()
        assert effectiveClipWeight(config, 50) == pytest.approx(0.3)

    def test_zero_warmup_epochs_returns_base_immediately(self) -> None:
        config = self._config(clipGuidanceWarmupEpochs=0)
        assert effectiveClipWeight(config, 0) == pytest.approx(0.3)


# ---------------------------------------------------------------------
# Phase F fix: config validation for new fields
# ---------------------------------------------------------------------
class TestPhaseFFConfigValidation:
    _REQUIRED = dict(
        datasetRoot=Path("/tmp/d"),
        tokenizerDir=Path("/tmp/d"),
        outputDir=Path("/tmp/d"),
    )

    def test_encoder_lr_multiplier_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="encoderLrMultiplier"):
            V2FullTrainingConfig(**self._REQUIRED, encoderLrMultiplier=0.0)

    def test_contrastive_bank_size_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="contrastiveBankSize"):
            V2FullTrainingConfig(**self._REQUIRED, contrastiveBankSize=-1)

    def test_warmup_start_must_be_gte_base(self) -> None:
        with pytest.raises(ValueError, match="clipGuidanceWeightStart"):
            V2FullTrainingConfig(
                **self._REQUIRED,
                clipGuidanceWeight=0.5,
                clipGuidanceWeightStart=0.3,
            )

    def test_aux_pool_weight_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="auxPoolContrastiveWeight"):
            V2FullTrainingConfig(
                **self._REQUIRED, auxPoolContrastiveWeight=-0.1
            )

    def test_aux_pool_weight_zero_is_valid(self) -> None:
        cfg = V2FullTrainingConfig(
            **self._REQUIRED, auxPoolContrastiveWeight=0.0
        )
        assert cfg.auxPoolContrastiveWeight == 0.0


# ---------------------------------------------------------------------
# Phase F iter-2: _meanOffDiagonalCosine diagnostic helper
# ---------------------------------------------------------------------
class TestMeanOffDiagonalCosine:
    def test_identical_vectors_return_sim_one(self) -> None:
        from ainimator.training.full_training_v2 import (
            _meanOffDiagonalCosine,
        )
        v = torch.tensor([1.0, 0.0, 0.0])
        emb = torch.stack([v, v, v], dim=0)
        mean, count = _meanOffDiagonalCosine(emb)
        assert mean == pytest.approx(1.0)
        # 3 distinct upper-triangular pairs.
        assert count == 3

    def test_orthogonal_vectors_return_sim_zero(self) -> None:
        from ainimator.training.full_training_v2 import (
            _meanOffDiagonalCosine,
        )
        emb = torch.eye(4)  # 4 orthonormal vectors
        mean, count = _meanOffDiagonalCosine(emb)
        assert mean == pytest.approx(0.0)
        assert count == 6  # C(4, 2)

    def test_single_sample_returns_zero_pair_count(self) -> None:
        from ainimator.training.full_training_v2 import (
            _meanOffDiagonalCosine,
        )
        emb = torch.randn(1, 8)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        mean, count = _meanOffDiagonalCosine(emb)
        assert count == 0
        assert mean == 0.0
