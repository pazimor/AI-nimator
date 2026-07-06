"""End-to-end tests for :mod:`ainimator.training.debug_runner`.

These tests exercise the full debug pipeline on a synthetic micro-dataset:

* :func:`buildDebugConfig` produces a :class:`V2TrainingConfig` with the
  expected reduced dims and ``healthEverySteps=1``.
* :func:`runDebug` trains end-to-end, writes a checkpoint, produces a
  generated artefact (``.dae``), and writes per-step health JSONL entries.

All tests run on CPU in well under a minute so they live in the regular
test suite (no marks needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ainimator.training.debug_runner import (
    DebugTrainingConfig,
    buildDebugConfig,
    runDebug,
    _DEBUG_HEALTH_EVERY_STEPS,
    _DEBUG_EPOCHS,
    _DEBUG_DENOISER_EMBED_DIM,
    _DEBUG_DENOISER_NUM_LAYERS,
)
from ainimator.training.training_v2 import V2TrainingConfig
from ainimator.text import CustomTokenizer, CustomTokenizerConfig


# =====================================================================
# Synthetic dataset helpers (mirrors test_training_v2 for isolation)
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
            "meta": {
                "fps": 30,
                "frames": frames,
                "joints": "SMPL-22",
            },
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
                    "path": str(
                        sampleShardPath.relative_to(root)
                    ),
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
        [{"sampleId": 0, "textId": 0, "pairBytes": 1}],
    )


def _buildTokenizerDir(root: Path) -> Path:
    """Train and save a tiny BPE tokenizer."""
    tokenizer = CustomTokenizer.train(
        [
            "a person walks forward.",
            "the woman dances.",
            "someone runs.",
        ]
        * 4,
        config=CustomTokenizerConfig(
            vocabSize=128,
            maxLength=16,
            minFrequency=1,
        ),
    )
    tokenizer.save(root)
    return root


# =====================================================================
# Tests
# =====================================================================

def test_build_debug_config_reduced_dims(tmp_path: Path) -> None:
    """buildDebugConfig must return a V2TrainingConfig with reduced dims."""
    debug = DebugTrainingConfig(
        datasetRoot=tmp_path / "data",
        tokenizerDir=tmp_path / "tok",
        outputDir=tmp_path / "out",
        device="cpu",
    )
    config = buildDebugConfig(debug)

    assert isinstance(config, V2TrainingConfig)
    assert config.denoiserEmbedDim == _DEBUG_DENOISER_EMBED_DIM
    assert config.denoiserNumLayers == _DEBUG_DENOISER_NUM_LAYERS
    # health probes must fire at every step in debug mode.
    assert config.healthEverySteps == _DEBUG_HEALTH_EVERY_STEPS
    assert config.healthEverySteps == 1
    assert config.epochs == _DEBUG_EPOCHS
    # Production defaults must not be mutated.
    from ainimator.training.training_v2 import V2TrainingConfig as _Prod
    prod_default = _Prod(
        datasetRoot=tmp_path / "data",
        tokenizerDir=tmp_path / "tok",
        outputDir=tmp_path / "out",
    )
    assert config.denoiserEmbedDim != prod_default.denoiserEmbedDim


def test_run_debug_produces_checkpoint_and_generation(
    tmp_path: Path,
) -> None:
    """runDebug must produce a checkpoint and a .dae generation artefact."""
    _buildSyntheticDataset(tmp_path / "data")
    tokenizerDir = _buildTokenizerDir(tmp_path / "tok")
    outputDir = tmp_path / "out"

    debug = DebugTrainingConfig(
        datasetRoot=tmp_path / "data",
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        sampleLinkIndex=0,
        seed=42,
        device="cpu",
    )
    result = runDebug(debug)

    # Checkpoint must exist on disk.
    assert result.checkpointPath.exists(), (
        f"Checkpoint not found: {result.checkpointPath}"
    )

    # Generation artefact must exist on disk.
    assert result.generationPath.exists(), (
        f"Generation artefact not found: {result.generationPath}"
    )
    assert result.generationPath.suffix == ".dae"

    # Elapsed time reported (no hard wall-clock assertion on CPU).
    assert result.elapsedSeconds > 0.0


def test_run_debug_writes_health_jsonl_per_step(
    tmp_path: Path,
) -> None:
    """Health JSONL must contain one entry per training step (everySteps=1)."""
    _buildSyntheticDataset(tmp_path / "data")
    tokenizerDir = _buildTokenizerDir(tmp_path / "tok")
    outputDir = tmp_path / "out"

    debug = DebugTrainingConfig(
        datasetRoot=tmp_path / "data",
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        device="cpu",
    )
    result = runDebug(debug)

    # At least one JSONL file must exist.
    healthDir = outputDir / "health"
    assert healthDir.exists(), "health/ subdirectory not found."
    jsonlFiles = list(healthDir.glob("*.jsonl"))
    assert jsonlFiles, "No health JSONL files found."

    # Count lines across all JSONL files.
    lines: list[str] = []
    for jsonlFile in jsonlFiles:
        for line in jsonlFile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                lines.append(line)

    # With everySteps=1 we expect at least _DEBUG_EPOCHS entries.
    assert len(lines) >= _DEBUG_EPOCHS, (
        f"Expected >= {_DEBUG_EPOCHS} JSONL lines (one per step), "
        f"got {len(lines)}."
    )

    # Each line must be valid JSON with a 'step' key.
    for raw in lines:
        record = json.loads(raw)
        assert "step" in record, f"Missing 'step' key in JSONL line: {raw}"


def test_run_debug_health_jsonl_path_returned(
    tmp_path: Path,
) -> None:
    """DebugRunResult.healthJsonlPath must point inside the health/ dir."""
    _buildSyntheticDataset(tmp_path / "data")
    tokenizerDir = _buildTokenizerDir(tmp_path / "tok")
    outputDir = tmp_path / "out"

    debug = DebugTrainingConfig(
        datasetRoot=tmp_path / "data",
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        device="cpu",
    )
    result = runDebug(debug)

    healthDir = outputDir / "health"
    # The returned path must be inside health/.
    assert str(result.healthJsonlPath).startswith(str(healthDir)), (
        f"healthJsonlPath {result.healthJsonlPath} is not inside "
        f"{healthDir}."
    )


def test_run_debug_health_verdicts_not_all_unknown(
    tmp_path: Path,
) -> None:
    """Debug run JSONL must contain non-UNKNOWN verdicts.

    A3 probe-attachment bug caused all 5 contracts to evaluate as
    UNKNOWN (probes never fired → no metric data).  After the A6
    fix the probe-derived contracts (update_ratio, loss_decomposition)
    must produce real verdicts (OK / WARNING / CRITICAL).

    Specifically asserts that:
    - ``verdict.update_ratio`` is NOT UNKNOWN (probe attached to
      denoiser.outputProjection and fired).
    - ``verdict.loss_decomposition`` is NOT UNKNOWN (loss_share
      is computed by trainStep and passed to hub.step).
    - ``verdict.post_norm_stats`` is NOT UNKNOWN (computed in
      runOverfit and passed to hub.step).
    """
    _buildSyntheticDataset(tmp_path / "data")
    tokenizerDir = _buildTokenizerDir(tmp_path / "tok")
    outputDir = tmp_path / "out"

    debug = DebugTrainingConfig(
        datasetRoot=tmp_path / "data",
        tokenizerDir=tokenizerDir,
        outputDir=outputDir,
        device="cpu",
    )
    runDebug(debug)

    healthDir = outputDir / "health"
    jsonlFiles = sorted(healthDir.glob("*.jsonl"))
    assert jsonlFiles, "No health JSONL files"

    # Collect all records.
    records = []
    for jsonlFile in jsonlFiles:
        for line in jsonlFile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert records, "No records in JSONL files"

    # Check the LAST record (most likely to have all metrics populated).
    lastRecord = records[-1]

    # --- Probe-derived verdict: update_ratio -----------------------
    # Requires the output_head probe to attach to
    # denoiser.outputProjection (not the non-existent denoiser.outputProj).
    updateRatioVerdict = lastRecord.get("verdict.update_ratio")
    assert updateRatioVerdict is not None, (
        "verdict.update_ratio missing from JSONL — contract not evaluated"
    )
    assert updateRatioVerdict != "UNKNOWN", (
        f"verdict.update_ratio is UNKNOWN — "
        f"probe likely failed to attach or produced no data. "
        f"Record keys: {list(lastRecord.keys())}"
    )

    # --- Loss-share verdict: loss_decomposition --------------------
    lossVerdict = lastRecord.get("verdict.loss_decomposition")
    assert lossVerdict is not None, (
        "verdict.loss_decomposition missing from JSONL"
    )
    assert lossVerdict != "UNKNOWN", (
        f"verdict.loss_decomposition is UNKNOWN — "
        f"loss_share not passed to hub.step. "
        f"Record keys: {list(lastRecord.keys())}"
    )

    # --- Post-norm verdict: post_norm_stats ------------------------
    postNormVerdict = lastRecord.get("verdict.post_norm_stats")
    assert postNormVerdict is not None, (
        "verdict.post_norm_stats missing from JSONL"
    )
    assert postNormVerdict != "UNKNOWN", (
        f"verdict.post_norm_stats is UNKNOWN — "
        f"post_norm_stats not wired into overfit loop. "
        f"Record keys: {list(lastRecord.keys())}"
    )
