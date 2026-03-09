"""Utilities to build generation text embedding caches."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from src.shared.constants.preprocessed import (
    PREPROCESSED_GENERATION_TEXT_CACHE_MANIFEST_FILENAME,
    PREPROCESSED_GENERATION_TEXT_CACHE_VERSION,
    PREPROCESSED_GENERATION_TEXT_EMBED_SHARDS_DIRNAME,
)
from src.shared.model.clip.core import ClipModel
from src.shared.preprocessed_dataset import (
    PreprocessedLinkDataset,
    computeClipCheckpointFingerprint,
    resolveGenerationTextCacheDir,
)
from src.shared.types import (
    GenerationTextCacheManifest,
    PreprocessedDatasetShardInfo,
)

LOGGER = logging.getLogger("shared.generation_text_cache")
DEFAULT_GENERATION_CACHE_BATCH_SIZE = 256


class _PooledTextEncoderStub(torch.nn.Module):
    """Minimal encoder stub used to construct ClipModel without HF weights."""

    def __init__(self, hiddenSize: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hiddenSize)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        del args, kwargs
        raise RuntimeError(
            "The pooled-text cache builder should not call the text encoder."
        )


def resolveGenerationTextCacheManifestPath(
    datasetRoot: Path,
    checkpointPath: Path,
) -> Path:
    """Return the manifest path for a checkpoint-specific generation cache."""
    clipFingerprint = computeClipCheckpointFingerprint(checkpointPath)
    cacheDir = resolveGenerationTextCacheDir(datasetRoot, clipFingerprint)
    return cacheDir / PREPROCESSED_GENERATION_TEXT_CACHE_MANIFEST_FILENAME


def ensureGenerationTextCache(
    datasetRoot: Path,
    checkpointPath: Path,
    batchSize: int = DEFAULT_GENERATION_CACHE_BATCH_SIZE,
) -> Path:
    """Return an existing generation cache or build it when missing."""
    manifestPath = resolveGenerationTextCacheManifestPath(
        datasetRoot=datasetRoot,
        checkpointPath=checkpointPath,
    )
    if manifestPath.exists():
        return manifestPath.parent
    return rebuildGenerationTextCache(
        datasetRoot=datasetRoot,
        checkpointPath=checkpointPath,
        batchSize=batchSize,
    )


def rebuildGenerationTextCache(
    datasetRoot: Path,
    checkpointPath: Path,
    batchSize: int = DEFAULT_GENERATION_CACHE_BATCH_SIZE,
) -> Path:
    """Build or rebuild the generation text cache for a CLIP checkpoint."""
    if batchSize <= 0:
        raise ValueError("batchSize must be strictly positive.")

    resolvedDatasetRoot = datasetRoot.expanduser().resolve()
    resolvedCheckpoint = checkpointPath.expanduser().resolve()
    clipFingerprint = computeClipCheckpointFingerprint(resolvedCheckpoint)
    cacheDir = resolveGenerationTextCacheDir(
        resolvedDatasetRoot,
        clipFingerprint,
    )
    manifestPath = (
        cacheDir / PREPROCESSED_GENERATION_TEXT_CACHE_MANIFEST_FILENAME
    )
    if cacheDir.exists():
        shutil.rmtree(cacheDir)

    LOGGER.info(
        "Building generation text cache for %s in %s",
        resolvedCheckpoint,
        cacheDir,
    )
    dataset = PreprocessedLinkDataset(resolvedDatasetRoot)
    model = _buildProjectionModel(dataset, resolvedCheckpoint)
    shardsDir = cacheDir / PREPROCESSED_GENERATION_TEXT_EMBED_SHARDS_DIRNAME
    shardsDir.mkdir(parents=True, exist_ok=True)
    shardInfos = _writeCacheShards(
        dataset=dataset,
        model=model,
        cacheDir=cacheDir,
        shardsDir=shardsDir,
        batchSize=batchSize,
    )
    manifest = GenerationTextCacheManifest(
        version=PREPROCESSED_GENERATION_TEXT_CACHE_VERSION,
        clipFingerprint=clipFingerprint,
        embedDim=model.textProj.out_features,
        totalTexts=len(dataset.textIndexEntries),
        shardSize=dataset.manifest.textShardSize,
        shards=shardInfos,
    )
    payload = {
        "version": manifest.version,
        "clipFingerprint": manifest.clipFingerprint,
        "embedDim": manifest.embedDim,
        "totalTexts": manifest.totalTexts,
        "shardSize": manifest.shardSize,
        "shards": [
            {"path": entry.path, "sampleCount": entry.sampleCount}
            for entry in manifest.shards
        ],
    }
    manifestPath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cacheDir


def _buildProjectionModel(
    dataset: PreprocessedLinkDataset,
    checkpointPath: Path,
) -> ClipModel:
    checkpoint = torch.load(
        checkpointPath,
        map_location="cpu",
        weights_only=False,
    )
    stateDict = checkpoint.get("model_state_dict")
    if not isinstance(stateDict, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")
    textProjWeight = stateDict.get("textProj.weight")
    textProjBias = stateDict.get("textProj.bias")
    if not isinstance(textProjWeight, torch.Tensor):
        raise ValueError("Checkpoint is missing textProj.weight.")
    if not isinstance(textProjBias, torch.Tensor):
        raise ValueError("Checkpoint is missing textProj.bias.")
    embedDim = int(textProjWeight.shape[0])
    model = ClipModel(
        tokenizer=object(),
        textEncoder=_PooledTextEncoderStub(dataset.manifest.pooledTextDim),
        freezeTextEncoder=False,
        embedDim=embedDim,
        numBones=1,
    )
    model.textProj.load_state_dict(
        {
            "weight": textProjWeight,
            "bias": textProjBias,
        }
    )
    model.eval()
    return model


def _writeCacheShards(
    dataset: PreprocessedLinkDataset,
    model: ClipModel,
    cacheDir: Path,
    shardsDir: Path,
    batchSize: int,
) -> list[PreprocessedDatasetShardInfo]:
    shardInfos: list[PreprocessedDatasetShardInfo] = []
    with torch.no_grad():
        for shardIndex in range(len(dataset.manifest.textShards)):
            textPayloads = dataset._loadTextShard(shardIndex)
            cacheEntries: list[dict[str, object]] = []
            for start in range(0, len(textPayloads), batchSize):
                chunk = textPayloads[start:start + batchSize]
                pooledText = torch.stack(
                    [entry["pooled_text"] for entry in chunk]
                )
                textEmbeddings, _ = model.encodePooledText(pooledText)
                for offset, entry in enumerate(chunk):
                    cacheEntries.append(
                        {
                            "text_id": int(entry["text_id"]),
                            "text_embedding": textEmbeddings[offset].detach().cpu(),
                        }
                    )
            shardPath = shardsDir / f"text_embed_shard_{shardIndex:05d}.pt"
            torch.save(cacheEntries, shardPath)
            shardInfos.append(
                PreprocessedDatasetShardInfo(
                    path=str(shardPath.relative_to(cacheDir)),
                    sampleCount=len(cacheEntries),
                )
            )
    return shardInfos
