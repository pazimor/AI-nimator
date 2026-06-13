"""CLI entry-point for generation-specific cached text embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from ainimator.training.generation_text_cache import (
    ensureGenerationTextCache,
    rebuildGenerationTextCache,
    resolveGenerationTextCacheManifestPath,
)


def buildArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute generation text embeddings for a CLIP checkpoint.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the V2 preprocessed dataset root.",
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        required=True,
        help="Path to the CLIP checkpoint used by train_generation.",
    )
    parser.add_argument(
        "--network-config",
        type=Path,
        default=None,
        help="Accepted for interface parity; unused by the V2 cache builder.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Accepted for interface parity; unused by the V2 cache builder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of unique texts projected per batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing cache for the same checkpoint fingerprint.",
    )
    return parser


def main() -> None:
    parser = buildArgumentParser()
    arguments = parser.parse_args()
    del arguments.network_config, arguments.profile

    if arguments.batch_size <= 0:
        parser.exit(1, "--batch-size must be strictly positive.\n")

    datasetRoot = arguments.dataset_root.expanduser().resolve()
    checkpointPath = arguments.clip_checkpoint.expanduser().resolve()
    manifestPath = resolveGenerationTextCacheManifestPath(
        datasetRoot=datasetRoot,
        checkpointPath=checkpointPath,
    )
    if manifestPath.exists() and not arguments.overwrite:
        parser.exit(
            1,
            "Generation text cache already exists for this checkpoint. "
            "Pass --overwrite to rebuild it.\n",
        )
    cacheDir = (
        rebuildGenerationTextCache(
            datasetRoot=datasetRoot,
            checkpointPath=checkpointPath,
            batchSize=arguments.batch_size,
        )
        if arguments.overwrite
        else ensureGenerationTextCache(
            datasetRoot=datasetRoot,
            checkpointPath=checkpointPath,
            batchSize=arguments.batch_size,
        )
    )
    parser.exit(
        0,
        f"Generation text cache written to {cacheDir}\n",
    )


if __name__ == "__main__":
    main()
