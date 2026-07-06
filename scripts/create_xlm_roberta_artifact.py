"""Bootstrap a frozen XLM-RoBERTa encoder artifact for controller text-cond.

Creates output/xlm_roberta_artifact/ with the encoder_config.yaml,
weights.pt, and hash.txt expected by loadEncoderArtifact().

Usage:
    poetry run python scripts/create_xlm_roberta_artifact.py
    poetry run python scripts/create_xlm_roberta_artifact.py \
        --output-dir output/xlm_roberta_artifact
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

MODEL_NAME = "xlm-roberta-base"
BACKBONE_HIDDEN_DIM = 768
OUTPUT_DIM = 768
MAX_LENGTH = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/xlm_roberta_artifact"),
    )
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from ainimator.text.clip_text_encoder import (
        ClipTextEncoder,
        ClipTextEncoderConfig,
    )
    from ainimator.text.artifact import saveEncoderArtifact

    print(f"Loading {MODEL_NAME} …")
    config = ClipTextEncoderConfig(
        modelName=MODEL_NAME,
        maxLength=MAX_LENGTH,
        outputDim=OUTPUT_DIM,
        clipHiddenDim=BACKBONE_HIDDEN_DIM,
        dropout=0.0,
        useNullEmbedding=True,
        l2NormalizeOutput=True,
    )
    encoder = ClipTextEncoder(config)

    print(f"Saving artifact to {args.output_dir} …")
    digest = saveEncoderArtifact(encoder, args.output_dir, tokenizer=None)
    print(f"Done. weights SHA-256: {digest[:16]}…")
    print(f"Artifact: {args.output_dir}")


if __name__ == "__main__":
    main()
