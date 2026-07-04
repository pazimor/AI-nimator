"""CLI — encode a text prompt into a ControlPreset embedding (B3-bis).

Zero logic: load the frozen text encoder artifact, encode one prompt
into a pooled embedding, and write it as JSON consumable by the engine
plugins (``prompt_emb`` field of ``control_preset.schema.json``, or fed
at runtime via ``SetPromptEmbedding``).

Example
-------
``python -m ainimator.cli.encode_prompt \\
    --prompt "a person walks forward" \\
    --encoder-artifact output/clip_text_artifact \\
    --output output/presets/walk_prompt.json``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from ainimator.training.controller_training_v2 import (
    encodeTextToPooled,
    loadFrozenTextEncoder,
)


def _parseArgs() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Encode a prompt into a preset prompt_emb JSON."
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--encoder-artifact",
        dest="encoderArtifact",
        type=Path,
        required=True,
        metavar="DIR",
        help="Frozen text encoder artifact directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="FILE",
        help="Destination JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Encode one prompt and write the embedding JSON."""
    logging.basicConfig(level=logging.INFO)
    args = _parseArgs()
    device = torch.device("cpu")
    encoder, tokenizer = loadFrozenTextEncoder(args.encoderArtifact, device)
    pooled = encodeTextToPooled([args.prompt], encoder, tokenizer, device)
    payload = {
        "prompt": args.prompt,
        "prompt_emb": pooled[0].detach().cpu().float().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    logging.info(
        "prompt encoded (dim=%d) -> %s", pooled.shape[-1], args.output
    )


if __name__ == "__main__":
    main()
