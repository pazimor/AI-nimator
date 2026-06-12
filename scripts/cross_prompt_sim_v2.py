"""Cross-prompt sensitivity probe for an AI-nimator v2 checkpoint.

The stock ``diagnose_generation_v2`` CLI only sweeps seeds and CFG for a
*single* prompt, so it cannot answer the most important question when a
model "ignores the text": **do two semantically different prompts produce
different motion?**  This read-only script fills that gap.

For a list of prompts it samples one motion each (same seed, same CFG)
and prints the pair-wise cosine similarity matrix of the flattened
rotation6d outputs.  Low off-diagonal values (≲ 0.6) mean the denoiser
actually uses the prompt; values ≳ 0.9 mean the conditioning path is
bypassed at inference even when the encoder itself discriminates.

Usage
-----
.. code-block:: bash

    python -m scripts.cross_prompt_sim_v2 \\
        --checkpoint output/generation_v2_phase2_v2/v2_full_best.pt \\
        --cfg 2.5 --seed 0 --frames 120 --num-steps 100 --device mps
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import torch

from src.features.generation.training_v2 import (
    EMPTY_PROMPT,
    loadCheckpointV2,
    resolveDevice,
)
from src.shared.model.generation.sampler_v2 import DDIMSamplerV2

LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPTS: tuple[str, ...] = (
    "a person walks forward",
    "a person is jumping",
    "a person sits down on a chair",
    "a person waves with the right hand",
    "a person runs quickly",
)


def _cosineSimilarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened 1-D tensors."""
    dot = float(torch.dot(a, b).item())
    norms = float((a.norm() * b.norm()).item())
    if norms == 0.0:
        return 0.0
    return dot / norms


def _centered(motion: torch.Tensor) -> torch.Tensor:
    """Subtract the per-channel temporal mean (removes the shared DC pose).

    Raw rotation6d carries a large constant component (the rest-pose-ish
    6D values shared by every plausible human motion), so the raw cosine
    floors around ~0.9 even for radically different motions.  Centering
    on the per-frame mean exposes the motion-specific variation.
    """
    flat = motion[: motion.shape[0]].float()
    return flat - flat.mean(dim=0, keepdim=True)


def _fkPositions(motion: torch.Tensor) -> torch.Tensor:
    """FK to pelvis-relative joint XYZ — a semantic motion descriptor."""
    from src.shared.model.components.ops import rot6dToJointXYZ

    xyz = rot6dToJointXYZ(motion.unsqueeze(0).float()).squeeze(0)
    return xyz - xyz[:, :1]


def _encode(tokenizer, encoder, prompt: str, device: torch.device):
    """Tokenise then encode a single prompt on ``device``."""
    encoded = tokenizer.encode(prompt)
    return encoder(
        encoded.inputIds.to(device),
        encoded.attentionMask.to(device),
    )


def _sampleBone(
    sampler: DDIMSamplerV2,
    denoiser,
    condOutput,
    uncondOutput,
    arguments: argparse.Namespace,
    device: torch.device,
    normalizer,
) -> torch.Tensor:
    """Sample one motion and return its flattened bone tensor (CPU)."""
    useCfg = float(arguments.cfg) != 1.0
    output = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=condOutput.hiddenStates,
        textKeyPaddingMask=condOutput.keyPaddingMask,
        unconditionalTextHiddenStates=(
            uncondOutput.hiddenStates if useCfg else None
        ),
        unconditionalTextKeyPaddingMask=(
            uncondOutput.keyPaddingMask if useCfg else None
        ),
        frames=int(arguments.frames),
        numSteps=int(arguments.numSteps),
        cfgScale=float(arguments.cfg),
        eta=0.0,
        device=device,
        guidanceRescale=float(arguments.guidanceRescale),
        seed=int(arguments.seed),
        normalizer=normalizer,
    )
    return output.boneMotion[0].detach().float().cpu()


def _offDiagonalAverage(
    motions: list[torch.Tensor],
    transform,
) -> float:
    """Mean off-diagonal cosine after applying ``transform`` to each motion."""
    transformed = [transform(motion).reshape(-1) for motion in motions]
    sims: list[float] = []
    for rowIndex in range(len(transformed)):
        for colIndex in range(rowIndex + 1, len(transformed)):
            sims.append(
                _cosineSimilarity(transformed[rowIndex], transformed[colIndex])
            )
    return sum(sims) / len(sims) if sims else 0.0


def _logMatrix(prompts: Sequence[str], motions: list[torch.Tensor]) -> None:
    """Report cross-prompt similarity under raw, centered and FK metrics.

    Reference values for *genuinely different* real training motions
    (so you know what "uses the prompt" actually looks like):
      raw ≈ 0.88–0.98   centered ≈ 0.0–0.25   fk ≈ -0.8…0.1
    The raw rotation6d cosine is dominated by the shared DC pose and is a
    poor discriminator; trust the centered and FK rows.
    """
    rawAvg = _offDiagonalAverage(motions, lambda m: m)
    cenAvg = _offDiagonalAverage(motions, _centered)
    fkAvg = _offDiagonalAverage(motions, _fkPositions)
    LOGGER.info("Cross-prompt off-diagonal similarity (lower = prompt used):")
    LOGGER.info("  raw rot6d   = %.4f   (real-motion ref ~0.9; misleading)",
                rawAvg)
    LOGGER.info("  centered    = %.4f   (real-motion ref ~0.1)", cenAvg)
    LOGGER.info("  FK positions= %.4f   (real-motion ref ~0.05)", fkAvg)
    meanStd = sum(float(m.std()) for m in motions) / len(motions)
    LOGGER.info("  mean rot6d std = %.4f   (train ref ~0.15)", meanStd)


def buildArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-prompt sensitivity probe for a v2 checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--num-steps", dest="numSteps", type=int, default=100)
    parser.add_argument(
        "--guidance-rescale",
        dest="guidanceRescale",
        type=float,
        default=0.0,
        help="Guidance-rescale strength in [0,1] (0 disables).",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Optional ';'-separated prompt list (overrides defaults).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = buildArgumentParser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not arguments.checkpoint.exists():
        LOGGER.error("Checkpoint not found: %s", arguments.checkpoint)
        return 1

    device = resolveDevice(arguments.device)
    tokenizer, encoder, denoiser, schedule, normalizer, payload = (
        loadCheckpointV2(arguments.checkpoint, device=device)
    )
    encoder.eval()
    denoiser.eval()
    sampler = DDIMSamplerV2(
        schedule,
        predictionMode=str(payload["training_config"]["predictionMode"]),
    )

    prompts = (
        tuple(p.strip() for p in arguments.prompts.split(";") if p.strip())
        if arguments.prompts
        else DEFAULT_PROMPTS
    )
    uncondOutput = _encode(tokenizer, encoder, EMPTY_PROMPT, device)
    LOGGER.info("cfg=%.1f seed=%d frames=%d", arguments.cfg,
                arguments.seed, arguments.frames)

    motions: list[torch.Tensor] = []
    with torch.no_grad():
        for prompt in prompts:
            condOutput = _encode(tokenizer, encoder, prompt, device)
            motions.append(
                _sampleBone(
                    sampler, denoiser, condOutput, uncondOutput,
                    arguments, device, normalizer,
                )
            )
    _logMatrix(prompts, motions)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
