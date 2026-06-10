"""CLI to diagnose a v2 checkpoint by sweeping seeds × CFG scales.

What this gives you
-------------------
* **Determinism check** — same seed → same output.  Run twice with
  ``--seeds 42 --cfg-scales 1.0`` and the stats must be byte-identical.
* **CFG sensitivity** — how much does a higher CFG affect rotation
  amplitude / displacement?  A model that ignores text will show flat
  rows.
* **Tremblement signal** — the Δ²rot/frame column.  A static pose has
  ~0, a tremor has high values relative to a known-good baseline.
* **Distribution sanity** — rot μ / σ should match the training stats
  (printed by ``train_generation_v2`` after ``Fitting normalizer …``).
  Far-off numbers point to a normalizer or schedule issue.

Usage example
-------------
.. code-block:: bash

    poetry run python -m src.cli.diagnose_generation_v2 \\
        --checkpoint output/generation_v2/v2_full_best.pt \\
        --prompt "a person walking forward." \\
        --frames 120 \\
        --num-steps 100 \\
        --seeds 0,42,123 \\
        --cfg-scales 1.0,2.5,3.5 \\
        --device mps

Add ``--export-dir <path>`` to also write one ``.dae`` per (seed, cfg)
combination so you can compare them visually in Blender.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import torch

from src.features.dataset_builder.animation_rebuilder import (
    AnimationRebuilder,
)
from src.features.generation.diagnose_v2 import (
    computeGenerationStats,
    formatStatsLine,
)
from src.features.generation.training_v2 import (
    EMPTY_PROMPT,
    loadCheckpointV2,
    resolveDevice,
)
from src.shared.constants.skeletons import SMPL22_BONE_ORDER
from src.shared.model.generation.sampler_v2 import DDIMSamplerV2
from src.shared.types import (
    DatasetBuilderConfig,
    DatasetBuilderPaths,
    DatasetBuilderProcessing,
)

LOGGER = logging.getLogger(__name__)


def _parseFloatList(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parseIntList(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def buildArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose a v2 checkpoint by sweeping seeds and CFG scales."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the v2 checkpoint .pt file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt used to condition the generation.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of motion frames to generate. (default: %(default)s)",
    )
    parser.add_argument(
        "--num-steps",
        dest="numSteps",
        type=int,
        default=100,
        help="DDIM sampling steps. (default: %(default)s)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,42,123",
        help=(
            "Comma-separated seeds for the initial noise. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--cfg-scales",
        dest="cfgScales",
        type=str,
        default="1.0,2.5,3.5",
        help=(
            "Comma-separated CFG scales to sweep. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--export-dir",
        dest="exportDir",
        type=Path,
        default=None,
        help=(
            "When set, write one .dae per (seed, cfg) combination "
            "named diag_seed{N}_cfg{X.X}.dae."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS written into the optional .dae exports.",
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = buildArgumentParser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.logLevel),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not arguments.checkpoint.exists():
        LOGGER.error("Checkpoint not found: %s", arguments.checkpoint)
        return 1
    seeds = _parseIntList(arguments.seeds)
    cfgScales = _parseFloatList(arguments.cfgScales)
    if not seeds or not cfgScales:
        LOGGER.error("--seeds and --cfg-scales must each have ≥ 1 value.")
        return 1

    device = resolveDevice(arguments.device)
    LOGGER.info("Loading checkpoint %s on %s.", arguments.checkpoint, device)
    (
        tokenizer,
        encoder,
        denoiser,
        schedule,
        normalizer,
        payload,
    ) = loadCheckpointV2(arguments.checkpoint, device=device)
    encoder.eval()
    denoiser.eval()
    sampler = DDIMSamplerV2(
        schedule,
        predictionMode=str(payload["training_config"]["predictionMode"]),
    )

    encodedCond = tokenizer.encode(arguments.prompt)
    condOutput = encoder(
        encodedCond.inputIds.to(device),
        encodedCond.attentionMask.to(device),
    )
    encodedUncond = tokenizer.encode(EMPTY_PROMPT)
    uncondOutput = encoder(
        encodedUncond.inputIds.to(device),
        encodedUncond.attentionMask.to(device),
    )

    # Sanity check on the encoder side: compute the average cosine
    # similarity between the conditional and unconditional pooled token
    # representations.  If this is ≈1.0, the encoder collapsed.  If
    # it is < 0.95 but the sampler still gives cfg_sim ≈ 1.0, the
    # collapse is on the denoiser cross-attention path instead.
    condTokens = condOutput.hiddenStates[0]
    uncondTokens = uncondOutput.hiddenStates[0]
    condRealMask = ~condOutput.keyPaddingMask[0]
    uncondRealMask = ~uncondOutput.keyPaddingMask[0]
    condPooled = (
        (condTokens * condRealMask.unsqueeze(-1).float()).sum(dim=0)
        / condRealMask.float().sum().clamp(min=1)
    )
    uncondPooled = (
        (uncondTokens * uncondRealMask.unsqueeze(-1).float()).sum(dim=0)
        / uncondRealMask.float().sum().clamp(min=1)
    )
    encoderSim = float(
        torch.nn.functional.cosine_similarity(
            condPooled.detach().float(),
            uncondPooled.detach().float(),
            dim=0,
        ).item()
    )
    LOGGER.info(
        "Encoder cond↔uncond similarity (pooled, masked): %.4f  "
        "(cond_norm=%.3f uncond_norm=%.3f)",
        encoderSim,
        float(condPooled.norm().item()),
        float(uncondPooled.norm().item()),
    )

    # Header
    LOGGER.info("=" * 78)
    LOGGER.info(
        "Diagnostic sweep: %d seed × %d cfg = %d configs",
        len(seeds),
        len(cfgScales),
        len(seeds) * len(cfgScales),
    )
    LOGGER.info("Prompt: %r", arguments.prompt)
    trainingPrompt = payload.get("training_sample", {}).get("rawText")
    if trainingPrompt:
        LOGGER.info("Training prompt (overfit only): %r", trainingPrompt)

    # Print the training distribution as a baseline — generated stats
    # that depart far from this baseline indicate a denormalisation or
    # mode-collapse issue.
    if normalizer is not None:
        boneMean = normalizer.boneMean.float().cpu()
        boneStd = normalizer.boneStd.float().cpu()
        LOGGER.info(
            "Training BONE stats (per-channel avg, the model targets "
            "this distribution): rot μ=%+.3f σ=%.3f  bone=[%.3f, %.3f]",
            float(boneMean.mean().item()),
            float(boneStd.mean().item()),
            float((boneMean - 3 * boneStd).min().item()),
            float((boneMean + 3 * boneStd).max().item()),
        )
        if normalizer.hasGlobalBranch:
            globalMean = normalizer.globalMean.float().cpu()
            globalStd = normalizer.globalStd.float().cpu()
            LOGGER.info(
                "Training GLOBAL stats: rt μ=%+.3f σ=%.3f  "
                "global=[%.3f, %.3f] (per-channel min/max of μ±3σ)",
                float(globalMean.mean().item()),
                float(globalStd.mean().item()),
                float((globalMean - 3 * globalStd).min().item()),
                float((globalMean + 3 * globalStd).max().item()),
            )
    LOGGER.info("-" * 78)

    rebuilder = None
    if arguments.exportDir is not None:
        arguments.exportDir.mkdir(parents=True, exist_ok=True)
        rebuilder = _buildRebuilder(arguments.exportDir / "diag.dae")

    # We collect every generated bone tensor so we can compute pair-wise
    # similarity at the end (mode-collapse signal).
    samplesByConfig: dict[tuple[int, float], torch.Tensor] = {}

    for seed in seeds:
        for cfg in cfgScales:
            output = sampler.sample(
                denoiser=denoiser,
                textHiddenStates=condOutput.hiddenStates,
                textKeyPaddingMask=condOutput.keyPaddingMask,
                unconditionalTextHiddenStates=(
                    uncondOutput.hiddenStates if cfg != 1.0 else None
                ),
                unconditionalTextKeyPaddingMask=(
                    uncondOutput.keyPaddingMask if cfg != 1.0 else None
                ),
                frames=int(arguments.frames),
                numSteps=int(arguments.numSteps),
                cfgScale=float(cfg),
                eta=0.0,
                device=device,
                seed=int(seed),
                normalizer=normalizer,
            )
            stats = computeGenerationStats(
                output.boneMotion[0],
                output.globalMotion[0]
                if output.globalMotion is not None
                else None,
            )
            LOGGER.info(
                "seed=%-4d cfg=%-4.1f  %s",
                seed,
                cfg,
                formatStatsLine(stats),
            )
            samplesByConfig[(seed, float(cfg))] = (
                output.boneMotion[0].detach().float().cpu()
            )

            if rebuilder is not None:
                exportPath = arguments.exportDir / (
                    f"diag_seed{seed}_cfg{cfg:.1f}.dae"
                )
                _exportDae(
                    output.boneMotion[0],
                    output.globalMotion[0]
                    if output.globalMotion is not None
                    else None,
                    fps=int(arguments.fps),
                    rebuilder=rebuilder,
                    outputPath=exportPath,
                    prompt=arguments.prompt,
                )

    LOGGER.info("-" * 78)
    _logCollapseDiagnostics(samplesByConfig, seeds, cfgScales)
    LOGGER.info("=" * 78)
    LOGGER.info("Diagnostic complete.")
    return 0


def _logCollapseDiagnostics(
    samples: dict[tuple[int, float], torch.Tensor],
    seeds: Sequence[int],
    cfgScales: Sequence[float],
) -> None:
    """Print pair-wise similarities to surface mode-collapse / CFG issues.

    * **seed similarity at fixed cfg**: how much does changing the
      initial noise change the output?  A healthy diffusion produces
      visibly different sequences (similarity ~0.3–0.7).  Mode-collapse
      pegs every seed-pair near 1.0 (the model ignores noise).
    * **cfg similarity at fixed seed**: how much does the conditioning
      strength change the output?  A model whose unconditional and
      conditional branches collapsed to the same answer will show ~1.0
      across all CFG values — meaning CFG can't help at inference.
    """
    if len(seeds) >= 2:
        LOGGER.info(
            "Seed sensitivity (cosine similarity of bone outputs across "
            "seeds at fixed cfg — high ≈1.0 means the model ignores "
            "the initial noise = mode collapse):"
        )
        for cfg in cfgScales:
            sims: list[float] = []
            for index in range(len(seeds) - 1):
                a = samples[(seeds[index], float(cfg))].flatten()
                b = samples[(seeds[index + 1], float(cfg))].flatten()
                sims.append(_cosineSimilarity(a, b))
            avg = sum(sims) / len(sims)
            LOGGER.info(
                "  cfg=%-4.1f  avg_seed_sim=%.4f  pairs=%s",
                cfg,
                avg,
                ["%.3f" % s for s in sims],
            )
    if len(cfgScales) >= 2:
        LOGGER.info(
            "CFG sensitivity (cosine similarity across cfg at fixed "
            "seed — high ≈1.0 means cfg has no effect = either uncond "
            "branch never trained, or the encoder produces the same "
            "embedding for prompt vs empty string):"
        )
        for seed in seeds:
            sims = []
            for index in range(len(cfgScales) - 1):
                a = samples[(seed, float(cfgScales[index]))].flatten()
                b = samples[(seed, float(cfgScales[index + 1]))].flatten()
                sims.append(_cosineSimilarity(a, b))
            avg = sum(sims) / len(sims)
            LOGGER.info(
                "  seed=%-4d avg_cfg_sim=%.4f  pairs=%s",
                seed,
                avg,
                ["%.3f" % s for s in sims],
            )


def _cosineSimilarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened 1-D tensors."""
    dot = float(torch.dot(a, b).item())
    norms = float((a.norm() * b.norm()).item())
    if norms == 0.0:
        return 0.0
    return dot / norms


def _buildRebuilder(referencePath: Path) -> AnimationRebuilder:
    parentDir = referencePath.parent.resolve()
    paths = DatasetBuilderPaths(
        animationRoot=parentDir,
        promptRoot=parentDir,
        promptSources=[parentDir],
        indexCsv=parentDir / "missing-index.csv",
        outputRoot=parentDir,
    )
    return AnimationRebuilder(
        DatasetBuilderConfig(
            paths=paths,
            processing=DatasetBuilderProcessing(),
        )
    )


def _exportDae(
    boneRotation6d: torch.Tensor,
    rootTranslation: torch.Tensor | None,
    fps: int,
    rebuilder: AnimationRebuilder,
    outputPath: Path,
    prompt: str,
) -> None:
    from src.cli.generate_animation_v2 import _buildAnimationSampleV2

    sample = _buildAnimationSampleV2(
        boneRotation6d=boneRotation6d,
        rootTranslation=rootTranslation,
        fps=fps,
        outputPath=outputPath,
        extras={"prompt": prompt, "diagnostic": True},
    )
    rebuilder.exportCollada(sample, outputPath)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
