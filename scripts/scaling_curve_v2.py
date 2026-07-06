"""Scaling curve: how many distinct prompts can the v2 model condition on?

For each N, train on N held-in ACCAD samples (train==val) with a roughly
constant optimiser-step budget, then regenerate a fixed probe set of 5
prompts and measure:

* fidelity     = mean FK-cosine(gen_i, GT_i)            (higher = better)
* retrieval    = fraction where gen_i's closest GT is GT_i
* distinctness = mean pairwise FK-cosine(gen_i, gen_j)  (lower = distinct)

Results are appended to /tmp/scaling_results.txt as they complete.
"""

from __future__ import annotations

import itertools
import math
import random
from pathlib import Path

import torch

from ainimator.training.full_training_v2 import (
    V2FullTrainingConfig,
    runFullTraining,
)
from ainimator.training.training_v2 import loadCheckpointV2, EMPTY_PROMPT
from ainimator.model.sampler_v2 import DDIMSamplerV2
from ainimator.data.preprocessed_dataset import PreprocessedLinkDataset
from ainimator.health.evaluation import (
    fkVector as _fk_health,
    cosineSim as _cos,
    pickProbes,
    evaluateAtCfg,
)

DATASET = Path("/Users/pazimor/dataset_preprocessed")
TOKENIZER = Path("output/text/custom_tokenizer")
DEVICE = "mps"
FRAMES = 200
TARGET_STEPS = 700
RESULTS = Path("/tmp/scaling_results.txt")
NS = [5, 20, 100, 400]


def _log(line: str) -> None:
    print(line, flush=True)
    with RESULTS.open("a") as handle:
        handle.write(line + "\n")


def _fk(motion: torch.Tensor) -> torch.Tensor:
    """FK vector — delegates to health/evaluation.py."""
    return _fk_health(motion, maxFrames=FRAMES)


def _pickProbes(
    dataset: PreprocessedLinkDataset, accad: list[int]
) -> list[int]:
    """Pick probe indices — delegates to health/evaluation.py."""
    return pickProbes(
        dataset, accad,
        keywords=("walk", "jump", "sit", "wave", "run", "kick", "turn"),
    )


def _trainConfig(indices: list[int], epochs: int, outDir: Path) -> V2FullTrainingConfig:
    batch = min(len(indices), 8)
    return V2FullTrainingConfig(
        datasetRoot=DATASET, tokenizerDir=TOKENIZER, outputDir=outDir,
        datasetFolders=("ACCAD",), sampleLinkIndices=tuple(indices),
        epochs=epochs, batchSize=batch, gradientAccumulation=1,
        maxSamplesPerEpoch=0, learningRate=1e-4, encoderLrMultiplier=3.0,
        condMaskProb=0.0, clipGuidanceWeight=0.3, auxPoolContrastiveWeight=0.5,
        contrastiveTemperature=0.1, jointPositionWeight=1.0,
        footContactWeight=0.5, velocityXyzWeight=0.1,
        useFilmConditioning=True, usePerBlockFilm=True, emaDecay=0.0,
        mirrorProb=0.0, textEncoderType="clip", normalizerFitMaxSamples=64,
        validateEveryEpochs=max(1, epochs), bestMetric="loss_diffusion",
        device=DEVICE, seed=0, logEvery=max(1, epochs // 4),
    )


def _evaluate(
    outDir: Path, probes: list[int], dataset, cfgScale: float = 1.0,
    checkpoint: str = "v2_full_best.pt",
) -> tuple[float, float, float]:
    """Evaluate fidelity/retrieval/distinctness — delegates to health/evaluation.py."""
    tok, enc, den, sch, norm, _ = loadCheckpointV2(
        outDir / checkpoint, device=DEVICE
    )
    enc.eval()
    den.eval()
    sampler = DDIMSamplerV2(sch, predictionMode="v")
    row = evaluateAtCfg(
        probeIndices=probes,
        dataset=dataset,
        tokenizer=tok,
        encoder=enc,
        denoiser=den,
        sampler=sampler,
        normalizer=norm,
        cfgScale=cfgScale,
        device=DEVICE,
        frames=FRAMES,
        numSteps=100,
        seed=0,
        emptyPrompt=EMPTY_PROMPT,
    )
    return row["fidelity"], row["retrieval"], row["distinctness"]


def main() -> None:
    RESULTS.write_text("")
    dataset = PreprocessedLinkDataset(DATASET)
    accad = [i for i, e in enumerate(dataset.linkEntries) if e.datasetFolder == "ACCAD"]
    probes = _pickProbes(dataset, accad)
    _log(f"probes={probes}")
    for index in probes:
        _log(f"  probe {index}: {dataset[index].get('raw_text')!r}")
    gtFk = [_fk(dataset[i]["motion"]) for i in probes]
    gtPairs = [
        _cos(gtFk[a], gtFk[b])
        for a, b in itertools.combinations(range(len(probes)), 2)
    ]
    _log(f"GT distinctness (reference) = {sum(gtPairs)/len(gtPairs):+.3f}")
    _log("N    epochs steps  fidelity retrieval distinct")
    pool = [i for i in accad if i not in probes]
    random.Random(0).shuffle(pool)
    for n in NS:
        indices = probes + pool[: n - len(probes)]
        batch = min(len(indices), 8)
        stepsPerEpoch = math.ceil(len(indices) / batch)
        epochs = max(15, round(TARGET_STEPS / stepsPerEpoch))
        steps = epochs * stepsPerEpoch
        outDir = Path(f"/tmp/scaling_N{n}")
        try:
            runFullTraining(_trainConfig(indices, epochs, outDir))
            fidelity, retrieval, distinct = _evaluate(outDir, probes, dataset)
            _log(
                f"{n:<4d} {epochs:<6d} {steps:<6d} {fidelity:+.3f}    "
                f"{retrieval:.2f}      {distinct:+.3f}"
            )
        except Exception as error:  # noqa: BLE001 — keep the sweep alive
            _log(f"{n:<4d} FAILED: {error}")
    _log("DONE")


if __name__ == "__main__":
    main()
