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

from src.features.generation.full_training_v2 import (
    V2FullTrainingConfig,
    runFullTraining,
)
from src.features.generation.training_v2 import loadCheckpointV2, EMPTY_PROMPT
from src.shared.model.components.ops import rot6dToJointXYZ
from src.shared.model.generation.sampler_v2 import DDIMSamplerV2
from src.shared.preprocessed_dataset import PreprocessedLinkDataset

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
    xyz = rot6dToJointXYZ(motion[:FRAMES].unsqueeze(0).float()).squeeze(0)
    return (xyz - xyz[:, :1]).reshape(-1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def _pickProbes(dataset: PreprocessedLinkDataset, accad: list[int]) -> list[int]:
    wants = ["walk", "jump", "sit", "wave", "run", "kick", "turn"]
    chosen: list[int] = []
    usedWord: set[str] = set()
    for index in accad:
        text = (dataset[index].get("raw_text") or "").lower()
        if dataset[index]["motion"].shape[0] < 80:
            continue
        for word in wants:
            if word in text and word not in usedWord:
                usedWord.add(word)
                chosen.append(index)
                break
        if len(chosen) >= 5:
            break
    return chosen


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
    tok, enc, den, sch, norm, _ = loadCheckpointV2(
        outDir / checkpoint, device=DEVICE
    )
    enc.eval(); den.eval()
    sampler = DDIMSamplerV2(sch, predictionMode="v")
    gts = [_fk(dataset[i]["motion"]) for i in probes]
    uncEnc = tok.encode(EMPTY_PROMPT)
    unc = enc(uncEnc.inputIds.to(DEVICE), uncEnc.attentionMask.to(DEVICE))
    useCfg = cfgScale != 1.0
    gens: list[torch.Tensor] = []
    for index in probes:
        encoded = tok.encode(dataset[index].get("raw_text") or EMPTY_PROMPT)
        out = enc(encoded.inputIds.to(DEVICE), encoded.attentionMask.to(DEVICE))
        sample = sampler.sample(
            denoiser=den, textHiddenStates=out.hiddenStates,
            textKeyPaddingMask=out.keyPaddingMask, frames=FRAMES,
            numSteps=100, cfgScale=cfgScale,
            unconditionalTextHiddenStates=unc.hiddenStates if useCfg else None,
            unconditionalTextKeyPaddingMask=unc.keyPaddingMask if useCfg else None,
            eta=0.0, device=DEVICE, seed=0, normalizer=norm,
        )
        gens.append(_fk(sample.boneMotion[0].detach().cpu()))
    fidelity = sum(_cos(gens[i], gts[i]) for i in range(len(probes))) / len(probes)
    hits = sum(
        1 for i in range(len(probes))
        if max(range(len(probes)), key=lambda j: _cos(gens[i], gts[j])) == i
    )
    retrieval = hits / len(probes)
    pairs = [
        _cos(gens[a], gens[b])
        for a, b in itertools.combinations(range(len(probes)), 2)
    ]
    distinct = sum(pairs) / len(pairs)
    return fidelity, retrieval, distinct


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
