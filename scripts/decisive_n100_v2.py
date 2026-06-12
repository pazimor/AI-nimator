"""Decisive test: N=100 trained to high per-sample exposure.

The scaling curve held the step budget constant, so per-sample exposure
collapsed with N (700→280→56→15) and N=100 fell to fidelity -0.15.  This
script retrains the SAME 100 ACCAD samples and SAME 5 probes with ~600
exposures/sample (≈ the N=5 budget).  If fidelity recovers, the cliff was
under-training, not a capacity ceiling.
"""

from __future__ import annotations

import random
from pathlib import Path

from scripts.scaling_curve_v2 import (
    _evaluate,
    _pickProbes,
    _trainConfig,
)
from src.features.generation.full_training_v2 import runFullTraining
from src.shared.preprocessed_dataset import PreprocessedLinkDataset

DATASET = Path("/Users/pazimor/dataset_preprocessed")
RESULTS = Path("/tmp/decisive_n100.txt")
EPOCHS = 600  # 100/8 ≈ 13 steps/epoch → ~7800 steps → ~600 exposures/sample


def _log(line: str) -> None:
    print(line, flush=True)
    with RESULTS.open("a") as handle:
        handle.write(line + "\n")


def main() -> None:
    RESULTS.write_text("")
    dataset = PreprocessedLinkDataset(DATASET)
    accad = [
        i for i, e in enumerate(dataset.linkEntries)
        if e.datasetFolder == "ACCAD"
    ]
    probes = _pickProbes(dataset, accad)
    pool = [i for i in accad if i not in probes]
    random.Random(0).shuffle(pool)
    indices = probes + pool[: 100 - len(probes)]
    _log(f"N=100 decisive run — probes={probes}")
    _log(f"epochs={EPOCHS}  (~600 exposures/sample vs 56 in the sweep)")
    outDir = Path("/tmp/decisive_N100")
    runFullTraining(_trainConfig(indices, EPOCHS, outDir))
    fidelity, retrieval, distinct = _evaluate(outDir, probes, dataset)
    _log("                fidelity retrieval distinct")
    _log(f"N=100@56  (sweep)   -0.151    0.20      +0.394")
    _log(
        f"N=100@600 (this)    {fidelity:+.3f}    {retrieval:.2f}      "
        f"{distinct:+.3f}"
    )
    _log("DONE")


if __name__ == "__main__":
    main()
