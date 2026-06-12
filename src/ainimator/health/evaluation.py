"""Evaluation helpers for fidelity / retrieval / distinctness.

These functions are absorbed from ``scripts/scaling_curve_v2.py``
(_evaluate / _fk / _cos / _pickProbes) and extended to support
multi-CFG evaluation as required by ROADMAP §3.5 and §5.2.

The standard probe set is: walk / jump / sit / wave / run (5 prompts).
Metrics are evaluated at cfg ∈ {1, 4, 6}; thresholds apply to the
best row.

Public surface
--------------
* :func:`fkVector` — FK→joint-XYZ flattened descriptor.
* :func:`cosineSim` — cosine similarity between two 1-D tensors.
* :func:`pickProbes` — select representative probe indices.
* :func:`evaluateAtCfg` — fidelity / retrieval / distinctness at one CFG.
* :func:`evaluateMultiCfg` — results dict at all canonical CFG scales.
* :const:`PROBE_KEYWORDS` — the 5 standard motion keywords.
* :const:`CANONICAL_CFG_SCALES` — [1, 4, 6].
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
PROBE_KEYWORDS: tuple[str, ...] = (
    "walk", "jump", "sit", "wave", "run",
)
CANONICAL_CFG_SCALES: tuple[float, ...] = (1.0, 4.0, 6.0)

_EPS = 1e-8
_DEFAULT_FRAMES = 200


# ------------------------------------------------------------------
# Core math helpers
# ------------------------------------------------------------------
def fkVector(
    motion: torch.Tensor,
    maxFrames: int = _DEFAULT_FRAMES,
) -> torch.Tensor:
    """Forward-kinematics → pelvis-relative joint-XYZ descriptor.

    Converts rotation6d to a flat 1-D feature vector suitable for
    cosine comparison.

    Parameters
    ----------
    motion : torch.Tensor
        Shape ``(F, numBones, 6)`` or ``(F, C)`` (flat).
        F may exceed ``maxFrames``; it will be truncated.
    maxFrames : int
        Cap on the number of frames used.

    Returns
    -------
    torch.Tensor
        1-D CPU float tensor.
    """
    from ainimator.geometry.components.ops import rot6dToJointXYZ

    flat = motion[:maxFrames].detach().float().cpu()
    if flat.ndim == 3:
        # (F, B, 6) already
        seq = flat
    else:
        # (F, B*6) → (F, B, 6) best-effort
        boneCount = flat.shape[1] // 6
        seq = flat.reshape(flat.shape[0], boneCount, 6)

    xyz = rot6dToJointXYZ(seq.unsqueeze(0).float()).squeeze(0)
    # Subtract pelvis (joint 0) to get pelvis-relative positions
    centered = xyz - xyz[:, :1]
    return centered.reshape(-1)


def cosineSim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors.

    Parameters
    ----------
    a, b : torch.Tensor
        1-D float tensors of equal length.

    Returns
    -------
    float
    """
    denom = float((a.norm() * b.norm()).item()) + _EPS
    return float(torch.dot(a, b).item()) / denom


def pickProbes(
    dataset: Any,
    candidateIndices: list[int],
    keywords: tuple[str, ...] = PROBE_KEYWORDS,
    minFrames: int = 80,
) -> list[int]:
    """Select probe indices covering each motion keyword.

    Parameters
    ----------
    dataset : PreprocessedLinkDataset
        Must support ``dataset[i]`` returning a dict with
        ``"raw_text"`` and ``"motion"`` keys.
    candidateIndices : list[int]
        Pool of link indices to search through.
    keywords : tuple[str, ...]
        Motion type keywords to cover.
    minFrames : int
        Minimum frame count to accept a candidate.

    Returns
    -------
    list[int]
        Up to ``len(keywords)`` probe indices, one per keyword.
    """
    chosen: list[int] = []
    usedKeyword: set[str] = set()
    for index in candidateIndices:
        sample = dataset[index]
        text = (sample.get("raw_text") or "").lower()
        frameCount = sample["motion"].shape[0]
        if frameCount < minFrames:
            continue
        for word in keywords:
            if word in text and word not in usedKeyword:
                usedKeyword.add(word)
                chosen.append(index)
                break
        if len(chosen) >= len(keywords):
            break
    return chosen


# ------------------------------------------------------------------
# Evaluation at one CFG scale
# ------------------------------------------------------------------
def evaluateAtCfg(
    probeIndices: list[int],
    dataset: Any,
    tokenizer: Any,
    encoder: Any,
    denoiser: Any,
    sampler: Any,
    normalizer: Any,
    cfgScale: float,
    device: Any,
    frames: int = _DEFAULT_FRAMES,
    numSteps: int = 100,
    seed: int = 0,
    emptyPrompt: str = "",
) -> dict[str, float]:
    """Compute fidelity / retrieval / distinctness at one CFG scale.

    Parameters
    ----------
    probeIndices : list[int]
        Dataset indices to evaluate (one per motion type).
    dataset : PreprocessedLinkDataset
        Must return dicts with ``"motion"`` and ``"raw_text"`` keys.
    tokenizer, encoder, denoiser, sampler, normalizer
        Live model components from ``loadCheckpointV2``.
    cfgScale : float
        CFG guidance scale.
    device : torch.device or str
        Inference device.
    frames : int
        Number of frames to generate per sample.
    numSteps : int
        DDIM steps.
    seed : int
        RNG seed for reproducible samples.
    emptyPrompt : str
        Unconditional text prompt.

    Returns
    -------
    dict with keys: fidelity, retrieval, distinctness, cfg_scale
    """
    gtVectors = [
        fkVector(dataset[i]["motion"]) for i in probeIndices
    ]
    uncEncoded = tokenizer.encode(emptyPrompt)
    uncOut = encoder(
        uncEncoded.inputIds.to(device),
        uncEncoded.attentionMask.to(device),
    )
    useCfg = cfgScale != 1.0
    genVectors: list[torch.Tensor] = []

    with torch.no_grad():
        for index in probeIndices:
            rawText = dataset[index].get("raw_text") or emptyPrompt
            condEncoded = tokenizer.encode(rawText)
            condOut = encoder(
                condEncoded.inputIds.to(device),
                condEncoded.attentionMask.to(device),
            )
            output = sampler.sample(
                denoiser=denoiser,
                textHiddenStates=condOut.hiddenStates,
                textKeyPaddingMask=condOut.keyPaddingMask,
                unconditionalTextHiddenStates=(
                    uncOut.hiddenStates if useCfg else None
                ),
                unconditionalTextKeyPaddingMask=(
                    uncOut.keyPaddingMask if useCfg else None
                ),
                frames=frames,
                numSteps=numSteps,
                cfgScale=cfgScale,
                eta=0.0,
                device=device,
                seed=seed,
                normalizer=normalizer,
            )
            genVectors.append(
                fkVector(output.boneMotion[0].detach().cpu())
            )

    numProbes = len(probeIndices)
    fidelity = sum(
        cosineSim(genVectors[i], gtVectors[i])
        for i in range(numProbes)
    ) / max(1, numProbes)

    hits = sum(
        1 for i in range(numProbes)
        if max(range(numProbes),
               key=lambda j: cosineSim(genVectors[i], gtVectors[j])) == i
    )
    retrieval = hits / max(1, numProbes)

    pairs = [
        cosineSim(genVectors[a], genVectors[b])
        for a, b in itertools.combinations(range(numProbes), 2)
    ]
    distinctness = (sum(pairs) / len(pairs)) if pairs else 0.0

    return {
        "fidelity": fidelity,
        "retrieval": retrieval,
        "distinctness": distinctness,
        "cfg_scale": cfgScale,
    }


def evaluateMultiCfg(
    probeIndices: list[int],
    dataset: Any,
    tokenizer: Any,
    encoder: Any,
    denoiser: Any,
    sampler: Any,
    normalizer: Any,
    device: Any,
    cfgScales: tuple[float, ...] = CANONICAL_CFG_SCALES,
    frames: int = _DEFAULT_FRAMES,
    numSteps: int = 100,
    seed: int = 0,
    emptyPrompt: str = "",
) -> list[dict[str, float]]:
    """Evaluate fidelity / retrieval / distinctness at multiple CFG scales.

    Parameters
    ----------
    cfgScales : tuple[float, ...]
        CFG scales to evaluate; defaults to CANONICAL_CFG_SCALES.

    Returns
    -------
    list of dicts, one per CFG scale (each has fidelity, retrieval,
    distinctness, cfg_scale keys).
    """
    results: list[dict[str, float]] = []
    for cfg in cfgScales:
        row = evaluateAtCfg(
            probeIndices=probeIndices,
            dataset=dataset,
            tokenizer=tokenizer,
            encoder=encoder,
            denoiser=denoiser,
            sampler=sampler,
            normalizer=normalizer,
            cfgScale=cfg,
            device=device,
            frames=frames,
            numSteps=numSteps,
            seed=seed,
            emptyPrompt=emptyPrompt,
        )
        results.append(row)
    return results
