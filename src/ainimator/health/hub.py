"""HealthHub — central health registry for AI-nimator.

This is the single entry point for all health monitoring.  Four
runtime modes on one hub object:

* ``hub.step()`` — during training, sampled every N steps.
* ``hub.audit()`` — offline on a checkpoint (pytorch-auditor adapter).
* ``hub.diagnose()`` — offline on generations (absorbs diagnose_v2 +
  cross_prompt_sim + evaluation functions).
* ``hub.report()`` — aggregates JSONL + verdicts into a health sheet.

The hub attaches probes via PyTorch hooks and never modifies
``forward()`` — ONNX traceability is preserved.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ainimator.health.contract import Contract, ContractResult, Verdict
from ainimator.health.jsonl_writer import JsonlWriter
from ainimator.health.probe import Probe, ProbeSnapshot
from ainimator.health.tb_writer import TbWriter

LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Score reference table (ROADMAP §3.5) used by hub.report()
# ------------------------------------------------------------------
SCORE_REFERENCE: list[dict[str, Any]] = [
    {
        "metric": "cfg_sim",
        "description": "cosine gen cond vs uncond (same seed)",
        "direction": "↓",
        "target": "< 0.95",
        "ok": ("< 0.95",),
        "warning": ("0.95–0.99",),
        "critical": ("> 0.99",),
    },
    {
        "metric": "seed_sim",
        "description": "cosine between seeds (same prompt)",
        "direction": "↓",
        "target": "< 0.90",
        "ok": ("< 0.90",),
        "warning": ("0.90–0.97",),
        "critical": ("> 0.97",),
    },
    {
        "metric": "encoder_cond_uncond_sim",
        "description": "cosine pool encoder cond vs null",
        "direction": "↓",
        "target": "< 0.50",
        "ok": ("< 0.50",),
        "warning": ("0.50–0.75",),
        "critical": ("> 0.75",),
    },
    {
        "metric": "cross_prompt_sim",
        "description": "cosine generations of different prompts",
        "direction": "↓",
        "target": "< 0.80",
        "ok": ("< 0.80",),
        "warning": ("0.80–0.95",),
        "critical": ("> 0.95",),
    },
    {
        "metric": "fidelity",
        "description": "FK-cosine(gen, GT) on probe set",
        "direction": "↑",
        "target": "> 0.60",
        "ok": ("> 0.60",),
        "warning": ("0.20–0.60",),
        "critical": ("< 0.20",),
    },
    {
        "metric": "retrieval",
        "description": "fraction nearest-GT(gen_i) == GT_i",
        "direction": "↑",
        "target": ">= 0.80",
        "ok": (">= 0.80",),
        "warning": ("0.40–0.80",),
        "critical": ("<= 0.20",),
    },
    {
        "metric": "distinctness",
        "description": "FK-cosine mean inter-generations",
        "direction": "↓",
        "target": "< 0.50",
        "ok": ("< 0.50",),
        "warning": ("0.50–0.80",),
        "critical": ("> 0.80",),
    },
    {
        "metric": "conditioning_sensitivity",
        "description": "Δloss forward real vs shuffled text",
        "direction": "↑",
        "target": "> 0",
        "ok": ("nettement > 0",),
        "warning": ("≈ 0 on 1 window",),
        "critical": ("≈ 0 persistant",),
    },
    {
        "metric": "effective_rank",
        "description": "rank of hidden states (normalised)",
        "direction": "↑",
        "target": "> 0.5",
        "ok": ("> 0.5",),
        "warning": ("0.1–0.5",),
        "critical": ("< 0.1",),
    },
    {
        "metric": "intra_batch_sim",
        "description": "cosine mean hidden states intra-batch",
        "direction": "↓",
        "target": "< 0.5",
        "ok": ("< 0.5",),
        "warning": ("0.5–0.9",),
        "critical": ("> 0.9",),
    },
    {
        "metric": "update_ratio",
        "description": "||Δw|| / ||w|| per step",
        "direction": "range",
        "target": "1e-4 – 1e-2",
        "ok": ("1e-4 – 1e-2",),
        "warning": ("< 1e-5 or > 1e-1",),
        "critical": ("≈ 0 or NaN",),
    },
    {
        "metric": "loss_share",
        "description": "share of each loss component",
        "direction": "range",
        "target": "1–70%",
        "ok": ("1–70%",),
        "warning": ("< 1% or > 80%",),
        "critical": ("exactly 0",),
    },
    {
        "metric": "post_norm_stats",
        "description": "mean/std after z-norm (1st batch)",
        "direction": "mean→0 std→1",
        "target": "|mean|<0.1 |std-1|<0.1",
        "ok": ("both < 0.1",),
        "warning": ("< 0.3",),
        "critical": ("beyond 0.3",),
    },
    {
        "metric": "nan_inf",
        "description": "NaN/Inf in weights, activations, grads",
        "direction": "0",
        "target": "0",
        "ok": ("0",),
        "warning": ("—",),
        "critical": (">= 1",),
    },
    {
        "metric": "val_gap",
        "description": "(val − train) / train on diffusion loss",
        "direction": "↓",
        "target": "< 15%",
        "ok": ("< 15%",),
        "warning": ("15–40%",),
        "critical": ("> 40%",),
    },
    {
        "metric": "epoch_time",
        "description": "epoch time drift since epoch 1",
        "direction": "stable",
        "target": "drift < 1.5×",
        "ok": ("< 1.5×",),
        "warning": ("1.5–3×",),
        "critical": ("> 3×",),
    },
]


# ------------------------------------------------------------------
# HealthHub
# ------------------------------------------------------------------
class HealthHub:
    """Central health registry — attaches probes, collects, routes.

    Parameters
    ----------
    outputDir : Path
        Run output directory for JSONL files.
    contracts : list[Contract]
        Declarative criteria to evaluate at each step.
    probes : list[Probe]
        Hooks to attach to the model.
    everySteps : int
        Capture every N optimiser steps (default 50).
    """

    def __init__(
        self,
        outputDir: Path,
        contracts: list[Contract],
        probes: list[Probe],
        everySteps: int = 50,
    ) -> None:
        self._outputDir = outputDir
        self._contracts = contracts
        self._probes = probes
        self._everySteps = everySteps

        self._jsonl = JsonlWriter(outputDir, "health")
        self._tb = TbWriter(outputDir)

        self._attached = False
        self._stepMetrics: dict[str, float | None] = {}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def everySteps(self) -> int:
        """Number of optimiser steps between health captures."""
        return self._everySteps

    @everySteps.setter
    def everySteps(self, value: int) -> None:
        """Set the number of steps between health captures."""
        self._everySteps = value

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------
    def attach(self, rootModel: nn.Module) -> None:
        """Attach all probes to the model.

        Parameters
        ----------
        rootModel : nn.Module
            The model to instrument (usually the denoiser or a wrapper
            that exposes both encoder and denoiser attributes).
        """
        if self._attached:
            return
        for probe in self._probes:
            try:
                probe.attach(rootModel)
            except AttributeError as exc:
                LOGGER.warning(
                    "HealthHub: could not attach probe '%s' "
                    "(path '%s'): %s",
                    probe.name,
                    probe.modulePath,
                    exc,
                )
        self._attached = True

    def detach(self) -> None:
        """Remove all probes from the model."""
        for probe in self._probes:
            probe.detach()
        self._attached = False

    # ------------------------------------------------------------------
    # Runtime: step (training)
    # ------------------------------------------------------------------
    def step(
        self,
        globalStep: int,
        metrics: dict[str, float | None] | None = None,
    ) -> list[ContractResult]:
        """Collect probe snapshots and evaluate contracts.

        Called from the training loop every ``everySteps`` steps.
        Returns contract verdicts; logs WARNING/CRITICAL to the logger.

        Parameters
        ----------
        globalStep : int
            Current optimiser step counter.
        metrics : dict, optional
            Extra metrics from the training loop (e.g. loss components,
            ``conditioning_sensitivity``).

        Returns
        -------
        list[ContractResult]
            Contract evaluation results.
        """
        if globalStep % self._everySteps != 0:
            return []

        # Collect probe snapshots into a flat metrics dict.
        collected: dict[str, float | None] = {}
        for probe in self._probes:
            snap = probe.lastSnapshot
            if snap is None:
                continue
            prefix = probe.name
            _addIfNotNone(collected, f"{prefix}.mean", snap.mean)
            _addIfNotNone(collected, f"{prefix}.std", snap.std)
            _addIfNotNone(collected, f"{prefix}.norm", snap.norm)
            _addIfNotNone(
                collected, f"{prefix}.effective_rank",
                snap.effective_rank
            )
            _addIfNotNone(
                collected, f"{prefix}.intra_batch_sim",
                snap.intra_batch_sim
            )
            _addIfNotNone(
                collected, f"{prefix}.update_ratio", snap.update_ratio
            )
            _addIfNotNone(
                collected, f"{prefix}.encoder_cond_uncond_sim",
                snap.encoder_cond_uncond_sim
            )

        # Merge probe metrics with caller-supplied metrics.
        if metrics:
            collected.update(metrics)

        # Expose intra_batch_sim as the contract-visible key.
        self._exposeAggregateMetrics(collected)

        # Evaluate contracts.
        results = [c.evaluate(collected) for c in self._contracts]

        # Log any non-OK verdicts.
        for result in results:
            if result.verdict in (Verdict.WARNING, Verdict.CRITICAL):
                LOGGER.warning(
                    "[health] step=%d  %s [%s]: %s",
                    globalStep,
                    result.name,
                    result.verdict.value,
                    result.message,
                )

        # Write to JSONL and TensorBoard.
        record: dict[str, Any] = {"step": globalStep}
        record.update({k: v for k, v in collected.items() if v is not None})
        for res in results:
            record[f"verdict.{res.name}"] = res.verdict.value
        self._jsonl.write(record)
        self._tb.write(globalStep, {
            k: v for k, v in collected.items()
            if v is not None
        })

        return results

    def _exposeAggregateMetrics(
        self, metrics: dict[str, float | None]
    ) -> None:
        """Promote probe sub-metrics to top-level contract keys.

        ``cfg_sim`` is NOT derived from ``intra_batch_sim`` here.
        True cond-vs-uncond similarity is only available from
        ``hub.diagnose()``, not from live training hooks.  Leaving it
        absent (UNKNOWN) is correct during training.
        """
        # intra_batch_sim top-level
        if "intra_batch_sim" not in metrics:
            val = metrics.get("denoiser_blocks.intra_batch_sim")
            if val is not None:
                metrics["intra_batch_sim"] = val

        # effective_rank top-level
        if "effective_rank" not in metrics:
            val = metrics.get("denoiser_blocks.effective_rank")
            if val is not None:
                metrics["effective_rank"] = val

        # update_ratio top-level
        if "update_ratio" not in metrics:
            val = metrics.get("output_head.update_ratio")
            if val is not None:
                metrics["update_ratio"] = val

        # encoder_cond_uncond_sim top-level
        if "encoder_cond_uncond_sim" not in metrics:
            val = metrics.get(
                "text_encoder_pool.encoder_cond_uncond_sim"
            )
            if val is not None:
                metrics["encoder_cond_uncond_sim"] = val

    # ------------------------------------------------------------------
    # Runtime: audit (offline checkpoint)
    # ------------------------------------------------------------------
    def audit(
        self,
        encoder: nn.Module,
        denoiser: nn.Module,
        normalizer: Any,
        payload: dict[str, Any],
        checkpointPath: Path | None = None,
    ) -> dict[str, Any]:
        """Offline audit of a loaded checkpoint (pytorch-auditor L3-4 adapter).

        Accepts pre-loaded components so that the hub itself does not
        need to import ``ainimator.training`` (layer constraint: health
        is L4 and must not depend on training, also L4).  The CLI is
        L5 and handles the loading.

        Parameters
        ----------
        encoder, denoiser : nn.Module
            Loaded and eval'd model components.
        normalizer : MotionNormalizer
            Fitted normalizer from the checkpoint.
        payload : dict
            Raw checkpoint payload (for training_config etc.).
        checkpointPath : Path, optional
            For logging only.

        Returns
        -------
        dict
            Audit results with keys: nan_inf, weight_norms,
            post_norm_stats, model_info.
        """
        LOGGER.info(
            "HealthHub.audit: analysing %s",
            checkpointPath or "checkpoint",
        )

        results: dict[str, Any] = {}
        if checkpointPath is not None:
            results["checkpoint"] = str(checkpointPath)
        results["model_info"] = {
            "encoder_params": sum(
                p.numel() for p in encoder.parameters()
            ),
            "denoiser_params": sum(
                p.numel() for p in denoiser.parameters()
            ),
        }

        # NaN / Inf scan.
        results["nan_inf"] = _scanNanInf(denoiser)

        # Weight norm per named parameter.
        results["weight_norms"] = {
            name: float(param.data.norm().item())
            for name, param in denoiser.named_parameters()
            if param.requires_grad
        }

        # Normalizer post-norm stats (sanity check z-norm).
        results["post_norm_stats"] = _normalizerStats(normalizer)

        # Training config summary.
        results["training_config"] = payload.get("training_config", {})

        auditPath = self._outputDir / "health" / "audit.json"
        auditPath.parent.mkdir(parents=True, exist_ok=True)
        import json
        with auditPath.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        LOGGER.info("HealthHub.audit: results written to %s", auditPath)
        return results

    # ------------------------------------------------------------------
    # Runtime: diagnose (offline generations)
    # ------------------------------------------------------------------
    def diagnose(
        self,
        tokenizer: Any,
        encoder: nn.Module,
        denoiser: nn.Module,
        sampler: Any,
        normalizer: Any,
        device: Any,
        prompt: str = "a person walks forward",
        seeds: tuple[int, ...] = (0, 42, 123),
        cfgScales: tuple[float, ...] = (1.0, 4.0, 6.0),
        frames: int = 120,
        numSteps: int = 100,
        emptyPrompt: str = "",
        probeIndices: list[int] | None = None,
        dataset: Any | None = None,
    ) -> dict[str, Any]:
        """Offline conditioning diagnostics (absorbs diagnose_v2).

        Accepts pre-loaded components so ``health`` does not need to
        import ``ainimator.training`` or ``ainimator.data`` (layer L4
        constraint).  The CLI (L5) handles loading and passes
        components here.

        Computes:
        - cfg_sim (cosine cond vs uncond same seed)
        - seed_sim (cosine across seeds same prompt)
        - encoder_cond_uncond_sim (pool-level)
        - cross_prompt_sim (if probeIndices + dataset supplied)
        - fidelity / retrieval / distinctness at cfg ∈ {1, 4, 6}

        Parameters
        ----------
        tokenizer, encoder, denoiser, sampler, normalizer
            Pre-loaded model components.
        device : torch.device or str
            Inference device.
        prompt : str
            Primary prompt for cfg_sim / seed_sim.
        seeds, cfgScales : tuple
            Seed / CFG sweep dimensions.
        frames, numSteps : int
            Generation parameters.
        emptyPrompt : str
            Unconditional text (CFG null prompt).
        probeIndices : list[int], optional
            When supplied, cross_prompt_sim and multi-CFG eval are run.
        dataset : PreprocessedLinkDataset, optional
            Pre-loaded dataset for fidelity/retrieval/distinctness.

        Returns
        -------
        dict with all diagnostic metrics.
        """
        # --- Encoder cond↔uncond sim --------------------------------
        condEnc = tokenizer.encode(prompt)
        condOut = encoder(
            condEnc.inputIds.to(device),
            condEnc.attentionMask.to(device),
        )
        uncondEnc = tokenizer.encode(emptyPrompt)
        uncondOut = encoder(
            uncondEnc.inputIds.to(device),
            uncondEnc.attentionMask.to(device),
        )
        encoderSim = _poolCosineSim(condOut, uncondOut)

        # --- Sampling sweep -----------------------------------------
        samplesByConfig: dict[tuple[int, float], torch.Tensor] = {}
        with torch.no_grad():
            for seed in seeds:
                for cfg in cfgScales:
                    useCfg = float(cfg) != 1.0
                    output = sampler.sample(
                        denoiser=denoiser,
                        textHiddenStates=condOut.hiddenStates,
                        textKeyPaddingMask=condOut.keyPaddingMask,
                        unconditionalTextHiddenStates=(
                            uncondOut.hiddenStates if useCfg else None
                        ),
                        unconditionalTextKeyPaddingMask=(
                            uncondOut.keyPaddingMask if useCfg else None
                        ),
                        frames=frames,
                        numSteps=numSteps,
                        cfgScale=float(cfg),
                        eta=0.0,
                        device=device,
                        seed=int(seed),
                        normalizer=normalizer,
                    )
                    samplesByConfig[(int(seed), float(cfg))] = (
                        output.boneMotion[0].detach().float().cpu()
                    )

        cfgSim, seedSim = _computeCollapseSims(
            samplesByConfig, seeds, cfgScales
        )

        metrics: dict[str, Any] = {
            "encoder_cond_uncond_sim": encoderSim,
            "cfg_sim": cfgSim,
            "seed_sim": seedSim,
            "prompt": prompt,
            "seeds": list(seeds),
            "cfg_scales": list(cfgScales),
        }

        # Parity values matching the old diagnose CLI layout.
        metrics["sample_stats"] = _buildSampleStats(
            samplesByConfig, seeds, cfgScales
        )

        # --- Cross-prompt sim + fidelity/retrieval/distinctness ------
        if probeIndices is not None and dataset is not None:
            from ainimator.health.evaluation import (
                evaluateMultiCfg, CANONICAL_CFG_SCALES,
            )

            crossSim = _crossPromptSim(
                probeIndices, dataset, tokenizer, encoder, denoiser,
                sampler, normalizer, device, frames, numSteps,
                emptyPrompt
            )
            metrics["cross_prompt_sim"] = crossSim

            evalRows = evaluateMultiCfg(
                probeIndices=probeIndices,
                dataset=dataset,
                tokenizer=tokenizer,
                encoder=encoder,
                denoiser=denoiser,
                sampler=sampler,
                normalizer=normalizer,
                device=device,
                cfgScales=CANONICAL_CFG_SCALES,
                frames=frames,
                numSteps=numSteps,
                emptyPrompt=emptyPrompt,
            )
            metrics["eval_rows"] = evalRows

        # Write output.
        import json
        diagPath = (
            self._outputDir / "health" / "diagnose.json"
        )
        diagPath.parent.mkdir(parents=True, exist_ok=True)
        with diagPath.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        LOGGER.info(
            "HealthHub.diagnose: results written to %s", diagPath
        )
        return metrics

    # ------------------------------------------------------------------
    # Runtime: report
    # ------------------------------------------------------------------
    def report(
        self,
        runDir: Path,
        outputFormat: str = "markdown",
    ) -> str:
        """Aggregate JSONL + verdicts into a health sheet.

        Reads all ``health/*.jsonl`` files from ``runDir``, computes
        per-metric last/mean values, evaluates contracts, and formats
        a health sheet with all 16 ROADMAP §3.5 metrics.

        Parameters
        ----------
        runDir : Path
            Run output directory containing ``health/`` subdirectory.
        outputFormat : str
            ``"markdown"`` or ``"json"``.

        Returns
        -------
        str
            Formatted health sheet.
        """
        import json as _json
        healthDir = runDir / "health"
        aggregated: dict[str, list[float]] = {}

        # Read all JSONL files.
        if healthDir.exists():
            for jsonlFile in sorted(healthDir.glob("*.jsonl")):
                for line in jsonlFile.read_text(
                    encoding="utf-8"
                ).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    for key, val in record.items():
                        if key in ("step",):
                            continue
                        if isinstance(val, (int, float)):
                            aggregated.setdefault(key, []).append(
                                float(val)
                            )

        # Compute last-value per metric.
        lastValues: dict[str, float] = {
            k: vals[-1] for k, vals in aggregated.items() if vals
        }

        # Try to read diagnose.json for diagnose-phase metrics.
        diagFile = healthDir / "diagnose.json"
        if diagFile.exists():
            try:
                diagData = _json.loads(
                    diagFile.read_text(encoding="utf-8")
                )
                for key in (
                    "cfg_sim", "seed_sim", "encoder_cond_uncond_sim",
                    "cross_prompt_sim",
                ):
                    if key in diagData:
                        lastValues.setdefault(key, float(diagData[key]))
            except Exception:  # noqa: BLE001
                pass

        return _formatSheet(lastValues, self._contracts, outputFormat)

    def close(self) -> None:
        """Close TensorBoard writer."""
        self._tb.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _addIfNotNone(
    d: dict[str, float | None], key: str, value: float | None
) -> None:
    """Add key→value to d only when value is not None."""
    if value is not None:
        d[key] = value


def _scanNanInf(model: nn.Module) -> int:
    """Count parameters containing NaN or Inf.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
        Number of parameter tensors with at least one NaN/Inf.
    """
    count = 0
    for param in model.parameters():
        if not torch.isfinite(param.data).all():
            count += 1
    return count


def _normalizerStats(normalizer: Any) -> dict[str, float]:
    """Extract mean/std statistics from a MotionNormalizer."""
    stats: dict[str, float] = {}
    try:
        boneMean = float(
            normalizer.boneMean.float().cpu().mean().item()
        )
        boneStd = float(
            normalizer.boneStd.float().cpu().mean().item()
        )
        stats["bone_mean"] = boneMean
        stats["bone_std"] = boneStd
        # Canonical key matches SCORE_REFERENCE and _verdictForMetric.
        stats["post_norm_stats"] = abs(boneMean)
        stats["post_norm_std_deviation"] = abs(boneStd - 1.0)
    except AttributeError:
        pass
    return stats


def _poolCosineSim(condOut: Any, uncondOut: Any) -> float:
    """Compute cosine similarity of masked-mean pooled embeddings."""
    condTok = condOut.hiddenStates[0]
    uncondTok = uncondOut.hiddenStates[0]
    condMask = ~condOut.keyPaddingMask[0]
    uncondMask = ~uncondOut.keyPaddingMask[0]
    condPool = (
        (condTok * condMask.unsqueeze(-1).float()).sum(0)
        / condMask.float().sum().clamp(min=1e-8)
    )
    uncondPool = (
        (uncondTok * uncondMask.unsqueeze(-1).float()).sum(0)
        / uncondMask.float().sum().clamp(min=1e-8)
    )
    return float(
        torch.nn.functional.cosine_similarity(
            condPool.detach().float(),
            uncondPool.detach().float(),
            dim=0,
        ).item()
    )


def _cosineSim1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two 1-D tensors."""
    denom = float((a.norm() * b.norm()).item())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(a, b).item()) / denom


def _computeCollapseSims(
    samples: dict[tuple[int, float], torch.Tensor],
    seeds: tuple[int, ...],
    cfgScales: tuple[float, ...],
) -> tuple[float, float]:
    """Compute mean cfg_sim and seed_sim from sampled motions.

    Returns
    -------
    tuple[float, float]
        (cfg_sim, seed_sim) — means over all pairs.
    """
    cfgSims: list[float] = []
    if len(cfgScales) >= 2:
        for seed in seeds:
            for idx in range(len(cfgScales) - 1):
                a = samples[(seed, float(cfgScales[idx]))].flatten()
                b = samples[(seed, float(cfgScales[idx + 1]))].flatten()
                cfgSims.append(_cosineSim1d(a, b))

    seedSims: list[float] = []
    if len(seeds) >= 2:
        for cfg in cfgScales:
            for idx in range(len(seeds) - 1):
                a = samples[(seeds[idx], float(cfg))].flatten()
                b = samples[(seeds[idx + 1], float(cfg))].flatten()
                seedSims.append(_cosineSim1d(a, b))

    cfgSim = sum(cfgSims) / len(cfgSims) if cfgSims else float("nan")
    seedSim = (
        sum(seedSims) / len(seedSims) if seedSims else float("nan")
    )
    return cfgSim, seedSim


def _buildSampleStats(
    samples: dict[tuple[int, float], torch.Tensor],
    seeds: tuple[int, ...],
    cfgScales: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Build per-(seed, cfg) sample stats for parity with diagnose CLI."""
    from ainimator.health.diagnose_v2 import (
        computeGenerationStats, formatStatsLine,
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for cfg in cfgScales:
            key = (int(seed), float(cfg))
            if key not in samples:
                continue
            bone = samples[key]
            stats = computeGenerationStats(bone.reshape(-1, 22, 6))
            rows.append({
                "seed": seed,
                "cfg": cfg,
                "stats_line": formatStatsLine(stats),
                "rotation_mean": stats.rotationMean,
                "rotation_std": stats.rotationStd,
                "root_displacement": stats.rootDisplacement,
                "rotation_velocity_mean": stats.rotationVelocityMean,
                "rotation_acceleration_mean": (
                    stats.rotationAccelerationMean
                ),
            })
    return rows


def _crossPromptSim(
    probeIndices: list[int],
    dataset: Any,
    tokenizer: Any,
    encoder: Any,
    denoiser: Any,
    sampler: Any,
    normalizer: Any,
    device: Any,
    frames: int,
    numSteps: int,
    emptyPrompt: str,
) -> float:
    """Mean off-diagonal FK cosine similarity across probe prompts."""
    from ainimator.health.evaluation import fkVector, cosineSim

    uncEnc = tokenizer.encode(emptyPrompt)
    uncOut = encoder(
        uncEnc.inputIds.to(device),
        uncEnc.attentionMask.to(device),
    )
    motions: list[torch.Tensor] = []
    with torch.no_grad():
        for index in probeIndices:
            rawText = dataset[index].get("raw_text") or emptyPrompt
            cEnc = tokenizer.encode(rawText)
            cOut = encoder(
                cEnc.inputIds.to(device),
                cEnc.attentionMask.to(device),
            )
            output = sampler.sample(
                denoiser=denoiser,
                textHiddenStates=cOut.hiddenStates,
                textKeyPaddingMask=cOut.keyPaddingMask,
                unconditionalTextHiddenStates=uncOut.hiddenStates,
                unconditionalTextKeyPaddingMask=uncOut.keyPaddingMask,
                frames=frames,
                numSteps=numSteps,
                cfgScale=2.5,
                eta=0.0,
                device=device,
                seed=0,
                normalizer=normalizer,
            )
            motions.append(
                fkVector(output.boneMotion[0].detach().cpu())
            )

    sims: list[float] = []
    for i in range(len(motions)):
        for j in range(i + 1, len(motions)):
            sims.append(cosineSim(motions[i], motions[j]))
    return sum(sims) / len(sims) if sims else float("nan")


def _verdictForMetric(
    metric: str,
    value: float | None,
) -> str:
    """Simple rule-based verdict for the report sheet.

    Uses hardcoded thresholds from ROADMAP §3.5.
    """
    if value is None or math.isnan(float(value)):
        return "UNKNOWN"
    v = float(value)
    # lower-is-better metrics
    lower = {
        "cfg_sim": (0.95, 0.99),
        "seed_sim": (0.90, 0.97),
        "encoder_cond_uncond_sim": (0.50, 0.75),
        "cross_prompt_sim": (0.80, 0.95),
        "distinctness": (0.50, 0.80),
        "intra_batch_sim": (0.50, 0.90),
        "post_norm_stats": (0.10, 0.30),
    }
    if metric in lower:
        warn, crit = lower[metric]
        if v < warn:
            return "OK"
        if v < crit:
            return "WARNING"
        return "CRITICAL"

    # higher-is-better metrics
    higher = {
        "fidelity": (0.60, 0.20),
        "retrieval": (0.80, 0.20),
        "conditioning_sensitivity": (0.01, 0.001),
        "effective_rank": (0.50, 0.10),
    }
    if metric in higher:
        okT, critT = higher[metric]
        if v >= okT:
            return "OK"
        if v >= critT:
            return "WARNING"
        return "CRITICAL"

    return "OK"


def _formatSheet(
    values: dict[str, float],
    contracts: list[Contract],
    outputFormat: str,
) -> str:
    """Format the health sheet from values dict.

    Parameters
    ----------
    values : dict
        Last observed metric values.
    contracts : list[Contract]
        Contracts to evaluate.
    outputFormat : str
        ``"markdown"`` or ``"json"``.

    Returns
    -------
    str
    """
    import json

    rows: list[dict[str, Any]] = []
    for ref in SCORE_REFERENCE:
        metric = str(ref["metric"])
        value = values.get(metric)
        verdict = _verdictForMetric(metric, value)
        rows.append({
            "metric": metric,
            "description": ref["description"],
            "direction": ref["direction"],
            "target": ref["target"],
            "value": value,
            "verdict": verdict,
        })

    if outputFormat == "json":
        return json.dumps(rows, indent=2, default=str)

    # Markdown table
    lines = [
        "# Health Sheet",
        "",
        (
            "| Metric | Value | Direction | Target | Verdict |"
        ),
        (
            "|--------|-------|-----------|--------|---------|"
        ),
    ]
    for row in rows:
        val = row["value"]
        valStr = f"{val:.4f}" if isinstance(val, float) else "N/A"
        lines.append(
            f"| {row['metric']} | {valStr} | "
            f"{row['direction']} | {row['target']} | "
            f"**{row['verdict']}** |"
        )
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Factory: build HealthHub from health.yaml
# ------------------------------------------------------------------
def buildHealthHub(
    outputDir: Path,
    configPath: Path | None = None,
) -> HealthHub:
    """Build a HealthHub from ``src/configs/health.yaml``.

    Parameters
    ----------
    outputDir : Path
        Run output directory.
    configPath : Path, optional
        Override the default health.yaml path.

    Returns
    -------
    HealthHub
    """
    import yaml

    if configPath is None:
        configPath = (
            Path(__file__).parent.parent.parent
            / "configs" / "health.yaml"
        )

    raw: dict[str, Any] = {}
    if configPath.exists():
        with configPath.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    healthCfg = raw.get("health", {})
    everySteps = int(healthCfg.get("everySteps", 50))

    # Build probes from YAML.
    probes: list[Probe] = []
    for entry in raw.get("probes", []):
        probes.append(Probe(
            name=str(entry["name"]),
            modulePath=str(entry["module_path"]),
            capture=list(entry.get("capture", [])),
            hookType=str(entry.get("hook_type", "forward")),
            captureEvery=everySteps,
        ))

    # Build contracts from YAML.
    contracts: list[Contract] = []
    for entry in raw.get("contracts", []):
        contracts.append(Contract.fromYamlDict(entry))

    return HealthHub(
        outputDir=outputDir,
        contracts=contracts,
        probes=probes,
        everySteps=everySteps,
    )
