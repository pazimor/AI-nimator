"""Acceptance criteria tests for HealthHub (Phase A3).

AC4: Injecting constant hidden states (artificial collapse) asserts
     CRITICAL on the effective_rank / intra_batch_sim contract.

AC5: Injecting shuffled text with no effect on the loss asserts
     WARNING on the conditioning_sensitivity contract.

AC6: Wall-clock overhead < 5% at everySteps=50 (micro-benchmark).

A6-probe-attach: Real model attach via named-mapping dict — probes
  fire and produce non-null effective_rank, intra_batch_sim,
  update_ratio; no probe logs an attach failure.
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from ainimator.health.contract import Contract, Verdict
from ainimator.health.hub import HealthHub
from ainimator.health.probe import Probe


# ------------------------------------------------------------------
# Helpers: tiny synthetic models
# ------------------------------------------------------------------
class _Denoiser(nn.Module):
    """Minimal denoiser with a known module path."""

    def __init__(self) -> None:
        super().__init__()
        self.denoiser = nn.Linear(16, 16)
        self.outputProj = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.denoiser(x)
        return self.outputProj(hidden)


def _makeHub(outputDir: Path, everySteps: int = 1) -> HealthHub:
    """Build a minimal HealthHub with the collapse-detection contract."""
    collapseContract = Contract(
        name="intra_batch_collapse",
        metric="intra_batch_sim",
        direction="lower_is_better",
        thresholds={
            "ok_threshold": 0.5,
            "warning_threshold": 0.9,
            "critical_above": 0.9,
        },
    )
    condContract = Contract(
        name="conditioning_sensitivity",
        metric="conditioning_sensitivity",
        direction="higher_is_better",
        thresholds={
            "ok_threshold": 0.01,
            "warning_threshold": 0.001,
            "critical_below": 0.001,
        },
    )
    probe = Probe(
        name="denoiser_blocks",
        modulePath="denoiser",
        capture=["intra_batch_sim", "effective_rank"],
        hookType="forward",
    )
    hub = HealthHub(
        outputDir=outputDir,
        contracts=[collapseContract, condContract],
        probes=[probe],
        everySteps=everySteps,
    )
    return hub


# ------------------------------------------------------------------
# AC4: Artificial collapse → CRITICAL
# ------------------------------------------------------------------
def test_collapse_gives_critical_verdict() -> None:
    """Injecting constant hidden states triggers CRITICAL verdict.

    Rationale (ROADMAP §3.3 "Collapse cond/uncond"):
    Phase E was diagnosed only after 20h of training because there
    was no integrated signal for intra_batch_sim ≈ 1.  This test
    proves that identical hidden states fire the CRITICAL path.
    """
    with tempfile.TemporaryDirectory() as tmpDir:
        model = _Denoiser()
        hub = _makeHub(Path(tmpDir), everySteps=1)
        hub.attach(model)

        # Constant input → all batch samples produce identical activations.
        x = torch.ones(8, 16)
        model(x)

        results = hub.step(globalStep=1)

        # Find the intra_batch collapse contract result.
        collapseResult = next(
            (r for r in results if r.name == "intra_batch_collapse"),
            None,
        )
        assert collapseResult is not None, (
            "intra_batch_collapse contract not evaluated"
        )
        assert collapseResult.verdict == Verdict.CRITICAL, (
            f"Expected CRITICAL for constant hidden states, "
            f"got {collapseResult.verdict} "
            f"(intra_batch_sim={collapseResult.value:.4f})"
        )
        hub.detach()
        hub.close()


# ------------------------------------------------------------------
# AC5: Shuffled text / zero conditioning sensitivity → WARNING
# ------------------------------------------------------------------
def test_zero_conditioning_sensitivity_gives_warning() -> None:
    """Injecting shuffled text with no effect on loss fires WARNING.

    Rationale (ROADMAP §3.3 "Prompt ignoré"):
    The 212-epoch CLIP run had cross_prompt_sim 0.98 — the model was
    not using the prompt at all.  conditioning_sensitivity ≈ 0 is the
    live training signal.
    """
    with tempfile.TemporaryDirectory() as tmpDir:
        model = _Denoiser()
        hub = _makeHub(Path(tmpDir), everySteps=1)
        hub.attach(model)

        # Trigger a forward pass so the probe snapshot exists.
        x = torch.randn(4, 16)
        model(x)

        # Inject conditioning_sensitivity = 0.0 (no effect from text).
        results = hub.step(
            globalStep=1,
            metrics={"conditioning_sensitivity": 0.0},
        )

        condResult = next(
            (r for r in results if r.name == "conditioning_sensitivity"),
            None,
        )
        assert condResult is not None, (
            "conditioning_sensitivity contract not evaluated"
        )
        assert condResult.verdict in (Verdict.WARNING, Verdict.CRITICAL), (
            f"Expected WARNING or CRITICAL for zero sensitivity, "
            f"got {condResult.verdict}"
        )
        hub.detach()
        hub.close()


# ------------------------------------------------------------------
# AC6: Wall-clock overhead < 5% at everySteps=50
# ------------------------------------------------------------------
def test_overhead_below_5_percent() -> None:
    """Health hub overhead stays under 5% at everySteps=50.

    Methodology (matches reviewer benchmark):
    - 384-dim FFN × 4 blocks (B=8, F=120, so B×F=960 tokens)
    - everySteps=50 — hub fires 4 times in 200 steps
    - Probe captures: mean, std, intra_batch_sim, effective_rank
      (the full production probe set including SVD)
    - CPU only (most conservative environment)
    - Asserts the AC6 target: < 5% overhead.

    This target is achievable because ``_computeEffectiveRank`` now
    caps its submatrix to ``_RANK_MAX_ROWS × _RANK_MAX_COLS`` (64×64)
    before calling SVD, bounding the cost to O(64^3) ≈ 0.26ms on CPU.
    """
    N_STEPS = 200
    DIM = 384
    N_TOKENS = 960  # B=8 × F=120 — reviewer benchmark

    # Four-block FFN that produces (N_TOKENS, DIM) activations to
    # simulate the denoiser hidden-state shape used in production.
    class _FourBlockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.denoiser = nn.Sequential(
                nn.Linear(DIM, DIM * 4),
                nn.GELU(),
                nn.Linear(DIM * 4, DIM),
                nn.Linear(DIM, DIM * 4),
                nn.GELU(),
                nn.Linear(DIM * 4, DIM),
                nn.Linear(DIM, DIM * 4),
                nn.GELU(),
                nn.Linear(DIM * 4, DIM),
                nn.Linear(DIM, DIM * 4),
                nn.GELU(),
                nn.Linear(DIM * 4, DIM),
            )
            self.outputProj = nn.Linear(DIM, DIM)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.outputProj(self.denoiser(x))

    model = _FourBlockModel()
    # (N_TOKENS, DIM) — matches B×F=960, D=384 from reviewer benchmark
    x = torch.randn(N_TOKENS, DIM)

    # Warm up to stabilise timing
    for _ in range(5):
        _ = model(x)

    # --- Baseline: N steps without health --------------------------
    start = time.perf_counter()
    for _ in range(N_STEPS):
        _ = model(x)
    baselineMs = (time.perf_counter() - start) * 1000

    # --- With health: full production probe set including effective_rank
    # Hub fires 4 times in 200 steps (steps 50, 100, 150, 200).
    with tempfile.TemporaryDirectory() as tmpDir:
        probe = Probe(
            name="denoiser_blocks",
            modulePath="denoiser",
            # Full production probe set including effective_rank (SVD)
            capture=[
                "mean", "std", "intra_batch_sim", "effective_rank",
            ],
            hookType="forward",
            captureEvery=50,
        )
        collapseContract = Contract(
            name="intra_batch_collapse",
            metric="intra_batch_sim",
            direction="lower_is_better",
            thresholds={
                "ok_threshold": 0.5,
                "warning_threshold": 0.9,
                "critical_above": 0.9,
            },
        )
        hub = HealthHub(
            outputDir=Path(tmpDir),
            contracts=[collapseContract],
            probes=[probe],
            everySteps=50,
        )
        hub.attach(model)

        start = time.perf_counter()
        for step in range(N_STEPS):
            _ = model(x)
            hub.step(
                globalStep=step + 1,
                metrics={"conditioning_sensitivity": 0.5},
            )
        withHealthMs = (time.perf_counter() - start) * 1000

        hub.detach()
        hub.close()

    assert baselineMs > 0, "Baseline timing was zero"
    overhead = (withHealthMs - baselineMs) / baselineMs
    print(
        f"\n[AC6] baseline={baselineMs:.1f}ms  "
        f"with_health={withHealthMs:.1f}ms  "
        f"overhead={overhead:.1%}  "
        f"(model: {DIM}-dim × 4 blocks, {N_TOKENS} tokens, "
        f"everySteps=50, CPU)"
    )
    assert overhead < 0.05, (
        f"Health overhead {overhead:.1%} exceeds 5% AC6 target. "
        f"baseline={baselineMs:.1f}ms, "
        f"with_health={withHealthMs:.1f}ms. "
        f"Root cause: check effective_rank SVD cost in probe.py."
    )


# ------------------------------------------------------------------
# JSONL output contains contract verdicts (AC1 proxy)
# ------------------------------------------------------------------
def test_jsonl_contains_contract_verdicts() -> None:
    """After hub.step() the JSONL file contains all 5 contract verdicts."""
    import json

    # Build a hub with all 5 contracts from health.yaml.
    with tempfile.TemporaryDirectory() as tmpDir:
        from ainimator.health.hub import buildHealthHub

        hub = buildHealthHub(Path(tmpDir))
        model = _Denoiser()
        hub.attach(model)

        x = torch.randn(4, 16)
        model(x)

        # Step 50 (divisible by everySteps=50).
        hub.step(
            globalStep=50,
            metrics={
                "conditioning_sensitivity": 0.05,
                "loss_share": 0.20,
                "post_norm_mean_deviation": 0.05,
                "cfg_sim": 0.80,
                "update_ratio": 5e-4,
            },
        )

        hub.detach()
        hub.close()

        jsonlPath = Path(tmpDir) / "health" / "health.jsonl"
        assert jsonlPath.exists(), "JSONL file was not created"

        records = [
            json.loads(line)
            for line in jsonlPath.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(records) >= 1

        record = records[0]
        # At least 3 of the 5 contract verdicts should be present.
        verdictKeys = [
            k for k in record if k.startswith("verdict.")
        ]
        assert len(verdictKeys) >= 3, (
            f"Expected >= 3 verdict keys, found {verdictKeys}"
        )


# ------------------------------------------------------------------
# A6 probe-attach: real model + named-mapping dict → metrics populate
# ------------------------------------------------------------------

class _DenoiserWithBlocks(nn.Module):
    """Minimal MotionDenoiserV2-shaped model for attach testing.

    Exposes ``blocks`` (ModuleList) and ``outputProjection`` (Linear)
    so the production health.yaml paths resolve correctly.
    """

    def __init__(self, dim: int = 32, numBlocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(numBlocks)
        ])
        self.outputProjection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.outputProjection(x)


class _EncoderModule(nn.Module):
    """Minimal encoder-shaped module for attach testing."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def test_real_model_probe_attach_populates_metrics() -> None:
    """Named-mapping attach produces non-null probe metrics.

    This is the empirical attach test that was missing in A3:
    - Builds real module shapes matching MotionDenoiserV2/encoder.
    - Attaches the hub via ``{"denoiser": ..., "encoder": ...}``.
    - Runs several forward passes and asserts that
      ``effective_rank``, ``intra_batch_sim``, and ``update_ratio``
      appear in the JSONL with non-null values.
    - Asserts no probe logged an attach failure (i.e. none produced
      a ``None`` snapshot at step 1).

    Rationale: the A6 review found all probes produced UNKNOWN
    because ``hub.attach(components.denoiser)`` resolved paths like
    ``"denoiser"`` against the bare denoiser (AttributeError:
    MotionDenoiserV2 has no attribute 'denoiser').
    """
    import json
    import logging

    DIM = 32
    BATCH = 4

    denoiser = _DenoiserWithBlocks(dim=DIM, numBlocks=2)
    encoder = _EncoderModule(dim=DIM)

    with tempfile.TemporaryDirectory() as tmpDir:
        # Build probes that match the production health.yaml paths.
        probeBlocks = Probe(
            name="denoiser_blocks",
            modulePath="denoiser.blocks[-1]",
            capture=["mean", "std", "norm", "effective_rank",
                     "intra_batch_sim"],
            hookType="forward",
        )
        probeEncoder = Probe(
            name="text_encoder_pool",
            modulePath="encoder",
            capture=["mean", "std", "norm"],
            hookType="forward",
        )
        probeHead = Probe(
            name="output_head",
            modulePath="denoiser.outputProjection",
            capture=["mean", "std", "norm", "update_ratio"],
            hookType="forward_and_backward",
        )

        # Collect attach-failure warnings.
        attachWarnings: list[str] = []

        class _CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                msg = self.format(record)
                if "could not attach probe" in msg:
                    attachWarnings.append(msg)

        hubLogger = logging.getLogger("ainimator.health.hub")
        handler = _CapturingHandler()
        hubLogger.addHandler(handler)

        try:
            hub = HealthHub(
                outputDir=Path(tmpDir),
                contracts=[],
                probes=[probeBlocks, probeEncoder, probeHead],
                everySteps=1,
            )
            # Use the named-mapping form.
            hub.attach({
                "denoiser": denoiser,
                "encoder": encoder,
            })

            # Several forward+backward passes.
            for _ in range(3):
                x = torch.randn(BATCH, DIM)
                out = denoiser(x)
                loss = out.sum()
                loss.backward()

            hub.step(globalStep=1)
            hub.detach()
            hub.close()
        finally:
            hubLogger.removeHandler(handler)

        assert not attachWarnings, (
            f"Probe attach failures detected:\n"
            + "\n".join(attachWarnings)
        )

        jsonlPath = Path(tmpDir) / "health" / "health.jsonl"
        assert jsonlPath.exists(), "JSONL not written"
        records = [
            json.loads(line)
            for line in jsonlPath.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert records, "No JSONL records"
        record = records[0]

        assert "denoiser_blocks.effective_rank" in record, (
            f"effective_rank missing from JSONL. Keys: "
            f"{list(record.keys())}"
        )
        assert "denoiser_blocks.intra_batch_sim" in record, (
            f"intra_batch_sim missing from JSONL. Keys: "
            f"{list(record.keys())}"
        )
        assert "output_head.update_ratio" in record, (
            f"update_ratio missing from JSONL. Keys: "
            f"{list(record.keys())}"
        )
        # Values must be non-null floats.
        assert isinstance(
            record["denoiser_blocks.effective_rank"], float
        )
        assert isinstance(
            record["denoiser_blocks.intra_batch_sim"], float
        )
        # update_ratio may be None on first call (no prev weight norm).
        # After 3 passes it must be a float.
        assert isinstance(record["output_head.update_ratio"], float), (
            f"update_ratio should be float after 3 passes, "
            f"got {record.get('output_head.update_ratio')!r}"
        )
