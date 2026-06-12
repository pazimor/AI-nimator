"""Tests for HealthHub.report() — AC2.

Verifies that ``hub.report()`` produces a health sheet containing
all 16 metrics from ROADMAP §3.5 table, each with value/target/verdict.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ainimator.health.hub import SCORE_REFERENCE, buildHealthHub


# ------------------------------------------------------------------
# AC2: health report produces 16 metrics
# ------------------------------------------------------------------
def test_report_contains_16_metrics_markdown() -> None:
    """health report markdown contains all 16 §3.5 metrics."""
    with tempfile.TemporaryDirectory() as tmpDir:
        tmpPath = Path(tmpDir)

        # Write a synthetic JSONL with several metrics.
        healthDir = tmpPath / "health"
        healthDir.mkdir()
        jsonlPath = healthDir / "health.jsonl"
        syntheticRecord = json.dumps({
            "step": 50,
            "cfg_sim": 0.80,
            "seed_sim": 0.85,
            "encoder_cond_uncond_sim": 0.40,
            "cross_prompt_sim": 0.70,
            "fidelity": 0.65,
            "retrieval": 0.80,
            "distinctness": 0.30,
            "conditioning_sensitivity": 0.05,
            "effective_rank": 0.60,
            "intra_batch_sim": 0.30,
            "update_ratio": 5e-4,
            "loss_share": 0.20,
            "post_norm_mean_deviation": 0.05,
            "nan_inf": 0,
            "val_gap": 0.10,
            "epoch_time": 1.1,
        })
        jsonlPath.write_text(syntheticRecord + "\n", encoding="utf-8")

        hub = buildHealthHub(tmpPath)
        sheet = hub.report(runDir=tmpPath, outputFormat="markdown")
        hub.close()

        # All 16 metrics must appear in the sheet.
        allMetrics = [ref["metric"] for ref in SCORE_REFERENCE]
        assert len(allMetrics) == 16, (
            f"SCORE_REFERENCE must have 16 rows, has {len(allMetrics)}"
        )
        for metric in allMetrics:
            assert metric in sheet, (
                f"Metric '{metric}' missing from health sheet.\n"
                f"Sheet:\n{sheet}"
            )

        # Sheet must contain direction and verdict columns.
        assert "Verdict" in sheet
        assert "Target" in sheet


def test_report_contains_16_metrics_json() -> None:
    """health report JSON contains 16 items, each with metric/verdict."""
    with tempfile.TemporaryDirectory() as tmpDir:
        tmpPath = Path(tmpDir)
        healthDir = tmpPath / "health"
        healthDir.mkdir()

        # Minimal JSONL.
        (healthDir / "health.jsonl").write_text(
            json.dumps({"step": 1, "cfg_sim": 0.75}) + "\n",
            encoding="utf-8",
        )

        hub = buildHealthHub(tmpPath)
        sheetJson = hub.report(runDir=tmpPath, outputFormat="json")
        hub.close()

        rows = json.loads(sheetJson)
        assert len(rows) == 16, (
            f"Expected 16 rows, got {len(rows)}"
        )
        for row in rows:
            assert "metric" in row
            assert "verdict" in row
            assert "target" in row
            assert "direction" in row


def test_report_missing_metric_shows_unknown() -> None:
    """Metrics absent from JSONL show as N/A (UNKNOWN verdict)."""
    with tempfile.TemporaryDirectory() as tmpDir:
        tmpPath = Path(tmpDir)
        healthDir = tmpPath / "health"
        healthDir.mkdir()
        # Empty JSONL — no metrics at all.
        (healthDir / "health.jsonl").write_text(
            json.dumps({"step": 1}) + "\n", encoding="utf-8"
        )

        hub = buildHealthHub(tmpPath)
        sheet = hub.report(runDir=tmpPath, outputFormat="markdown")
        hub.close()

        # Sheet renders without crashing and contains N/A markers.
        assert "N/A" in sheet
