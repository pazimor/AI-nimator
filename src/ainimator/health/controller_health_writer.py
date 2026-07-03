"""JSONL health stream for the controller training loops.

The controller loops (A1 overfit, A6 generalization) compute their
health metrics inline via :mod:`ainimator.health.controller_metrics`
instead of going through HealthHub probes.  This module bridges them
onto the published record schema (:mod:`ainimator.health.record_schema`)
so a controller run writes the same ``outputDir/health/health.jsonl``
stream the Streamlit monitor (``apps/monitor``) already consumes for
diffusion runs.
"""

from __future__ import annotations

from pathlib import Path

from ainimator.health.contract import Verdict
from ainimator.health.jsonl_writer import JsonlWriter
from ainimator.health.record_schema import (
    LOSS_PREFIX,
    LOSS_SHARE_PREFIX,
    STEP_KEY,
    VERDICT_PREFIX,
)

# Bare summary key: minimum weighted share among active loss components.
_LOSS_SHARE_BARE = "loss_share"
_LOSS_TOTAL_KEY = f"{LOSS_PREFIX}total"


class ControllerHealthWriter:
    """Append controller training records to ``health/health.jsonl``.

    Parameters
    ----------
    outputDir : Path
        Run output directory; the stream lands in
        ``outputDir/health/health.jsonl``.
    """

    def __init__(self, outputDir: Path) -> None:
        self._jsonl = JsonlWriter(outputDir, "health")

    @property
    def path(self) -> Path:
        """Absolute path to the JSONL file."""
        return self._jsonl.path

    def writeLosses(
        self,
        step: int,
        totalLoss: float,
        weightedComponents: dict[str, float],
    ) -> None:
        """Write one loss record (``loss_*`` + ``loss_share.*`` keys).

        Parameters
        ----------
        step : int
            Monotonic step (epoch index for the controller loops).
        totalLoss : float
            Weighted total loss (backprop objective).
        weightedComponents : dict[str, float]
            Per-term *weighted* losses keyed with the ``loss_`` prefix
            already applied (e.g. ``loss_velocity``), matching
            :class:`~ainimator.model.losses_controller_v2.ControllerLossResult`
            component names.
        """
        record: dict[str, float | int] = {
            STEP_KEY: step,
            _LOSS_TOTAL_KEY: totalLoss,
        }
        shares: list[float] = []
        for name, value in weightedComponents.items():
            record[name] = value
            if totalLoss > 0.0:
                share = value / totalLoss
                component = name.removeprefix(LOSS_PREFIX)
                record[f"{LOSS_SHARE_PREFIX}{component}"] = share
                shares.append(share)
        if shares:
            record[_LOSS_SHARE_BARE] = min(shares)
        self._jsonl.write(record)

    def writeEvaluation(
        self,
        step: int,
        metrics: dict[str, float],
        verdicts: dict[str, Verdict],
    ) -> None:
        """Write one evaluation record (metrics + ``verdict.*`` keys).

        Parameters
        ----------
        step : int
            Monotonic step at which the evaluation ran.
        metrics : dict[str, float]
            Flat controller metrics (``control_sensitivity``,
            ``mean_collapse_rank``, ``rollout_drift``, ...).
        verdicts : dict[str, Verdict]
            Contract verdicts keyed by contract name.
        """
        record: dict[str, float | int | str] = {STEP_KEY: step}
        record.update(metrics)
        for name, verdict in verdicts.items():
            record[f"{VERDICT_PREFIX}{name}"] = verdict.value
        self._jsonl.write(record)
