"""Contract: declarative YAML criteria → OK / WARNING / CRITICAL.

A Contract binds one health metric to a threshold specification read
from ``src/configs/health.yaml``.  At evaluation time it takes a
dict of metric floats and returns a :class:`Verdict`.

Verdict vocabulary is aligned with the pytorch-auditor skill so the
two tools share a common output format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


# ------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------
class Verdict(str, Enum):
    """Health verdict, ordered by severity (OK < WARNING < CRITICAL).

    Values are strings so they serialize naturally to JSON / YAML.
    """

    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"  # metric was not produced


# ------------------------------------------------------------------
# Contract data container
# ------------------------------------------------------------------
@dataclass(frozen=True)
class ContractResult:
    """Result of evaluating one contract against a metric value.

    Attributes
    ----------
    name : str
        Contract identifier.
    metric : str
        Name of the metric key that was evaluated.
    value : float or None
        The observed metric value (None when unavailable).
    verdict : Verdict
        OK / WARNING / CRITICAL / UNKNOWN.
    message : str
        Human-readable explanation (shown in logs and health sheet).
    """

    name: str
    metric: str
    value: float | None
    verdict: Verdict
    message: str


# ------------------------------------------------------------------
# Contract
# ------------------------------------------------------------------
class Contract:
    """Declarative health criterion for one metric.

    Supports three evaluation modes:
    - ``lower_is_better``: thresholds are upper bounds
      (ok < ok_threshold, ok_threshold ≤ value < warning_threshold,
       value ≥ critical_above → CRITICAL).
    - ``higher_is_better``: thresholds are lower bounds
      (value > ok_threshold → OK, critical_below → CRITICAL).
    - ``range``: healthy window is [ok_min, ok_max].

    Parameters
    ----------
    name : str
        Human-readable identifier.
    metric : str
        Key in the metrics dict passed to :meth:`evaluate`.
    direction : str
        One of ``"lower_is_better"``, ``"higher_is_better"``,
        ``"range"``.
    thresholds : dict
        Threshold values (keys depend on direction).
    """

    def __init__(
        self,
        name: str,
        metric: str,
        direction: str,
        thresholds: dict[str, float],
    ) -> None:
        self.name = name
        self.metric = metric
        self.direction = direction
        self.thresholds = thresholds

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        metrics: dict[str, float | None],
    ) -> ContractResult:
        """Evaluate this contract against a metrics dict.

        Parameters
        ----------
        metrics : dict
            Keys are metric names; values are floats (or None when
            the metric could not be produced).

        Returns
        -------
        ContractResult
        """
        value = metrics.get(self.metric)
        if value is None or math.isnan(float(value)):
            return ContractResult(
                name=self.name,
                metric=self.metric,
                value=None,
                verdict=Verdict.UNKNOWN,
                message=f"Metric '{self.metric}' was not produced.",
            )
        return self._applyThresholds(float(value))

    def _applyThresholds(self, value: float) -> ContractResult:
        """Apply direction-specific thresholds to a concrete value."""
        if self.direction == "lower_is_better":
            return self._evalLower(value)
        if self.direction == "higher_is_better":
            return self._evalHigher(value)
        if self.direction == "range":
            return self._evalRange(value)
        return ContractResult(
            name=self.name,
            metric=self.metric,
            value=value,
            verdict=Verdict.UNKNOWN,
            message=f"Unknown direction '{self.direction}'.",
        )

    def _evalLower(self, value: float) -> ContractResult:
        """Evaluate a lower-is-better metric."""
        critAbove = self.thresholds.get("critical_above")
        warnThresh = self.thresholds.get("warning_threshold")
        okThresh = self.thresholds.get("ok_threshold")

        if critAbove is not None and value > critAbove:
            return ContractResult(
                name=self.name,
                metric=self.metric,
                value=value,
                verdict=Verdict.CRITICAL,
                message=(
                    f"{self.metric}={value:.4f} > {critAbove:.4f} "
                    "(CRITICAL threshold)."
                ),
            )
        if warnThresh is not None and value >= warnThresh:
            return ContractResult(
                name=self.name,
                metric=self.metric,
                value=value,
                verdict=Verdict.WARNING,
                message=(
                    f"{self.metric}={value:.4f} in WARNING zone "
                    f"[{warnThresh:.4f}, {critAbove}]."
                ),
            )
        return ContractResult(
            name=self.name,
            metric=self.metric,
            value=value,
            verdict=Verdict.OK,
            message=(
                f"{self.metric}={value:.4f} below ok threshold "
                f"{okThresh}."
            ),
        )

    def _evalHigher(self, value: float) -> ContractResult:
        """Evaluate a higher-is-better metric."""
        critBelow = self.thresholds.get("critical_below")
        warnThresh = self.thresholds.get("warning_threshold")
        okThresh = self.thresholds.get("ok_threshold")

        if critBelow is not None and value < critBelow:
            return ContractResult(
                name=self.name,
                metric=self.metric,
                value=value,
                verdict=Verdict.CRITICAL,
                message=(
                    f"{self.metric}={value:.4f} < {critBelow:.4f} "
                    "(CRITICAL: near zero / dead conditioning)."
                ),
            )
        if warnThresh is not None and value < warnThresh:
            return ContractResult(
                name=self.name,
                metric=self.metric,
                value=value,
                verdict=Verdict.WARNING,
                message=(
                    f"{self.metric}={value:.4f} below WARNING "
                    f"threshold {warnThresh:.4f}."
                ),
            )
        if okThresh is not None and value < okThresh:
            return ContractResult(
                name=self.name,
                metric=self.metric,
                value=value,
                verdict=Verdict.WARNING,
                message=(
                    f"{self.metric}={value:.4f} below OK "
                    f"threshold {okThresh:.4f}."
                ),
            )
        return ContractResult(
            name=self.name,
            metric=self.metric,
            value=value,
            verdict=Verdict.OK,
            message=f"{self.metric}={value:.4f} (OK).",
        )

    def _evalRange(self, value: float) -> ContractResult:
        """Evaluate a range-based metric."""
        critBelow = self.thresholds.get("critical_below")
        critAbove = self.thresholds.get("critical_above")
        warnMin = self.thresholds.get("warning_min")
        warnMax = self.thresholds.get("warning_max")

        if critBelow is not None and value < critBelow:
            return ContractResult(
                name=self.name,
                metric=self.metric,
                value=value,
                verdict=Verdict.CRITICAL,
                message=(
                    f"{self.metric}={value:.6f} < {critBelow:.6f} "
                    "(CRITICAL: effectively zero / NaN)."
                ),
            )
        if critAbove is not None and value > critAbove:
            return ContractResult(
                name=self.name,
                metric=self.metric,
                value=value,
                verdict=Verdict.CRITICAL,
                message=(
                    f"{self.metric}={value:.6f} > {critAbove:.6f} "
                    "(CRITICAL: exploding)."
                ),
            )
        inRange = True
        if warnMin is not None and value < warnMin:
            inRange = False
        if warnMax is not None and value > warnMax:
            inRange = False
        if not inRange:
            return ContractResult(
                name=self.name,
                metric=self.metric,
                value=value,
                verdict=Verdict.WARNING,
                message=(
                    f"{self.metric}={value:.6f} outside healthy "
                    f"range [{warnMin}, {warnMax}]."
                ),
            )
        return ContractResult(
            name=self.name,
            metric=self.metric,
            value=value,
            verdict=Verdict.OK,
            message=f"{self.metric}={value:.6f} (OK).",
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def fromYamlDict(cls, data: dict[str, Any]) -> "Contract":
        """Construct a Contract from a health.yaml contract entry.

        Parameters
        ----------
        data : dict
            One entry from the ``contracts`` list in health.yaml.

        Returns
        -------
        Contract
        """
        name = str(data["name"])
        metric = str(data["metric"])
        direction = str(data["direction"])
        thresholds = {
            key: float(data[key])
            for key in (
                "ok_threshold",
                "warning_threshold",
                "critical_above",
                "critical_below",
                "ok_min",
                "ok_max",
                "warning_min",
                "warning_max",
            )
            if key in data
        }
        return cls(
            name=name,
            metric=metric,
            direction=direction,
            thresholds=thresholds,
        )
