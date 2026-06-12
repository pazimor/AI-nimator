"""Health monitoring subsystem — Probe / Contract / HealthHub.

Public surface:
    HealthHub   — central registry (step / audit / diagnose / report)
    buildHealthHub — factory from health.yaml config
    Probe       — forward/backward hook capturing scalar stats
    Contract    — declarative YAML criteria → verdict
    Verdict     — OK / WARNING / CRITICAL / UNKNOWN
    ProbeSnapshot — one captured stat set from a Probe
    ContractResult — verdict + message from a Contract evaluation
"""

from ainimator.health.contract import Contract, ContractResult, Verdict
from ainimator.health.hub import HealthHub, buildHealthHub
from ainimator.health.probe import Probe, ProbeSnapshot

__all__ = [
    "Contract",
    "ContractResult",
    "HealthHub",
    "Probe",
    "ProbeSnapshot",
    "Verdict",
    "buildHealthHub",
]
