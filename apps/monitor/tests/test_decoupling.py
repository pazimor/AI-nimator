"""Decoupling gate: apps/monitor must only import ainimator.health.record_schema.

This test AST-parses every non-test Python file in apps/monitor/ and
asserts that no import reaches into any other ainimator.* submodule.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MONITOR_DIR = Path(__file__).parent.parent
_ALLOWED_AINIMATOR_MODULE = "ainimator.health.record_schema"


def _collect_monitor_sources() -> list[Path]:
    """Return all non-test .py files in apps/monitor/ (not tests/)."""
    return [
        p for p in _MONITOR_DIR.glob("*.py")
        if not p.name.startswith("test_")
    ]


def _forbidden_imports(path: Path) -> list[str]:
    """Return list of forbidden ainimator.* import strings in a file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_forbidden_ainimator(module):
                forbidden.append(f"{path.name}: from {module} import …")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_ainimator(alias.name):
                    forbidden.append(f"{path.name}: import {alias.name}")
    return forbidden


def _is_forbidden_ainimator(module: str) -> bool:
    """Return True if module is ainimator.* but not the allowed one."""
    if not module.startswith("ainimator"):
        return False
    return module != _ALLOWED_AINIMATOR_MODULE


def test_monitor_imports_only_record_schema() -> None:
    """No apps/monitor source file imports any ainimator.* except record_schema."""
    violations: list[str] = []
    for source_file in _collect_monitor_sources():
        violations.extend(_forbidden_imports(source_file))

    assert not violations, (
        "apps/monitor must only import ainimator.health.record_schema.\n"
        "Violations found:\n"
        + "\n".join(f"  • {v}" for v in violations)
    )
