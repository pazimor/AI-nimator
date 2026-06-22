"""Negative import test for the controller (ROADMAP_DETERMINIST §3.1).

``model`` must never import ``data`` (import-linter contract
``model_no_data``).  This static AST check guards the controller modules
specifically so a stray ascending import is caught in unit tests, not
only by ``lint-imports``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MODEL_FILES = (
    Path("src/ainimator/model/controller_v2.py"),
    Path("src/ainimator/model/controller_rollout.py"),
    Path("src/ainimator/model/losses_controller_v2.py"),
)


def _importedModules(path: Path) -> set[str]:
    """Return the set of fully-qualified imported module names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("path", _MODEL_FILES, ids=lambda p: p.name)
def test_controller_model_never_imports_data(path: Path) -> None:
    imported = _importedModules(path)
    offending = {m for m in imported if m.startswith("ainimator.data")}
    assert not offending, f"{path.name} imports data: {offending}"
