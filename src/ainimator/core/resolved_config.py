"""Write ``resolved_config.yaml`` into a run's output directory.

Every training run must call :func:`writeResolvedConfig` immediately
after creating ``outputDir`` so that any future reader can reproduce
the exact configuration that produced the run's artefacts.

The file includes:

* The full resolved config (all fields, including defaults not supplied
  on the command line).
* The current git SHA (``HEAD``), or ``"untracked"`` when the repo is
  not available.
* The run timestamp (ISO-8601 UTC).

Design notes
------------
:func:`writeResolvedConfig` accepts either a Pydantic ``BaseModel``
(uses ``model_dump()``) or a plain Python dataclass (uses
``dataclasses.asdict()``).  Path values are stringified so the YAML
is human-readable.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import subprocess
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel

RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"

LOGGER = logging.getLogger(__name__)


def _gitSha() -> str:
    """Return the current ``HEAD`` SHA or ``"untracked"`` on failure.

    Returns
    -------
    str
        7-character short SHA, or ``"untracked"`` when the working
        directory is not inside a git repository or ``git`` is not
        available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "untracked"


def _toSerializable(value: Any) -> Any:
    """Recursively convert a value to a YAML-serializable type.

    Pydantic models, ``Path`` objects, and ``tuple`` values are
    converted to ``dict``, ``str``, and ``list`` respectively.
    ``None`` is kept as-is.

    Parameters
    ----------
    value : Any
        Arbitrary value from a Pydantic model dump.

    Returns
    -------
    Any
        YAML-serializable representation.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _toSerializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_toSerializable(item) for item in value]
    return value


def _configToDict(
    config: Union[BaseModel, object],
) -> dict[str, Any]:
    """Convert ``config`` to a plain dict for YAML serialisation.

    Accepts Pydantic ``BaseModel`` (uses ``model_dump()``) or any
    Python ``dataclass`` (uses ``dataclasses.asdict()``).

    Parameters
    ----------
    config : BaseModel | dataclass
        Configuration object to convert.

    Returns
    -------
    dict[str, Any]
        Serialisable dictionary.

    Raises
    ------
    TypeError
        If ``config`` is neither a Pydantic model nor a dataclass.
    """
    if isinstance(config, BaseModel):
        return config.model_dump()
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)  # type: ignore[arg-type]
    raise TypeError(
        f"config must be a Pydantic BaseModel or a dataclass; "
        f"got {type(config).__name__}."
    )


def writeResolvedConfig(
    config: Union[BaseModel, object],
    outputDir: Path,
) -> Path:
    """Serialise ``config`` to ``outputDir/resolved_config.yaml``.

    The file includes the full config dump (all fields, not just those
    that differ from the defaults), the git SHA of the current
    ``HEAD``, and the UTC timestamp of the call.

    Parameters
    ----------
    config : BaseModel | dataclass
        Any Pydantic model or frozen dataclass carrying the run
        configuration.
    outputDir : Path
        Directory to write into.  Must already exist (the caller is
        responsible for creating it).

    Returns
    -------
    Path
        Absolute path to the written file.

    Raises
    ------
    FileNotFoundError
        If ``outputDir`` does not exist.
    """
    if not outputDir.is_dir():
        raise FileNotFoundError(
            f"outputDir does not exist: {outputDir}"
        )

    payload: dict[str, Any] = {
        "git_sha": _gitSha(),
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "config": _toSerializable(_configToDict(config)),
    }

    destPath = outputDir / RESOLVED_CONFIG_FILENAME
    with destPath.open("w", encoding="utf-8") as fileHandle:
        yaml.dump(
            payload,
            fileHandle,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=True,
        )
    LOGGER.info("Resolved config written to %s.", destPath)
    return destPath
