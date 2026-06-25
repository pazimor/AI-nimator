"""CLI entry point for the health subsystem.

Usage
-----
.. code-block:: bash

    # Follow a running training run's JSONL in real time
    python -m ainimator.cli.health watch <runDir>

    # Offline audit of a checkpoint
    python -m ainimator.cli.health audit <checkpoint>

    # Offline conditioning diagnostics on a checkpoint
    # (routes to controller or diffusion branch based on checkpoint model-type)
    python -m ainimator.cli.health diagnose <checkpoint>
    python -m ainimator.cli.health diagnose --profile controller_default <ckpt>

    # Aggregate JSONL + verdicts into a health sheet
    python -m ainimator.cli.health report <runDir>

This is a zero-logic CLI: it parses args and delegates to
:class:`~ainimator.health.hub.HealthHub`.

Routing
-------
``diagnose`` reads the ``model-type`` from the checkpoint payload (key
``"model_config"`` present → controller; otherwise diffusion).  If a
``resolved_config.yaml`` exists beside the checkpoint or in the run
directory, it is read first and takes precedence.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import yaml

LOGGER = logging.getLogger(__name__)

# Default diagnose profile names (network.yaml → diagnose_profiles)
_CONTROLLER_DEFAULT_PROFILE = "controller_default"
_DIFFUSION_DEFAULT_PROFILE = "diffusion_default"
_NETWORK_YAML = Path(__file__).parent.parent.parent / "configs" / "network.yaml"


# ------------------------------------------------------------------
# Model-type detection helpers
# ------------------------------------------------------------------
def _detectModelType(checkpointPath: Path) -> str:
    """Detect model-type from a checkpoint file.

    Resolution order:
    1. ``resolved_config.yaml`` in the same directory as the checkpoint
       (or in the parent of the ``checkpoints/`` subdir).
    2. Keys present in the checkpoint payload (``"model_config"`` key
       → controller; ``"training_config"`` with diffusion markers →
       diffusion).
    3. Default: ``"diffusion"`` (canonical default per ROADMAP §3.2).

    Parameters
    ----------
    checkpointPath : Path
        Path to the ``.pt`` file.

    Returns
    -------
    str
        ``"controller"`` or ``"diffusion"``.
    """
    # 1. Try resolved_config.yaml
    for searchDir in _resolvedConfigSearchDirs(checkpointPath):
        resolvedPath = searchDir / "resolved_config.yaml"
        if resolvedPath.exists():
            modelType = _modelTypeFromResolvedConfig(resolvedPath)
            if modelType is not None:
                LOGGER.debug(
                    "model-type=%s from %s", modelType, resolvedPath
                )
                return modelType

    # 2. Sniff checkpoint payload keys
    try:
        import torch
        payload: dict[str, Any] = torch.load(
            checkpointPath,
            map_location="cpu",
            weights_only=False,
        )
        if "model_config" in payload:
            LOGGER.debug(
                "model-type=controller detected from checkpoint keys"
            )
            return "controller"
        if "training_config" in payload:
            LOGGER.debug(
                "model-type=diffusion detected from checkpoint keys"
            )
            return "diffusion"
    except Exception as exc:
        LOGGER.warning("Could not sniff checkpoint payload: %s", exc)

    LOGGER.debug("model-type defaulting to diffusion")
    return "diffusion"


def _resolvedConfigSearchDirs(checkpointPath: Path) -> list[Path]:
    """Return candidate directories for ``resolved_config.yaml``.

    Checks the checkpoint's own directory, then the parent (for when
    the checkpoint is inside a ``checkpoints/`` subdir).
    """
    ckptDir = checkpointPath.parent
    candidates = [ckptDir]
    if ckptDir.name == "checkpoints":
        candidates.append(ckptDir.parent)
    return candidates


def _modelTypeFromResolvedConfig(resolvedPath: Path) -> str | None:
    """Parse ``model-type`` from a ``resolved_config.yaml``.

    Returns ``None`` if the file does not contain the key.
    """
    try:
        raw = yaml.safe_load(resolvedPath.read_text(encoding="utf-8")) or {}
        configSection = raw.get("config", {})
        # Controller training writes a ControllerTrainingConfig dataclass;
        # diffusion writes a V2FullTrainingConfig or similar.
        # Look for the model-type field in a few known locations.
        for candidate in (
            configSection.get("modelType"),
            configSection.get("model-type"),
            configSection.get("model_type"),
        ):
            if candidate is not None:
                return str(candidate)
        # Heuristic: if the config has 'model_config' or 'phaseMode',
        # it is a controller run.
        if "phaseMode" in configSection or "contextFrames" in configSection:
            return "controller"
    except Exception as exc:
        LOGGER.warning(
            "Could not parse resolved_config.yaml at %s: %s",
            resolvedPath,
            exc,
        )
    return None


# ------------------------------------------------------------------
# Profile loader
# ------------------------------------------------------------------
def _loadDiagnoseProfile(
    profileName: str,
    networkYaml: Path = _NETWORK_YAML,
) -> dict[str, Any]:
    """Load a diagnose profile from ``network.yaml``.

    Parameters
    ----------
    profileName : str
        Key under ``diagnose_profiles`` in ``network.yaml``.
    networkYaml : Path
        Path to the YAML file (injectable for tests).

    Returns
    -------
    dict
        Merged profile dict (empty dict if profile not found).
    """
    if not networkYaml.exists():
        LOGGER.warning(
            "network.yaml not found at %s; using empty profile", networkYaml
        )
        return {}
    raw = yaml.safe_load(networkYaml.read_text(encoding="utf-8")) or {}
    profiles = raw.get("diagnose_profiles", {})
    if profileName not in profiles:
        LOGGER.warning(
            "Diagnose profile '%s' not found in network.yaml; "
            "using defaults",
            profileName,
        )
        return {}
    return profiles[profileName]


# ------------------------------------------------------------------
# Sub-command implementations
# ------------------------------------------------------------------
def _cmdWatch(args: argparse.Namespace) -> int:
    """Stream health JSONL from a running run.

    Polls ``runDir/health/health.jsonl`` once per second and prints
    new lines until interrupted.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    runDir = Path(args.runDir)
    jsonlPath = runDir / "health" / "health.jsonl"
    LOGGER.info("Watching %s (Ctrl-C to stop)", jsonlPath)

    offset = 0
    try:
        while True:
            if jsonlPath.exists():
                content = jsonlPath.read_text(encoding="utf-8")
                lines = content.splitlines()
                for line in lines[offset:]:
                    if line.strip():
                        print(line, flush=True)
                offset = len(lines)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    return 0


def _cmdAudit(args: argparse.Namespace) -> int:
    """Run offline checkpoint audit.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    import json

    checkpointPath = Path(args.checkpoint)
    if not checkpointPath.exists():
        LOGGER.error("Checkpoint not found: %s", checkpointPath)
        return 1

    outputDir = Path(args.output_dir) if args.output_dir else (
        checkpointPath.parent
    )

    # L5 CLI loads the checkpoint; health/ (L4) never imports training/.
    from ainimator.training.training_v2 import loadCheckpointV2
    from ainimator.health.hub import buildHealthHub

    _, encoder, denoiser, _, normalizer, payload = loadCheckpointV2(
        checkpointPath, device=str(args.device)
    )
    encoder.eval()
    denoiser.eval()

    hub = buildHealthHub(outputDir)
    results = hub.audit(
        encoder=encoder,
        denoiser=denoiser,
        normalizer=normalizer,
        payload=payload,
        checkpointPath=checkpointPath,
    )
    hub.close()
    print(json.dumps(results, indent=2, default=str))
    return 0


def _cmdDiagnose(args: argparse.Namespace) -> int:
    """Run offline generation diagnostics, routing by model-type.

    Detects ``model-type`` from the checkpoint (via ``resolved_config.yaml``
    or checkpoint payload sniffing), then delegates to the controller or
    diffusion branch.  Profile values fill in any missing args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    checkpointPath = Path(args.checkpoint)
    if not checkpointPath.exists():
        LOGGER.error("Checkpoint not found: %s", checkpointPath)
        return 1

    modelType = _detectModelType(checkpointPath)
    LOGGER.info("diagnose: detected model-type=%s", modelType)

    if modelType == "controller":
        return _cmdDiagnoseController(args, checkpointPath)
    return _cmdDiagnoseDiffusion(args, checkpointPath)


def _cmdDiagnoseController(
    args: argparse.Namespace,
    checkpointPath: Path,
) -> int:
    """Controller branch of ``diagnose``.

    Loads profile from ``network.yaml`` (``diagnose_profiles``), applies
    arg overrides, and calls ``hub.diagnoseController()``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    checkpointPath : Path
        Validated checkpoint path.

    Returns
    -------
    int
        Exit code.
    """
    import json

    # Load profile (CLI --profile overrides the default).
    profileName = getattr(args, "profile", None) or _CONTROLLER_DEFAULT_PROFILE
    profile = _loadDiagnoseProfile(profileName)
    diagDefaults = profile.get("diagnose", {})

    # Resolve values: profile first, then hardcoded defaults (G-PROFILES:
    # hyperparameter flags removed from CLI; all knobs live in the profile).
    controlSpec = _resolve(
        getattr(args, "control", None),
        diagDefaults.get("control"),
        "forward:1.0",
    )
    rolloutFrames = int(_resolve(
        getattr(args, "rollout_frames", None),
        diagDefaults.get("rollout-frames"),
        120,
    ))
    # --shuffle-control uses store_true so it's always bool, never None.
    # Profile value applies only when the CLI flag was not explicitly set
    # (heuristic: flag is False and profile says True → use profile).
    cliShuffleControl = getattr(args, "shuffle_control", False)
    profileShuffleControl = diagDefaults.get("shuffle-control", False)
    shuffleControl = bool(cliShuffleControl or profileShuffleControl)
    phaseModeArg = _resolve(
        getattr(args, "phase_mode", None),
        diagDefaults.get("phase-mode"),
        None,
    )
    seedsStr = _resolve(
        getattr(args, "seeds", None),
        diagDefaults.get("seeds"),
        "0,42,123",
    )
    seeds = tuple(int(s) for s in str(seedsStr).split(",") if s.strip())

    outputDir = (
        Path(args.output_dir) if getattr(args, "output_dir", None)
        else checkpointPath.parent
    )

    # L5 CLI loads; health/ (L4) never imports training/.
    from ainimator.training.controller_training_v2 import (
        loadControllerCheckpoint,
        resolveControllerDevice,
    )
    from ainimator.health.hub import buildHealthHub

    deviceStr = getattr(args, "device", "auto")
    dev = resolveControllerDevice(str(deviceStr))
    model, stateNorm, deltaNorm, _controlMean, _controlStd = (
        loadControllerCheckpoint(checkpointPath, device=dev)
    )
    model.eval()

    hub = buildHealthHub(outputDir)
    results = hub.diagnoseController(
        model=model,
        stateNormalizer=stateNorm,
        deltaNormalizer=deltaNorm,
        device=dev,
        controlSpec=str(controlSpec),
        rolloutFrames=rolloutFrames,
        shuffleControl=shuffleControl,
        seeds=seeds,
        phaseMode=phaseModeArg if phaseModeArg else None,
    )
    hub.close()
    print(json.dumps(results, indent=2, default=str))
    return 0


def _cmdDiagnoseDiffusion(
    args: argparse.Namespace,
    checkpointPath: Path,
) -> int:
    """Diffusion branch of ``diagnose`` (unchanged from pre-LOT4).

    Loads the profile from ``network.yaml`` when ``--profile`` is
    supplied; otherwise uses CLI flag values directly.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    checkpointPath : Path
        Validated checkpoint path.

    Returns
    -------
    int
        Exit code.
    """
    import json

    # Load profile when provided; all knobs come from the profile (G-PROFILES).
    profileName = getattr(args, "profile", None)
    diagDefaults: dict[str, Any] = {}
    if profileName:
        profile = _loadDiagnoseProfile(profileName)
        diagDefaults = profile.get("diagnose", {})

    prompt = _resolve(
        getattr(args, "prompt", None),
        diagDefaults.get("prompt"),
        "a person walks forward",
    )
    seedsStr = _resolve(
        getattr(args, "seeds", None),
        diagDefaults.get("seeds"),
        "0,42,123",
    )
    cfgScalesStr = _resolve(
        getattr(args, "cfg_scales", None),
        diagDefaults.get("cfg-scales"),
        "1.0,4.0,6.0",
    )
    frames = int(_resolve(
        getattr(args, "frames", None),
        diagDefaults.get("frames"),
        120,
    ))
    numSteps = int(_resolve(
        getattr(args, "num_steps", None),
        diagDefaults.get("num-steps"),
        100,
    ))

    seeds = tuple(int(s) for s in str(seedsStr).split(",") if s.strip())
    cfgScales = tuple(
        float(c) for c in str(cfgScalesStr).split(",") if c.strip()
    )
    outputDir = (
        Path(args.output_dir) if getattr(args, "output_dir", None)
        else checkpointPath.parent
    )

    # L5 CLI loads the checkpoint; health/ (L4) never imports training/.
    from ainimator.training.training_v2 import (
        EMPTY_PROMPT, loadCheckpointV2, resolveDevice,
    )
    from ainimator.model.sampler_v2 import DDIMSamplerV2
    from ainimator.health.hub import buildHealthHub

    devObj = resolveDevice(str(getattr(args, "device", "auto")))
    tok, enc, den, sched, norm, payload = loadCheckpointV2(
        checkpointPath, device=devObj
    )
    enc.eval()
    den.eval()
    sampler = DDIMSamplerV2(
        sched,
        predictionMode=str(payload["training_config"]["predictionMode"]),
    )

    hub = buildHealthHub(outputDir)
    results = hub.diagnose(
        tokenizer=tok,
        encoder=enc,
        denoiser=den,
        sampler=sampler,
        normalizer=norm,
        device=devObj,
        prompt=str(prompt),
        seeds=seeds,
        cfgScales=cfgScales,
        frames=frames,
        numSteps=numSteps,
        emptyPrompt=EMPTY_PROMPT,
    )
    hub.close()
    print(json.dumps(results, indent=2, default=str))
    return 0


# ------------------------------------------------------------------
# Value resolution helpers
# ------------------------------------------------------------------
def _resolve(
    cliValue: Any,
    profileValue: Any,
    default: Any,
) -> Any:
    """Return the first non-None value among cli, profile, default."""
    if cliValue is not None:
        return cliValue
    if profileValue is not None:
        return profileValue
    return default


def _resolveBool(
    cliValue: Any,
    profileValue: Any,
    default: bool,
) -> bool:
    """Resolve a boolean through cli → profile → default."""
    resolved = _resolve(cliValue, profileValue, default)
    if isinstance(resolved, bool):
        return resolved
    return str(resolved).lower() in ("true", "1", "yes")


def _cmdReport(args: argparse.Namespace) -> int:
    """Generate health sheet for a completed run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    runDir = Path(args.runDir)
    if not runDir.exists():
        LOGGER.error("runDir not found: %s", runDir)
        return 1

    from ainimator.health.hub import buildHealthHub

    hub = buildHealthHub(runDir)
    sheet = hub.report(
        runDir=runDir,
        outputFormat=str(args.format),
    )
    hub.close()
    print(sheet)
    return 0


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------
def _buildParser() -> argparse.ArgumentParser:
    """Build the argument parser for the health CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m ainimator.cli.health",
        description="AI-nimator health monitoring CLI.",
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # watch
    watchP = sub.add_parser(
        "watch", help="Stream health JSONL from a running run."
    )
    watchP.add_argument("runDir", type=str, help="Run output directory.")

    # audit
    auditP = sub.add_parser(
        "audit", help="Offline audit of a checkpoint."
    )
    auditP.add_argument("checkpoint", type=str, help="Checkpoint .pt path.")
    auditP.add_argument("--device", default="cpu")
    auditP.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Where to write audit JSON (default: checkpoint dir).",
    )

    # diagnose
    diagP = sub.add_parser(
        "diagnose",
        help=(
            "Offline diagnostics (routes to controller or diffusion "
            "branch based on checkpoint model-type)."
        ),
    )
    diagP.add_argument("checkpoint", type=str, help="Checkpoint .pt path.")
    diagP.add_argument(
        "--profile",
        default=None,
        help=(
            "Diagnose profile name from network.yaml diagnose_profiles "
            "(e.g. 'controller_default', 'controller_short', "
            "'diffusion_default').  Profile values fill in defaults; "
            "any explicit flag overrides the profile."
        ),
    )
    diagP.add_argument("--device", default="auto")
    diagP.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Where to write diagnose JSON (default: checkpoint dir).",
    )

    # Hyperparameter flags removed (G-PROFILES): all run knobs are
    # controlled via --profile <name> in network.yaml diagnose_profiles.
    # The _cmdDiagnoseController / _cmdDiagnoseDiffusion handlers read
    # profile values and fall back to hardcoded defaults when no profile
    # is specified.

    # report
    reportP = sub.add_parser(
        "report", help="Aggregate JSONL into a health sheet."
    )
    reportP.add_argument("runDir", type=str, help="Run output directory.")
    reportP.add_argument(
        "--format", default="markdown", choices=["markdown", "json"],
    )

    return parser


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code.
    """
    parser = _buildParser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.logLevel),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    dispatch = {
        "watch": _cmdWatch,
        "audit": _cmdAudit,
        "diagnose": _cmdDiagnose,
        "report": _cmdReport,
    }
    handler = dispatch.get(args.subcommand)
    if handler is None:
        LOGGER.error("Unknown subcommand: %s", args.subcommand)
        return 1
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
