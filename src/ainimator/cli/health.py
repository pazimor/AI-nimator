"""CLI entry point for the health subsystem.

Usage
-----
.. code-block:: bash

    # Follow a running training run's JSONL in real time
    python -m ainimator.cli.health watch <runDir>

    # Offline audit of a checkpoint
    python -m ainimator.cli.health audit <checkpoint>

    # Offline conditioning diagnostics on a checkpoint
    python -m ainimator.cli.health diagnose <checkpoint>

    # Aggregate JSONL + verdicts into a health sheet
    python -m ainimator.cli.health report <runDir>

This is a zero-logic CLI: it parses args and delegates to
:class:`~ainimator.health.hub.HealthHub`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger(__name__)


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
    """Run offline generation diagnostics.

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
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    cfgScales = tuple(
        float(c) for c in args.cfg_scales.split(",") if c.strip()
    )

    # L5 CLI loads the checkpoint; health/ (L4) never imports training/.
    from ainimator.training.training_v2 import (
        EMPTY_PROMPT, loadCheckpointV2, resolveDevice,
    )
    from ainimator.model.sampler_v2 import DDIMSamplerV2
    from ainimator.health.hub import buildHealthHub

    devObj = resolveDevice(str(args.device))
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
        prompt=str(args.prompt),
        seeds=seeds,
        cfgScales=cfgScales,
        frames=int(args.frames),
        numSteps=int(args.num_steps),
        emptyPrompt=EMPTY_PROMPT,
    )
    hub.close()
    print(json.dumps(results, indent=2, default=str))
    return 0


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
        "diagnose", help="Offline generation diagnostics."
    )
    diagP.add_argument("checkpoint", type=str, help="Checkpoint .pt path.")
    diagP.add_argument(
        "--prompt", default="a person walks forward",
        help="Primary conditioning prompt.",
    )
    diagP.add_argument("--seeds", default="0,42,123")
    diagP.add_argument("--cfg-scales", dest="cfg_scales",
                       default="1.0,2.5,3.5")
    diagP.add_argument("--frames", type=int, default=120)
    diagP.add_argument("--num-steps", dest="num_steps", type=int,
                       default=100)
    diagP.add_argument("--device", default="auto")
    diagP.add_argument(
        "--output-dir", dest="output_dir", default=None,
    )

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
