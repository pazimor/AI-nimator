"""Plugin build orchestrator — export → validate → deliver → build (B5).

One command produces an engine plugin carrying a **fresh** bundle from
the checkpoint given on the command line (no default, no auto-detect —
vérité §2.9). Usage::

    python -m apps.build.build_plugin --target unity \\
        --checkpoint output/<run>/checkpoints/<ckpt>.pt

Steps (ROADMAP_PLUGINS §3.4):

1. **export**  — ``python -m ainimator.cli.export_onnx bundle`` into a
   temporary directory (subprocess: the orchestrator only invokes the
   CLI, it never imports model code).
2. **validate** — ``manifest.json`` against
   ``apps/spec/manifest.schema.json`` + contract-version compatibility
   with the target plugin (prefix+major match), presets against
   ``apps/spec/control_preset.schema.json``.
3. **deliver** — copy the bundle into the target plugin's gitignored
   resource folder.
4. **build**   — engine packaging step (``Unity -batchmode`` /
   ``RunUAT BuildPlugin``); requires the engine location via an
   environment variable and can be skipped with ``--skip-engine-build``
   on machines without the engine.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("apps.build.build_plugin")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "apps" / "spec"
MANIFEST_SCHEMA = SPEC_DIR / "manifest.schema.json"
PRESET_SCHEMA = SPEC_DIR / "control_preset.schema.json"

# Contract versions each plugin target accepts: prefix + major of
# ``bundle_version`` must match one of these (mirrors the in-engine
# checks of BundleVersion.cs / FBundleLoader).
ACCEPTED_CONTRACT_VERSIONS: dict[str, tuple[str, ...]] = {
    "unity": ("A7",),
    "unreal": ("A7",),
}

_VERSION_PATTERN = re.compile(r"^([A-Z]?)(\d+)\.(\d+)$")


@dataclass(frozen=True)
class TargetConfig:
    """Delivery + engine-build description of one plugin target.

    Attributes
    ----------
    name : str
        Target key (``unity`` / ``unreal``).
    deliveryDir : Path
        Gitignored plugin folder that receives the fresh bundle.
    engineEnvVar : str
        Environment variable pointing at the engine installation.
    """

    name: str
    deliveryDir: Path
    engineEnvVar: str


TARGETS: dict[str, TargetConfig] = {
    "unity": TargetConfig(
        name="unity",
        deliveryDir=(
            REPO_ROOT / "apps" / "unity-sentis"
            / "com.ainimator.controller" / "StreamingAssets"
            / "AInimatorBundle"
        ),
        engineEnvVar="UNITY_PATH",
    ),
    "unreal": TargetConfig(
        name="unreal",
        deliveryDir=(
            REPO_ROOT / "apps" / "unreal-nne" / "AInimator" / "Content"
            / "AInimatorBundle"
        ),
        engineEnvVar="UE_ROOT",
    ),
}


class BuildError(SystemExit):
    """Fatal orchestration error with an explicit message."""

    def __init__(self, message: str) -> None:
        super().__init__(f"build_plugin: {message}")


def exportBundle(
    checkpoint: Path,
    bundleDir: Path,
    encoderArtifact: Path | None = None,
) -> None:
    """Run the export CLI to produce a fresh bundle (step 1).

    Parameters
    ----------
    checkpoint : Path
        Controller checkpoint (mandatory, validated to exist).
    bundleDir : Path
        Destination directory for the bundle.
    encoderArtifact : Path | None
        Frozen CLIP text-encoder artifact.  When given the bundle
        ships the in-engine text encoder (B7 / A7.1 —
        ``apps/spec/text_encoding.md``).
    """
    if not checkpoint.exists():
        raise BuildError(f"checkpoint not found: {checkpoint}")
    if encoderArtifact is not None and not encoderArtifact.exists():
        raise BuildError(
            f"encoder artifact not found: {encoderArtifact}"
        )
    resolvedConfig = checkpoint.parent.parent / "resolved_config.yaml"
    command = [
        sys.executable, "-m", "ainimator.cli.export_onnx", "bundle",
        "--checkpoint", str(checkpoint),
        "--output-dir", str(bundleDir),
    ]
    if resolvedConfig.exists():
        command += ["--resolved-config", str(resolvedConfig)]
    if encoderArtifact is not None:
        command += ["--encoder-artifact", str(encoderArtifact)]
    LOGGER.info("[1/4] export: %s", " ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise BuildError("bundle export failed (see output above).")


def _contractKey(bundleVersion: str) -> str:
    """Return the ``prefix+major`` compatibility key of a version."""
    match = _VERSION_PATTERN.match(bundleVersion)
    if match is None:
        raise BuildError(
            f"manifest bundle_version {bundleVersion!r} does not match "
            "the expected '<PREFIX><major>.<minor>' pattern."
        )
    return f"{match.group(1)}{match.group(2)}"


def validateBundle(bundleDir: Path, target: TargetConfig) -> None:
    """Validate manifest + presets against the spec schemas (step 2).

    Parameters
    ----------
    bundleDir : Path
        Freshly exported bundle directory.
    target : TargetConfig
        Target whose accepted contract versions gate the build.
    """
    import jsonschema

    LOGGER.info("[2/4] validate: %s", bundleDir)
    manifest = json.loads((bundleDir / "manifest.json").read_text())
    schema = json.loads(MANIFEST_SCHEMA.read_text())
    jsonschema.validate(manifest, schema)
    key = _contractKey(manifest["bundle_version"])
    accepted = ACCEPTED_CONTRACT_VERSIONS[target.name]
    if key not in accepted:
        raise BuildError(
            f"bundle contract version {manifest['bundle_version']!r} "
            f"(key {key!r}) is not accepted by the {target.name} plugin "
            f"(accepted: {', '.join(accepted)}). STOP."
        )
    presetSchema = json.loads(PRESET_SCHEMA.read_text())
    for presetPath in sorted((bundleDir / "presets").glob("*.json")):
        jsonschema.validate(
            json.loads(presetPath.read_text()), presetSchema
        )
    LOGGER.info("manifest + presets valid (contract %s).", key)


def deliverBundle(bundleDir: Path, target: TargetConfig) -> Path:
    """Copy the validated bundle into the plugin resources (step 3).

    The destination is wiped first so no stale bundle file survives —
    the delivered bundle is always exactly the fresh export.

    Returns
    -------
    Path
        The delivery directory.
    """
    destination = target.deliveryDir
    LOGGER.info("[3/4] deliver: %s", destination)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(bundleDir, destination)
    return destination


def _unityBuildCommand(enginePath: str) -> list[str]:
    """Assemble the Unity batchmode packaging command."""
    projectPath = os.environ.get(
        "UNITY_PROJECT",
        str(REPO_ROOT / "apps" / "unity-sentis" / "TestProject"),
    )
    packMethod = os.environ.get(
        "UNITY_PACK_METHOD",
        "AInimator.Controller.Editor.BundleDeliveryTools.PackPlugin",
    )
    return [
        enginePath, "-batchmode", "-quit",
        "-projectPath", projectPath,
        "-executeMethod", packMethod,
    ]


def _unrealBuildCommand(engineRoot: str) -> list[str]:
    """Assemble the RunUAT BuildPlugin packaging command."""
    runUat = Path(engineRoot) / "Engine" / "Build" / "BatchFiles" / (
        "RunUAT.bat" if os.name == "nt" else "RunUAT.sh"
    )
    uplugin = (
        REPO_ROOT / "apps" / "unreal-nne" / "AInimator"
        / "AInimator.uplugin"
    )
    package = REPO_ROOT / "output" / "plugin_builds" / "unreal"
    return [
        str(runUat), "BuildPlugin",
        f"-Plugin={uplugin}",
        f"-Package={package}",
    ]


def buildEngine(target: TargetConfig) -> None:
    """Run the engine packaging step (step 4) — needs the engine.

    Parameters
    ----------
    target : TargetConfig
        Target whose engine env var locates the toolchain.
    """
    engine = os.environ.get(target.engineEnvVar)
    if not engine:
        raise BuildError(
            f"engine step needs ${target.engineEnvVar} (path to the "
            f"{target.name} toolchain). Set it, or pass "
            "--skip-engine-build to stop after delivery."
        )
    command = (
        _unityBuildCommand(engine) if target.name == "unity"
        else _unrealBuildCommand(engine)
    )
    LOGGER.info("[4/4] build: %s", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise BuildError(f"{target.name} engine build failed.")


def _parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the orchestrator CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m apps.build.build_plugin",
        description="Export → validate → deliver → build (B5, §3.4).",
    )
    parser.add_argument(
        "--target", choices=sorted(TARGETS), required=True,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, metavar="FILE",
        help="Controller checkpoint (.pt). Mandatory — no default.",
    )
    parser.add_argument(
        "--skip-engine-build", action="store_true",
        help="Stop after delivery (machine without the engine).",
    )
    parser.add_argument(
        "--encoder-artifact", type=Path, default=None, metavar="DIR",
        dest="encoderArtifact",
        help=(
            "Frozen CLIP text-encoder artifact.  Ships the in-engine "
            "text encoder in the bundle (B7 / A7.1). Optional."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full pipeline for one target."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parseArgs(argv)
    target = TARGETS[args.target]
    with tempfile.TemporaryDirectory(prefix="ainimator_bundle_") as tmp:
        bundleDir = Path(tmp) / "bundle"
        exportBundle(args.checkpoint, bundleDir, args.encoderArtifact)
        validateBundle(bundleDir, target)
        delivered = deliverBundle(bundleDir, target)
    if args.skip_engine_build:
        LOGGER.info(
            "engine build skipped; fresh bundle delivered at %s",
            delivered,
        )
        return
    buildEngine(target)


if __name__ == "__main__":
    main()
