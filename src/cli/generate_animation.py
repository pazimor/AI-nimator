"""CLI entry point to generate animations from a checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.features.generation.generate_animation import (
    createAnimationFromCheckpoint,
    loadInferenceSettings,
)
from src.shared.types import (
    GenerationInferenceConfig,
    GenerationModelSettings,
    GenerationOutputOptions,
    validateTag,
)

DEFAULT_GENERATION_CONFIG_PATH = Path("src/configs/train_generation.yaml")
DEFAULT_DATASET_CONFIG_PATH = Path("src/configs/dataset.yaml")
DEFAULT_OUTPUT_PATH = Path("output/generated_animation")
DEFAULT_DEVICE = "auto"
DEFAULT_DDIM_STEPS = 50

JSON_SUFFIX = ".json"
DAE_SUFFIX = ".dae"


def buildArgumentParser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Generate an animation from a trained checkpoint.",
    )
    addCoreArguments(parser)
    addOutputArguments(parser)
    addModelArguments(parser)
    addSamplingArguments(parser)
    return parser


def addCoreArguments(parser: argparse.ArgumentParser) -> None:
    """
    Register core generation arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser instance to update.
    """
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the generation checkpoint file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt used for generation.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag label (must match the known list).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        required=True,
        help="Number of frames to generate.",
    )


def addOutputArguments(parser: argparse.ArgumentParser) -> None:
    """
    Register output-related arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser instance to update.
    """
    addOutputPathArguments(parser)
    addOutputFpsArgument(parser)
    addTranslationArguments(parser)


def addOutputPathArguments(parser: argparse.ArgumentParser) -> None:
    """
    Register output path arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser instance to update.
    """
    parser.add_argument(
        "--output",
        dest="outputPath",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output base path for JSON/DAE files "
            "(default: output/generated_animation)."
        ),
    )
    parser.add_argument(
        "--dataset-config",
        dest="datasetConfigPath",
        type=Path,
        default=DEFAULT_DATASET_CONFIG_PATH,
        help="Path to dataset config used for Collada export.",
    )


def addOutputFpsArgument(parser: argparse.ArgumentParser) -> None:
    """
    Register FPS override argument.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser instance to update.
    """
    parser.add_argument(
        "--fps",
        dest="fps",
        type=int,
        default=None,
        help="Override FPS (defaults to dataset config fallback).",
    )


def addTranslationArguments(parser: argparse.ArgumentParser) -> None:
    """
    Register translation export flags.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser instance to update.
    """
    parser.add_argument(
        "--zero-root-translation",
        dest="zeroRootTranslation",
        action="store_true",
        help="Zero root translation during Collada export.",
    )
    parser.add_argument(
        "--anchor-root-translation",
        dest="anchorRootTranslation",
        action="store_true",
        help="Anchor root translation during Collada export.",
    )


def addModelArguments(parser: argparse.ArgumentParser) -> None:
    """
    Register model and config override arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser instance to update.
    """
    addConfigArguments(parser)
    addModelOverrideArguments(parser)


def addConfigArguments(parser: argparse.ArgumentParser) -> None:
    """
    Register config path arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser instance to update.
    """
    parser.add_argument(
        "--config",
        dest="configPath",
        type=Path,
        default=DEFAULT_GENERATION_CONFIG_PATH,
        help="Path to the generation training config file.",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        type=str,
        default=None,
        help="Network profile to use from network.yaml.",
    )


def addModelOverrideArguments(parser: argparse.ArgumentParser) -> None:
    """
    Register model override arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser instance to update.
    """
    parser.add_argument(
        "--model-name",
        dest="modelName",
        type=str,
        default=None,
        help="Override the text encoder model name.",
    )
    parser.add_argument(
        "--clip-checkpoint",
        dest="clipCheckpoint",
        type=Path,
        default=None,
        help="Override CLIP checkpoint path from config.",
    )
    parser.add_argument(
        "--network-config",
        dest="networkConfigPath",
        type=Path,
        default=None,
        help="Override network config path from config.",
    )


def addSamplingArguments(parser: argparse.ArgumentParser) -> None:
    """
    Register sampling arguments for inference.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser instance to update.
    """
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device backend: auto, cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--ddim-steps",
        dest="ddimSteps",
        type=int,
        default=DEFAULT_DDIM_STEPS,
        help="Number of DDIM steps used for sampling.",
    )


def resolveOutputPaths(outputPath: Path) -> tuple[Path, Path]:
    """
    Resolve JSON and DAE output paths from a base path.

    Parameters
    ----------
    outputPath : Path
        Base path or file path with extension.

    Returns
    -------
    tuple[Path, Path]
        JSON and DAE output paths.
    """
    suffix = outputPath.suffix.lower()
    if suffix == JSON_SUFFIX:
        return outputPath, outputPath.with_suffix(DAE_SUFFIX)
    if suffix == DAE_SUFFIX:
        return outputPath.with_suffix(JSON_SUFFIX), outputPath
    if suffix == "":
        return outputPath.with_suffix(JSON_SUFFIX), outputPath.with_suffix(
            DAE_SUFFIX
        )
    raise ValueError(f"Unsupported output suffix: {suffix}")


def validateTagValue(tag: Optional[str]) -> Optional[str]:
    """
    Validate tag value when provided.

    Parameters
    ----------
    tag : Optional[str]
        Tag string from CLI.

    Returns
    -------
    Optional[str]
        Validated tag or None.
    """
    if tag is None:
        return None
    return validateTag(tag)


def resolveModelSettings(
    configPath: Path,
    profile: Optional[str],
    modelNameOverride: Optional[str],
    clipCheckpointOverride: Optional[Path],
    networkConfigOverride: Optional[Path],
) -> GenerationModelSettings:
    """
    Resolve model settings from config file and overrides.

    Parameters
    ----------
    configPath : Path
        Path to generation training config.
    profile : Optional[str]
        Profile name within the config.
    modelNameOverride : Optional[str]
        CLI override for the model name.
    clipCheckpointOverride : Optional[Path]
        CLI override for the CLIP checkpoint path.
    networkConfigOverride : Optional[Path]
        CLI override for the network config path.

    Returns
    -------
    GenerationModelSettings
        Normalized model settings.
    """
    clipCheckpoint, networkConfig, modelName = loadInferenceSettings(
        configPath,
        profile,
    )
    resolvedModelName = modelNameOverride or modelName
    resolvedClip = clipCheckpointOverride or clipCheckpoint
    resolvedNetwork = networkConfigOverride or networkConfig
    return GenerationModelSettings(
        modelName=resolvedModelName,
        clipCheckpoint=resolvedClip,
        networkConfigPath=resolvedNetwork,
        profile=profile,
    )


def buildInferenceConfig(
    arguments: argparse.Namespace,
    jsonPath: Path,
    tagValue: Optional[str],
) -> GenerationInferenceConfig:
    """
    Build inference config from CLI arguments.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed CLI arguments.
    jsonPath : Path
        Output JSON path for the inference config.
    tagValue : Optional[str]
        Validated tag value.

    Returns
    -------
    GenerationInferenceConfig
        Inference configuration for generation.
    """
    return GenerationInferenceConfig(
        checkpoint=arguments.checkpoint,
        prompt=arguments.prompt,
        tag=tagValue,
        frames=arguments.frames,
        output=jsonPath,
        device=arguments.device,
        ddimSteps=arguments.ddimSteps,
    )


def buildOutputOptions(
    jsonPath: Path,
    daePath: Path,
    fps: Optional[int],
    zeroRootTranslation: bool,
    anchorRootTranslation: bool,
) -> GenerationOutputOptions:
    """
    Build output options for JSON and Collada export.

    Parameters
    ----------
    jsonPath : Path
        Destination JSON path.
    daePath : Path
        Destination Collada path.
    fps : Optional[int]
        Optional FPS override.
    zeroRootTranslation : bool
        Zero root translation flag.
    anchorRootTranslation : bool
        Anchor root translation flag.

    Returns
    -------
    GenerationOutputOptions
        Output options for export.
    """
    return GenerationOutputOptions(
        jsonPath=jsonPath,
        daePath=daePath,
        fps=fps,
        zeroRootTranslation=zeroRootTranslation,
        anchorRootTranslation=anchorRootTranslation,
    )


def validateTranslationFlags(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """
    Validate mutually exclusive translation flags.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed CLI arguments.
    parser : argparse.ArgumentParser
        Parser instance for error reporting.
    """
    if arguments.zeroRootTranslation and arguments.anchorRootTranslation:
        parser.error(
            "Use either --zero-root-translation or "
            "--anchor-root-translation, not both."
        )


def buildGenerationInputs(
    arguments: argparse.Namespace,
) -> tuple[
    GenerationInferenceConfig,
    GenerationModelSettings,
    GenerationOutputOptions,
]:
    """
    Build inference, model, and output configs from CLI arguments.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    tuple[
        GenerationInferenceConfig,
        GenerationModelSettings,
        GenerationOutputOptions,
    ]
        Configs used by the generation pipeline.
    """
    jsonPath, daePath = resolveOutputPaths(arguments.outputPath)
    tagValue = validateTagValue(arguments.tag)
    inferenceConfig = buildInferenceConfig(arguments, jsonPath, tagValue)
    modelSettings = resolveModelSettings(
        configPath=arguments.configPath,
        profile=arguments.profile,
        modelNameOverride=arguments.modelName,
        clipCheckpointOverride=arguments.clipCheckpoint,
        networkConfigOverride=arguments.networkConfigPath,
    )
    outputOptions = buildOutputOptions(
        jsonPath=jsonPath,
        daePath=daePath,
        fps=arguments.fps,
        zeroRootTranslation=arguments.zeroRootTranslation,
        anchorRootTranslation=arguments.anchorRootTranslation,
    )
    return inferenceConfig, modelSettings, outputOptions


def runGeneration(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """
    Execute the generation workflow from CLI arguments.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed CLI arguments.
    parser : argparse.ArgumentParser
        Parser instance for error handling.
    """
    validateTranslationFlags(arguments, parser)
    inferenceConfig, modelSettings, outputOptions = buildGenerationInputs(
        arguments
    )
    createAnimationFromCheckpoint(
        inferenceConfig=inferenceConfig,
        modelSettings=modelSettings,
        outputOptions=outputOptions,
        datasetConfigPath=arguments.datasetConfigPath,
    )
    parser.exit(
        0,
        f"Generated JSON: {outputOptions.jsonPath}\n"
        f"Generated DAE: {outputOptions.daePath}\n",
    )


def main() -> None:
    """
    CLI entry point for animation generation.

    Returns
    -------
    None
        Exits the process after running.
    """
    logging.basicConfig(level=logging.INFO)
    parser = buildArgumentParser()
    arguments = parser.parse_args()
    try:
        runGeneration(arguments, parser)
    except Exception as error:  # noqa: BLE001
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
