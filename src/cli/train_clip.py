"""CLI entry point for CLIP training experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from src.features.clip.config_loader import loadTrainingConfig
from src.features.clip.train_clip import (
    buildOptimizer,
    buildTrainValDataloaders,
    evaluateValidation,
    loadCheckpoint,
    saveCheckpoint,
    trainOneEpoch,
)
from src.shared.model.clip.core import ClipModel
from src.shared.types import ClipTrainingConfig, ClipTrainingResult

LOGGER = logging.getLogger("clip.train_cli")
DEFAULT_CONFIG_PATH = Path("src/configs/train_clip.yaml")


def buildArgumentParser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser mirroring other project tools.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance ready for execution.
    """
    parser = argparse.ArgumentParser(
        description="Train the text-motion CLIP model on converted datasets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the CLIP training YAML configuration file.",
    )
    return parser


def main() -> None:
    """
    CLI entry point orchestrating configuration loading and training.
    """
    logging.basicConfig(level=logging.INFO)
    parser = buildArgumentParser()
    arguments = parser.parse_args()
    try:
        configPath = _validateConfigPath(arguments.config)
        config = loadTrainingConfig(configPath)
        result = _runTraining(config)
        parser.exit(
            0,
            f"Training finished with loss={result.finalLoss:.4f} "
            f"after {result.epochsRun} epochs on {result.device}.\n",
        )
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Training failed")
        parser.exit(1, f"{error}\n")


def _runTraining(
    config: ClipTrainingConfig,
) -> ClipTrainingResult:
    """
    Execute the end-to-end training workflow with early stopping and checkpointing.

    Parameters
    ----------
    config : ClipTrainingConfig
        Parsed training configuration.

    Returns
    -------
    ClipTrainingResult
        Dataclass describing the training outcome.
    """
    device = _resolveDevice(config.training.device)
    trainLoader, valLoader = buildTrainValDataloaders(
        promptRoot=config.paths.promptRoot,
        animationRoot=config.paths.animationRoot,
        maxLength=config.training.maxPromptLength,
        batchSize=config.training.batchSize,
        modelName=config.training.modelName,
        validationSplit=config.training.validationSplit,
    )
    model = ClipModel(
        modelName=config.training.modelName,
        embedDim=config.training.embedDim,
    ).to(device)
    optimizer = buildOptimizer(
        model=model,
        learningRate=config.training.learningRate,
    )

    bestValLoss: Optional[float] = None
    epochsWithoutImprovement = 0
    startEpoch = 0
    trainLoss = 0.0

    # Resume from checkpoint if specified
    if config.training.resumeCheckpoint is not None:
        LOGGER.info("Resuming from checkpoint: %s", config.training.resumeCheckpoint)
        resumedEpoch, resumedLoss = loadCheckpoint(
            checkpointPath=config.training.resumeCheckpoint,
            model=model,
            optimizer=optimizer,
        )
        startEpoch = resumedEpoch
        bestValLoss = resumedLoss
        LOGGER.info(
            "Resumed from epoch %s with loss %.4f",
            resumedEpoch,
            resumedLoss,
        )

    for epochIndex in range(startEpoch, config.training.epochs):
        epochsRun = epochIndex + 1 - startEpoch
        LOGGER.info(
            "Starting epoch %s/%s",
            epochsRun,
            config.training.epochs,
        )
        trainLoss = trainOneEpoch(trainLoader, model, optimizer, device)
        LOGGER.info("Epoch %s train loss: %.4f", epochsRun, trainLoss)

        # Validation evaluation
        if valLoader is not None:
            valLoss = evaluateValidation(valLoader, model, device)
            LOGGER.info("Epoch %s val loss: %.4f", epochsRun, valLoss)

            # Checkpointing - save best model
            if bestValLoss is None or valLoss < bestValLoss:
                bestValLoss = valLoss
                epochsWithoutImprovement = 0
                if config.training.checkpointDir is not None:
                    checkpointPath = saveCheckpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epochsRun,
                        loss=valLoss,
                        checkpointDir=config.training.checkpointDir,
                    )
                    LOGGER.info("Saved best model to %s", checkpointPath)
            else:
                epochsWithoutImprovement += 1
                LOGGER.info(
                    "No improvement for %s epoch(s)",
                    epochsWithoutImprovement,
                )

            # Early stopping check
            if epochsWithoutImprovement >= config.training.earlyStoppingPatience:
                LOGGER.info(
                    "Early stopping triggered after %s epochs without improvement",
                    epochsWithoutImprovement,
                )
                break

    finalLoss = bestValLoss if bestValLoss is not None else trainLoss
    return ClipTrainingResult(
        epochsRun=epochsRun,
        finalLoss=finalLoss,
        device=device.type,
    )


def _resolveDevice(choice: str) -> torch.device:
    """
    Resolve the torch.device based on CLI arguments.

    Parameters
    ----------
    choice : str
        Requested backend ("auto", "cuda", "cpu", "mps").

    Returns
    -------
    torch.device
        Device satisfying the request.

    Raises
    ------
    RuntimeError
        Raised when the requested backend is unavailable.
    """
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend not available.")
        return torch.device("cuda")
    if choice == "mps":
        hasMps = hasattr(torch.backends, "mps")
        available = torch.backends.mps.is_available() if hasMps else False
        if not available:
            raise RuntimeError("Apple MPS backend not available.")
        return torch.device("mps")
    return torch.device(choice)


def _validateConfigPath(path: Path) -> Path:
    """
    Validate that the requested configuration path exists.

    Parameters
    ----------
    path : Path
        Path provided via CLI argument.

    Returns
    -------
    Path
        Resolved configuration path.

    Raises
    ------
    FileNotFoundError
        Raised when the configuration file is missing.
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    return resolved


if __name__ == "__main__":
    main()
