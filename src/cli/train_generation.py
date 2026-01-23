"""CLI entry point for motion generation training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Mapping, Optional

import torch

from src.shared.config_loader import loadGenerationConfig
from src.features.generation.train_generation import (
    buildOptimizer,
    evaluateValidation,
    loadCheckpoint,
    saveCheckpoint,
    trainOneEpoch,
)
from src.shared.dataset_manager import (
    DatasetManager,
    MemoryManagerConfig,
    estimateModelBytes,
)
from src.shared.config_loader import loadNetworkConfig
from src.shared.learning_rate import LearningRateScheduler, LearningRateConfig
from src.shared.model.generation.ddim import DDIM
from src.shared.model.generation.motion_generator import MotionGenerator
from src.shared.types import GenerationTrainingConfig, GenerationTrainingResult

LOGGER = logging.getLogger("generation.train_cli")
DEFAULT_CONFIG_PATH = Path("src/configs/train_generation.yaml")


def buildArgumentParser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Train the diffusion-based motion generation model.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the generation training YAML configuration file.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Configuration profile to use (e.g., 'spark' for DGX Spark). "
             "If not specified, uses 'training' section.",
    )
    return parser


def main() -> None:
    """CLI entry point orchestrating configuration loading and training."""
    logging.basicConfig(level=logging.INFO)
    parser = buildArgumentParser()
    arguments = parser.parse_args()

    try:
        configPath = _validateConfigPath(arguments.config)
        config = loadGenerationConfig(configPath, profile=arguments.profile)
        if arguments.profile:
            LOGGER.info("Using profile: %s", arguments.profile)
        result = _runTraining(config, profile=arguments.profile)
        
        parser.exit(
            0,
            f"Training finished with loss={result.finalLoss:.4f} "
            f"after {result.epochsRun} epochs on {result.device}.\n",
        )
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Training failed")
        parser.exit(1, f"{error}\n")


def _runTraining(
    config: GenerationTrainingConfig,
    profile: Optional[str] = None,
) -> GenerationTrainingResult:
    """
    Execute the end-to-end training workflow.

    Parameters
    ----------
    config : GenerationTrainingConfig
        Parsed training configuration.
    profile : Optional[str]
        Optional network profile name to apply.

    Returns
    -------
    GenerationTrainingResult
        Training outcome.
    """
    device = _resolveDevice(config.training.device)
    LOGGER.info("Using device: %s", device)
    
    # Log effective batch size
    effectiveBatchSize = (
        config.training.batchSize * config.training.gradientAccumulation
    )
    LOGGER.info(
        "Batch size: %d x %d accumulation = %d effective",
        config.training.batchSize,
        config.training.gradientAccumulation,
        effectiveBatchSize,
    )

    # Load network configuration
    networkConfig = loadNetworkConfig(
        configPath=config.networkConfigPath,
        profile=profile,
    )
    LOGGER.info(
        "Network config: embed_dim=%d, heads=%d, layers=%d, spatial=%d, "
        "channels=%d, repr=%s, mode=%s, steps=%d",
        networkConfig.embedDim,
        networkConfig.generation.numHeads,
        networkConfig.generation.numLayers,
        networkConfig.generation.numSpatialLayers,
        networkConfig.generation.motionChannels,
        networkConfig.generation.rotationRepr,
        networkConfig.generation.spatiotemporalMode,
        networkConfig.generation.diffusionSteps,
    )

    # Build model with network config
    model = MotionGenerator(
        embedDim=networkConfig.embedDim,
        numHeads=networkConfig.generation.numHeads,
        numLayers=networkConfig.generation.numLayers,
        numSpatialLayers=networkConfig.generation.numSpatialLayers,
        motionChannels=networkConfig.generation.motionChannels,
        rotationRepr=networkConfig.generation.rotationRepr,
        spatiotemporalMode=networkConfig.generation.spatiotemporalMode,
        numBones=networkConfig.generation.numBones,
        diffusionSteps=networkConfig.generation.diffusionSteps,
        modelName=config.training.modelName,
        clipCheckpoint=config.paths.clipCheckpoint,
        geodesicWeight=config.training.geodesicWeight,
        geodesicWeightSchedule=config.training.geodesicWeightSchedule,
    ).to(device)
    LOGGER.info(
        "Model initialized with CLIP from %s",
        config.paths.clipCheckpoint,
    )

    modelMemoryBytes = estimateModelBytes(model)
    memoryConfig = MemoryManagerConfig(
        MM_memoryLimitGB=config.training.MM_memoryLimitGB,
    )
    datasetManager = DatasetManager(
        datasetRoot=config.paths.datasetRoot,
        batchSize=config.training.batchSize,
        validationSplit=config.training.validationSplit,
        modelMemoryBytes=modelMemoryBytes,
        memoryConfig=memoryConfig,
        device=device,
        validationIndicesPath=config.paths.validationIndices,
        maxSamplesPerEpoch=config.training.maxSamplesPerEpoch,
    )
    datasetManager.dataset.validateCompatibility(
        modelName=config.training.modelName,
        maxPromptLength=config.training.maxPromptLength,
    )
    LOGGER.info("Dataset indexed: %d total samples", datasetManager.totalSize)

    # Build optimizer
    optimizer = buildOptimizer(
        model=model,
        learningRate=config.training.learningRate,
    )

    # Build DDIM scheduler and move to device
    ddim = DDIM(
        num_timesteps=networkConfig.generation.diffusionSteps
    ).to(device)

    # Learning rate scheduler with configurable warmup and decay
    lrConfig = LearningRateConfig(
        initialLR=config.training.learningRate,
        minLR=config.training.lrMin,
        warmupEpochs=config.training.lrWarmupEpochs,
        scheduleType=config.training.lrSchedule,
        decayEpochs=config.training.lrDecayEpochs,
    )
    scheduler = LearningRateScheduler(
        optimizer=optimizer,
        config=lrConfig,
        totalEpochs=config.training.epochs,
    )

    bestValLoss: Optional[float] = None
    epochsWithoutImprovement = 0
    startEpoch = 0
    trainLoss = 0.0
    epochsRun = 0
    if config.training.fixedTrainChunk:
        LOGGER.info("Using fixed training chunk for overfit testing.")

    # Resume from checkpoint if specified
    if config.training.resumeCheckpoint is not None:
        LOGGER.info(
            "Resuming from checkpoint: %s",
            config.training.resumeCheckpoint,
        )
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

    # Training loop
    for epochIndex in range(startEpoch, config.training.epochs):
        epochsRun = epochIndex + 1 - startEpoch
        currentLr = optimizer.param_groups[0]["lr"]
        
        # Get dataloaders for this epoch (auto-rotating)
        dataloaderIndex = 0 if config.training.fixedTrainChunk else epochIndex
        trainLoader, valLoader, chunkInfo = (
            datasetManager.getDataloadersForEpoch(dataloaderIndex)
        )

        trainLoss, trainComponents = trainOneEpoch(
            trainLoader,
            model,
            optimizer,
            ddim,
            device,
            gradientAccumulation=config.training.gradientAccumulation,
            epoch=epochIndex + 1,
            totalEpochs=config.training.epochs,
            chunkInfo=chunkInfo,
            memoryLimitGB=config.training.MM_memoryLimitGB,
        )
        LOGGER.info(
            "Epoch %s train loss: %.4f (lr=%.6f)",
            epochIndex + 1,
            trainLoss,
            currentLr,
        )
        _logLossComponents("train", epochIndex + 1, trainComponents)
        
        # Update learning rate
        scheduler.step()

        # Validation evaluation
        if valLoader is not None:
            valLoss, valComponents = evaluateValidation(
                valLoader,
                model,
                ddim,
                device,
            )
            LOGGER.info("Epoch %s val loss: %.4f", epochIndex + 1, valLoss)
            _logLossComponents("val", epochIndex + 1, valComponents)

            # Checkpointing - save best model
            if bestValLoss is None or valLoss < bestValLoss:
                bestValLoss = valLoss
                epochsWithoutImprovement = 0
                checkpointPath = saveCheckpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epochIndex + 1,
                    loss=valLoss,
                    checkpointDir=config.paths.checkpointDir,
                )
                LOGGER.info("Saved best model to %s", checkpointPath)
            else:
                epochsWithoutImprovement += 1
                LOGGER.info(
                    "No improvement for %s epoch(s)",
                    epochsWithoutImprovement,
                )

            # Early stopping check
            if (
                epochsWithoutImprovement
                >= config.training.earlyStoppingPatience
            ):
                LOGGER.info(
                    "Early stopping triggered after %s epochs "
                    "without improvement",
                    epochsWithoutImprovement,
                )
                break
        else:
            # No validation, save periodically
            if (epochIndex + 1) % 10 == 0:
                checkpointPath = saveCheckpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epochIndex + 1,
                    loss=trainLoss,
                    checkpointDir=config.paths.checkpointDir,
                )
                LOGGER.info("Saved checkpoint to %s", checkpointPath)

    finalLoss = bestValLoss if bestValLoss is not None else trainLoss
    return GenerationTrainingResult(
        epochsRun=epochsRun,
        finalLoss=finalLoss,
        device=device.type,
    )


def _logLossComponents(
    phase: str,
    epoch: int,
    components: Mapping[str, float],
) -> None:
    """
    Log loss component breakdown for the given phase.

    Parameters
    ----------
    phase : str
        Training phase label (e.g., "train", "val").
    epoch : int
        Current epoch number (1-based).
    components : Mapping[str, float]
        Loss component averages.
    """
    if not components:
        return
    LOGGER.info(
        "Epoch %s %s components: diff=%.4f geo=%.4f "
        "vel=%.4f acc=%.4f",
        epoch,
        phase,
        components.get("loss_diffusion", 0.0),
        components.get("loss_geodesic", 0.0),
        components.get("loss_velocity", 0.0),
        components.get("loss_acceleration", 0.0),
    )


def _resolveDevice(choice: str) -> torch.device:
    """
    Resolve the torch.device based on CLI arguments.

    Parameters
    ----------
    choice : str
        Requested backend.

    Returns
    -------
    torch.device
        Device satisfying the request.
    """
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
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
    Validate that the configuration path exists.

    Parameters
    ----------
    path : Path
        Path provided via CLI.

    Returns
    -------
    Path
        Resolved configuration path.
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    return resolved


if __name__ == "__main__":
    main()
