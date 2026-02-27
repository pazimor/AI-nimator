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
    parser.add_argument(
        "--dataset-folders",
        type=str,
        default=None,
        help=(
            "Comma-separated top-level dataset folders to include "
            "(example: KIT,CMU,ACCAD)."
        ),
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=None,
        help=(
            "Enable overfit debug mode on a fixed subset size "
            "(for example: 16)."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Override the optimizer learning rate for this run. "
            "This value is reapplied after resume."
        ),
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
        selectedProfile = arguments.profile or "training"
        LOGGER.info("Using profile: %s", selectedProfile)
        result = _runTraining(
            config,
            profile=arguments.profile,
            datasetFolders=_parseFolderList(arguments.dataset_folders),
            overfitSamples=arguments.overfit_samples,
            learningRateOverride=arguments.learning_rate,
        )
        
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
    datasetFolders: Optional[list[str]] = None,
    overfitSamples: Optional[int] = None,
    learningRateOverride: Optional[float] = None,
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
        "Network config: clip_embed_dim=%d, gen_embed_dim=%d, "
        "num_heads=%d, num_layers=%d, diffusion_steps=%d",
        networkConfig.embedDim,
        networkConfig.generation.embedDim,
        networkConfig.generation.numHeads,
        networkConfig.generation.numLayers,
        networkConfig.generation.diffusionSteps,
    )

    # Build model with network config
    model = MotionGenerator(
        embedDim=networkConfig.embedDim,
        generationEmbedDim=networkConfig.generation.embedDim,
        numHeads=networkConfig.generation.numHeads,
        numLayers=networkConfig.generation.numLayers,
        numBones=networkConfig.generation.numBones,
        diffusionSteps=networkConfig.generation.diffusionSteps,
        modelName=config.training.modelName,
        clipCheckpoint=config.paths.clipCheckpoint,
        xyzWeight=config.training.xyzWeight,
        xyzWeightSchedule=config.training.xyzWeightSchedule,
        velXyzWeight=config.training.velXyzWeight,
        diffusionWeight=config.training.diffusionWeight,
        accelerationWeight=config.training.accelerationWeight,
        numSpatialLayers=networkConfig.generation.numSpatialLayers,
        numHierarchyLayers=networkConfig.generation.numHierarchyLayers,
        numSpatioTemporalLayers=networkConfig.generation.numSpatioTemporalLayers,
        maxPromptLength=config.training.maxPromptLength,
        predictionTarget=config.training.predictionTarget,
    ).to(device)
    LOGGER.info(
        "Model initialized with CLIP from %s",
        config.paths.clipCheckpoint,
    )
    LOGGER.info(
        "Rotation-only training input enabled (6D joints, no root "
        "translation fed to the network)."
    )
    xyzSchedule = config.training.xyzWeightSchedule.lower()
    effectiveXyzWeight = config.training.xyzWeight
    if xyzSchedule == "timestep":
        effectiveXyzWeight *= 0.5
    LOGGER.info(
        "XYZ optimization: base_weight=%.4f, schedule=%s, "
        "approx_effective_weight=%.4f",
        config.training.xyzWeight,
        config.training.xyzWeightSchedule,
        effectiveXyzWeight,
    )
    LOGGER.info(
        "Loss weights: diffusion=%.4f, xyz=%.4f, vel_xyz=%.4f, acc=%.4f",
        config.training.diffusionWeight,
        config.training.xyzWeight,
        config.training.velXyzWeight,
        config.training.accelerationWeight,
    )
    LOGGER.info(
        "Diffusion parameterization: prediction_target=%s",
        config.training.predictionTarget,
    )
    if config.training.accelerationWeight > 0.1:
        LOGGER.warning(
            "Acceleration weight %.4f is high for rotation-only 6D "
            "training and can dominate optimization. Start near 0.02 "
            "unless you have a measured reason to increase it.",
            config.training.accelerationWeight,
        )

    modelMemoryBytes = estimateModelBytes(model)
    memoryConfig = MemoryManagerConfig(
        MM_memoryLimitGB=config.training.MM_memoryLimitGB,
        clearMpsCache=config.training.clearMpsCache,
    )
    if not config.training.clearMpsCache:
        LOGGER.info("MPS cache clearing disabled by config.")

    requestedOverfitSamples = (
        overfitSamples
        if overfitSamples is not None
        else config.training.overfitSamples
    )
    if (
        requestedOverfitSamples is not None
        and requestedOverfitSamples <= 0
    ):
        raise ValueError(
            "overfit-samples must be a positive integer when provided."
        )

    useFixedTrainChunk = config.training.fixedTrainChunk
    selectedValidationSplit = config.training.validationSplit
    selectedMaxSamplesPerEpoch = config.training.maxSamplesPerEpoch
    selectedValidationIndicesPath = config.paths.validationIndices
    selectedResumeCheckpoint = config.training.resumeCheckpoint
    selectedNumWorkers = config.training.numWorkers
    if requestedOverfitSamples is not None:
        useFixedTrainChunk = True
        selectedValidationSplit = 0.0
        selectedMaxSamplesPerEpoch = requestedOverfitSamples
        selectedValidationIndicesPath = None
        if selectedNumWorkers is None:
            selectedNumWorkers = 0
        LOGGER.info(
            "Overfit mode enabled: %d samples, fixed chunk, "
            "validation disabled.",
            requestedOverfitSamples,
        )
        if selectedNumWorkers == 0:
            LOGGER.info(
                "Overfit mode: using num_workers=0 to avoid worker startup latency."
            )
    if selectedValidationSplit <= 0.0:
        LOGGER.info(
            "Validation disabled: periodic checkpoints are latest training "
            "state only; they are not selected on validation."
        )

    selectedFolders = (
        datasetFolders
        if datasetFolders is not None
        else config.paths.datasetFolders
    )
    datasetManager = DatasetManager(
        datasetRoot=config.paths.datasetRoot,
        batchSize=config.training.batchSize,
        validationSplit=selectedValidationSplit,
        modelMemoryBytes=modelMemoryBytes,
        memoryConfig=memoryConfig,
        device=device,
        validationIndicesPath=selectedValidationIndicesPath,
        maxSamplesPerEpoch=selectedMaxSamplesPerEpoch,
        datasetFolders=selectedFolders,
        numWorkers=selectedNumWorkers,
    )
    datasetManager.dataset.validateCompatibility(
        modelName=config.training.modelName,
        maxPromptLength=config.training.maxPromptLength,
    )
    LOGGER.info("Dataset indexed: %d total samples", datasetManager.totalSize)

    selectedLearningRate = (
        float(learningRateOverride)
        if learningRateOverride is not None
        else float(config.training.learningRate)
    )
    if selectedLearningRate <= 0.0:
        raise ValueError("learning-rate must be strictly positive.")

    # Build optimizer
    optimizer = buildOptimizer(
        model=model,
        learningRate=selectedLearningRate,
    )
    learningRateSource = (
        "cli override"
        if learningRateOverride is not None
        else "config"
    )
    LOGGER.info(
        "Learning rate (%s): %.6f",
        learningRateSource,
        selectedLearningRate,
    )

    # Build DDIM scheduler and move to device
    ddim = DDIM(
        num_timesteps=networkConfig.generation.diffusionSteps
    ).to(device)

    bestValLoss: Optional[float] = None
    epochsWithoutImprovement = 0
    startEpoch = 0
    trainLoss = 0.0
    epochsRun = 0
    if useFixedTrainChunk:
        LOGGER.info("Using fixed training chunk for overfit testing.")

    # Resume from checkpoint if specified
    if selectedResumeCheckpoint is not None:
        LOGGER.info(
            "Resuming from checkpoint: %s",
            selectedResumeCheckpoint,
        )
        try:
            resumedEpoch, resumedLoss = loadCheckpoint(
                checkpointPath=selectedResumeCheckpoint,
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
            resumedLearningRate = float(optimizer.param_groups[0]["lr"])
            if abs(resumedLearningRate - selectedLearningRate) > 1e-12:
                LOGGER.warning(
                    "Checkpoint optimizer LR %.6f differs from requested "
                    "LR %.6f. Reapplying requested LR.",
                    resumedLearningRate,
                    selectedLearningRate,
                )
            _setOptimizerLearningRate(optimizer, selectedLearningRate)
            LOGGER.info(
                "Learning rate after resume: %.6f",
                float(optimizer.param_groups[0]["lr"]),
            )
        except RuntimeError as error:
            firstLine = str(error).splitlines()[0]
            LOGGER.warning(
                "Skipping resume checkpoint due to incompatible checkpoint "
                "state: %s",
                firstLine,
            )
            startEpoch = 0
            bestValLoss = None

    # Training loop
    for epochIndex in range(startEpoch, config.training.epochs):
        epochsRun = epochIndex + 1 - startEpoch
        currentLr = optimizer.param_groups[0]["lr"]
        
        # Get dataloaders for this epoch (auto-rotating)
        dataloaderIndex = 0 if useFixedTrainChunk else epochIndex
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
            clearMpsCache=config.training.clearMpsCache,
        )
        valLoss: Optional[float] = None
        valComponents = _nanLossComponents()
        
        # Validation evaluation
        if valLoader is not None:
            valLoss, valComponents = evaluateValidation(
                valLoader,
                model,
                ddim,
                device,
            )
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
                LOGGER.info(
                    "Saved periodic checkpoint to %s "
                    "(validation disabled; not a best model)",
                    checkpointPath,
                )

        _logEpochSummary(
            epoch=epochIndex + 1,
            totalEpochs=config.training.epochs,
            learningRate=currentLr,
            trainComponents=trainComponents,
            valComponents=valComponents,
        )

        if (
            valLoader is not None
            and epochsWithoutImprovement
            >= config.training.earlyStoppingPatience
        ):
            LOGGER.info(
                "Early stopping triggered after %s epochs without "
                "improvement",
                epochsWithoutImprovement,
            )
            break

    finalLoss = bestValLoss if bestValLoss is not None else trainLoss
    return GenerationTrainingResult(
        epochsRun=epochsRun,
        finalLoss=finalLoss,
        device=device.type,
    )


def _parseFolderList(rawValue: str | None) -> list[str] | None:
    """Parse comma-separated folder names from CLI."""
    if rawValue is None:
        return None
    folders = [item.strip() for item in rawValue.split(",")]
    normalized = [item for item in folders if item]
    if not normalized:
        return None
    return normalized


def _setOptimizerLearningRate(
    optimizer: torch.optim.Optimizer,
    learningRate: float,
) -> None:
    """Apply the requested learning rate to every optimizer param group."""
    for paramGroup in optimizer.param_groups:
        paramGroup["lr"] = learningRate


def _nanLossComponents() -> dict[str, float]:
    """Return NaN placeholders for epoch summary formatting."""
    return {
        "loss_diffusion": float("nan"),
        "loss_xyz": float("nan"),
        "loss_vel_xyz": float("nan"),
        "loss_acceleration": float("nan"),
        "contrib_diffusion": float("nan"),
        "contrib_xyz": float("nan"),
        "contrib_vel_xyz": float("nan"),
        "contrib_acceleration": float("nan"),
    }


def _formatComponents(components: Mapping[str, float]) -> str:
    """Format train/val component block for epoch summary log."""
    if _componentsDisabled(
        components,
        (
            "loss_diffusion",
            "loss_xyz",
            "loss_vel_xyz",
            "loss_acceleration",
        ),
    ):
        return "disabled"
    return (
        "diff= %.4f, xyz= %.4f, vel_xyz= %.4f, acc= %.4f"
        % (
            components.get("loss_diffusion", float("nan")),
            components.get("loss_xyz", float("nan")),
            components.get("loss_vel_xyz", float("nan")),
            components.get("loss_acceleration", float("nan")),
        )
    )


def _safePercent(numerator: float, denominator: float) -> float:
    """Return a stable percentage value."""
    if denominator <= 0.0 or numerator != numerator:
        return float("nan")
    return 100.0 * numerator / denominator


def _formatImpact(components: Mapping[str, float]) -> str:
    """Format weighted contribution ratios for each loss term."""
    if _componentsDisabled(
        components,
        (
            "contrib_diffusion",
            "contrib_xyz",
            "contrib_vel_xyz",
            "contrib_acceleration",
        ),
    ):
        return "disabled"
    diff = components.get("contrib_diffusion", float("nan"))
    xyz = components.get("contrib_xyz", float("nan"))
    velXyz = components.get("contrib_vel_xyz", float("nan"))
    acc = components.get("contrib_acceleration", float("nan"))
    total = diff + xyz + velXyz + acc
    return (
        "diff= %.1f%%, xyz= %.1f%%, vel_xyz= %.1f%%, acc= %.1f%%"
        % (
            _safePercent(diff, total),
            _safePercent(xyz, total),
            _safePercent(velXyz, total),
            _safePercent(acc, total),
        )
    )


def _componentsDisabled(
    components: Mapping[str, float],
    keys: tuple[str, ...],
) -> bool:
    """Detect placeholder metric blocks used when validation is disabled."""
    return all(
        components.get(key, float("nan"))
        != components.get(key, float("nan"))
        for key in keys
    )


def _logEpochSummary(
    epoch: int,
    totalEpochs: int,
    learningRate: float,
    trainComponents: Mapping[str, float],
    valComponents: Mapping[str, float],
) -> None:
    """
    Log one compact epoch line following the requested format.
    """
    LOGGER.info(
        "epoch: %s/%s, lr = %.6f, train: [ %s ], val: [ %s ]",
        epoch,
        totalEpochs,
        learningRate,
        _formatComponents(trainComponents),
        _formatComponents(valComponents),
    )
    LOGGER.info(
        "epoch: %s impact train: [ %s ], val: [ %s ]",
        epoch,
        _formatImpact(trainComponents),
        _formatImpact(valComponents),
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
