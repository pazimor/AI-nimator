"""CLI entry point for motion generation training."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import torch

from src.shared.config_loader import loadGenerationConfig
from src.features.generation.train_generation import (
    computeMotionStatistics,
    disableDropoutModules,
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
from src.shared.preprocessed_dataset import computeClipCheckpointFingerprint
from src.shared.config_loader import loadNetworkConfig
from src.shared.model.components import buildEnabledComponents
from src.shared.model.generation.ddim import DDIM
from src.shared.model.generation.motion_generator import MotionGenerator
from src.shared.types import GenerationTrainingConfig, GenerationTrainingResult

LOGGER = logging.getLogger("generation.train_cli")
DEFAULT_CONFIG_PATH = Path("src/configs/train_generation.yaml")


@dataclass(frozen=True)
class OverfitSelection:
    """Normalized overfit selector."""

    raw: str
    sampleCount: Optional[int] = None
    rangeStart: Optional[int] = None
    rangeEnd: Optional[int] = None

    @property
    def isRange(self) -> bool:
        """Return True when the selector targets an explicit range."""
        return self.rangeStart is not None and self.rangeEnd is not None

    @property
    def sampleTotal(self) -> int:
        """Return the number of samples represented by the selector."""
        if self.sampleCount is not None:
            return self.sampleCount
        if self.rangeStart is None or self.rangeEnd is None:
            raise RuntimeError("Invalid overfit selector state.")
        return self.rangeEnd - self.rangeStart + 1


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
        type=str,
        default=None,
        help=(
            "Enable overfit debug mode on a fixed subset selector "
            "(for example: 16 or 3:6, range is 1-based inclusive)."
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
    parser.add_argument(
        "--network-config",
        type=Path,
        default=None,
        help=(
            "Optional override for the shared network.yaml file used "
            "for architecture and motion feature toggles."
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
            networkConfigPathOverride=arguments.network_config,
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
    overfitSamples: Optional[str] = None,
    learningRateOverride: Optional[float] = None,
    networkConfigPathOverride: Optional[Path] = None,
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
    resolvedNetworkConfigPath = (
        networkConfigPathOverride.expanduser()
        if networkConfigPathOverride is not None
        else config.networkConfigPath
    )
    networkConfig = loadNetworkConfig(
        configPath=resolvedNetworkConfigPath,
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
    enabledComponents = buildEnabledComponents(networkConfig.generation.boneData)
    enabledComponentKeys = [component.key for component in enabledComponents]
    clipMotionComponents = (
        buildEnabledComponents(networkConfig.clip.boneData)
        if networkConfig.clip.boneData is not None
        else ()
    )
    if networkConfig.clip.boneData is not None and not clipMotionComponents:
        raise ValueError(
            "clip.bone-data is configured but enables no motion features. "
            "Enable at least one component or remove clip.bone-data to keep "
            "the legacy rotation-only CLIP input."
        )
    clipMotionKeys = (
        [component.key for component in clipMotionComponents]
        if clipMotionComponents
        else ["rotation6d"]
    )
    if "rotation6d" not in enabledComponentKeys:
        raise ValueError(
            "The current generation model still requires `rotation6d` in "
            "bone-data because `motion` is the only input consumed by the "
            "denoiser."
        )
    LOGGER.info(
        "Enabled motion components: %s",
        ", ".join(enabledComponentKeys),
    )
    LOGGER.info("CLIP motion components: %s", ", ".join(clipMotionKeys))
    LOGGER.info(
        "CLIP motion encoder: heads=%d, layers=%d",
        networkConfig.clip.motionNumHeads,
        networkConfig.clip.motionNumLayers,
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
        clipMotionNumHeads=networkConfig.clip.motionNumHeads,
        clipMotionNumLayers=networkConfig.clip.motionNumLayers,
        xyzWeight=config.training.xyzWeight,
        xyzWeightSchedule=config.training.xyzWeightSchedule,
        velXyzWeight=config.training.velXyzWeight,
        diffusionWeight=config.training.diffusionWeight,
        accelerationWeight=config.training.accelerationWeight,
        clipGuidanceWeight=config.training.clipGuidanceWeight,
        numSpatialLayers=networkConfig.generation.numSpatialLayers,
        numSpatioTemporalLayers=networkConfig.generation.numSpatioTemporalLayers,
        maxPromptLength=config.training.maxPromptLength,
        clipMotionComponents=clipMotionComponents,
        generationMotionComponents=enabledComponents,
    ).to(device)
    LOGGER.info(
        "Model initialized with CLIP from %s",
        config.paths.clipCheckpoint,
    )
    if config.training.disableDropout:
        updatedDropouts = disableDropoutModules(model.denoiser)
        LOGGER.info(
            "Overfit dropout override enabled: %d dropout modules set to 0.0",
            updatedDropouts,
        )
    clipFingerprint = computeClipCheckpointFingerprint(config.paths.clipCheckpoint)
    LOGGER.info(
        "Generation text cache fingerprint: %s",
        clipFingerprint,
    )
    if enabledComponentKeys == ["rotation6d"]:
        LOGGER.info(
            "Rotation-only training input enabled (6D joints only)."
        )
    else:
        LOGGER.info(
            "Auxiliary generation supervision enabled for: %s",
            ", ".join(
                key for key in enabledComponentKeys if key != "rotation6d"
            ),
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
    if config.training.clipGuidanceWeight > 0.0:
        LOGGER.info(
            "CLIP guidance enabled with weight %.4f",
            config.training.clipGuidanceWeight,
        )
    if config.training.deterministicCorruption:
        LOGGER.info(
            "Deterministic diffusion corruption enabled for stable overfit runs."
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

    requestedOverfitSpec = (
        overfitSamples
        if overfitSamples is not None
        else config.training.overfitSamples
    )
    overfitSelection = _parseOverfitSelection(requestedOverfitSpec)

    useFixedTrainChunk = config.training.fixedTrainChunk
    selectedValidationSplit = config.training.validationSplit
    selectedMaxSamplesPerEpoch = config.training.maxSamplesPerEpoch
    selectedValidationIndicesPath = config.paths.validationIndices
    selectedResumeCheckpoint = config.training.resumeCheckpoint
    selectedFixedSampleRange: Optional[tuple[int, int]] = None
    if overfitSelection is not None:
        useFixedTrainChunk = True
        selectedValidationSplit = 0.0
        selectedValidationIndicesPath = None
        if overfitSelection.isRange:
            selectedMaxSamplesPerEpoch = None
            if (
                overfitSelection.rangeStart is None
                or overfitSelection.rangeEnd is None
            ):
                raise RuntimeError("Invalid overfit range selector.")
            selectedFixedSampleRange = (
                overfitSelection.rangeStart,
                overfitSelection.rangeEnd,
            )
            LOGGER.info(
                "Overfit mode enabled: fixed range %s (%d samples), "
                "validation disabled.",
                overfitSelection.raw,
                overfitSelection.sampleTotal,
            )
        else:
            selectedMaxSamplesPerEpoch = overfitSelection.sampleCount
            LOGGER.info(
                "Overfit mode enabled: first %d samples, fixed chunk, "
                "validation disabled.",
                overfitSelection.sampleTotal,
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
        fixedSampleRange=selectedFixedSampleRange,
        generationCacheCheckpoint=config.paths.clipCheckpoint,
        includeTokenizedText=True,
        preloadEpochChunks=True,
    )
    datasetManager.dataset.validateCompatibility(
        modelName=config.training.modelName,
        maxPromptLength=config.training.maxPromptLength,
        requiredComponents=sorted(
            set(enabledComponentKeys) | set(clipMotionKeys)
        ),
        expectedPooledTextDim=model.clip.textEncoder.config.hidden_size,
        expectedGenerationEmbedDim=model.embedDim,
        requireGenerationTextEmbedding=True,
    )
    LOGGER.info("Dataset indexed: %d total samples", datasetManager.totalSize)
    if overfitSelection is not None:
        _logSelectedOverfitPrompts(
            datasetManager=datasetManager,
            selection=overfitSelection,
            tokenizer=model.clip.tokenizer,
        )

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

    # Compute Z-normalization statistics from training data and store on model.
    # These buffers are saved/restored with the checkpoint.
    if not _hasMotionStatistics(model):
        LOGGER.info("Computing motion Z-normalization statistics ...")
        statsLoader, _, _ = datasetManager.getDataloadersForEpoch(0)
        motionMean, motionStd = computeMotionStatistics(
            statsLoader,
            device=torch.device("cpu"),
            maxBatches=500,
        )
        model.setMotionStatistics(motionMean.to(device), motionStd.to(device))
        LOGGER.info(
            "Motion statistics set: mean range [%.4f, %.4f], "
            "std range [%.4f, %.4f]",
            float(motionMean.min()),
            float(motionMean.max()),
            float(motionStd.min()),
            float(motionStd.max()),
        )

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
            deterministicCorruption=config.training.deterministicCorruption,
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
                deterministicCorruption=config.training.deterministicCorruption,
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
            trainLoss=trainLoss,
            valLoss=valLoss,
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


def _hasMotionStatistics(model: MotionGenerator) -> bool:
    """Return True when the model already has non-default Z-norm stats."""
    return not (
        torch.all(model.motion_mean == 0.0)
        and torch.all(model.motion_std == 1.0)
    )


def _parseOverfitSelection(raw: Optional[str]) -> Optional[OverfitSelection]:
    """Parse an overfit selector from CLI/config."""
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or value.lower() == "null":
        return None
    if ":" not in value:
        count = _parsePositiveInt(value, label="overfit-samples")
        return OverfitSelection(raw=value, sampleCount=count)

    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(
            "overfit-samples range must use the format start:end "
            "(example: 3:6)."
        )
    start = _parsePositiveInt(
        parts[0].strip(),
        label="overfit-samples range start",
    )
    end = _parsePositiveInt(
        parts[1].strip(),
        label="overfit-samples range end",
    )
    if start > end:
        raise ValueError(
            "overfit-samples range start must be <= end "
            f"(got {value!r})."
        )
    return OverfitSelection(
        raw=value,
        rangeStart=start,
        rangeEnd=end,
    )


def _parsePositiveInt(value: str, label: str) -> int:
    """Parse a strictly positive integer from text."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer.") from error
    if parsed <= 0:
        raise ValueError(f"{label} must be strictly positive.")
    return parsed


def _logSelectedOverfitPrompts(
    datasetManager: DatasetManager,
    selection: OverfitSelection,
    tokenizer: object,
) -> None:
    """Log the prompts selected for overfit mode."""
    selectedIndices, chunkInfo = datasetManager.getEpochSampleIndices(0)
    if not selectedIndices:
        LOGGER.warning("Overfit selection is empty.")
        return
    LOGGER.info(
        "Selected overfit prompts (%s): %s",
        selection.raw,
        chunkInfo,
    )
    dataset = datasetManager.dataset
    for order, datasetIndex in enumerate(selectedIndices, start=1):
        sample = dataset[datasetIndex]
        inputIds = sample.get("input_ids")
        if isinstance(inputIds, torch.Tensor):
            tokenIds = inputIds.detach().cpu().tolist()
        else:
            tokenIds = list(inputIds) if inputIds is not None else []
        promptText = tokenizer.decode(
            tokenIds,
            skip_special_tokens=True,
        ).strip()
        sourceFile = dataset.indexEntries[datasetIndex].sourceFile or "unknown"
        LOGGER.info(
            "Overfit prompt %d/%d [dataset=%d source=%s]: %s",
            order,
            len(selectedIndices),
            datasetIndex,
            sourceFile,
            promptText or "<empty prompt>",
        )


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
    formatted = [
        "diff= %.4f" % components.get("loss_diffusion", float("nan")),
        "xyz= %.4f" % components.get("loss_xyz", float("nan")),
        "vel_xyz= %.4f" % components.get("loss_vel_xyz", float("nan")),
        "acc= %.4f" % components.get("loss_acceleration", float("nan")),
    ]
    if "loss_components" in components:
        formatted.append(
            "aux= %.4f" % components.get("loss_components", float("nan"))
        )
    if "loss_root_translation" in components:
        formatted.append(
            "rtrans= %.4f"
            % components.get("loss_root_translation", float("nan"))
        )
    if "loss_root_velocity" in components:
        formatted.append(
            "rvel= %.4f"
            % components.get("loss_root_velocity", float("nan"))
        )
    if "loss_clip_guidance" in components:
        formatted.append(
            "clip= %.4f"
            % components.get("loss_clip_guidance", float("nan"))
        )
    return ", ".join(formatted)


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
    trainLoss: float,
    valLoss: float | None,
    trainComponents: Mapping[str, float],
    valComponents: Mapping[str, float],
) -> None:
    """
    Log one compact epoch line following the requested format.
    """
    LOGGER.info(
        "epoch: %s/%s, lr = %.6f, train_loss= %.4f, val_loss= %s, train: [ %s ], val: [ %s ]",
        epoch,
        totalEpochs,
        learningRate,
        trainLoss,
        "disabled" if valLoss is None else f"{valLoss:.4f}",
        _formatComponents(trainComponents),
        _formatComponents(valComponents),
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
