"""CLI entry point for CLIP training experiments."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import torch
from src.shared.config_loader import loadTrainingConfig
from src.features.clip.train_clip import (
    disableDropoutModules,
    buildOptimizer,
    evaluateValidation,
    loadCheckpoint,
    saveCheckpoint,
    trainOneEpoch,
    trainOneEpochWithAccumulation,
)
from src.shared.dataset_manager import (
    DatasetManager,
    MemoryManagerConfig,
    estimateModelBytes,
)
from src.shared.config_loader import loadNetworkConfig
from src.shared.model.components import buildEnabledComponents
from src.shared.model.clip.core import ClipModel
from src.shared.types import ClipTrainingConfig, ClipTrainingResult

LOGGER = logging.getLogger("clip.train_cli")
DEFAULT_CONFIG_PATH = Path("src/configs/train_clip.yaml")


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
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Configuration profile to use (e.g., 'spark' for DGX Spark).",
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
        config = loadTrainingConfig(configPath, profile=arguments.profile)
        result = _runTraining(
            config,
            profile=arguments.profile,
            datasetFolders=_parseFolderList(arguments.dataset_folders),
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
    config: ClipTrainingConfig,
    profile: Optional[str] = None,
    datasetFolders: Optional[list[str]] = None,
) -> ClipTrainingResult:
    """
    Execute the end-to-end training workflow with early stopping.

    Parameters
    ----------
    config : ClipTrainingConfig
        Parsed training configuration.
    profile : Optional[str]
        Optional network profile name to apply.

    Returns
    -------
    ClipTrainingResult
        Dataclass describing the training outcome.
    """
    device = _resolveDevice(config.training.device)
    
    # Load network architecture config
    networkConfig = loadNetworkConfig(
        configPath=config.networkConfigPath,
        profile=profile,
    )
    
    # Build model with motion encoder parameters
    LOGGER.info(
        "Building CLIP model with embed_dim=%d, motion_heads=%d, "
        "motion_layers=%d",
        networkConfig.embedDim,
        networkConfig.clip.motionNumHeads,
        networkConfig.clip.motionNumLayers,
    )
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
    LOGGER.info("CLIP motion components: %s", ", ".join(clipMotionKeys))
    model = ClipModel(
        modelName=config.training.modelName,
        embedDim=networkConfig.embedDim,
        motionNumHeads=networkConfig.clip.motionNumHeads,
        motionNumLayers=networkConfig.clip.motionNumLayers,
        motionComponents=clipMotionComponents,
        numBones=networkConfig.generation.numBones,
    ).to(device)
    
    # Log model size
    totalParams = sum(p.numel() for p in model.parameters())
    trainableParams = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    LOGGER.info(
        "Model params: %s total, %s trainable",
        f"{totalParams:,}",
        f"{trainableParams:,}",
    )
    if config.training.disableDropout:
        updatedDropouts = disableDropoutModules(model.motionBackbone)
        LOGGER.info(
            "Overfit dropout override enabled: %d dropout modules set to 0.0",
            updatedDropouts,
        )
    
    optimizer = buildOptimizer(
        model=model,
        learningRate=config.training.learningRate,
        weightDecay=config.training.weightDecay,
    )

    modelMemoryBytes = estimateModelBytes(model)
    memoryConfig = MemoryManagerConfig(
        MM_memoryLimitGB=config.training.MM_memoryLimitGB,
    )
    selectedFolders = (
        datasetFolders
        if datasetFolders is not None
        else config.paths.datasetFolders
    )
    overfitSelection = _parseOverfitSelection(config.training.overfitSamples)
    selectedFixedSampleRange: tuple[int, int] | None = None
    selectedMaxSamplesPerEpoch = config.training.maxSamplesPerEpoch
    if overfitSelection is not None:
        if overfitSelection.isRange:
            selectedFixedSampleRange = (
                int(overfitSelection.rangeStart),
                int(overfitSelection.rangeEnd),
            )
        else:
            selectedFixedSampleRange = (1, int(overfitSelection.sampleCount))
        selectedMaxSamplesPerEpoch = None
        LOGGER.info(
            "Overfit mode enabled: fixed range %s, validation %s.",
            overfitSelection.raw,
            "disabled" if config.training.validationSplit <= 0.0 else "enabled",
        )
    datasetManager = DatasetManager(
        datasetRoot=config.paths.datasetRoot,
        batchSize=config.training.batchSize,
        validationSplit=config.training.validationSplit,
        modelMemoryBytes=modelMemoryBytes,
        memoryConfig=memoryConfig,
        device=device,
        validationIndicesPath=config.training.validationIndicesPath,
        maxSamplesPerEpoch=selectedMaxSamplesPerEpoch,
        datasetFolders=selectedFolders,
        fixedSampleRange=selectedFixedSampleRange,
        includeTokenizedText=False,
        preloadEpochChunks=True,
    )
    datasetManager.dataset.validateCompatibility(
        modelName=config.training.modelName,
        maxPromptLength=config.training.maxPromptLength,
        requiredComponents=list(model.requiredMotionComponentKeys()),
        expectedPooledTextDim=model.textEncoder.config.hidden_size,
    )
    LOGGER.info("Dataset indexed: %d total samples", datasetManager.totalSize)
    useFixedTrainChunk = (
        config.training.fixedTrainChunk or overfitSelection is not None
    )
    if useFixedTrainChunk:
        LOGGER.info("Using fixed training chunk for CLIP overfit testing.")
    if overfitSelection is not None:
        _warnDegenerateOverfitSelection(datasetManager)

    bestValLoss: Optional[float] = None
    epochsWithoutImprovement = 0
    startEpoch = 0
    trainLoss = 0.0
    epochsRun = 0

    # Resume from checkpoint if specified
    if config.training.resumeCheckpoint is not None:
        LOGGER.info(
            "Resuming from checkpoint: %s",
            config.training.resumeCheckpoint,
        )
        try:
            resumedEpoch, resumedLoss = loadCheckpoint(
                checkpointPath=config.training.resumeCheckpoint,
                model=model,
                optimizer=optimizer,
            )
            _applyLearningRate(
                optimizer=optimizer,
                learningRate=config.training.learningRate,
            )
            startEpoch = resumedEpoch
            bestValLoss = resumedLoss
            LOGGER.info(
                "Resumed from epoch %s with loss %.4f",
                resumedEpoch,
                resumedLoss,
            )
            LOGGER.info(
                "Re-applied configured LR after resume: %.6f",
                config.training.learningRate,
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

    useAccumulation = config.training.gradientAccumulation > 1

    for epochIndex in range(startEpoch, config.training.epochs):
        epochsRun = epochIndex + 1
        
        # Get dataloaders (auto-rotating)
        dataloaderIndex = 0 if useFixedTrainChunk else epochIndex
        trainLoader, valLoader, chunkInfo = (
            datasetManager.getDataloadersForEpoch(dataloaderIndex)
        )

        # Training
        if useAccumulation:
            trainLoss, trainComponents = trainOneEpochWithAccumulation(
                trainLoader,
                model,
                optimizer,
                device,
                accumulationSteps=config.training.gradientAccumulation,
                epoch=epochsRun,
                totalEpochs=config.training.epochs,
                chunkInfo=chunkInfo,
                memoryLimitGB=config.training.MM_memoryLimitGB,
            )
        else:
            trainLoss, trainComponents = trainOneEpoch(
                trainLoader,
                model,
                optimizer,
                device,
                epoch=epochsRun,
                totalEpochs=config.training.epochs,
                chunkInfo=chunkInfo,
            )
        
        LOGGER.info(
            "Epoch %s train loss: %.4f [%s]",
            epochsRun,
            trainLoss,
            _formatLossComponents(trainComponents),
        )

        # Validation evaluation
        if valLoader is not None:
            valLoss, retrieval, valComponents = evaluateValidation(
                valLoader,
                model,
                device,
            )
            LOGGER.info(
                "Epoch %s val loss: %.4f [%s]",
                epochsRun,
                valLoss,
                _formatLossComponents(valComponents),
            )
            LOGGER.info(
                "Retrieval: t2m@1=%.4f t2m@5=%.4f m2t@1=%.4f m2t@5=%.4f",
                retrieval["t2m_top1"],
                retrieval["t2m_top5"],
                retrieval["m2t_top1"],
                retrieval["m2t_top5"],
            )

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

        LOGGER.info("LR: %.6f", optimizer.param_groups[0]["lr"])

    finalLoss = bestValLoss if bestValLoss is not None else trainLoss
    return ClipTrainingResult(
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


def _parseOverfitSelection(raw: Optional[str]) -> Optional[OverfitSelection]:
    """Parse an overfit selector from config."""
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


def _formatLossComponents(components: Mapping[str, float]) -> str:
    """Format the CLIP loss breakdown compactly for logs."""
    return (
        "text= %.4f, motion= %.4f, cosine= %.4f"
        % (
            components.get("loss_text_contrastive", float("nan")),
            components.get("loss_motion_contrastive", float("nan")),
            components.get("loss_motion_cosine", float("nan")),
        )
    )


def _warnDegenerateOverfitSelection(datasetManager: DatasetManager) -> None:
    """Warn when the CLIP overfit subset collapses the contrastive objective."""
    selectedIndices, chunkInfo = datasetManager.getEpochSampleIndices(0)
    if not selectedIndices:
        return
    dataset = datasetManager.dataset
    textIds: set[int] = set()
    for datasetIndex in selectedIndices:
        sample = dataset[datasetIndex]
        textId = sample.get("text_id")
        if isinstance(textId, int):
            textIds.add(textId)
        elif isinstance(textId, torch.Tensor) and textId.numel() == 1:
            textIds.add(int(textId.item()))
    if len(textIds) <= 1 and len(selectedIndices) > 1:
        LOGGER.warning(
            "CLIP overfit selection %s contains %d samples but only %d unique "
            "text_id. The contrastive mask becomes degenerate and the cosine "
            "term is the only meaningful learning signal.",
            chunkInfo,
            len(selectedIndices),
            len(textIds),
        )


def _applyLearningRate(
    optimizer: torch.optim.Optimizer,
    learningRate: float,
) -> None:
    """Apply the configured learning rate after optional checkpoint resume."""
    for paramGroup in optimizer.param_groups:
        paramGroup["lr"] = learningRate


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
