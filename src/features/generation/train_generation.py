"""Training loop for the motion generation model."""

from __future__ import annotations

import gc
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset

# Limit CPU threads to reduce memory usage
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(2)

# macOS: Disable sudden termination to prevent watchdog kills
if sys.platform == "darwin":
    try:
        import Foundation  # type: ignore
        Foundation.NSProcessInfo.processInfo().disableSuddenTermination()
        Foundation.NSProcessInfo.processInfo().disableAutomaticTermination_(
            "Training in progress"
        )
    except ImportError:
        pass  # pyobjc not installed, skip

from src.shared.model.clip.data import MotionTextClipDataset, motionTextCollate
from src.shared.model.generation.ddim import DDIM
from src.shared.model.generation.losses import diffusionLoss
from src.shared.model.generation.motion_generator import MotionGenerator
from src.shared.progress import TrainingProgressBar
from src.shared.types import GenerationTrainingConfig

from transformers import XLMRobertaTokenizerFast

BatchDict = Mapping[str, object]
LOGGER = logging.getLogger("generation.train")


class RotatingDatasetManager:
    """
    Manages rotating through dataset chunks across epochs.
    
    Allows training on a small chunk per epoch while eventually
    seeing the entire dataset over multiple epochs.
    """
    
    def __init__(
        self,
        datasetRoot: Path,
        maxLength: int,
        batchSize: int,
        modelName: str,
        validationSplit: float,
        chunkSize: int,
    ) -> None:
        self.datasetRoot = datasetRoot
        self.maxLength = maxLength
        self.batchSize = batchSize
        self.modelName = modelName
        self.validationSplit = validationSplit
        self.chunkSize = chunkSize
        self.currentOffset = 0
        self._dataset: Optional[MotionTextClipDataset] = None
        self._totalSize: Optional[int] = None
        
    def _ensureDataset(self) -> MotionTextClipDataset:
        """Lazy-load dataset metadata (without loading motions)."""
        if self._dataset is None:
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.modelName)
            self._dataset = MotionTextClipDataset(
                rootPrompts=self.datasetRoot,
                rootAnimations=self.datasetRoot,
                tokenizer=tokenizer,
                maxLength=self.maxLength,
                cacheMotion=False,
            )
            self._totalSize = len(self._dataset)
            LOGGER.info("Dataset indexed: %d total samples", self._totalSize)
        return self._dataset
    
    @property
    def totalSize(self) -> int:
        """Get total dataset size."""
        self._ensureDataset()
        return self._totalSize or 0
    
    def getDataloadersForEpoch(
        self, epochIndex: int
    ) -> Tuple[DataLoader, Optional[DataLoader], int, int]:
        """
        Get dataloaders for a specific epoch with rotated chunk.
        
        Returns
        -------
        Tuple containing:
            - Train dataloader
            - Validation dataloader (or None)
            - Start index of chunk
            - End index of chunk
        """
        dataset = self._ensureDataset()
        totalSize = self._totalSize or len(dataset)
        
        # Calculate chunk boundaries
        startIdx = (epochIndex * self.chunkSize) % totalSize
        endIdx = min(startIdx + self.chunkSize, totalSize)
        
        # Handle wrap-around
        if startIdx + self.chunkSize > totalSize:
            # Wrap around to beginning
            indices = list(range(startIdx, totalSize)) + list(range(0, (startIdx + self.chunkSize) % totalSize))
        else:
            indices = list(range(startIdx, endIdx))
        
        chunkSubset = Subset(dataset, indices)
        
        if self.validationSplit <= 0.0 or len(indices) < 2:
            trainLoader = DataLoader(
                chunkSubset,
                batch_size=self.batchSize,
                shuffle=True,
                collate_fn=motionTextCollate,
                num_workers=0,
                pin_memory=False,
            )
            return trainLoader, None, startIdx, endIdx
        
        valSize = max(1, int(len(indices) * self.validationSplit))
        trainSize = len(indices) - valSize
        trainSubset, valSubset = random_split(chunkSubset, [trainSize, valSize])
        
        trainLoader = DataLoader(
            trainSubset,
            batch_size=self.batchSize,
            shuffle=True,
            collate_fn=motionTextCollate,
            num_workers=0,
            pin_memory=False,
        )
        valLoader = DataLoader(
            valSubset,
            batch_size=self.batchSize,
            shuffle=False,
            collate_fn=motionTextCollate,
            num_workers=0,
            pin_memory=False,
        )
        
        return trainLoader, valLoader, startIdx, endIdx


def buildTrainValDataloaders(
    datasetRoot: Path,
    maxLength: int,
    batchSize: int,
    modelName: str,
    validationSplit: float,
    maxSamples: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train and validation dataloaders.

    Parameters
    ----------
    datasetRoot : Path
        Root directory containing prompt and animation files.
    maxLength : int
        Maximum token length for prompts.
    batchSize : int
        Batch size for training.
    modelName : str
        Hugging Face model name for tokenizer.
    validationSplit : float
        Fraction of data for validation.
    maxSamples : Optional[int]
        Maximum number of samples to use (None = use all).

    Returns
    -------
    Tuple[DataLoader, Optional[DataLoader]]
        Training and optional validation dataloaders.
    """
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(modelName)
    dataset = MotionTextClipDataset(
        rootPrompts=datasetRoot,
        rootAnimations=datasetRoot,
        tokenizer=tokenizer,
        maxLength=maxLength,
        cacheMotion=False,  # Disable caching to reduce memory usage
    )

    if len(dataset) == 0:
        raise ValueError(f"No samples found under {datasetRoot}")

    # Limit dataset size if maxSamples is specified
    effectiveDataset = dataset
    if maxSamples is not None and maxSamples < len(dataset):
        indices = list(range(maxSamples))
        effectiveDataset = Subset(dataset, indices)
        LOGGER.info("Dataset limited to %d samples (was %d)", maxSamples, len(dataset))

    if validationSplit <= 0.0 or len(effectiveDataset) < 2:
        return _makeDataloader(effectiveDataset, batchSize, shuffle=True), None

    valSize = max(1, int(len(effectiveDataset) * validationSplit))
    trainSize = len(effectiveDataset) - valSize
    trainSubset, valSubset = random_split(effectiveDataset, [trainSize, valSize])

    trainLoader = DataLoader(
        trainSubset,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=motionTextCollate,
        num_workers=0,
        pin_memory=False,
    )
    valLoader = DataLoader(
        valSubset,
        batch_size=batchSize,
        shuffle=False,
        collate_fn=motionTextCollate,
        num_workers=0,
        pin_memory=False,
    )

    return trainLoader, valLoader


def _makeDataloader(
    dataset: MotionTextClipDataset,
    batchSize: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a dataloader with custom collation."""
    return DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        collate_fn=motionTextCollate,
        num_workers=0,
        pin_memory=False,
    )


def buildOptimizer(
    model: MotionGenerator,
    learningRate: float,
) -> torch.optim.Optimizer:
    """
    Create optimizer for trainable parameters only.

    Parameters
    ----------
    model : MotionGenerator
        The generation model.
    learningRate : float
        Learning rate.

    Returns
    -------
    torch.optim.Optimizer
        AdamW optimizer for denoiser parameters.
    """
    # Only train denoiser parameters (CLIP is frozen)
    trainableParams = list(model.denoiser.parameters())
    return torch.optim.AdamW(trainableParams, lr=learningRate)


def trainOneEpoch(
    dataloader: Iterable[BatchDict],
    model: MotionGenerator,
    optimizer: torch.optim.Optimizer,
    ddim: DDIM,
    device: torch.device,
    gradientAccumulation: int = 1,
    epoch: int = 1,
    totalEpochs: int = 1,
    chunkInfo: Optional[str] = None,
    memoryLimitGB: float = 0.0,
) -> float:
    """
    Run a single training epoch.

    Parameters
    ----------
    dataloader : Iterable[BatchDict]
        Training dataloader.
    model : MotionGenerator
        Generation model.
    optimizer : torch.optim.Optimizer
        Optimizer.
    ddim : DDIM
        Diffusion scheduler.
    device : torch.device
        Training device.
    gradientAccumulation : int, optional
        Number of batches to accumulate gradients over, by default 1.
    epoch : int
        Current epoch number (1-based).
    totalEpochs : int
        Total number of epochs.
    chunkInfo : Optional[str]
        Optional description of current dataset chunk.
    memoryLimitGB : float
        Maximum memory usage in GB before triggering cleanup (0 = disabled).

    Returns
    -------
    float
        Average training loss.
    """
    from src.shared.dataset_manager import MemoryManager, MemoryManagerConfig
    
    model.train()
    numBatches = 0
    
    # Setup memory manager
    memoryConfig = MemoryManagerConfig(MM_memoryLimitGB=memoryLimitGB)
    memoryManager = MemoryManager(memoryConfig, device)
    memoryManager.logMemoryStatus("epoch start")

    optimizer.zero_grad(set_to_none=True)
    accumSteps = 0

    with TrainingProgressBar(
        dataloader,
        epoch=epoch,
        totalEpochs=totalEpochs,
        desc="Generation",
        device=device,
        chunkInfo=chunkInfo,
    ) as pbar:
        for batch in pbar:
            loss = _runBatchAccumulate(
                batch, model, ddim, device, gradientAccumulation
            )
            pbar.updateLoss(loss)
            numBatches += 1
            accumSteps += 1

            # Step optimizer after accumulation
            if accumSteps >= gradientAccumulation:
                torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumSteps = 0
                
                # Clear MPS cache after optimizer step
                if device.type == "mps":
                    torch.mps.empty_cache()
                    
            # Check memory and cleanup if needed
            memoryManager.checkAndCleanup(numBatches)

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        return pbar.metrics.avgLoss


def _runBatch(
    batch: BatchDict,
    model: MotionGenerator,
    optimizer: torch.optim.Optimizer,
    ddim: DDIM,
    device: torch.device,
) -> float:
    """
    Run forward/backward pass for a single batch.

    Parameters
    ----------
    batch : BatchDict
        Batch from dataloader.
    model : MotionGenerator
        Generation model.
    optimizer : torch.optim.Optimizer
        Optimizer.
    ddim : DDIM
        Diffusion scheduler.
    device : torch.device
        Device.

    Returns
    -------
    float
        Batch loss value.
    """
    optimizer.zero_grad(set_to_none=True)  # More memory-efficient than zero_grad()

    # Move data to device
    inputIds = batch["input_ids"].to(device)
    attentionMask = batch["attention_mask"].to(device)
    motion = batch["motion"].to(device)
    tags = batch["tag"]

    # Delete batch reference early
    del batch

    batchSize = motion.shape[0]

    # Sample random timesteps
    timesteps = torch.randint(
        0, ddim.num_timesteps,
        (batchSize,),
        device=device,
        dtype=torch.long,
    )

    # Sample noise
    noise = torch.randn_like(motion)

    # Add noise to motion
    noisyMotion = ddim.q_sample(motion, timesteps, noise)
    
    # Delete original motion to save memory
    del motion

    # Predict noise
    outputs = model(
        textInputIds=inputIds,
        textAttentionMask=attentionMask,
        tags=tags,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetNoise=noise,
    )
    
    # Delete inputs early
    del inputIds, attentionMask, noisyMotion, tags

    loss = outputs["loss"]
    
    # Delete outputs dict early, keep only loss
    del outputs

    loss.backward()
    
    # Clip gradients to prevent memory spikes
    torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), max_norm=1.0)
    
    optimizer.step()

    lossValue = float(loss.detach().item())

    # Explicitly delete remaining tensors to free memory
    del noise, timesteps, loss

    return lossValue


def _runBatchAccumulate(
    batch: BatchDict,
    model: MotionGenerator,
    ddim: DDIM,
    device: torch.device,
    gradientAccumulation: int = 1,
) -> float:
    """
    Run forward/backward pass for gradient accumulation (no optimizer step).

    Parameters
    ----------
    batch : BatchDict
        Batch from dataloader.
    model : MotionGenerator
        Generation model.
    ddim : DDIM
        Diffusion scheduler.
    device : torch.device
        Device.
    gradientAccumulation : int
        Number of steps to accumulate (for loss scaling).

    Returns
    -------
    float
        Batch loss value.
    """
    # Move data to device
    inputIds = batch["input_ids"].to(device)
    attentionMask = batch["attention_mask"].to(device)
    motion = batch["motion"].to(device)
    tags = batch["tag"]

    del batch

    batchSize = motion.shape[0]

    # Sample random timesteps
    timesteps = torch.randint(
        0, ddim.num_timesteps,
        (batchSize,),
        device=device,
        dtype=torch.long,
    )

    # Sample noise
    noise = torch.randn_like(motion)

    # Add noise to motion
    noisyMotion = ddim.q_sample(motion, timesteps, noise)
    del motion

    # Predict noise
    outputs = model(
        textInputIds=inputIds,
        textAttentionMask=attentionMask,
        tags=tags,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetNoise=noise,
    )
    
    del inputIds, attentionMask, noisyMotion, tags

    loss = outputs["loss"]
    del outputs

    # Scale loss for gradient accumulation
    scaledLoss = loss / gradientAccumulation
    scaledLoss.backward()

    lossValue = float(loss.detach().item())
    del noise, timesteps, loss, scaledLoss

    return lossValue


def evaluateValidation(
    dataloader: DataLoader,
    model: MotionGenerator,
    ddim: DDIM,
    device: torch.device,
) -> float:
    """
    Compute average loss on validation set.

    Parameters
    ----------
    dataloader : DataLoader
        Validation dataloader.
    model : MotionGenerator
        Generation model.
    ddim : DDIM
        Diffusion scheduler.
    device : torch.device
        Device.

    Returns
    -------
    float
        Average validation loss.
    """
    model.eval()
    totalLoss = 0.0
    numBatches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputIds = batch["input_ids"].to(device)
            attentionMask = batch["attention_mask"].to(device)
            motion = batch["motion"].to(device)
            tags = batch["tag"]

            batchSize = motion.shape[0]

            timesteps = torch.randint(
                0, ddim.num_timesteps,
                (batchSize,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(motion)
            noisyMotion = ddim.q_sample(motion, timesteps, noise)

            outputs = model(
                textInputIds=inputIds,
                textAttentionMask=attentionMask,
                tags=tags,
                noisyMotion=noisyMotion,
                timesteps=timesteps,
                targetNoise=noise,
            )

            totalLoss += float(outputs["loss"].item())
            numBatches += 1

            # Free memory in validation loop
            del inputIds, attentionMask, motion, noisyMotion, noise, timesteps, outputs

    gc.collect()
    model.train()
    return totalLoss / max(numBatches, 1)


def saveCheckpoint(
    model: MotionGenerator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpointDir: Path,
    filename: str = "best_model.pt",
) -> Path:
    """
    Save model checkpoint.

    Parameters
    ----------
    model : MotionGenerator
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer state.
    epoch : int
        Current epoch.
    loss : float
        Best loss achieved.
    checkpointDir : Path
        Directory for checkpoints.
    filename : str, optional
        Checkpoint filename.

    Returns
    -------
    Path
        Path to saved checkpoint.
    """
    checkpointDir.mkdir(parents=True, exist_ok=True)
    checkpointPath = checkpointDir / filename

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "denoiser_state_dict": model.denoiser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpointPath,
    )

    return checkpointPath


def loadCheckpoint(
    checkpointPath: Path,
    model: MotionGenerator,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float]:
    """
    Load model checkpoint.

    Parameters
    ----------
    checkpointPath : Path
        Path to checkpoint file.
    model : MotionGenerator
        Model to load into.
    optimizer : Optional[torch.optim.Optimizer], optional
        Optimizer to load state into.

    Returns
    -------
    Tuple[int, float]
        Epoch and loss from checkpoint.
    """
    checkpoint = torch.load(checkpointPath, weights_only=False, map_location="cpu")

    # Try to load full model state first, fallback to denoiser only
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "denoiser_state_dict" in checkpoint:
        model.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))
