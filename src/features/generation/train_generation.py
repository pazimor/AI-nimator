"""Training loop for the motion generation model."""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader

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

from src.shared.model.generation.ddim import DDIM
from src.shared.model.generation.motion_generator import MotionGenerator
from src.shared.progress import TrainingProgressBar

BatchDict = Mapping[str, object]
LossComponents = dict[str, float]
LOGGER = logging.getLogger("generation.train")
LOSS_COMPONENT_KEYS = (
    "loss_diffusion",
    "loss_geodesic",
    "loss_velocity",
    "loss_acceleration",
)
LOSS_COMPONENT_LABELS = {
    "loss_diffusion": "diff",
    "loss_geodesic": "geo",
    "loss_velocity": "vel",
    "loss_acceleration": "acc",
}


def _initLossComponents() -> LossComponents:
    """
    Initialize loss component accumulators.

    Returns
    -------
    LossComponents
        Zero-initialized component dictionary.
    """
    return {key: 0.0 for key in LOSS_COMPONENT_KEYS}


def _extractLossComponents(
    outputs: Mapping[str, torch.Tensor],
) -> LossComponents:
    """
    Extract loss component values from model outputs.

    Parameters
    ----------
    outputs : Mapping[str, torch.Tensor]
        Output dictionary from the model forward pass.

    Returns
    -------
    LossComponents
        Dictionary with detached component values.
    """
    components: LossComponents = {}
    for key in LOSS_COMPONENT_KEYS:
        value = outputs.get(key)
        if value is not None:
            components[key] = float(value.detach().item())
    return components


def _updateLossComponents(
    componentSums: LossComponents,
    components: Mapping[str, float],
) -> None:
    """
    Accumulate loss components into running sums.

    Parameters
    ----------
    componentSums : LossComponents
        Running sums to update.
    components : Mapping[str, float]
        Latest batch loss components.
    """
    for key, value in components.items():
        if key in componentSums:
            componentSums[key] += value


def _averageLossComponents(
    componentSums: Mapping[str, float],
    batchCount: int,
) -> LossComponents:
    """
    Compute average loss components from sums.

    Parameters
    ----------
    componentSums : Mapping[str, float]
        Running loss sums.
    batchCount : int
        Number of batches contributing to the sums.

    Returns
    -------
    LossComponents
        Average loss components per batch.
    """
    safeCount = max(batchCount, 1)
    return {
        key: value / safeCount for key, value in componentSums.items()
    }


def _buildLossPostfix(
    lossValue: float,
    avgLoss: float,
    avgComponents: Mapping[str, float],
) -> dict[str, str]:
    """
    Build progress bar postfix values for loss breakdown.

    Parameters
    ----------
    lossValue : float
        Current batch loss.
    avgLoss : float
        Average loss so far.
    avgComponents : Mapping[str, float]
        Average component losses.

    Returns
    -------
    dict[str, str]
        Postfix values for tqdm.
    """
    postfix = {
        "loss": f"{lossValue:.4f}",
        "avg": f"{avgLoss:.4f}",
    }
    for key, label in LOSS_COMPONENT_LABELS.items():
        if key in avgComponents:
            postfix[label] = f"{avgComponents[key]:.4f}"
    return postfix


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
) -> tuple[float, LossComponents]:
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
    tuple[float, LossComponents]
        Average training loss and average component losses.
    """
    from src.shared.dataset_manager import MemoryManager, MemoryManagerConfig
    
    model.train()
    numBatches = 0
    componentSums = _initLossComponents()
    
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
            lossValue, lossComponents = _runBatchAccumulate(
                batch, model, ddim, device, gradientAccumulation
            )
            pbar.updateLoss(lossValue)
            _updateLossComponents(componentSums, lossComponents)
            numBatches += 1
            accumSteps += 1
            avgComponents = _averageLossComponents(componentSums, numBatches)
            pbar.setPostfix(
                **_buildLossPostfix(
                    lossValue=lossValue,
                    avgLoss=pbar.metrics.avgLoss,
                    avgComponents=avgComponents,
                )
            )

            # Step optimizer after accumulation
            if accumSteps >= gradientAccumulation:
                torch.nn.utils.clip_grad_norm_(
                    model.denoiser.parameters(),
                    max_norm=1.0,
                )
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
        avgComponents = _averageLossComponents(componentSums, numBatches)
        return pbar.metrics.avgLoss, avgComponents


def _runBatch(
    batch: BatchDict,
    model: MotionGenerator,
    optimizer: torch.optim.Optimizer,
    ddim: DDIM,
    device: torch.device,
) -> tuple[float, LossComponents]:
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
    tuple[float, LossComponents]
        Batch loss value and component breakdown.
    """
    optimizer.zero_grad(
        set_to_none=True
    )  # More memory-efficient than zero_grad()

    # Move data to device
    inputIds = batch["input_ids"].to(device)
    attentionMask = batch["attention_mask"].to(device)
    motion = batch["motion"].to(device)
    motionMask = batch.get("motion_mask")
    if motionMask is not None:
        motionMask = motionMask.to(device)
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

    # Predict noise
    outputs = model(
        textInputIds=inputIds,
        textAttentionMask=attentionMask,
        tags=tags,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetNoise=noise,
        targetMotion=motion,
        motionMask=motionMask,
    )
    
    # Delete inputs early
    del inputIds, attentionMask, noisyMotion, tags, motion, motionMask

    loss = outputs["loss"]
    lossComponents = _extractLossComponents(outputs)
    
    # Delete outputs dict early, keep only loss
    del outputs

    loss.backward()
    
    # Clip gradients to prevent memory spikes
    torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), max_norm=1.0)
    
    optimizer.step()

    lossValue = float(loss.detach().item())

    # Explicitly delete remaining tensors to free memory
    del noise, timesteps, loss

    return lossValue, lossComponents


def _runBatchAccumulate(
    batch: BatchDict,
    model: MotionGenerator,
    ddim: DDIM,
    device: torch.device,
    gradientAccumulation: int = 1,
) -> tuple[float, LossComponents]:
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
    tuple[float, LossComponents]
        Batch loss value and component breakdown.
    """
    # Move data to device
    inputIds = batch["input_ids"].to(device)
    attentionMask = batch["attention_mask"].to(device)
    motion = batch["motion"].to(device)
    motionMask = batch.get("motion_mask")
    if motionMask is not None:
        motionMask = motionMask.to(device)
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

    # Predict noise
    outputs = model(
        textInputIds=inputIds,
        textAttentionMask=attentionMask,
        tags=tags,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetNoise=noise,
        targetMotion=motion,
        motionMask=motionMask,
    )
    
    del inputIds, attentionMask, noisyMotion, tags, motion, motionMask

    loss = outputs["loss"]
    lossComponents = _extractLossComponents(outputs)
    del outputs

    # Scale loss for gradient accumulation
    scaledLoss = loss / gradientAccumulation
    scaledLoss.backward()

    lossValue = float(loss.detach().item())
    del noise, timesteps, loss, scaledLoss

    return lossValue, lossComponents


def evaluateValidation(
    dataloader: DataLoader,
    model: MotionGenerator,
    ddim: DDIM,
    device: torch.device,
) -> tuple[float, LossComponents]:
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
    tuple[float, LossComponents]
        Average validation loss and component breakdown.
    """
    model.eval()
    totalLoss = 0.0
    numBatches = 0
    componentSums = _initLossComponents()

    with torch.no_grad():
        for batch in dataloader:
            inputIds = batch["input_ids"].to(device)
            attentionMask = batch["attention_mask"].to(device)
            motion = batch["motion"].to(device)
            motionMask = batch.get("motion_mask")
            if motionMask is not None:
                motionMask = motionMask.to(device)
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
                targetMotion=motion,
                motionMask=motionMask,
            )

            totalLoss += float(outputs["loss"].item())
            _updateLossComponents(
                componentSums,
                _extractLossComponents(outputs),
            )
            numBatches += 1

            # Free memory in validation loop
            del (
                inputIds,
                attentionMask,
                motion,
                noisyMotion,
                noise,
                timesteps,
                outputs,
                motionMask,
            )

    gc.collect()
    model.train()
    avgLoss = totalLoss / max(numBatches, 1)
    avgComponents = _averageLossComponents(componentSums, numBatches)
    return avgLoss, avgComponents


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
    checkpoint = torch.load(
        checkpointPath,
        weights_only=False,
        map_location="cpu",
    )

    # Try to load full model state first, fallback to denoiser only
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "denoiser_state_dict" in checkpoint:
        model.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))
