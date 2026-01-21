"""Minimal training loop for the text-motion CLIP model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.shared.constants.clip import DEFAULT_LEARNING_RATE
from src.shared.model.clip.core import ClipModel
from src.shared.progress import TrainingProgressBar

BatchDict = Mapping[str, object]



def buildOptimizer(
    model: ClipModel,
    learningRate: float = DEFAULT_LEARNING_RATE,
    weightDecay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Create the AdamW optimizer covering learnable modules.

    Parameters
    ----------
    model : ClipModel
        Model containing the trainable parameters.
    learningRate : float, optional
        Learning rate provided to AdamW.
    weightDecay : float, optional
        L2 regularization weight (default 0.0).

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer instance.
    """
    return torch.optim.AdamW(
        _trainableParameters(model), 
        lr=learningRate,
        weight_decay=weightDecay,
    )


def trainOneEpoch(
    dataloader: Iterable[BatchDict],
    model: ClipModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    totalEpochs: int = 1,
    chunkInfo: Optional[str] = None,
) -> float:
    """
    Run a single training epoch and return the average loss.

    Parameters
    ----------
    dataloader : Iterable[BatchDict]
        Iterable yielding preprocessed text-motion batches.
    model : ClipModel
        Model under training.
    optimizer : torch.optim.Optimizer
        Optimizer handling gradient updates.
    device : torch.device
        Target device where tensors are moved.
    epoch : int
        Current epoch number (1-based).
    totalEpochs : int
        Total number of epochs.
    chunkInfo : Optional[str]
        Optional description of current dataset chunk.

    Returns
    -------
    float
        Average training loss across the epoch.
    """
    model.train()

    with TrainingProgressBar(
        dataloader,
        epoch=epoch,
        totalEpochs=totalEpochs,
        desc="CLIP",
        device=device,
        chunkInfo=chunkInfo,
    ) as pbar:
        for batch in pbar:
            batchLoss, outputs = _runBatch(batch, model, optimizer, device)
            pbar.updateLoss(batchLoss)

        return pbar.metrics.avgLoss


def _runBatch(
    batch: BatchDict,
    model: ClipModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, Mapping[str, torch.Tensor]]:
    """
    Execute a forward/backward pass for a single batch.

    Parameters
    ----------
    batch : BatchDict
        Mini-batch emitted by the dataloader.
    model : ClipModel
        CLIP model under training.
    optimizer : torch.optim.Optimizer
        Optimizer managing the trainable parameters.
    device : torch.device
        Target device where tensors reside.

    Returns
    -------
    tuple[float, Mapping[str, torch.Tensor]]
        Detached loss value and model outputs.
    """
    optimizer.zero_grad()
    outputs = model(
        textInputIds=_toDevice(batch["input_ids"], device),
        textAttentionMask=_toDevice(batch["attention_mask"], device),
        motionInput=_toDevice(batch["motion"], device),
        computeLoss=True,
    )
    loss = outputs["clip_loss"]
    loss.backward()
    optimizer.step()
    return float(loss.detach().item()), outputs


def _runBatchAccumulate(
    batch: BatchDict,
    model: ClipModel,
    device: torch.device,
    accumulationSteps: int,
) -> tuple[float, Mapping[str, torch.Tensor]]:
    """
    Execute a forward/backward pass for gradient accumulation.

    Parameters
    ----------
    batch : BatchDict
        Mini-batch emitted by the dataloader.
    model : ClipModel
        CLIP model under training.
    device : torch.device
        Target device where tensors reside.
    accumulationSteps : int
        Number of steps to accumulate gradients over.

    Returns
    -------
    tuple[float, Mapping[str, torch.Tensor]]
        Detached loss value and model outputs.
    """
    outputs = model(
        textInputIds=_toDevice(batch["input_ids"], device),
        textAttentionMask=_toDevice(batch["attention_mask"], device),
        motionInput=_toDevice(batch["motion"], device),
        computeLoss=True,
    )
    loss = outputs["clip_loss"] / accumulationSteps
    loss.backward()
    return float(outputs["clip_loss"].detach().item()), outputs


def trainOneEpochWithAccumulation(
    dataloader: Iterable[BatchDict],
    model: ClipModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulationSteps: int = 1,
    epoch: int = 1,
    totalEpochs: int = 1,
    chunkInfo: Optional[str] = None,
    memoryLimitGB: float = 0.0,
) -> float:
    """
    Run a single training epoch with gradient accumulation.

    Parameters
    ----------
    dataloader : Iterable[BatchDict]
        Iterable yielding preprocessed text-motion batches.
    model : ClipModel
        Model under training.
    optimizer : torch.optim.Optimizer
        Optimizer handling gradient updates.
    device : torch.device
        Target device where tensors are moved.
    accumulationSteps : int
        Number of batches to accumulate before stepping.
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
        Average training loss across the epoch.
    """
    model.train()
    optimizer.zero_grad()
    
    # Setup memory manager
    from src.shared.dataset_manager import MemoryManager, MemoryManagerConfig
    memoryConfig = MemoryManagerConfig(MM_memoryLimitGB=memoryLimitGB)
    memoryManager = MemoryManager(memoryConfig, device)

    with TrainingProgressBar(
        dataloader,
        epoch=epoch,
        totalEpochs=totalEpochs,
        desc="CLIP",
        device=device,
        chunkInfo=chunkInfo,
    ) as pbar:
        for batchIndex, batch in enumerate(pbar):
            batchLoss, outputs = _runBatchAccumulate(
                batch, model, device, accumulationSteps
            )
            pbar.updateLoss(batchLoss)

            # Step optimizer every accumulationSteps batches
            if (batchIndex + 1) % accumulationSteps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Check memory and cleanup if needed
            memoryManager.checkAndCleanup(batchIndex)

        # Handle remaining gradients
        if len(dataloader) % accumulationSteps != 0:
            optimizer.step()
            optimizer.zero_grad()

        return pbar.metrics.avgLoss


def _trainableParameters(model: ClipModel) -> list[torch.nn.Parameter]:
    """
    Return the modules updated by the optimizer.

    Parameters
    ----------
    model : ClipModel
        CLIP model under training.

    Returns
    -------
    list[torch.nn.Parameter]
        Trainable parameter list.
    """
    return [
        *model.textProj.parameters(),
        *model.motionProj.parameters(),
        *model.motionBackbone.parameters(),
        model.logitScale,
    ]


def _toDevice(value: object, device: torch.device) -> torch.Tensor:
    """
    Move tensors from CPU to the requested device.

    Parameters
    ----------
    value : object
        Tensor extracted from the batch.
    device : torch.device
        Target device used for computation.

    Returns
    -------
    torch.Tensor
        Tensor moved to the requested device.

    Raises
    ------
    TypeError
        Raised when the provided value is not a tensor.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected tensor batch entry.")
    return value.to(device)


def evaluateValidation(
    dataloader: DataLoader,
    model: ClipModel,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """
    Compute average loss on the validation set without gradients.

    Parameters
    ----------
    dataloader : DataLoader
        Validation dataloader.
    model : ClipModel
        Trained CLIP model.
    device : torch.device
        Target device for computation.

    Returns
    -------
    tuple[float, dict[str, float]]
        Average validation loss and retrieval metrics.
    """
    model.eval()
    totalLoss = 0.0
    retrieval = {
        "t2m_top1": 0.0,
        "t2m_top5": 0.0,
        "m2t_top1": 0.0,
        "m2t_top5": 0.0,
        "count": 0.0,
    }
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                textInputIds=_toDevice(batch["input_ids"], device),
                textAttentionMask=_toDevice(batch["attention_mask"], device),
                motionInput=_toDevice(batch["motion"], device),
                computeLoss=True,
            )
            totalLoss += float(outputs["clip_loss"].item())
            _accumulateRetrieval(retrieval, outputs)
    model.train()
    avgLoss = totalLoss / max(len(dataloader), 1)
    metrics = _finalizeRetrieval(retrieval)
    return avgLoss, metrics


def _accumulateRetrieval(
    metrics: dict[str, float],
    outputs: Mapping[str, object],
) -> None:
    logitsText = outputs.get("logits_per_text")
    logitsMotion = outputs.get("logits_per_motion")
    if not isinstance(logitsText, torch.Tensor):
        return
    if not isinstance(logitsMotion, torch.Tensor):
        return
    batchSize = logitsText.shape[0]
    if batchSize == 0:
        return

    target = torch.arange(batchSize, device=logitsText.device)
    for k in (1, 5):
        k = min(k, batchSize)
        correctText = (
            logitsText.topk(k, dim=1).indices == target[:, None]
        ).any(dim=1).sum()
        correctMotion = (
            logitsMotion.topk(k, dim=1).indices == target[:, None]
        ).any(dim=1).sum()
        if k == 1:
            metrics["t2m_top1"] += float(correctText.item())
            metrics["m2t_top1"] += float(correctMotion.item())
        else:
            metrics["t2m_top5"] += float(correctText.item())
            metrics["m2t_top5"] += float(correctMotion.item())
    metrics["count"] += float(batchSize)


def _finalizeRetrieval(metrics: dict[str, float]) -> dict[str, float]:
    count = metrics.get("count", 0.0)
    if count <= 0:
        return {
            "t2m_top1": 0.0,
            "t2m_top5": 0.0,
            "m2t_top1": 0.0,
            "m2t_top5": 0.0,
        }
    return {
        "t2m_top1": metrics["t2m_top1"] / count,
        "t2m_top5": metrics["t2m_top5"] / count,
        "m2t_top1": metrics["m2t_top1"] / count,
        "m2t_top5": metrics["m2t_top5"] / count,
    }


def saveCheckpoint(
    model: ClipModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpointDir: Path,
    filename: str = "best_model.pt",
) -> Path:
    """
    Save model and optimizer state to a checkpoint file.

    Parameters
    ----------
    model : ClipModel
        Trained CLIP model.
    optimizer : torch.optim.Optimizer
        Optimizer state to save.
    epoch : int
        Current epoch number.
    loss : float
        Best validation loss achieved.
    checkpointDir : Path
        Directory to save the checkpoint.
    filename : str, optional
        Name of the checkpoint file.

    Returns
    -------
    Path
        Path to the saved checkpoint.
    """
    checkpointDir.mkdir(parents=True, exist_ok=True)
    checkpointPath = checkpointDir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpointPath,
    )
    return checkpointPath


def loadCheckpoint(
    checkpointPath: Path,
    model: ClipModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float]:
    """
    Load model and optimizer state from a checkpoint file.

    Parameters
    ----------
    checkpointPath : Path
        Path to the checkpoint file.
    model : ClipModel
        Model to load the state into.
    optimizer : Optional[torch.optim.Optimizer], optional
        Optimizer to load state into.

    Returns
    -------
    Tuple[int, float]
        Epoch number and loss from the checkpoint.
    """
    checkpoint = torch.load(checkpointPath, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]
