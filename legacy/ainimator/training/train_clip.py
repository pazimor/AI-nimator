"""Minimal training loop for the text-motion CLIP model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ainimator.core.constants.clip import DEFAULT_LEARNING_RATE
from ainimator.core.checkpoint_io import saveTorchObjectAtomically
from ainimator.model.clip.core import ClipModel
from ainimator.core.progress import TrainingProgressBar

BatchDict = Mapping[str, object]
LossComponents = dict[str, float]
LOSS_COMPONENT_KEYS = (
    "loss_text_contrastive",
    "loss_motion_contrastive",
    "loss_motion_cosine",
)



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


def disableDropoutModules(module: nn.Module) -> int:
    """Set every dropout probability in ``module`` to zero."""
    updated = 0
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = 0.0
            updated += 1
    return updated


def trainOneEpoch(
    dataloader: Iterable[BatchDict],
    model: ClipModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    totalEpochs: int = 1,
    chunkInfo: Optional[str] = None,
) -> tuple[float, LossComponents]:
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
    tuple[float, LossComponents]
        Average training loss across the epoch and its components.
    """
    model.train()
    componentSums = _initLossComponents()
    numBatches = 0

    with TrainingProgressBar(
        dataloader,
        epoch=epoch,
        totalEpochs=totalEpochs,
        desc="CLIP",
        device=device,
        chunkInfo=chunkInfo,
    ) as pbar:
        for batch in pbar:
            batchLoss, batchComponents = _runBatch(
                batch,
                model,
                optimizer,
                device,
            )
            pbar.updateLoss(batchLoss)
            _updateLossComponents(componentSums, batchComponents)
            numBatches += 1

        return pbar.metrics.avgLoss, _averageLossComponents(
            componentSums,
            numBatches,
        )


def _runBatch(
    batch: BatchDict,
    model: ClipModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, LossComponents]:
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
    tuple[float, LossComponents]
        Detached loss value and loss components.
    """
    optimizer.zero_grad(set_to_none=True)
    motionInput = model.buildMotionInput(batch).to(device)
    positiveMask = _buildPositiveMask(batch, device)
    outputs = model(
        textInputIds=None,
        textAttentionMask=None,
        motionInput=motionInput,
        motionMask=_optionalMotionMask(batch, device),
        positiveMask=positiveMask,
        computeLoss=True,
        pooledText=_toDevice(batch["pooled_text"], device),
    )
    loss = outputs["clip_loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(_trainableParameters(model), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().item()), _extractLossComponents(outputs)


def _runBatchAccumulate(
    batch: BatchDict,
    model: ClipModel,
    device: torch.device,
    accumulationSteps: int,
) -> tuple[float, LossComponents]:
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
    tuple[float, LossComponents]
        Detached loss value and loss components.
    """
    motionInput = model.buildMotionInput(batch).to(device)
    positiveMask = _buildPositiveMask(batch, device)
    outputs = model(
        textInputIds=None,
        textAttentionMask=None,
        motionInput=motionInput,
        motionMask=_optionalMotionMask(batch, device),
        positiveMask=positiveMask,
        computeLoss=True,
        pooledText=_toDevice(batch["pooled_text"], device),
    )
    loss = outputs["clip_loss"] / accumulationSteps
    loss.backward()
    return (
        float(outputs["clip_loss"].detach().item()),
        _extractLossComponents(outputs),
    )


def trainOneEpochWithAccumulation(
    dataloader: Iterable[BatchDict],
    model: ClipModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulationSteps: int = 1,
    epoch: int = 1,
    totalEpochs: int = 1,
    chunkInfo: Optional[str] = None,
) -> tuple[float, LossComponents]:
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

    Returns
    -------
    tuple[float, LossComponents]
        Average training loss across the epoch and its components.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    componentSums = _initLossComponents()
    numBatches = 0

    with TrainingProgressBar(
        dataloader,
        epoch=epoch,
        totalEpochs=totalEpochs,
        desc="CLIP",
        device=device,
        chunkInfo=chunkInfo,
    ) as pbar:
        for batchIndex, batch in enumerate(pbar):
            batchLoss, batchComponents = _runBatchAccumulate(
                batch,
                model,
                device,
                accumulationSteps,
            )
            pbar.updateLoss(batchLoss)
            _updateLossComponents(componentSums, batchComponents)
            numBatches += 1

            # Step optimizer every accumulationSteps batches
            if (batchIndex + 1) % accumulationSteps == 0:
                torch.nn.utils.clip_grad_norm_(
                    _trainableParameters(model),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Handle remaining gradients
        if len(dataloader) % accumulationSteps != 0:
            torch.nn.utils.clip_grad_norm_(
                _trainableParameters(model),
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return pbar.metrics.avgLoss, _averageLossComponents(
            componentSums,
            numBatches,
        )


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


def _optionalMotionMask(
    batch: BatchDict,
    device: torch.device,
) -> torch.Tensor | None:
    """Return the batched motion padding mask when available."""
    motionMask = batch.get("motion_mask")
    if motionMask is None:
        return None
    if not isinstance(motionMask, torch.Tensor):
        raise TypeError("Expected motion_mask batch entry to be a tensor.")
    return motionMask.to(device=device, dtype=torch.bool)


def _initLossComponents() -> LossComponents:
    """Return zero-initialized CLIP loss accumulators."""
    return {key: 0.0 for key in LOSS_COMPONENT_KEYS}


def _extractLossComponents(
    outputs: Mapping[str, object],
) -> LossComponents:
    """Extract detached CLIP loss components from one forward pass."""
    components: LossComponents = {}
    for key in LOSS_COMPONENT_KEYS:
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            components[key] = float(value.detach().item())
    return components


def _updateLossComponents(
    componentSums: LossComponents,
    components: Mapping[str, float],
) -> None:
    """Accumulate CLIP loss components."""
    for key, value in components.items():
        if key in componentSums:
            componentSums[key] += value


def _averageLossComponents(
    componentSums: Mapping[str, float],
    batchCount: int,
) -> LossComponents:
    """Average CLIP loss components across the epoch."""
    safeCount = max(batchCount, 1)
    return {key: value / safeCount for key, value in componentSums.items()}


def evaluateValidation(
    dataloader: DataLoader,
    model: ClipModel,
    device: torch.device,
) -> tuple[float, dict[str, float], LossComponents]:
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
    tuple[float, dict[str, float], LossComponents]
        Average validation loss, retrieval metrics, and loss components.
    """
    model.eval()
    totalLoss = 0.0
    componentSums = _initLossComponents()
    retrievalGallery: dict[str, list[torch.Tensor]] = {
        "text_embeds": [],
        "motion_embeds": [],
        "sample_ids": [],
        "text_ids": [],
    }
    with torch.no_grad():
        for batch in dataloader:
            positiveMask = _buildPositiveMask(batch, device)
            outputs = model(
                textInputIds=None,
                textAttentionMask=None,
                motionInput=model.buildMotionInput(batch).to(device),
                motionMask=_optionalMotionMask(batch, device),
                positiveMask=positiveMask,
                computeLoss=True,
                pooledText=_toDevice(batch["pooled_text"], device),
            )
            totalLoss += float(outputs["clip_loss"].item())
            _updateLossComponents(
                componentSums,
                _extractLossComponents(outputs),
            )
            _accumulateValidationGallery(
                storage=retrievalGallery,
                outputs=outputs,
                batch=batch,
            )
    model.train()
    avgLoss = totalLoss / max(len(dataloader), 1)
    metrics = _computeGlobalRetrieval(retrievalGallery)
    return avgLoss, metrics, _averageLossComponents(
        componentSums,
        len(dataloader),
    )


def _accumulateValidationGallery(
    storage: dict[str, list[torch.Tensor]],
    outputs: Mapping[str, object],
    batch: BatchDict,
) -> None:
    textEmbeds = outputs.get("text_embeds")
    motionEmbeds = outputs.get("motion_embeds")
    sampleIds = batch.get("sample_id")
    textIds = batch.get("text_id")
    if not isinstance(textEmbeds, torch.Tensor):
        return
    if not isinstance(motionEmbeds, torch.Tensor):
        return
    if not isinstance(sampleIds, torch.Tensor):
        raise TypeError("Validation batch is missing sample_id tensor.")
    if not isinstance(textIds, torch.Tensor):
        raise TypeError("Validation batch is missing text_id tensor.")
    batchSize = int(textEmbeds.shape[0])
    if (
        batchSize != int(motionEmbeds.shape[0])
        or batchSize != int(sampleIds.shape[0])
        or batchSize != int(textIds.shape[0])
    ):
        raise ValueError("Validation embeddings and sample_ids must share batch size.")
    if batchSize == 0:
        return
    storage["text_embeds"].append(textEmbeds.detach().cpu())
    storage["motion_embeds"].append(motionEmbeds.detach().cpu())
    storage["sample_ids"].append(sampleIds.detach().cpu())
    storage["text_ids"].append(textIds.detach().cpu())


def _computeGlobalRetrieval(
    storage: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    textChunks = storage.get("text_embeds", [])
    motionChunks = storage.get("motion_embeds", [])
    sampleIdChunks = storage.get("sample_ids", [])
    textIdChunks = storage.get("text_ids", [])
    if not textChunks or not motionChunks or not sampleIdChunks:
        return {
            "t2m_top1": 0.0,
            "t2m_top5": 0.0,
            "m2t_top1": 0.0,
            "m2t_top5": 0.0,
        }
    textEmbeds = torch.cat(textChunks, dim=0)
    motionEmbeds = torch.cat(motionChunks, dim=0)
    sampleIds = torch.cat(sampleIdChunks, dim=0).to(dtype=torch.long)
    textIds = None
    if textIdChunks:
        textIds = torch.cat(textIdChunks, dim=0).to(dtype=torch.long)
    count = int(textEmbeds.shape[0])
    if count == 0:
        return {
            "t2m_top1": 0.0,
            "t2m_top5": 0.0,
            "m2t_top1": 0.0,
            "m2t_top5": 0.0,
        }
    if (
        count != int(motionEmbeds.shape[0])
        or count != int(sampleIds.shape[0])
        or (textIds is not None and count != int(textIds.shape[0]))
    ):
        raise ValueError(
            "Global validation retrieval expects matching text/motion/sample counts."
        )

    logitsText = torch.matmul(textEmbeds, motionEmbeds.t())
    logitsMotion = logitsText.t()
    positiveMask = sampleIds[:, None].eq(sampleIds[None, :])
    if textIds is not None:
        positiveMask = positiveMask | textIds[:, None].eq(textIds[None, :])

    metrics = {
        "t2m_top1": 0.0,
        "t2m_top5": 0.0,
        "m2t_top1": 0.0,
        "m2t_top5": 0.0,
    }
    for k in (1, 5):
        k = min(k, count)
        topTextIndices = logitsText.topk(k, dim=1).indices
        topMotionIndices = logitsMotion.topk(k, dim=1).indices
        correctText = positiveMask.gather(1, topTextIndices).any(dim=1).sum()
        correctMotion = positiveMask.t().gather(1, topMotionIndices).any(dim=1).sum()
        if k == 1:
            metrics["t2m_top1"] = float(correctText.item()) / count
            metrics["m2t_top1"] = float(correctMotion.item()) / count
        else:
            metrics["t2m_top5"] = float(correctText.item()) / count
            metrics["m2t_top5"] = float(correctMotion.item()) / count
    return {
        "t2m_top1": metrics["t2m_top1"],
        "t2m_top5": metrics["t2m_top5"],
        "m2t_top1": metrics["m2t_top1"],
        "m2t_top5": metrics["m2t_top5"],
    }


def _buildPositiveMask(
    batch: BatchDict,
    device: torch.device,
) -> torch.Tensor:
    """
    Mark links sharing the same sample or raw text as positives.
    """
    sampleIds = batch.get("sample_id")
    textIds = batch.get("text_id")
    if not isinstance(sampleIds, torch.Tensor):
        raise TypeError("Expected tensor batch entry.")
    sampleIds = sampleIds.to(device=device, dtype=torch.long)
    positiveMask = sampleIds[:, None].eq(sampleIds[None, :])
    if isinstance(textIds, torch.Tensor):
        textIds = textIds.to(device=device, dtype=torch.long)
        positiveMask = positiveMask | textIds[:, None].eq(textIds[None, :])
    return positiveMask


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
    checkpointPath = checkpointDir / filename
    saveTorchObjectAtomically(
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
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise RuntimeError(
            "Failed to load the CLIP checkpoint. The checkpoint likely does "
            "not match the current CLIP architecture or motion component layout. "
            f"Original error: {error}"
        ) from error
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as error:
            raise RuntimeError(
                "Optimizer state is incompatible with the current CLIP "
                "parameter set."
            ) from error
    return checkpoint["epoch"], checkpoint["loss"]
