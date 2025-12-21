"""Minimal training loop for the text-motion CLIP model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from transformers import XLMRobertaTokenizerFast

from src.shared.constants.clip import (
    BATCH_LOG_FREQUENCY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBED_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_MAX_LENGTH,
    DEFAULT_VALIDATION_SPLIT,
)
from src.shared.logging_utils import logClipBatchStats
from src.shared.model.clip.core import ClipModel
from src.shared.model.clip.data import MotionTextClipDataset, motionTextCollate

BatchDict = Mapping[str, object]


def buildDataloader(
    promptRoot: Path,
    animationRoot: Path,
    maxLength: int = DEFAULT_PROMPT_MAX_LENGTH,
    batchSize: int = DEFAULT_BATCH_SIZE,
    modelName: str = DEFAULT_MODEL_NAME,
) -> DataLoader:
    """
    Instantiate dataset and dataloader for CLIP training.

    Parameters
    ----------
    promptRoot : Path
        Directory containing prompt.json files.
    animationRoot : Path
        Directory containing animation payloads.
    maxLength : int, optional
        Maximum token length for padding/truncation.
    batchSize : int, optional
        Batch size used during training.
    modelName : str, optional
        Hugging Face identifier of the tokenizer.

    Returns
    -------
    DataLoader
        Configured dataloader using the shared collate function.
    """
    dataset = _buildDataset(
        promptRoot=promptRoot,
        animationRoot=animationRoot,
        tokenizer=_buildTokenizer(modelName),
        maxLength=maxLength,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No samples found under {promptRoot}. "
            "Verify the dataset-root/prompt-root configuration.",
        )
    return _makeDataloader(dataset, batchSize)


def buildOptimizer(
    model: ClipModel,
    learningRate: float = DEFAULT_LEARNING_RATE,
) -> torch.optim.Optimizer:
    """
    Create the AdamW optimizer covering learnable modules.

    Parameters
    ----------
    model : ClipModel
        Model containing the trainable parameters.
    learningRate : float, optional
        Learning rate provided to AdamW.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer instance.
    """
    return torch.optim.AdamW(_trainableParameters(model), lr=learningRate)


def trainOneEpoch(
    dataloader: Iterable[BatchDict],
    model: ClipModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run a single training epoch and return the average loss.

    Parameters
    ----------
    dataloader : Iterable[BatchDict]
        Iterable yielding MotionTextClipDataset batches.
    model : ClipModel
        Model under training.
    optimizer : torch.optim.Optimizer
        Optimizer handling gradient updates.
    device : torch.device
        Target device where tensors are moved.

    Returns
    -------
    float
        Average training loss across the epoch.
    """
    model.train()
    totalLoss = 0.0
    for batchIndex, batch in enumerate(dataloader):
        batchLoss, outputs = _runBatch(batch, model, optimizer, device)
        totalLoss += batchLoss
        if batchIndex % BATCH_LOG_FREQUENCY == 0:
            logClipBatchStats(batchIndex, outputs)
    return totalLoss / max(len(dataloader), 1)


def demoTrainingRun(
    promptRoot: Path,
    animationRoot: Path,
    device: torch.device | None = None,
) -> Tuple[ClipModel, float]:
    """
    Launch a minimal training loop for smoke testing purposes.

    Parameters
    ----------
    promptRoot : Path
        Root directory containing prompt.json files.
    animationRoot : Path
        Directory containing animation payloads.
    device : torch.device | None, optional
        Requested device; defaults to CUDA when available.

    Returns
    -------
    Tuple[ClipModel, float]
        Trained model and last epoch loss value.
    """
    resolvedDevice = _resolveDevice(device)
    dataloader = buildDataloader(promptRoot, animationRoot)
    model = ClipModel()
    model.to(resolvedDevice)
    optimizer = buildOptimizer(model=model)
    loss = trainOneEpoch(dataloader, model, optimizer, resolvedDevice)
    return model, loss


def _buildTokenizer(modelName: str) -> XLMRobertaTokenizerFast:
    """
    Return the tokenizer expected by ClipModel.

    Parameters
    ----------
    modelName : str
        Hugging Face identifier of the tokenizer.

    Returns
    -------
    XLMRobertaTokenizerFast
        Tokenizer preloaded with vocabulary files.
    """
    return XLMRobertaTokenizerFast.from_pretrained(modelName)


def _buildDataset(
    promptRoot: Path,
    animationRoot: Path,
    tokenizer: XLMRobertaTokenizerFast,
    maxLength: int,
) -> MotionTextClipDataset:
    """
    Index prompt segments without pre-loading motions.

    Parameters
    ----------
    promptRoot : Path
        Directory containing prompt.json files.
    animationRoot : Path
        Directory containing animation payloads.
    tokenizer : XLMRobertaTokenizerFast
        Tokenizer used to encode prompts.
    maxLength : int
        Maximum token sequence length.

    Returns
    -------
    MotionTextClipDataset
        Dataset yielding text-motion pairs.
    """
    return MotionTextClipDataset(
        rootPrompts=promptRoot,
        rootAnimations=animationRoot,
        tokenizer=tokenizer,
        maxLength=maxLength,
    )


def _makeDataloader(
    dataset: MotionTextClipDataset,
    batchSize: int,
) -> DataLoader:
    """
    Build the dataloader with custom collation.

    Parameters
    ----------
    dataset : MotionTextClipDataset
        Dataset created by `_buildDataset`.
    batchSize : int
        Batch size used during training.

    Returns
    -------
    DataLoader
        Dataloader yielding padded batches.
    """
    return DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=motionTextCollate,
    )


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


def _resolveDevice(device: torch.device | None) -> torch.device:
    """
    Select CUDA when available with a CPU fallback.

    Parameters
    ----------
    device : torch.device | None
        Requested device override.

    Returns
    -------
    torch.device
        Device used for the training loop.
    """
    if device is not None:
        return device
    backend = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(backend)


def buildTrainValDataloaders(
    promptRoot: Path,
    animationRoot: Path,
    maxLength: int = DEFAULT_PROMPT_MAX_LENGTH,
    batchSize: int = DEFAULT_BATCH_SIZE,
    modelName: str = DEFAULT_MODEL_NAME,
    validationSplit: float = DEFAULT_VALIDATION_SPLIT,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train and validation dataloaders with a split.

    Parameters
    ----------
    promptRoot : Path
        Directory containing prompt.json files.
    animationRoot : Path
        Directory containing animation payloads.
    maxLength : int, optional
        Maximum token length for padding/truncation.
    batchSize : int, optional
        Batch size used during training.
    modelName : str, optional
        Hugging Face identifier of the tokenizer.
    validationSplit : float, optional
        Fraction of data to use for validation (0.0-1.0).

    Returns
    -------
    Tuple[DataLoader, Optional[DataLoader]]
        Training dataloader and optional validation dataloader.
    """
    dataset = _buildDataset(
        promptRoot=promptRoot,
        animationRoot=animationRoot,
        tokenizer=_buildTokenizer(modelName),
        maxLength=maxLength,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No samples found under {promptRoot}. "
            "Verify the dataset-root/prompt-root configuration.",
        )
    if validationSplit <= 0.0 or len(dataset) < 2:
        return _makeDataloader(dataset, batchSize), None
    valSize = max(1, int(len(dataset) * validationSplit))
    trainSize = len(dataset) - valSize
    trainSubset, valSubset = random_split(dataset, [trainSize, valSize])
    trainLoader = DataLoader(
        trainSubset,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=motionTextCollate,
    )
    valLoader = DataLoader(
        valSubset,
        batch_size=batchSize,
        shuffle=False,
        collate_fn=motionTextCollate,
    )
    return trainLoader, valLoader


def evaluateValidation(
    dataloader: DataLoader,
    model: ClipModel,
    device: torch.device,
) -> float:
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
    float
        Average validation loss.
    """
    model.eval()
    totalLoss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                textInputIds=_toDevice(batch["input_ids"], device),
                textAttentionMask=_toDevice(batch["attention_mask"], device),
                motionInput=_toDevice(batch["motion"], device),
                computeLoss=True,
            )
            totalLoss += float(outputs["clip_loss"].item())
    model.train()
    return totalLoss / max(len(dataloader), 1)


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
