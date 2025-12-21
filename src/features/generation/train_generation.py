"""Training loop for the motion generation model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from src.shared.model.clip.data import MotionTextClipDataset, motionTextCollate
from src.shared.model.generation.ddim import DDIM
from src.shared.model.generation.losses import diffusionLoss
from src.shared.model.generation.motion_generator import MotionGenerator
from src.shared.types import GenerationTrainingConfig

from transformers import XLMRobertaTokenizerFast

BatchDict = Mapping[str, object]


def buildTrainValDataloaders(
    datasetRoot: Path,
    maxLength: int,
    batchSize: int,
    modelName: str,
    validationSplit: float,
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
    )

    if len(dataset) == 0:
        raise ValueError(f"No samples found under {datasetRoot}")

    if validationSplit <= 0.0 or len(dataset) < 2:
        return _makeDataloader(dataset, batchSize, shuffle=True), None

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

    Returns
    -------
    float
        Average training loss.
    """
    model.train()
    totalLoss = 0.0
    numBatches = 0

    for batch in dataloader:
        loss = _runBatch(batch, model, optimizer, ddim, device)
        totalLoss += loss
        numBatches += 1

    return totalLoss / max(numBatches, 1)


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
    optimizer.zero_grad()

    # Move data to device
    inputIds = batch["input_ids"].to(device)
    attentionMask = batch["attention_mask"].to(device)
    motion = batch["motion"].to(device)
    tags = batch["tag"]

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
    )

    loss = outputs["loss"]
    loss.backward()
    optimizer.step()

    return float(loss.detach().item())


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
