"""Training loop for the motion generation model."""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.types.generation import PREDICTION_TARGET_X0

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
    "loss_xyz",
    "loss_vel_xyz",
    "loss_acceleration",
    "loss_clip_guidance",
    "loss_root_translation",
    "loss_root_velocity",
    "loss_components",
    "loss_foot_skating",
)
LOSS_COMPONENT_LABELS = {
    "loss_xyz": "xyz",
    "loss_vel_xyz": "vel_xyz",
    "loss_clip_guidance": "clip",
    "loss_root_translation": "rtrans",
    "loss_root_velocity": "rvel",
    "loss_components": "aux",
    "loss_foot_skating": "skate",
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
    for key, value in outputs.items():
        if key == "loss" or not key.startswith("loss_"):
            continue
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
    # Train the denoiser and any auxiliary generation heads (CLIP is frozen).
    trainableParams = list(model.trainableParameters())
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
    clearMpsCache: bool = True,
    deterministicCorruption: bool = False,
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
    clearMpsCache : bool
        When False, skip explicit torch.mps.empty_cache() calls.

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
    memoryConfig = MemoryManagerConfig(
        MM_memoryLimitGB=memoryLimitGB,
        clearMpsCache=clearMpsCache,
    )
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
                batch,
                model,
                ddim,
                device,
                gradientAccumulation,
                deterministicCorruption=deterministicCorruption,
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
                    model.trainableParameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumSteps = 0
                
                # Clear MPS cache after optimizer step
                if clearMpsCache and device.type == "mps":
                    torch.mps.empty_cache()
                    
            # Check memory and cleanup if needed
            memoryManager.checkAndCleanup(numBatches)

        gc.collect()
        if clearMpsCache and device.type == "mps":
            torch.mps.empty_cache()
        avgComponents = _averageLossComponents(componentSums, numBatches)
        return pbar.metrics.avgLoss, avgComponents


def _runBatch(
    batch: BatchDict,
    model: MotionGenerator,
    optimizer: torch.optim.Optimizer,
    ddim: DDIM,
    device: torch.device,
    deterministicCorruption: bool = False,
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
    textEmbedding = batch["generation_text_embedding"].to(device)
    # Classifier-Free Guidance: randomly zero text embeddings
    if model.training and model.condMaskProb > 0.0:
        cfgDropMask = (
            torch.rand(textEmbedding.shape[0], device=device)
            < model.condMaskProb
        )
        textEmbedding = textEmbedding.clone()
        textEmbedding[cfgDropMask] = 0.0
    motion = batch["motion"].to(device)
    motionMask = batch.get("motion_mask")
    if motionMask is not None:
        motionMask = motionMask.to(device)
    clipMotionContext = model.clip.extractMotionContext(batch)
    componentTargets = _extractComponentTargets(batch, model, device)
    batchSize = motion.shape[0]
    _ensureRotationOnlyInput(motion)

    # Z-normalize motion for diffusion; raw motion stays as loss target.
    normalizedMotion = model.normalizeMotion(motion)

    timesteps, noise, noisyMotion = _prepareDiffusionInputs(
        batch=batch,
        motion=normalizedMotion,
        ddim=ddim,
        device=device,
        deterministicCorruption=deterministicCorruption,
    )

    # Delete batch reference early
    del batch

    # Predict noise
    outputs = model(
        textEmbedding=textEmbedding,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetNoise=noise,
        targetMotion=motion,
        motionMask=motionMask,
        clipMotionContext=clipMotionContext,
        componentTargets=componentTargets,
    )
    
    # Delete inputs early
    del (
        textEmbedding,
        noisyMotion,
        motion,
        motionMask,
        clipMotionContext,
        componentTargets,
    )

    loss = outputs["loss"]
    lossComponents = _extractLossComponents(outputs)
    
    # Delete outputs dict early, keep only loss
    del outputs

    loss.backward()
    
    # Clip gradients to prevent memory spikes
    torch.nn.utils.clip_grad_norm_(model.trainableParameters(), max_norm=1.0)
    
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
    deterministicCorruption: bool = False,
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
    textEmbedding = batch["generation_text_embedding"].to(device)
    # Classifier-Free Guidance: randomly zero text embeddings
    if model.training and model.condMaskProb > 0.0:
        cfgDropMask = (
            torch.rand(textEmbedding.shape[0], device=device)
            < model.condMaskProb
        )
        textEmbedding = textEmbedding.clone()
        textEmbedding[cfgDropMask] = 0.0
    motion = batch["motion"].to(device)
    motionMask = batch.get("motion_mask")
    if motionMask is not None:
        motionMask = motionMask.to(device)
    clipMotionContext = model.clip.extractMotionContext(batch)
    componentTargets = _extractComponentTargets(batch, model, device)
    batchSize = motion.shape[0]
    _ensureRotationOnlyInput(motion)

    # Z-normalize motion for diffusion; raw motion stays as loss target.
    normalizedMotion = model.normalizeMotion(motion)

    timesteps, noise, noisyMotion = _prepareDiffusionInputs(
        batch=batch,
        motion=normalizedMotion,
        ddim=ddim,
        device=device,
        deterministicCorruption=deterministicCorruption,
    )

    del batch

    # Predict noise
    outputs = model(
        textEmbedding=textEmbedding,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetNoise=noise,
        targetMotion=motion,
        motionMask=motionMask,
        clipMotionContext=clipMotionContext,
        componentTargets=componentTargets,
    )
    
    del (
        textEmbedding,
        noisyMotion,
        motion,
        motionMask,
        clipMotionContext,
        componentTargets,
    )

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
    deterministicCorruption: bool = False,
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
            textEmbedding = batch["generation_text_embedding"].to(device)
            motion = batch["motion"].to(device)
            motionMask = batch.get("motion_mask")
            if motionMask is not None:
                motionMask = motionMask.to(device)
            clipMotionContext = model.clip.extractMotionContext(batch)
            componentTargets = _extractComponentTargets(batch, model, device)
            batchSize = motion.shape[0]
            _ensureRotationOnlyInput(motion)

            # Z-normalize for diffusion; raw motion stays as loss target.
            normalizedMotion = model.normalizeMotion(motion)

            timesteps, noise, noisyMotion = _prepareDiffusionInputs(
                batch=batch,
                motion=normalizedMotion,
                ddim=ddim,
                device=device,
                deterministicCorruption=deterministicCorruption,
            )

            outputs = model(
                textEmbedding=textEmbedding,
                noisyMotion=noisyMotion,
                timesteps=timesteps,
                targetNoise=noise,
                targetMotion=motion,
                motionMask=motionMask,
                clipMotionContext=clipMotionContext,
                componentTargets=componentTargets,
            )

            totalLoss += float(outputs["loss"].item())
            _updateLossComponents(
                componentSums,
                _extractLossComponents(outputs),
            )
            numBatches += 1

            # Free memory in validation loop
            del (
                textEmbedding,
                motion,
                noisyMotion,
                noise,
                timesteps,
                outputs,
                motionMask,
                clipMotionContext,
                componentTargets,
            )

    gc.collect()
    model.train()
    avgLoss = totalLoss / max(numBatches, 1)
    avgComponents = _averageLossComponents(componentSums, numBatches)
    return avgLoss, avgComponents


def _ensureRotationOnlyInput(motion: torch.Tensor) -> None:
    """
    Ensure model inputs only contain 6D rotations (no translation channels).
    """
    if motion.shape[-1] != MotionGenerator.MOTION_ROTATION_CHANNELS:
        raise ValueError(
            "Expected motion input with 6 channels (rotation-only), "
            f"got {motion.shape[-1]}."
        )


def _extractComponentTargets(
    batch: BatchDict,
    model: MotionGenerator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collect auxiliary generation targets available in the batch."""
    targets: dict[str, torch.Tensor] = {}
    for component in model.generationMotionComponents:
        if component.key == "rotation6d":
            continue
        value = batch.get(component.sampleKey)
        if isinstance(value, torch.Tensor):
            targets[component.sampleKey] = value.to(device)
    return targets


def disableDropoutModules(module: nn.Module) -> int:
    """Set every dropout probability in ``module`` to zero."""
    updated = 0
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = 0.0
            updated += 1
    return updated


def _prepareDiffusionInputs(
    batch: BatchDict,
    motion: torch.Tensor,
    ddim: DDIM,
    device: torch.device,
    deterministicCorruption: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build timesteps/noise pairs for one batch."""
    batchSize = motion.shape[0]
    if not deterministicCorruption:
        timesteps = torch.randint(
            0,
            ddim.num_timesteps,
            (batchSize,),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(motion)
        return timesteps, noise, ddim.q_sample(motion, timesteps, noise)

    sampleIds = _resolveDeterministicSampleIds(batch, batchSize)
    timesteps = _deterministicTimesteps(
        sampleIds=sampleIds,
        numTimesteps=ddim.num_timesteps,
        device=device,
    )
    noise = _deterministicNoise(
        sampleIds=sampleIds,
        motionShape=motion.shape,
        dtype=motion.dtype,
        device=device,
    )
    return timesteps, noise, ddim.q_sample(motion, timesteps, noise)


def _resolveDeterministicSampleIds(
    batch: BatchDict,
    batchSize: int,
) -> list[int]:
    """Return stable per-sample ids used to seed deterministic corruption."""
    sampleIds = batch.get("sample_id")
    if isinstance(sampleIds, torch.Tensor) and sampleIds.numel() == batchSize:
        return [int(value) for value in sampleIds.detach().cpu().view(-1)]
    return list(range(batchSize))


def _deterministicTimesteps(
    sampleIds: list[int],
    numTimesteps: int,
    device: torch.device,
) -> torch.Tensor:
    """Map sample ids to stable diffusion timesteps."""
    values = [
        ((int(sampleId) * 1103515245 + 12345) & 0x7FFFFFFF) % numTimesteps
        for sampleId in sampleIds
    ]
    return torch.tensor(values, device=device, dtype=torch.long)


def _deterministicNoise(
    sampleIds: list[int],
    motionShape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build stable Gaussian noise per sample on CPU and move it to device."""
    sampleShape = tuple(motionShape[1:])
    noiseSamples: list[torch.Tensor] = []
    for sampleId in sampleIds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17_171 + int(sampleId))
        sampleNoise = torch.randn(
            sampleShape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        noiseSamples.append(sampleNoise)
    noise = torch.stack(noiseSamples, dim=0)
    return noise.to(device=device, dtype=dtype)


def computeMotionStatistics(
    dataloader: Iterable[BatchDict],
    device: torch.device,
    maxBatches: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean/std of motion tensors for Z-normalization.

    Returns tensors shaped ``(1, 1, bones, 6)`` suitable for broadcasting.
    """
    count = 0
    runningSum: Optional[torch.Tensor] = None
    runningSumSq: Optional[torch.Tensor] = None

    for i, batch in enumerate(dataloader):
        if i >= maxBatches:
            break
        motion = batch["motion"]  # (B, F, bones, 6)
        # Flatten to (N, bones, 6)
        flat = motion.reshape(-1, motion.shape[2], motion.shape[3]).float()
        if runningSum is None:
            runningSum = flat.sum(dim=0)
            runningSumSq = (flat ** 2).sum(dim=0)
        else:
            runningSum = runningSum + flat.sum(dim=0)
            runningSumSq = runningSumSq + (flat ** 2).sum(dim=0)
        count += flat.shape[0]

    if count == 0 or runningSum is None or runningSumSq is None:
        raise RuntimeError("Cannot compute statistics on an empty dataset.")
    mean = runningSum / count
    std = torch.sqrt(runningSumSq / count - mean ** 2).clamp(min=1e-5)
    # Reshape to (1, 1, bones, 6) for broadcasting
    return mean.unsqueeze(0).unsqueeze(0), std.unsqueeze(0).unsqueeze(0)


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
            "prediction_target": PREDICTION_TARGET_X0,
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
    checkpointPredictionTarget = checkpoint.get("prediction_target")
    if checkpointPredictionTarget is None:
        raise RuntimeError(
            "Checkpoint is missing prediction_target metadata. "
            "Older epsilon checkpoints are not supported by the "
            "x0-only generation pipeline."
        )
    checkpointPredictionTarget = str(
        checkpointPredictionTarget
    ).strip().lower()
    if checkpointPredictionTarget != PREDICTION_TARGET_X0:
        raise RuntimeError(
            "Checkpoint prediction_target "
            f"{checkpointPredictionTarget!r} is not supported. "
            "Only x0 checkpoints can be loaded."
        )

    # Try to load full model state first, fallback to denoiser only
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "denoiser_state_dict" in checkpoint:
        model.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as error:
            raise RuntimeError(
                "Optimizer state is incompatible with the current generation "
                "parameter set."
            ) from error

    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))
