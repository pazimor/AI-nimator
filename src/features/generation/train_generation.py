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
            # Combine epoch + batch index into a unique step counter used by
            # deterministic corruption.  Same (epoch, batch) → same (t, noise)
            # pair, but successive batches and epochs cover every timestep,
            # which is required for DDIM inference to work after overfit.
            globalStep = (max(int(epoch) - 1, 0) * 10_000_000) + numBatches
            lossValue, lossComponents = _runBatchAccumulate(
                batch,
                model,
                ddim,
                device,
                gradientAccumulation,
                deterministicCorruption=deterministicCorruption,
                stepCounter=globalStep,
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
    # Assemble combined bone and global feature tensors.
    boneFeatures = model.assembleBoneFeatures(batch, device)
    globalFeatures = model.assembleGlobalFeatures(batch, device)
    motionMask = batch.get("motion_mask")
    if motionMask is not None:
        motionMask = motionMask.to(device)
    clipMotionContext = model.clip.extractMotionContext(batch)

    # Z-normalize for diffusion; raw features stay as loss targets.
    normalizedBone = model.normalizeMotion(boneFeatures)
    normalizedGlobal = (
        model.normalizeGlobalFeatures(globalFeatures)
        if globalFeatures is not None
        else None
    )

    timesteps, noise, noisyMotion = _prepareDiffusionInputs(
        batch=batch,
        motion=normalizedBone,
        ddim=ddim,
        device=device,
        deterministicCorruption=deterministicCorruption,
    )
    noisyGlobal: torch.Tensor | None = None
    if normalizedGlobal is not None:
        globalNoise = torch.randn_like(normalizedGlobal)
        noisyGlobal = ddim.q_sample(normalizedGlobal, timesteps, globalNoise)

    # Delete batch reference early
    del batch

    # Predict noise
    outputs = model(
        textEmbedding=textEmbedding,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetMotion=boneFeatures,
        motionMask=motionMask,
        clipMotionContext=clipMotionContext,
        noisyGlobalFeatures=noisyGlobal,
        targetGlobalFeatures=globalFeatures,
    )
    
    # Delete inputs early
    del (
        textEmbedding,
        noisyMotion,
        boneFeatures,
        motionMask,
        clipMotionContext,
        globalFeatures,
        noisyGlobal,
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
    stepCounter: int = 0,
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
    # Assemble combined bone and global feature tensors.
    boneFeatures = model.assembleBoneFeatures(batch, device)
    globalFeatures = model.assembleGlobalFeatures(batch, device)
    motionMask = batch.get("motion_mask")
    if motionMask is not None:
        motionMask = motionMask.to(device)
    clipMotionContext = model.clip.extractMotionContext(batch)

    # Z-normalize for diffusion; raw features stay as loss targets.
    normalizedBone = model.normalizeMotion(boneFeatures)
    normalizedGlobal = (
        model.normalizeGlobalFeatures(globalFeatures)
        if globalFeatures is not None
        else None
    )

    timesteps, noise, noisyMotion = _prepareDiffusionInputs(
        batch=batch,
        motion=normalizedBone,
        ddim=ddim,
        device=device,
        deterministicCorruption=deterministicCorruption,
        stepCounter=stepCounter,
    )
    noisyGlobal: torch.Tensor | None = None
    if normalizedGlobal is not None:
        # Keep global noise in sync with bone noise: deterministic when
        # requested, independent Gaussian otherwise.  Using the same (t,
        # stepCounter) reproducibility rule ensures all feature scopes
        # travel the same corruption trajectory during an overfit run.
        if deterministicCorruption:
            _, globalNoise = _seededTimestepsAndNoise(
                sampleIds=_resolveDeterministicSampleIds(
                    batch, normalizedGlobal.shape[0],
                ),
                numTimesteps=ddim.num_timesteps,
                motionShape=normalizedGlobal.shape,
                dtype=normalizedGlobal.dtype,
                device=device,
                # Shift the step counter so bone and global seeds differ
                # and the two scopes do not share identical noise patterns.
                stepCounter=stepCounter + 1,
            )
        else:
            globalNoise = torch.randn_like(normalizedGlobal)
        noisyGlobal = ddim.q_sample(normalizedGlobal, timesteps, globalNoise)

    del batch

    # Predict noise
    outputs = model(
        textEmbedding=textEmbedding,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetMotion=boneFeatures,
        motionMask=motionMask,
        clipMotionContext=clipMotionContext,
        noisyGlobalFeatures=noisyGlobal,
        targetGlobalFeatures=globalFeatures,
    )
    
    del (
        textEmbedding,
        noisyMotion,
        boneFeatures,
        motionMask,
        clipMotionContext,
        globalFeatures,
        noisyGlobal,
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
            boneFeatures = model.assembleBoneFeatures(batch, device)
            globalFeatures = model.assembleGlobalFeatures(batch, device)
            motionMask = batch.get("motion_mask")
            if motionMask is not None:
                motionMask = motionMask.to(device)
            clipMotionContext = model.clip.extractMotionContext(batch)

            normalizedBone = model.normalizeMotion(boneFeatures)
            normalizedGlobal = (
                model.normalizeGlobalFeatures(globalFeatures)
                if globalFeatures is not None
                else None
            )

            timesteps, noise, noisyMotion = _prepareDiffusionInputs(
                batch=batch,
                motion=normalizedBone,
                ddim=ddim,
                device=device,
                deterministicCorruption=deterministicCorruption,
            )
            noisyGlobal: torch.Tensor | None = None
            if normalizedGlobal is not None:
                globalNoise = torch.randn_like(normalizedGlobal)
                noisyGlobal = ddim.q_sample(normalizedGlobal, timesteps, globalNoise)

            outputs = model(
                textEmbedding=textEmbedding,
                noisyMotion=noisyMotion,
                timesteps=timesteps,
                targetMotion=boneFeatures,
                motionMask=motionMask,
                clipMotionContext=clipMotionContext,
                noisyGlobalFeatures=noisyGlobal,
                targetGlobalFeatures=globalFeatures,
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
                boneFeatures,
                noisyMotion,
                noise,
                timesteps,
                outputs,
                motionMask,
                clipMotionContext,
                globalFeatures,
                noisyGlobal,
            )

    gc.collect()
    model.train()
    avgLoss = totalLoss / max(numBatches, 1)
    avgComponents = _averageLossComponents(componentSums, numBatches)
    return avgLoss, avgComponents


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
    stepCounter: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build timesteps/noise pairs for one batch.

    When ``deterministicCorruption`` is True, corruption is reproducible
    for a given ``(sampleId, stepCounter)`` pair but **varies across
    training steps**.  This is critical: the previous implementation hashed
    ``sampleId`` only, which meant overfit runs on a fixed sample saw the
    exact same timestep and noise forever -- the denoiser never learned to
    predict x0 at other timesteps, so DDIM inference collapsed to a noisy
    "mean pose".  Mixing in ``stepCounter`` preserves reproducibility while
    guaranteeing full timestep coverage over time.
    """
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
    timesteps, noise = _seededTimestepsAndNoise(
        sampleIds=sampleIds,
        numTimesteps=ddim.num_timesteps,
        motionShape=motion.shape,
        dtype=motion.dtype,
        device=device,
        stepCounter=stepCounter,
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


def _seededTimestepsAndNoise(
    sampleIds: list[int],
    numTimesteps: int,
    motionShape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    stepCounter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (timestep, noise) reproducibly from (sampleId, stepCounter).

    Each ``(sampleId, stepCounter)`` pair seeds a dedicated CPU generator so
    the same ``stepCounter`` twice produces identical corruption (useful for
    stable loss curves) while successive steps cycle through different
    timesteps and noises (required for DDIM inference to work).
    """
    sampleShape = tuple(motionShape[1:])
    timestepsList: list[int] = []
    noiseList: list[torch.Tensor] = []
    for sampleId in sampleIds:
        seed = (
            int(sampleId) * 2654435761
            + int(stepCounter) * 1103515245
            + 12345
        ) & 0x7FFFFFFF
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        timestep = int(
            torch.randint(
                0, numTimesteps, (1,), generator=generator,
            ).item()
        )
        sampleNoise = torch.randn(
            sampleShape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        timestepsList.append(timestep)
        noiseList.append(sampleNoise)
    timesteps = torch.tensor(timestepsList, device=device, dtype=torch.long)
    noise = torch.stack(noiseList, dim=0).to(device=device, dtype=dtype)
    return timesteps, noise


def computeMotionStatistics(
    dataloader: Iterable[BatchDict],
    device: torch.device,
    maxBatches: int = 500,
    model: Optional[MotionGenerator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean/std of bone feature tensors for Z-normalization.

    When *model* is provided the statistics are computed over the full
    assembled bone feature tensor (all enabled bone-scoped components).
    Otherwise falls back to the legacy ``batch["motion"]`` key.

    Returns tensors shaped ``(1, 1, bones, boneChannels)`` suitable for
    broadcasting.
    """
    count = 0
    runningSum: Optional[torch.Tensor] = None
    runningSumSq: Optional[torch.Tensor] = None

    for i, batch in enumerate(dataloader):
        if i >= maxBatches:
            break
        if model is not None and model.generationMotionComponents:
            motion = model.assembleBoneFeatures(batch, device)
        else:
            motion = batch["motion"]  # (B, F, bones, 6)
        # Flatten to (N, bones, channels)
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
    # Reshape to (1, 1, bones, channels) for broadcasting
    return mean.unsqueeze(0).unsqueeze(0), std.unsqueeze(0).unsqueeze(0)


def computeGlobalStatistics(
    dataloader: Iterable[BatchDict],
    device: torch.device,
    model: MotionGenerator,
    maxBatches: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean/std of global feature tensors.

    Returns tensors shaped ``(1, 1, globalChannels)``.
    """
    count = 0
    runningSum: Optional[torch.Tensor] = None
    runningSumSq: Optional[torch.Tensor] = None

    for i, batch in enumerate(dataloader):
        if i >= maxBatches:
            break
        globalFeatures = model.assembleGlobalFeatures(batch, device)
        if globalFeatures is None:
            break
        flat = globalFeatures.reshape(-1, globalFeatures.shape[-1]).float()
        if runningSum is None:
            runningSum = flat.sum(dim=0)
            runningSumSq = (flat ** 2).sum(dim=0)
        else:
            runningSum = runningSum + flat.sum(dim=0)
            runningSumSq = runningSumSq + (flat ** 2).sum(dim=0)
        count += flat.shape[0]

    if count == 0 or runningSum is None or runningSumSq is None:
        raise RuntimeError("Cannot compute global statistics on an empty dataset.")
    mean = runningSum / count
    std = torch.sqrt(runningSumSq / count - mean ** 2).clamp(min=1e-5)
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
