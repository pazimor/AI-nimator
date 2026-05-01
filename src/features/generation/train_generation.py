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

from src.shared.checkpoint_io import saveTorchObjectAtomically
from src.shared import diagnostics as diag
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
from src.shared.model.generation.ema import ExponentialMovingAverage
from src.shared.model.generation.motion_generator import MotionGenerator
from src.shared.progress import TrainingProgressBar

BatchDict = Mapping[str, object]
LossComponents = dict[str, float]
LOGGER = logging.getLogger("generation.train")
LOSS_COMPONENT_KEYS = (
    "loss_diffusion",
    "loss_bone_diffusion",
    "loss_global_diffusion",
    "loss_xyz",
    "loss_vel_xyz",
    "loss_acceleration",
    "loss_clip_guidance",
    "loss_root_translation",
    "loss_root_velocity",
    "loss_joint_xyz",
    "loss_joint_velocity",
    "loss_foot_contact",
    "loss_end_effector_velocity",
    "loss_components",
    "loss_foot_skating",
)
LOSS_COMPONENT_LABELS = {
    "loss_diffusion": "diff",
    "loss_bone_diffusion": "bone_d",
    "loss_global_diffusion": "glob_d",
    "loss_xyz": "xyz",
    "loss_vel_xyz": "vel_xyz",
    "loss_acceleration": "acc",
    "loss_clip_guidance": "clip",
    "loss_root_translation": "rtrans",
    "loss_root_velocity": "rvel",
    "loss_joint_xyz": "jxyz",
    "loss_joint_velocity": "jvel",
    "loss_foot_contact": "fcont",
    "loss_end_effector_velocity": "eevel",
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
    weightDecay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Create optimizer for trainable parameters only.

    Parameters
    ----------
    model : MotionGenerator
        The generation model.
    learningRate : float
        Learning rate.
    weightDecay : float
        L2 weight decay passed to AdamW. 0.0 disables regularization
        (historical default).  1e-4 is a conservative value that narrows
        the train/val gap on small datasets.

    Returns
    -------
    torch.optim.Optimizer
        AdamW optimizer for denoiser parameters.
    """
    # Train the denoiser and any auxiliary generation heads (CLIP is frozen).
    trainableParams = list(model.trainableParameters())
    return torch.optim.AdamW(
        trainableParams,
        lr=learningRate,
        weight_decay=float(weightDecay),
    )


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
    clearMpsCache: bool = True,
    deterministicCorruption: bool = False,
    ema: Optional[ExponentialMovingAverage] = None,
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
    clearMpsCache : bool
        When False, skip explicit torch.mps.empty_cache() calls.
    ema : Optional[ExponentialMovingAverage]
        When provided, the EMA shadow is updated after every real
        optimizer step (i.e. after gradient accumulation completes).

    Returns
    -------
    tuple[float, LossComponents]
        Average training loss and average component losses.
    """
    from src.shared.dataset_manager import MemoryManager, MemoryManagerConfig
    
    model.train()
    numBatches = 0
    componentSums = _initLossComponents()
    # Per-bucket loss accumulators (10 buckets of diffusion timesteps).
    bucketSums: dict[int, float] = {}
    bucketCounts: dict[int, int] = {}
    numBuckets = 10
    
    # Setup memory manager for status logging / cache eviction.
    memoryConfig = MemoryManagerConfig(clearMpsCache=clearMpsCache)
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
            lossValue, lossComponents, batchTimesteps = _runBatchAccumulate(
                batch,
                model,
                ddim,
                device,
                gradientAccumulation,
                deterministicCorruption=deterministicCorruption,
                stepCounter=globalStep,
                epoch=epoch,
                batchIdx=numBatches,
            )
            pbar.updateLoss(lossValue)
            _updateLossComponents(componentSums, lossComponents)
            # Aggregate per-timestep-bucket loss (bucket = t // (T/numBuckets)).
            if batchTimesteps:
                bucketWidth = max(ddim.num_timesteps // numBuckets, 1)
                for t in batchTimesteps:
                    bucket = min(int(t) // bucketWidth, numBuckets - 1)
                    bucketSums[bucket] = bucketSums.get(bucket, 0.0) + lossValue
                    bucketCounts[bucket] = bucketCounts.get(bucket, 0) + 1
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
                # Advance the EMA shadow AFTER every real optimizer step
                # (never on grad-accumulation micro-steps): the shadow must
                # track the trajectory of actually-applied updates.
                if ema is not None:
                    ema.update(model.trainableParameters())
                accumSteps = 0
                
                # Clear MPS cache after optimizer step
                if clearMpsCache and device.type == "mps":
                    torch.mps.empty_cache()

        gc.collect()
        if clearMpsCache and device.type == "mps":
            torch.mps.empty_cache()
        avgComponents = _averageLossComponents(componentSums, numBatches)
        if diag.get_logger() is not None:
            bucketAverages = {
                bucket: {
                    "avg_loss": bucketSums[bucket] / max(bucketCounts[bucket], 1),
                    "count": bucketCounts[bucket],
                }
                for bucket in sorted(bucketSums.keys())
            }
            # valLoss is intentionally None here: validation runs *after*
            # this training-epoch summary in the CLI loop, and is logged
            # separately via diagnostics.logValSummary so each event carries
            # the metrics available at its own emission point.  Downstream
            # post-processing should join epoch_summary + val_summary on
            # the `epoch` key rather than expecting val_loss on this record.
            diag.logEpochSummary(
                epoch=epoch,
                trainLoss=pbar.metrics.avgLoss,
                valLoss=None,
                components=avgComponents,
                timestepBuckets=bucketAverages,
                learningRate=float(
                    optimizer.param_groups[0]["lr"]
                    if optimizer.param_groups
                    else 0.0
                ),
            )
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

    # Move data to device.  Cross-attention requires the per-token text
    # representation, which the cached pooled ``generation_text_embedding``
    # doesn't contain.  Feed raw tokens; ``MotionGenerator.forward`` runs
    # CLIP fresh (no_grad, frozen text encoder) and produces both the
    # pooled vector and the per-token sequence in one pass.
    textInputIds = batch["input_ids"].to(device)
    textAttentionMask = batch["attention_mask"].to(device)
    # CFG dropout mask: applied AFTER the CLIP encode by
    # ``MotionGenerator.forward`` so pooled + tokens + key-padding mask
    # are zeroed in lockstep on the dropped samples.  Before the cross-
    # attention refactor this dropout was applied to the cached pooled
    # vector directly, which no longer drives the full conditioning path.
    condDropoutMask: Optional[torch.Tensor] = None
    if model.training and model.condMaskProb > 0.0:
        condDropoutMask = (
            torch.rand(textInputIds.shape[0], device=device)
            < model.condMaskProb
        )
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
        textInputIds=textInputIds,
        textAttentionMask=textAttentionMask,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetMotion=boneFeatures,
        motionMask=motionMask,
        clipMotionContext=clipMotionContext,
        noisyGlobalFeatures=noisyGlobal,
        targetGlobalFeatures=globalFeatures,
        condDropoutMask=condDropoutMask,
    )

    # Delete inputs early
    del (
        textInputIds,
        textAttentionMask,
        condDropoutMask,
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
    epoch: int = 1,
    batchIdx: int = 0,
) -> tuple[float, LossComponents, list[int]]:
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
    # Move data to device.  Same fresh-CLIP path as ``trainOneEpoch`` —
    # cross-attention requires the per-token text representation that the
    # cached pooled embedding doesn't carry.
    textInputIds = batch["input_ids"].to(device)
    textAttentionMask = batch["attention_mask"].to(device)
    # CFG dropout is applied post-encode by ``MotionGenerator.forward``.
    condDropoutMask: Optional[torch.Tensor] = None
    if model.training and model.condMaskProb > 0.0:
        condDropoutMask = (
            torch.rand(textInputIds.shape[0], device=device)
            < model.condMaskProb
        )
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
        textInputIds=textInputIds,
        textAttentionMask=textAttentionMask,
        noisyMotion=noisyMotion,
        timesteps=timesteps,
        targetMotion=boneFeatures,
        motionMask=motionMask,
        clipMotionContext=clipMotionContext,
        noisyGlobalFeatures=noisyGlobal,
        targetGlobalFeatures=globalFeatures,
        condDropoutMask=condDropoutMask,
    )

    diagActive = diag.get_logger() is not None
    predictedMotionForDiag = (
        outputs.get("predicted_motion").detach()
        if diagActive and outputs.get("predicted_motion") is not None
        else None
    )
    normalizedBoneForDiag = normalizedBone.detach() if diagActive else None
    targetMotionForDiag = boneFeatures.detach() if diagActive else None
    noisyMotionForDiag = noisyMotion.detach() if diagActive else None
    noiseForDiag = noise.detach() if diagActive else None

    del (
        textInputIds,
        textAttentionMask,
        condDropoutMask,
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
    timestepsList = [int(v) for v in timesteps.detach().cpu().view(-1).tolist()]

    if diagActive:
        gradNorm = diag.computeGradientNorm(model.trainableParameters())
        diag.logTrainBatch(
            epoch=epoch,
            batchIdx=batchIdx,
            globalStep=stepCounter,
            timesteps=timesteps.detach(),
            loss=lossValue,
            components=lossComponents,
            normalizedTarget=normalizedBoneForDiag,
            noisyInput=noisyMotionForDiag,
            noise=noiseForDiag,
            predictedMotion=predictedMotionForDiag,
            targetMotion=targetMotionForDiag,
            gradNorm=gradNorm,
        )

    del noise, timesteps, loss, scaledLoss

    return lossValue, lossComponents, timestepsList


def evaluateValidation(
    dataloader: DataLoader,
    model: MotionGenerator,
    ddim: DDIM,
    device: torch.device,
    deterministicCorruption: bool = False,
    ema: Optional[ExponentialMovingAverage] = None,
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
    ema : Optional[ExponentialMovingAverage]
        When provided, the model's online parameters are swapped with the
        EMA shadow for the duration of validation (then restored).  This
        is what produces the stable, inference-aligned val_loss series.

    Returns
    -------
    tuple[float, LossComponents]
        Average validation loss and component breakdown.
    """
    # Swap EMA weights into the model BEFORE eval() / no_grad() — the swap
    # itself is already under torch.no_grad via the EMA manager.  The
    # try/finally guarantees the online parameters are always restored, so
    # the next training step resumes from the correct trajectory even if
    # validation raises.
    if ema is not None:
        ema.storeAndSwap(model.trainableParameters())
    try:
        model.eval()
        totalLoss = 0.0
        numBatches = 0
        componentSums = _initLossComponents()

        with torch.no_grad():
            for batch in dataloader:
                # Validation runs the same fresh-CLIP path so the
                # cross-attention conditioning is identical to training.
                # No CFG dropout in eval mode.
                textInputIds = batch["input_ids"].to(device)
                textAttentionMask = batch["attention_mask"].to(device)
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
                    noisyGlobal = ddim.q_sample(
                        normalizedGlobal, timesteps, globalNoise,
                    )

                outputs = model(
                    textInputIds=textInputIds,
                    textAttentionMask=textAttentionMask,
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
                    textInputIds,
                    textAttentionMask,
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
    finally:
        # Always restore online weights, even if the validation loop raised.
        # Training in the next epoch must resume from the online trajectory,
        # not from the EMA shadow.
        if ema is not None:
            ema.restore(model.trainableParameters())


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

    Padded frames are excluded via ``batch["motion_mask"]`` when available so
    the statistics are not biased toward zero.  Including padding used to
    under-estimate ``motion_std`` and shrink every feature toward the mean,
    which made the normalized diffusion target a tiny fraction of a unit and
    starved the denoiser of learning signal at high timesteps.

    Returns tensors shaped ``(1, 1, bones, boneChannels)`` suitable for
    broadcasting.
    """
    count = 0.0
    runningSum: Optional[torch.Tensor] = None
    runningSumSq: Optional[torch.Tensor] = None

    for i, batch in enumerate(dataloader):
        if i >= maxBatches:
            break
        if model is not None and model.generationMotionComponents:
            motion = model.assembleBoneFeatures(batch, device)
        else:
            motion = batch["motion"]  # (B, F, bones, 6)
        motionFloat = motion.float()
        mask = batch.get("motion_mask")
        if isinstance(mask, torch.Tensor):
            # (B, F) → (B, F, 1, 1) for broadcasting over bones/channels.
            frameMask = mask.to(device=motionFloat.device).float()
            frameMask = frameMask.unsqueeze(-1).unsqueeze(-1)
            weighted = motionFloat * frameMask
            weightedSq = (motionFloat ** 2) * frameMask
            batchSum = weighted.sum(dim=(0, 1))
            batchSumSq = weightedSq.sum(dim=(0, 1))
            batchCount = frameMask.sum().item()
        else:
            flat = motionFloat.reshape(-1, motion.shape[2], motion.shape[3])
            batchSum = flat.sum(dim=0)
            batchSumSq = (flat ** 2).sum(dim=0)
            batchCount = float(flat.shape[0])
        if runningSum is None:
            runningSum = batchSum
            runningSumSq = batchSumSq
        else:
            runningSum = runningSum + batchSum
            runningSumSq = runningSumSq + batchSumSq
        count += batchCount

    if count == 0 or runningSum is None or runningSumSq is None:
        raise RuntimeError("Cannot compute statistics on an empty dataset.")
    mean = runningSum / count
    variance = (runningSumSq / count - mean ** 2).clamp(min=0.0)
    std = torch.sqrt(variance).clamp(min=1e-5)
    # Reshape to (1, 1, bones, channels) for broadcasting
    return mean.unsqueeze(0).unsqueeze(0), std.unsqueeze(0).unsqueeze(0)


def computeGlobalStatistics(
    dataloader: Iterable[BatchDict],
    device: torch.device,
    model: MotionGenerator,
    maxBatches: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean/std of global feature tensors.

    Padded frames are excluded via ``batch["motion_mask"]`` when available.

    Channels belonging to components with ``skipNormalization = True``
    (e.g. ``FootContactComponent``) are forced to mean=0 / std=1 so the
    denoiser sees raw values instead of z-normalized ones.

    Returns tensors shaped ``(1, 1, globalChannels)``.
    """
    count = 0.0
    runningSum: Optional[torch.Tensor] = None
    runningSumSq: Optional[torch.Tensor] = None

    for i, batch in enumerate(dataloader):
        if i >= maxBatches:
            break
        globalFeatures = model.assembleGlobalFeatures(batch, device)
        if globalFeatures is None:
            break
        globalFloat = globalFeatures.float()
        mask = batch.get("motion_mask")
        if isinstance(mask, torch.Tensor):
            frameMask = mask.to(device=globalFloat.device).float()
            frameMask = frameMask.unsqueeze(-1)
            weighted = globalFloat * frameMask
            weightedSq = (globalFloat ** 2) * frameMask
            batchSum = weighted.sum(dim=(0, 1))
            batchSumSq = weightedSq.sum(dim=(0, 1))
            batchCount = frameMask.sum().item()
        else:
            flat = globalFloat.reshape(-1, globalFeatures.shape[-1])
            batchSum = flat.sum(dim=0)
            batchSumSq = (flat ** 2).sum(dim=0)
            batchCount = float(flat.shape[0])
        if runningSum is None:
            runningSum = batchSum
            runningSumSq = batchSumSq
        else:
            runningSum = runningSum + batchSum
            runningSumSq = runningSumSq + batchSumSq
        count += batchCount

    if count == 0 or runningSum is None or runningSumSq is None:
        raise RuntimeError("Cannot compute global statistics on an empty dataset.")
    mean = runningSum / count
    variance = (runningSumSq / count - mean ** 2).clamp(min=0.0)
    std = torch.sqrt(variance).clamp(min=1e-5)

    # Force identity normalization (mean=0, std=1) on channels whose
    # component has skipNormalization=True.  This lets those channels
    # pass through the z-norm/denorm unchanged so the denoiser sees
    # raw values (e.g. binary {0,1} for foot_contact).
    offset = 0
    for component in model._globalComponents:
        if getattr(component, "skipNormalization", False):
            mean[offset:offset + component.channels] = 0.0
            std[offset:offset + component.channels] = 1.0
            LOGGER.info(
                "Global stats: skipping z-norm for %s (channels %d-%d) "
                "→ mean=0, std=1 (identity).",
                component.key,
                offset,
                offset + component.channels - 1,
            )
        offset += component.channels

    return mean.unsqueeze(0).unsqueeze(0), std.unsqueeze(0).unsqueeze(0)


def saveCheckpoint(
    model: MotionGenerator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpointDir: Path,
    filename: str = "best_model.pt",
    ema: Optional[ExponentialMovingAverage] = None,
) -> Path:
    """
    Save model checkpoint.

    When ``ema`` is provided, the EMA shadow weights are written as
    ``model_state_dict`` (so ``generate_animation.py`` loads them by
    default) and the true online weights are preserved under
    ``online_state_dict`` so the next ``loadCheckpoint`` call can resume
    training from the exact trajectory.

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
    ema : Optional[ExponentialMovingAverage]
        When provided, the EMA shadow is saved alongside the online state
        and the primary ``model_state_dict`` key points at the EMA weights.

    Returns
    -------
    Path
        Path to saved checkpoint.
    """
    onlineStateDict = model.state_dict()
    denoiserStateDict = model.denoiser.state_dict()
    payload: dict[str, object] = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "prediction_target": PREDICTION_TARGET_X0,
    }
    if ema is not None:
        # Materialise the EMA weights as a full model.state_dict() shape by
        # swapping in-place, snapshotting, then restoring — no second model
        # instance, no DDP quirks.
        ema.storeAndSwap(model.trainableParameters())
        try:
            emaAsStateDict = model.state_dict()
            emaDenoiserStateDict = model.denoiser.state_dict()
        finally:
            ema.restore(model.trainableParameters())
        payload["model_state_dict"] = emaAsStateDict
        payload["denoiser_state_dict"] = emaDenoiserStateDict
        payload["online_state_dict"] = onlineStateDict
        payload["online_denoiser_state_dict"] = denoiserStateDict
        payload["ema_state_dict"] = ema.stateDict()
    else:
        payload["model_state_dict"] = onlineStateDict
        payload["denoiser_state_dict"] = denoiserStateDict

    checkpointPath = checkpointDir / filename
    saveTorchObjectAtomically(payload, checkpointPath)

    return checkpointPath


def loadCheckpoint(
    checkpointPath: Path,
    model: MotionGenerator,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
) -> Tuple[int, float]:
    """
    Load model checkpoint.

    When ``ema`` is provided (training resume), the EMA shadow is
    restored from ``ema_state_dict`` and the online weights are loaded
    from ``online_state_dict`` — the true training trajectory, not the
    EMA-smoothed snapshot that lives under ``model_state_dict``.  When
    ``ema`` is ``None`` (inference), the primary ``model_state_dict`` is
    used directly, which already contains the EMA weights if the
    checkpoint was produced with EMA enabled.

    Parameters
    ----------
    checkpointPath : Path
        Path to checkpoint file.
    model : MotionGenerator
        Model to load into.
    optimizer : Optional[torch.optim.Optimizer], optional
        Optimizer to load state into.
    ema : Optional[ExponentialMovingAverage], optional
        EMA manager to restore.  Signals "this is a training resume".

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

    # Resume logic:
    #  - Training resume (ema given): prefer online_state_dict so the next
    #    training step picks up the exact trajectory; fall back to the old
    #    flat model_state_dict for backward compat with pre-EMA checkpoints.
    #  - Inference (ema is None): load model_state_dict directly.  On an
    #    EMA-aware checkpoint that is already the EMA-smoothed snapshot,
    #    which is what we want at generation time.
    # ``strict=False`` on every branch keeps old checkpoints loadable after
    # architecture upgrades (new / dropped parameters tolerated).
    if ema is not None and "online_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["online_state_dict"], strict=False,
        )
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "denoiser_state_dict" in checkpoint:
        model.denoiser.load_state_dict(
            checkpoint["denoiser_state_dict"], strict=False,
        )

    # After loading checkpoint weights, enforce skipNormalization on global
    # stats buffers.  Old checkpoints may contain stale z-norm stats for
    # components that now opt out (e.g. foot_contact BCE fix).  This forces
    # mean=0 / std=1 for those channels so the denoiser sees raw values.
    model.enforceSkipNormalization()

    # Restore EMA shadow when we are resuming training.
    if ema is not None:
        if "ema_state_dict" in checkpoint:
            try:
                ema.loadStateDict(checkpoint["ema_state_dict"])
            except RuntimeError as error:
                LOGGER.warning(
                    "EMA state in checkpoint is incompatible (%s); "
                    "re-initialising EMA from the loaded online weights.",
                    error,
                )
                ema.copyTo(model.trainableParameters())  # no-op on params
        else:
            LOGGER.warning(
                "Checkpoint predates EMA support; re-initialising EMA "
                "shadow from the online weights."
            )
            # Re-seed shadow with the freshly loaded online params by
            # overwriting the internal shadow tensors.
            for shadow, param in zip(
                ema.shadow, list(model.trainableParameters())
            ):
                shadow.data.copy_(param.detach())

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as error:
            # A ValueError here typically means the model has gained or
            # lost parameters since the checkpoint (e.g. after an
            # architecture fix).  Restart the optimizer from scratch
            # rather than aborting -- re-initialising Adam moments is
            # preferable to losing the trained weights.
            LOGGER.warning(
                "Optimizer state incompatible with current parameters "
                "(%s); resuming model weights but reinitialising the "
                "optimizer state.",
                error,
            )

    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))
