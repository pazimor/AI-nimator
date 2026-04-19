"""Full motion generation pipeline combining CLIP and diffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn

from src.shared import diagnostics as diag
from src.shared.model.clip.core import ClipModel
from src.shared.model.components.base import MotionComponent, SCOPE_BONE, SCOPE_GLOBAL
from src.shared.model.components.registry import computeFeatureLayout
from src.shared.model.generation.ddim import DDIM
from src.shared.model.generation.denoiser import MotionDenoiser
from src.shared.model.generation.losses import (
    DEFAULT_VEL_XYZ_SCHEDULE,
    DEFAULT_VELOCITY_XYZ_WEIGHT,
    DEFAULT_XYZ_WEIGHT,
    XYZ_SCHEDULE_NONE,
)
from src.shared.model.layers.correction import (
    Renormalization,
    Smoothing,
    VelocityRegularization,
)
from src.shared.types.generation import PREDICTION_TARGET_X0


class MotionGenerator(nn.Module):
    """
    Complete motion generation pipeline.

    Combines frozen CLIP text encoder with trainable diffusion denoiser
    and post-processing correction layers.
    """
    MOTION_ROTATION_CHANNELS = 6
    ROOT_TRANSLATION_CHANNELS = 3

    def __init__(
        self,
        embedDim: int = 64,
        numHeads: int = 4,
        numLayers: int = 6,
        numBones: int = 65,
        diffusionSteps: int = 1000,
        modelName: str = "xlm-roberta-base",
        clipCheckpoint: Optional[Path] = None,
        smoothingKernel: int = 3,
        maxVelocity: Optional[float] = None,
        xyzWeight: float = DEFAULT_XYZ_WEIGHT,
        xyzWeightSchedule: str = XYZ_SCHEDULE_NONE,
        velXyzWeight: float = DEFAULT_VELOCITY_XYZ_WEIGHT,
        velXyzWeightSchedule: str = DEFAULT_VEL_XYZ_SCHEDULE,
        diffusionWeight: float = 1.0,
        accelerationWeight: float = 0.0,
        clipGuidanceWeight: float = 0.0,
        footSkatingWeight: float = 0.0,
        condMaskProb: float = 0.1,
        minSnrGamma: float = 5.0,
        numSpatialLayers: int = 1,
        numSpatioTemporalLayers: int = 1,
        maxPromptLength: int = 64,
        generationEmbedDim: Optional[int] = None,
        clipMotionNumHeads: int = 4,
        clipMotionNumLayers: int = 2,
        clipMotionComponents: Optional[tuple[MotionComponent, ...]] = None,
        generationMotionComponents: Optional[tuple[MotionComponent, ...]] = None,
    ) -> None:
        super().__init__()
        self.embedDim = embedDim
        self.generationEmbedDim = (
            embedDim
            if generationEmbedDim is None
            else int(generationEmbedDim)
        )
        self.predictionTarget = PREDICTION_TARGET_X0
        self.numBones = numBones
        self.diffusionSteps = diffusionSteps
        self.xyzWeight = xyzWeight
        self.xyzWeightSchedule = xyzWeightSchedule
        self.velXyzWeight = velXyzWeight
        self.velXyzWeightSchedule = velXyzWeightSchedule
        self.diffusionWeight = diffusionWeight
        self.accelerationWeight = accelerationWeight
        self.clipGuidanceWeight = clipGuidanceWeight
        # Foot-skating penalty is OFF by default -- it applies FK on noisy
        # rot6d predictions at every diffusion timestep, which produces huge,
        # chaotic gradients early in training.  Enable it only once the base
        # diffusion loss has converged to reasonable motion.
        self.footSkatingWeight = float(footSkatingWeight)
        self.condMaskProb = condMaskProb
        # Min-SNR gamma down-weights high-t x0-MSE samples that otherwise
        # dominate the gradient and pull predictions toward a temporally
        # flat mean pose.  See Hang et al. 2023.  Set to 0 to disable.
        self.minSnrGamma = float(minSnrGamma)
        self.maxPromptLength = max(1, int(maxPromptLength))
        self.generationMotionComponents = tuple(generationMotionComponents or ())

        # Compute feature layout from enabled components.
        if self.generationMotionComponents:
            boneChannels, globalChannels, boneComponents, globalComponents = (
                computeFeatureLayout(self.generationMotionComponents)
            )
        else:
            boneChannels = self.MOTION_ROTATION_CHANNELS
            globalChannels = 0
            boneComponents = ()
            globalComponents = ()

        self.boneChannels = boneChannels
        self.globalChannels = globalChannels
        self._boneComponents = boneComponents
        self._globalComponents = globalComponents
        self._generationComponentsBySampleKey = {
            component.sampleKey: component
            for component in self.generationMotionComponents
            if component.key != "rotation6d"
        }

        # Determine if rotation6d is present (needed for FK-based losses).
        self._hasRotation6d = any(
            c.key == "rotation6d" for c in self.generationMotionComponents
        ) or not self.generationMotionComponents
        # Channel offset of rotation6d within bone features (always first
        # when present due to stable component ordering).
        self._rotation6dOffset = 0
        self._rotation6dChannels = self.MOTION_ROTATION_CHANNELS

        # CLIP text encoder (frozen)
        self.clip = ClipModel(
            modelName=modelName,
            embedDim=embedDim,
            freezeTextEncoder=True,
            motionNumHeads=clipMotionNumHeads,
            motionNumLayers=clipMotionNumLayers,
            motionComponents=clipMotionComponents,
            numBones=numBones,
        )
        if clipCheckpoint is not None:
            self._loadClipCheckpoint(clipCheckpoint)
        self._freezeClip()

        # Diffusion components
        self.ddim = DDIM(num_timesteps=diffusionSteps)
        self.denoiser = MotionDenoiser(
            embedDim=self.generationEmbedDim,
            numHeads=numHeads,
            numLayers=numLayers,
            numBones=numBones,
            motionChannels=boneChannels,
            globalChannels=globalChannels,
            numSpatialLayers=numSpatialLayers,
            numSpatioTemporalLayers=numSpatioTemporalLayers,
            textEmbedDim=embedDim,
        )

        # Z-normalization buffers for bone features.
        self.register_buffer(
            "motion_mean",
            torch.zeros(1, 1, numBones, boneChannels),
        )
        self.register_buffer(
            "motion_std",
            torch.ones(1, 1, numBones, boneChannels),
        )
        # Z-normalization buffers for global features.
        if globalChannels > 0:
            self.register_buffer(
                "global_mean",
                torch.zeros(1, 1, globalChannels),
            )
            self.register_buffer(
                "global_std",
                torch.ones(1, 1, globalChannels),
            )

        # Post-processing (inference only)
        self.renorm = Renormalization()
        self.smoothing = Smoothing(channels=numBones * 6, kernel_size=smoothingKernel)
        self.velocityReg = VelocityRegularization(max_velocity=maxVelocity)

    def forward(
        self,
        textInputIds: Optional[torch.Tensor] = None,
        textAttentionMask: Optional[torch.Tensor] = None,
        noisyMotion: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        targetNoise: Optional[torch.Tensor] = None,
        targetMotion: Optional[torch.Tensor] = None,
        motionMask: Optional[torch.Tensor] = None,
        clipMotionContext: Optional[Mapping[str, object]] = None,
        textEmbedding: Optional[torch.Tensor] = None,
        componentTargets: Optional[Mapping[str, torch.Tensor]] = None,
        noisyGlobalFeatures: Optional[torch.Tensor] = None,
        targetGlobalFeatures: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Parameters
        ----------
        noisyMotion : Optional[torch.Tensor]
            Noisy bone-scoped features shaped (batch, frames, bones, boneChannels).
        timesteps : Optional[torch.Tensor]
            Diffusion timesteps shaped (batch,).
        targetMotion : Optional[torch.Tensor], optional
            Ground truth clean bone features for losses.
        motionMask : Optional[torch.Tensor], optional
            Boolean mask indicating valid (non-padded) frames.
        textEmbedding : Optional[torch.Tensor], optional
            Precomputed CLIP text embedding used during training.
        componentTargets : Optional[Mapping[str, torch.Tensor]], optional
            Auxiliary generation targets keyed by sample tensor name.
        noisyGlobalFeatures : Optional[torch.Tensor], optional
            Noisy global-scoped features shaped (batch, frames, globalChannels).
        targetGlobalFeatures : Optional[torch.Tensor], optional
            Clean global features for losses.
        """
        if noisyMotion is None or timesteps is None:
            raise ValueError("noisyMotion and timesteps are required.")
        if textEmbedding is not None:
            if textInputIds is not None or textAttentionMask is not None:
                raise ValueError(
                    "Provide either textEmbedding or tokenized inputs, not both."
                )
            textEmbeds = textEmbedding
        else:
            if textInputIds is None or textAttentionMask is None:
                raise ValueError(
                    "textEmbedding or tokenized text inputs are required."
                )
            with torch.no_grad():
                textEmbeds, _ = self.clip.encodeText(
                    inputIds=textInputIds,
                    attentionMask=textAttentionMask,
                )

        padMask = None
        if motionMask is not None:
            padMask = ~motionMask.bool()

        boneOutput, globalOutput = self.denoiser(
            noisyMotion=noisyMotion,
            textEmbedding=textEmbeds,
            timesteps=timesteps,
            mask=padMask,
            noisyGlobalFeatures=noisyGlobalFeatures,
        )

        # The denoiser's direct output is the predicted x0 in **normalized**
        # space (same space as its input).  Keep a reference before
        # denormalizing so the base diffusion MSE stays scale-invariant
        # across channels -- denormalizing before the MSE would implicitly
        # weight each channel by std^2, drowning low-variance channels like
        # rotation6d under high-variance ones (joint xyz in meters, root
        # translation, etc.) and collapsing the model towards a mean pose.
        normalizedBonePred = boneOutput
        normalizedGlobalPred = globalOutput

        # Resolve x0 predictions for bone features.
        predictedBoneNoise, predictedBoneMotion = self._resolveModelPredictions(
            noisyMotion=noisyMotion,
            timesteps=timesteps,
            modelOutput=boneOutput,
        )
        predictedBoneMotion = self.denormalizeMotion(predictedBoneMotion)

        # Resolve x0 predictions for global features.
        predictedGlobal: Optional[torch.Tensor] = None
        if globalOutput is not None and noisyGlobalFeatures is not None:
            _, predictedGlobal = self._resolveModelPredictions(
                noisyMotion=noisyGlobalFeatures,
                timesteps=timesteps,
                modelOutput=globalOutput,
            )
            predictedGlobal = self.denormalizeGlobalFeatures(predictedGlobal)

        result: dict[str, torch.Tensor] = {
            "predicted_noise": predictedBoneNoise,
            "predicted_motion": predictedBoneMotion,
        }
        if predictedGlobal is not None:
            result["predicted_global"] = predictedGlobal

        if targetMotion is not None:
            from src.shared.model.generation.losses import (
                combinedGenerationLoss,
                minSnrLossWeights,
                startMotionLoss,
            )

            # Compute per-sample Min-SNR weights once; they apply to every
            # x0-MSE term below so all diffusion objectives share the same
            # timestep reweighting.
            perSampleWeights: Optional[torch.Tensor] = None
            if self.minSnrGamma > 0.0 and timesteps is not None:
                perSampleWeights = minSnrLossWeights(
                    timesteps=timesteps,
                    alphasCumprod=self.ddim.alphas_cumprod,
                    gamma=self.minSnrGamma,
                )

            # Extract rotation6d for FK-based losses if available.
            rot6dPredicted = self._extractRotation6d(predictedBoneMotion)
            rot6dTarget = self._extractRotation6d(targetMotion)

            footContact = (
                componentTargets.get("foot_contact")
                if componentTargets
                else None
            )
            # Extract ground-truth foot contact from target global features
            # (not from the model prediction which is unreliable during training).
            if footContact is None and targetGlobalFeatures is not None:
                footContact = self._extractGlobalComponent(
                    targetGlobalFeatures, "foot_contact",
                )

            loss, components = combinedGenerationLoss(
                predictedMotion=rot6dPredicted,
                targetMotion=rot6dTarget,
                diffusionWeight=self.diffusionWeight,
                xyzWeight=self.xyzWeight if self._hasRotation6d else 0.0,
                xyzWeightSchedule=self.xyzWeightSchedule,
                velocityXyzWeight=self.velXyzWeight if self._hasRotation6d else 0.0,
                velXyzWeightSchedule=self.velXyzWeightSchedule,
                accelerationWeight=self.accelerationWeight,
                timesteps=timesteps,
                numTimesteps=self.ddim.num_timesteps,
                motionMask=motionMask,
                footContact=footContact,
                footSkatingWeight=self.footSkatingWeight,
                perSampleWeights=perSampleWeights,
            )

            # Component losses on bone features (excluding rotation6d
            # which is already covered by the diffusion loss above).
            boneSplits = self._splitBonePredictions(predictedBoneMotion)
            boneTargetSplits = self._splitBonePredictions(targetMotion)
            componentLoss, componentLosses = self._multiFeatureLoss(
                boneSplits, boneTargetSplits,
                predictedGlobal, targetGlobalFeatures,
                motionMask,
            )
            if componentLoss is not None:
                loss = loss + componentLoss
                result.update(componentLosses)

            # When the bone feature tensor includes channels beyond rotation6d
            # (e.g. joint_xyz, joint_velocity), the rotation6d-only diffusion
            # MSE from combinedGenerationLoss does not supervise those extra
            # channels.  Add a full-bone MSE so every denoised channel gets a
            # proper denoising signal — noisy auxiliary channels would
            # otherwise contaminate rotation6d predictions through the shared
            # transformer during DDIM sampling.
            #
            # Critical: compute this MSE in **normalized** space so every
            # channel contributes with unit variance.  Before this change the
            # loss was computed on denormalized predictions, which scaled each
            # channel's gradient by std^2 and let meter-scale features
            # (joint_xyz, root translation) dominate rotation channels,
            # biasing the model toward a mean pose.
            if (
                self._hasRotation6d
                and self.boneChannels > self._rotation6dChannels
            ):
                normalizedBoneTarget = self.normalizeMotion(targetMotion)
                fullBoneDiffusion = startMotionLoss(
                    normalizedBonePred, normalizedBoneTarget, motionMask,
                    perSampleWeights=perSampleWeights,
                )
                loss = loss + self.diffusionWeight * fullBoneDiffusion
                result["loss_bone_diffusion"] = fullBoneDiffusion.detach()

            # Same for global features: the denoiser predicts them as a
            # diffusion target, so they need a proper denoising MSE beyond
            # the per-component auxiliary losses.  Computed in normalized
            # space for the same scale-invariance reason as bone features.
            if (
                normalizedGlobalPred is not None
                and targetGlobalFeatures is not None
            ):
                normalizedGlobalTarget = self.normalizeGlobalFeatures(
                    targetGlobalFeatures,
                )
                globalDiffusion = startMotionLoss(
                    normalizedGlobalPred, normalizedGlobalTarget, motionMask,
                    perSampleWeights=perSampleWeights,
                )
                loss = loss + self.diffusionWeight * globalDiffusion
                result["loss_global_diffusion"] = globalDiffusion.detach()

            if self.clipGuidanceWeight > 0.0 and self._hasRotation6d:
                clipMotionInput = self.clip.buildMotionInputFromMotion(
                    motion=rot6dPredicted,
                    context=clipMotionContext,
                )
                clipMotionEmbeds = self.clip.encodeMotion(
                    clipMotionInput,
                    motionMask=motionMask,
                )
                clipGuidanceLoss = 1.0 - (
                    torch.sum(textEmbeds * clipMotionEmbeds, dim=-1).mean()
                )
                loss = loss + (self.clipGuidanceWeight * clipGuidanceLoss)
                result["loss_clip_guidance"] = clipGuidanceLoss.detach()
            result["loss"] = loss
            result.update(components)
        elif targetNoise is not None:
            raise ValueError(
                "targetNoise without targetMotion is not supported by "
                "the fixed x0 prediction target."
            )

        return result

    def train(self, mode: bool = True) -> MotionGenerator:
        """
        Keep the frozen CLIP tower in eval mode while toggling the generator.
        """
        super().train(mode)
        self.clip.eval()
        return self

    @torch.no_grad()
    def generateSample(
        self,
        prompt: str,
        numFrames: int,
        ddimSteps: int = 50,
        device: Optional[torch.device] = None,
        applyPostProcessing: bool = True,
        cfgScale: float = 2.5,
    ) -> dict[str, torch.Tensor]:
        """
        Generate a motion sample and any exported auxiliary features.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        encoded = self.clip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.maxPromptLength,
            return_tensors="pt",
        )
        inputIds = encoded["input_ids"].to(device)
        attentionMask = encoded["attention_mask"].to(device)

        textEmbeds, _ = self.clip.encodeText(inputIds, attentionMask)

        # Initialize noise for bone and global features.
        xBone = torch.randn(
            1, numFrames, self.numBones, self.boneChannels, device=device,
        )
        xGlobal: Optional[torch.Tensor] = None
        if self.globalChannels > 0:
            xGlobal = torch.randn(
                1, numFrames, self.globalChannels, device=device,
            )

        resolvedSteps = max(1, min(int(ddimSteps), self.diffusionSteps))
        timestepSequence = torch.linspace(
            self.diffusionSteps - 1, 0, resolvedSteps,
        ).long().tolist()

        # Classifier-Free Guidance: null embedding for unconditional pass
        nullTextEmbeds = (
            torch.zeros_like(textEmbeds) if cfgScale > 1.0 else None
        )

        diagActive = diag.get_logger() is not None
        generationId = f"gen_{int(torch.randint(0, 1_000_000, (1,)).item())}"

        for i, t in enumerate(timestepSequence):
            tBatch = torch.full((1,), t, device=device, dtype=torch.long)
            # Conditional pass
            condBone, condGlobal = self.denoiser(
                noisyMotion=xBone,
                textEmbedding=textEmbeds,
                timesteps=tBatch,
                noisyGlobalFeatures=xGlobal,
            )
            _, condBoneX0 = self._resolveModelPredictions(
                noisyMotion=xBone, timesteps=tBatch, modelOutput=condBone,
            )
            condGlobalX0: Optional[torch.Tensor] = None
            if condGlobal is not None and xGlobal is not None:
                _, condGlobalX0 = self._resolveModelPredictions(
                    noisyMotion=xGlobal, timesteps=tBatch, modelOutput=condGlobal,
                )

            if nullTextEmbeds is not None:
                # Unconditional pass
                uncondBone, uncondGlobal = self.denoiser(
                    noisyMotion=xBone,
                    textEmbedding=nullTextEmbeds,
                    timesteps=tBatch,
                    noisyGlobalFeatures=xGlobal,
                )
                _, uncondBoneX0 = self._resolveModelPredictions(
                    noisyMotion=xBone, timesteps=tBatch, modelOutput=uncondBone,
                )
                guidedBoneX0 = uncondBoneX0 + cfgScale * (condBoneX0 - uncondBoneX0)
                if condGlobalX0 is not None and xGlobal is not None:
                    _, uncondGlobalX0 = self._resolveModelPredictions(
                        noisyMotion=xGlobal, timesteps=tBatch, modelOutput=uncondGlobal,
                    )
                    guidedGlobalX0 = uncondGlobalX0 + cfgScale * (condGlobalX0 - uncondGlobalX0)
                else:
                    guidedGlobalX0 = condGlobalX0
            else:
                guidedBoneX0 = condBoneX0
                guidedGlobalX0 = condGlobalX0

            if diagActive:
                uncondForLog = (
                    condBoneX0 if nullTextEmbeds is None else uncondBoneX0
                )
                diag.logDdimStep(
                    generationId=generationId,
                    stepIdx=i,
                    timestep=int(t),
                    xBone=xBone,
                    condX0=condBoneX0,
                    uncondX0=uncondForLog,
                    guidedX0=guidedBoneX0,
                    cfgScale=cfgScale,
                )

            # DDIM step for bone features.
            guidedBoneNoise = self.ddim.predict_noise_from_start(
                xBone, tBatch, guidedBoneX0,
            )
            xBone = self._ddimStep(xBone, guidedBoneNoise, t, timestepSequence, i)

            # DDIM step for global features.
            if xGlobal is not None and guidedGlobalX0 is not None:
                guidedGlobalNoise = self.ddim.predict_noise_from_start(
                    xGlobal, tBatch, guidedGlobalX0,
                )
                xGlobal = self._ddimStep(xGlobal, guidedGlobalNoise, t, timestepSequence, i)

        # Denormalize from z-space.
        xBone = self.denormalizeMotion(xBone)
        if xGlobal is not None:
            xGlobal = self.denormalizeGlobalFeatures(xGlobal)

        # Extract rotation6d for quaternion conversion.
        rot6d = self._extractRotation6d(xBone)
        motion6d = rot6d
        if applyPostProcessing and rot6d is not None:
            batch, frames, bones, channels = motion6d.shape
            motionFlat = motion6d.reshape(batch, frames, bones * channels)
            smoothed = self.smoothing(motionFlat)
            motion6d = smoothed.reshape(batch, frames, bones, channels)

        sample: dict[str, torch.Tensor] = {}
        if motion6d is not None:
            motionQuat = self._sixdToQuaternion(motion6d)
            if applyPostProcessing:
                motionQuat = self.velocityReg(motionQuat)
            sample["motion_quat"] = motionQuat

        # Export root translation from global predictions if available.
        rootTranslation = self._extractGlobalComponent(xGlobal, "root_translation")
        if rootTranslation is not None:
            sample["root_translation"] = rootTranslation

        if diagActive:
            diag.logGenerationSummary(
                generationId=generationId,
                prompt=prompt,
                numFrames=numFrames,
                ddimSteps=resolvedSteps,
                cfgScale=cfgScale,
                motionQuat=sample.get("motion_quat"),
            )

        return sample

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        numFrames: int,
        ddimSteps: int = 50,
        device: Optional[torch.device] = None,
        applyPostProcessing: bool = True,
        cfgScale: float = 2.5,
    ) -> torch.Tensor:
        """
        Generate motion from text prompt using DDIM sampling.

        Parameters
        ----------
        prompt : str
            Text description of the motion.
        numFrames : int
            Number of frames to generate.
        ddimSteps : int, optional
            Number of DDIM sampling steps, by default 50.
        device : Optional[torch.device], optional
            Device for generation.

        Returns
        -------
        torch.Tensor
            Generated motion shaped (1, frames, bones, 4) as quaternions.
        """
        sample = self.generateSample(
            prompt=prompt,
            numFrames=numFrames,
            ddimSteps=ddimSteps,
            device=device,
            applyPostProcessing=applyPostProcessing,
            cfgScale=cfgScale,
        )
        motionQuat = sample.get("motion_quat")
        if motionQuat is None:
            raise RuntimeError(
                "generate() requires rotation6d in the enabled bone-data "
                "components. Use generateSample() for feature-only models."
            )
        return motionQuat

    def _ddimStep(
        self,
        xt: torch.Tensor,
        predictedNoise: torch.Tensor,
        t: int,
        timestepSequence: list[int],
        stepIdx: int,
    ) -> torch.Tensor:
        """
        Perform a single DDIM sampling step.

        Parameters
        ----------
        xt : torch.Tensor
            Current noisy sample.
        predictedNoise : torch.Tensor
            Predicted noise at timestep t.
        t : int
            Current timestep.
        timestepSequence : list[int]
            Full sequence of timesteps.
        stepIdx : int
            Current step index.

        Returns
        -------
        torch.Tensor
            Denoised sample for next step.
        """
        device = xt.device
        tTensor = torch.tensor([t], device=device)

        # Predict x0
        x0Pred = self.ddim.predict_start_from_noise(xt, tTensor, predictedNoise)

        if stepIdx >= len(timestepSequence) - 1:
            return x0Pred

        # Get next timestep
        tPrev = timestepSequence[stepIdx + 1]

        # Get alpha values
        alphaCumprodT = self.ddim.alphas_cumprod[t].to(device)
        alphaCumprodTprev = self.ddim.alphas_cumprod[tPrev].to(device)

        # DDIM formula
        sqrtAlphaTprev = torch.sqrt(alphaCumprodTprev)
        sqrtOneMinusAlphaTprev = torch.sqrt(1 - alphaCumprodTprev)

        # Direction pointing to xt
        dirXt = sqrtOneMinusAlphaTprev * predictedNoise

        # Predicted sample at t-1
        xPrev = sqrtAlphaTprev * x0Pred + dirXt

        return xPrev

    def _resolveModelPredictions(
        self,
        noisyMotion: torch.Tensor,
        timesteps: torch.Tensor,
        modelOutput: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the denoiser x0 output into both epsilon and x0 views.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Predicted epsilon and predicted clean motion x0.
        """
        predictedMotion = modelOutput
        predictedNoise = self.ddim.predict_noise_from_start(
            noisyMotion,
            timesteps,
            predictedMotion,
        )
        return predictedNoise, predictedMotion

    # ------------------------------------------------------------------
    # Z-normalization helpers
    # ------------------------------------------------------------------

    def setMotionStatistics(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """Store dataset mean/std for bone feature Z-normalization."""
        self.motion_mean.copy_(mean.view(self.motion_mean.shape))
        self.motion_std.copy_(std.view(self.motion_std.shape))

    def setGlobalStatistics(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """Store dataset mean/std for global feature Z-normalization."""
        if self.globalChannels == 0:
            return
        self.global_mean.copy_(mean.view(self.global_mean.shape))
        self.global_std.copy_(std.view(self.global_std.shape))

    def normalizeMotion(self, motion: torch.Tensor) -> torch.Tensor:
        """Normalize raw bone features to zero-mean unit-variance."""
        return (motion - self.motion_mean) / self.motion_std.clamp(min=1e-5)

    def denormalizeMotion(self, motion: torch.Tensor) -> torch.Tensor:
        """Inverse of normalizeMotion — map back to raw bone feature space."""
        return motion * self.motion_std + self.motion_mean

    def normalizeGlobalFeatures(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize raw global features to zero-mean unit-variance."""
        if self.globalChannels == 0:
            return features
        return (features - self.global_mean) / self.global_std.clamp(min=1e-5)

    def denormalizeGlobalFeatures(self, features: torch.Tensor) -> torch.Tensor:
        """Inverse of normalizeGlobalFeatures."""
        if self.globalChannels == 0:
            return features
        return features * self.global_std + self.global_mean

    def trainableParameters(self) -> tuple[nn.Parameter, ...]:
        """Return every parameter optimized during generation training."""
        return tuple(self.denoiser.parameters())

    # ------------------------------------------------------------------
    # Feature assembly and splitting
    # ------------------------------------------------------------------

    def assembleBoneFeatures(
        self,
        batch: Mapping[str, object],
        device: torch.device,
    ) -> torch.Tensor:
        """Concatenate all bone-scoped features from a batch.

        Returns a tensor shaped (batch, frames, bones, boneChannels).
        """
        parts: list[torch.Tensor] = []
        for component in self._boneComponents:
            tensor = batch.get(component.sampleKey)
            if not isinstance(tensor, torch.Tensor):
                raise KeyError(
                    f"Batch missing required bone feature {component.sampleKey!r}."
                )
            parts.append(tensor.to(device))
        if not parts:
            # Legacy fallback: rotation6d only.
            motion = batch.get("motion")
            if not isinstance(motion, torch.Tensor):
                raise KeyError("Batch missing 'motion' tensor.")
            return motion.to(device)
        return torch.cat(parts, dim=-1)

    def assembleGlobalFeatures(
        self,
        batch: Mapping[str, object],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Concatenate all global-scoped features from a batch.

        Returns a tensor shaped (batch, frames, globalChannels) or None.
        """
        if not self._globalComponents:
            return None
        parts: list[torch.Tensor] = []
        for component in self._globalComponents:
            tensor = batch.get(component.sampleKey)
            if not isinstance(tensor, torch.Tensor):
                raise KeyError(
                    f"Batch missing required global feature {component.sampleKey!r}."
                )
            parts.append(tensor.to(device))
        return torch.cat(parts, dim=-1)

    def _extractRotation6d(self, boneFeatures: torch.Tensor) -> torch.Tensor:
        """Extract the rotation6d slice from concatenated bone features."""
        if not self._hasRotation6d:
            return boneFeatures
        return boneFeatures[
            ...,
            self._rotation6dOffset:self._rotation6dOffset + self._rotation6dChannels,
        ]

    def _splitBonePredictions(
        self,
        boneFeatures: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Split concatenated bone features into per-component tensors."""
        result: dict[str, torch.Tensor] = {}
        offset = 0
        for component in self._boneComponents:
            result[component.sampleKey] = boneFeatures[
                ..., offset:offset + component.channels
            ]
            offset += component.channels
        return result

    def _splitGlobalPredictions(
        self,
        globalFeatures: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Split concatenated global features into per-component tensors."""
        result: dict[str, torch.Tensor] = {}
        offset = 0
        for component in self._globalComponents:
            result[component.sampleKey] = globalFeatures[
                ..., offset:offset + component.channels
            ]
            offset += component.channels
        return result

    def _extractGlobalComponent(
        self,
        globalFeatures: Optional[torch.Tensor],
        sampleKey: str,
    ) -> Optional[torch.Tensor]:
        """Extract one component from concatenated global features."""
        if globalFeatures is None:
            return None
        offset = 0
        for component in self._globalComponents:
            if component.sampleKey == sampleKey:
                return globalFeatures[..., offset:offset + component.channels]
            offset += component.channels
        return None

    # ------------------------------------------------------------------
    # Multi-feature loss
    # ------------------------------------------------------------------

    def _multiFeatureLoss(
        self,
        boneSplits: dict[str, torch.Tensor],
        boneTargetSplits: dict[str, torch.Tensor],
        predictedGlobal: Optional[torch.Tensor],
        targetGlobal: Optional[torch.Tensor],
        motionMask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], dict[str, torch.Tensor]]:
        """Compute per-component losses for all non-rotation6d features."""
        totalLoss: Optional[torch.Tensor] = None
        lossCount = 0
        losses: dict[str, torch.Tensor] = {}

        # Bone-scoped component losses (skip rotation6d, handled by diffusion loss).
        for component in self._boneComponents:
            if component.key == "rotation6d":
                continue
            predicted = boneSplits.get(component.sampleKey)
            target = boneTargetSplits.get(component.sampleKey)
            if predicted is None or target is None:
                continue
            componentLoss = component.loss(predicted, target, motionMask)
            weight = self._componentLossWeight(component.key)
            weighted = weight * componentLoss
            if totalLoss is None:
                totalLoss = weighted
            else:
                totalLoss = totalLoss + weighted
            losses[f"loss_{component.key}"] = componentLoss.detach()
            lossCount += 1

        # Global-scoped component losses.
        if predictedGlobal is not None and targetGlobal is not None:
            predictedGlobalSplits = self._splitGlobalPredictions(predictedGlobal)
            targetGlobalSplits = self._splitGlobalPredictions(targetGlobal)
            for component in self._globalComponents:
                predicted = predictedGlobalSplits.get(component.sampleKey)
                target = targetGlobalSplits.get(component.sampleKey)
                if predicted is None or target is None:
                    continue
                componentLoss = component.loss(predicted, target, motionMask)
                weight = self._componentLossWeight(component.key)
                weighted = weight * componentLoss
                if totalLoss is None:
                    totalLoss = weighted
                else:
                    totalLoss = totalLoss + weighted
                losses[f"loss_{component.key}"] = componentLoss.detach()
                lossCount += 1

        if totalLoss is None or lossCount == 0:
            return None, {}
        # Do NOT average by lossCount: each component already has a
        # calibrated weight from _componentLossWeight.  Dividing by the
        # number of active auxiliary losses silently shrinks every signal
        # as more features are enabled, which was one of the causes of the
        # model collapsing towards a mean pose (the rotation6d diffusion
        # loss would dwarf the diluted auxiliary signals).
        losses["loss_components"] = totalLoss.detach()
        return totalLoss, losses

    def _componentLossWeight(self, componentKey: str) -> float:
        """Resolve the default weight for one auxiliary motion component."""
        if componentKey in {"root_translation", "joint_xyz", "pelvis_height"}:
            return max(1.0, float(self.xyzWeight))
        if componentKey in {
            "root_velocity",
            "joint_velocity",
            "end_effector_velocity",
            "root_yaw_velocity",
        }:
            return max(1.0, float(self.velXyzWeight))
        if componentKey == "root_yaw":
            return max(0.5, float(self.xyzWeight))
        if componentKey in {"foot_contact", "hand_contact"}:
            return 1.0
        return 1.0

    def _sixdToQuaternion(self, sixd: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation to quaternion.

        Parameters
        ----------
        sixd : torch.Tensor
            6D rotation shaped (..., 6).

        Returns
        -------
        torch.Tensor
            Quaternion shaped (..., 4).
        """
        # First get rotation matrix
        rotMat = self.renorm(sixd)

        # Convert rotation matrix to quaternion
        return self._rotationMatrixToQuaternion(rotMat)

    def _rotationMatrixToQuaternion(self, rotMat: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to quaternion.

        Parameters
        ----------
        rotMat : torch.Tensor
            Rotation matrix shaped (..., 3, 3).

        Returns
        -------
        torch.Tensor
            Quaternion shaped (..., 4) in (w, x, y, z) order.
        """
        batch_shape = rotMat.shape[:-2]

        m00 = rotMat[..., 0, 0]
        m01 = rotMat[..., 0, 1]
        m02 = rotMat[..., 0, 2]
        m10 = rotMat[..., 1, 0]
        m11 = rotMat[..., 1, 1]
        m12 = rotMat[..., 1, 2]
        m20 = rotMat[..., 2, 0]
        m21 = rotMat[..., 2, 1]
        m22 = rotMat[..., 2, 2]

        trace = m00 + m11 + m22

        # Initialize quaternion components
        qw = torch.zeros(batch_shape, device=rotMat.device, dtype=rotMat.dtype)
        qx = torch.zeros_like(qw)
        qy = torch.zeros_like(qw)
        qz = torch.zeros_like(qw)

        # Case 1: trace > 0
        cond1 = trace > 0
        s1 = torch.sqrt(trace[cond1] + 1.0) * 2
        qw[cond1] = 0.25 * s1
        qx[cond1] = (m21[cond1] - m12[cond1]) / s1
        qy[cond1] = (m02[cond1] - m20[cond1]) / s1
        qz[cond1] = (m10[cond1] - m01[cond1]) / s1

        # Case 2: m00 > m11 and m00 > m22
        cond2 = ~cond1 & (m00 > m11) & (m00 > m22)
        s2 = torch.sqrt(1.0 + m00[cond2] - m11[cond2] - m22[cond2]) * 2
        qw[cond2] = (m21[cond2] - m12[cond2]) / s2
        qx[cond2] = 0.25 * s2
        qy[cond2] = (m01[cond2] + m10[cond2]) / s2
        qz[cond2] = (m02[cond2] + m20[cond2]) / s2

        # Case 3: m11 > m22
        cond3 = ~cond1 & ~cond2 & (m11 > m22)
        s3 = torch.sqrt(1.0 + m11[cond3] - m00[cond3] - m22[cond3]) * 2
        qw[cond3] = (m02[cond3] - m20[cond3]) / s3
        qx[cond3] = (m01[cond3] + m10[cond3]) / s3
        qy[cond3] = 0.25 * s3
        qz[cond3] = (m12[cond3] + m21[cond3]) / s3

        # Case 4: else
        cond4 = ~cond1 & ~cond2 & ~cond3
        s4 = torch.sqrt(1.0 + m22[cond4] - m00[cond4] - m11[cond4]) * 2
        qw[cond4] = (m10[cond4] - m01[cond4]) / s4
        qx[cond4] = (m02[cond4] + m20[cond4]) / s4
        qy[cond4] = (m12[cond4] + m21[cond4]) / s4
        qz[cond4] = 0.25 * s4

        return torch.stack([qw, qx, qy, qz], dim=-1)

    def _loadClipCheckpoint(self, checkpointPath: Path) -> None:
        """
        Load CLIP model weights from checkpoint.

        Parameters
        ----------
        checkpointPath : Path
            Path to the CLIP checkpoint file.
        """
        checkpoint = torch.load(checkpointPath, weights_only=False, map_location="cpu")
        try:
            self.clip.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as error:
            raise RuntimeError(
                "Failed to load the CLIP checkpoint. The checkpoint motion "
                "encoder likely does not match the current CLIP architecture "
                "(clip.motion-num-heads / clip.motion-num-layers) or "
                "clip.bone-data layout. Re-train CLIP or point generation "
                f"to a matching checkpoint. Original error: {error}"
            ) from error

    def _freezeClip(self) -> None:
        """Freeze all CLIP parameters."""
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()
