"""Full motion generation pipeline combining CLIP and diffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn

from src.shared.model.clip.core import ClipModel
from src.shared.model.components import buildMotionFeatureTensors
from src.shared.model.components.base import MotionComponent
from src.shared.model.generation.ddim import DDIM
from src.shared.model.generation.denoiser import MotionDenoiser
from src.shared.model.generation.losses import (
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
        diffusionWeight: float = 1.0,
        accelerationWeight: float = 0.0,
        clipGuidanceWeight: float = 0.0,
        numSpatialLayers: int = 1,
        numSpatioTemporalLayers: int = 1,
        maxPromptLength: int = 64,
        generationEmbedDim: Optional[int] = None,
        clipMotionNumHeads: int = 4,
        clipMotionNumLayers: int = 2,
        clipMotionComponents: Optional[tuple[MotionComponent, ...]] = None,
        generationMotionComponents: Optional[tuple[MotionComponent, ...]] = None,
    ) -> None:
        """
        Initialize MotionGenerator.

        Parameters
        ----------
        embedDim : int, optional
            CLIP embedding dimension, by default 64.
        numHeads : int, optional
            Number of attention heads, by default 4.
        numLayers : int, optional
            Number of denoising layers, by default 6.
        numBones : int, optional
            Number of skeleton bones, by default 65.
        diffusionSteps : int, optional
            Number of diffusion timesteps, by default 1000.
        modelName : str, optional
            XLM-Roberta model name, by default "xlm-roberta-base".
        clipCheckpoint : Optional[Path], optional
            Path to pre-trained CLIP checkpoint, by default None.
        smoothingKernel : int, optional
            Kernel size for temporal smoothing, by default 3.
        maxVelocity : Optional[float], optional
            Maximum velocity for regularization, by default None.
        xyzWeight : float, optional
            Base XYZ loss weight, by default 0.1.
        xyzWeightSchedule : str, optional
            Schedule for XYZ weighting, by default "none".
        velXyzWeight : float, optional
            Weight for velocity matching in XYZ space, by default 0.01.
        diffusionWeight : float, optional
            Weight for diffusion loss, by default 1.0.
        accelerationWeight : float, optional
            Weight for acceleration loss, by default 0.0.
        clipGuidanceWeight : float, optional
            Weight for the auxiliary CLIP text-motion guidance loss.
        numSpatialLayers : int, optional
            Number of spatial GCN blocks.
        numSpatioTemporalLayers : int, optional
            Number of local spatio-temporal mixing blocks.
        maxPromptLength : int, optional
            Tokenizer max length used during inference tokenization.
        generationEmbedDim : Optional[int], optional
            Internal denoiser width. When omitted, defaults to ``embedDim``
            for backward compatibility.
        clipMotionNumHeads : int, optional
            Number of attention heads in the frozen CLIP motion encoder.
        clipMotionNumLayers : int, optional
            Number of transformer layers in the frozen CLIP motion encoder.
        clipMotionComponents : Optional[tuple[MotionComponent, ...]], optional
            Optional CLIP motion feature layout. When omitted, CLIP uses the
            legacy rotation-only motion tensor.
        generationMotionComponents : Optional[tuple[MotionComponent, ...]], optional
            Motion features supervised during generation training. The denoiser
            still predicts the base 6D rotation tensor, while auxiliary heads
            and losses can supervise extra motion signals such as root motion.
        """
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
        self.diffusionWeight = diffusionWeight
        self.accelerationWeight = accelerationWeight
        self.clipGuidanceWeight = clipGuidanceWeight
        self.maxPromptLength = max(1, int(maxPromptLength))
        self.generationMotionComponents = tuple(generationMotionComponents or ())
        self._generationComponentsBySampleKey = {
            component.sampleKey: component
            for component in self.generationMotionComponents
            if component.key != "rotation6d"
        }

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
            numSpatialLayers=numSpatialLayers,
            numSpatioTemporalLayers=numSpatioTemporalLayers,
            textEmbedDim=embedDim,
        )
        self.rootTranslationHead: Optional[nn.Module]
        if self._requiresRootTranslationHead():
            flatMotionDim = self.numBones * self.MOTION_ROTATION_CHANNELS
            self.rootTranslationHead = nn.Sequential(
                nn.LayerNorm(flatMotionDim),
                nn.Linear(flatMotionDim, self.generationEmbedDim),
                nn.SiLU(),
                nn.Linear(
                    self.generationEmbedDim,
                    self.ROOT_TRANSLATION_CHANNELS,
                ),
            )
        else:
            self.rootTranslationHead = None

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
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Parameters
        ----------
        textInputIds : Optional[torch.Tensor]
            Tokenized text input IDs.
        textAttentionMask : Optional[torch.Tensor]
            Text attention mask.
        noisyMotion : Optional[torch.Tensor]
            Noisy motion shaped (batch, frames, bones, 6).
        timesteps : Optional[torch.Tensor]
            Diffusion timesteps shaped (batch,).
        targetNoise : Optional[torch.Tensor], optional
            Unused legacy argument kept for backward compatibility.
        targetMotion : Optional[torch.Tensor], optional
            Ground truth clean motion for x0-based losses.
        motionMask : Optional[torch.Tensor], optional
            Boolean mask indicating valid (non-padded) frames.
        clipMotionContext : Optional[Mapping[str, object]], optional
            Auxiliary tensors used to rebuild the CLIP motion input for
            guidance when some CLIP components are not directly predicted.
        textEmbedding : Optional[torch.Tensor], optional
            Precomputed CLIP text embedding used during training.
        componentTargets : Optional[Mapping[str, torch.Tensor]], optional
            Auxiliary generation targets keyed by sample tensor name
            (for example ``root_translation`` or ``joint_xyz``).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with predicted noise and optional loss.
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

        # Predict denoiser output
        if noisyMotion.shape[-1] != self.MOTION_ROTATION_CHANNELS:
            raise ValueError(
                "Expected rotation-only motion with 6 channels "
                "(no translation), got "
                f"{noisyMotion.shape[-1]}."
            )
        padMask = None
        if motionMask is not None:
            padMask = ~motionMask.bool()

        denoiserOutput = self.denoiser(
            noisyMotion=noisyMotion,
            textEmbedding=textEmbeds,
            timesteps=timesteps,
            mask=padMask,
        )
        predictedNoise, predictedMotion = self._resolveModelPredictions(
            noisyMotion=noisyMotion,
            timesteps=timesteps,
            modelOutput=denoiserOutput,
        )
        predictedRootTranslation = self._predictRootTranslation(predictedMotion)

        result = {
            "predicted_noise": predictedNoise,
            "predicted_motion": predictedMotion,
        }
        if predictedRootTranslation is not None:
            result["predicted_root_translation"] = predictedRootTranslation

        if targetMotion is not None:
            from src.shared.model.generation.losses import combinedGenerationLoss

            loss, components = combinedGenerationLoss(
                predictedMotion=predictedMotion,
                targetMotion=targetMotion,
                diffusionWeight=self.diffusionWeight,
                xyzWeight=self.xyzWeight,
                xyzWeightSchedule=self.xyzWeightSchedule,
                velocityXyzWeight=self.velXyzWeight,
                accelerationWeight=self.accelerationWeight,
                timesteps=timesteps,
                numTimesteps=self.ddim.num_timesteps,
                motionMask=motionMask,
            )
            componentLoss, componentLosses = self._generationComponentLoss(
                predictedMotion=predictedMotion,
                predictedRootTranslation=predictedRootTranslation,
                componentTargets=componentTargets,
                motionMask=motionMask,
            )
            if componentLoss is not None:
                loss = loss + componentLoss
                result.update(componentLosses)
            if self.clipGuidanceWeight > 0.0:
                clipMotionInput = self.clip.buildMotionInputFromMotion(
                    motion=predictedMotion,
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

        x = torch.randn(1, numFrames, self.numBones, 6, device=device)
        resolvedSteps = max(1, min(int(ddimSteps), self.diffusionSteps))
        stepRatio = max(1, self.diffusionSteps // resolvedSteps)
        timestepSequence = list(range(0, self.diffusionSteps, stepRatio))[::-1]

        for i, t in enumerate(timestepSequence):
            tBatch = torch.full((1,), t, device=device, dtype=torch.long)
            denoiserOutput = self.denoiser(
                noisyMotion=x,
                textEmbedding=textEmbeds,
                timesteps=tBatch,
            )
            predictedNoise, _ = self._resolveModelPredictions(
                noisyMotion=x,
                timesteps=tBatch,
                modelOutput=denoiserOutput,
            )
            x = self._ddimStep(x, predictedNoise, t, timestepSequence, i)

        rawMotion6d = x
        predictedRootTranslation = self._predictRootTranslation(rawMotion6d)
        motion6d = rawMotion6d
        if applyPostProcessing:
            batch, frames, bones, channels = motion6d.shape
            motionFlat = motion6d.view(batch, frames, bones * channels)
            smoothed = self.smoothing(motionFlat)
            motion6d = smoothed.view(batch, frames, bones, channels)

        motionQuat = self._sixdToQuaternion(motion6d)
        if applyPostProcessing:
            motionQuat = self.velocityReg(motionQuat)
        sample = {"motion_quat": motionQuat}

        if predictedRootTranslation is not None:
            sample["root_translation"] = predictedRootTranslation
        return sample

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        numFrames: int,
        ddimSteps: int = 50,
        device: Optional[torch.device] = None,
        applyPostProcessing: bool = True,
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
        return self.generateSample(
            prompt=prompt,
            numFrames=numFrames,
            ddimSteps=ddimSteps,
            device=device,
            applyPostProcessing=applyPostProcessing,
        )["motion_quat"]

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

    def trainableParameters(self) -> tuple[nn.Parameter, ...]:
        """Return every parameter optimized during generation training."""
        parameters = list(self.denoiser.parameters())
        if self.rootTranslationHead is not None:
            parameters.extend(self.rootTranslationHead.parameters())
        return tuple(parameters)

    def _requiresRootTranslationHead(self) -> bool:
        """Return True when generation supervision needs explicit root motion."""
        requiredKeys = {"root_translation", "root_velocity"}
        return any(
            component.key in requiredKeys
            for component in self.generationMotionComponents
        )

    def _predictRootTranslation(
        self,
        motion: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict root translation from the generated 6D motion sequence."""
        if self.rootTranslationHead is None:
            return None
        flatMotion = motion.reshape(motion.shape[0], motion.shape[1], -1)
        return self.rootTranslationHead(flatMotion)

    def _generationComponentLoss(
        self,
        predictedMotion: torch.Tensor,
        predictedRootTranslation: Optional[torch.Tensor],
        componentTargets: Optional[Mapping[str, torch.Tensor]],
        motionMask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], dict[str, torch.Tensor]]:
        """Compute auxiliary losses for enabled generation components."""
        if not componentTargets or not self._generationComponentsBySampleKey:
            return None, {}

        predictedTargets = self._buildPredictedComponentTargets(
            predictedMotion=predictedMotion,
            predictedRootTranslation=predictedRootTranslation,
        )
        totalLoss = torch.tensor(0.0, device=predictedMotion.device)
        lossCount = 0
        losses: dict[str, torch.Tensor] = {}

        for sampleKey, target in componentTargets.items():
            component = self._generationComponentsBySampleKey.get(sampleKey)
            if component is None:
                continue
            predicted = predictedTargets.get(sampleKey)
            if predicted is None:
                continue
            componentLoss = component.loss(
                predicted=predicted,
                target=target,
                motionMask=motionMask,
            )
            weightedLoss = self._componentLossWeight(component.key) * componentLoss
            totalLoss = totalLoss + weightedLoss
            losses[f"loss_{component.key}"] = componentLoss.detach()
            lossCount += 1

        if lossCount == 0:
            return None, {}
        totalLoss = totalLoss / float(lossCount)
        losses["loss_components"] = totalLoss.detach()
        return totalLoss, losses

    def _buildPredictedComponentTargets(
        self,
        predictedMotion: torch.Tensor,
        predictedRootTranslation: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build auxiliary targets from the predicted clean motion."""
        if not self.generationMotionComponents:
            return {}

        stacked: dict[str, list[torch.Tensor]] = {}
        for batchIndex in range(predictedMotion.shape[0]):
            extras: dict[str, object] = {}
            if predictedRootTranslation is not None:
                extras["trans"] = predictedRootTranslation[batchIndex]
            sampleFeatures = buildMotionFeatureTensors(
                motion=predictedMotion[batchIndex],
                extras=extras,
                enabledComponents=self.generationMotionComponents,
            )
            for sampleKey, value in sampleFeatures.items():
                stacked.setdefault(sampleKey, []).append(value)
        return {
            sampleKey: torch.stack(values, dim=0)
            for sampleKey, values in stacked.items()
        }

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
