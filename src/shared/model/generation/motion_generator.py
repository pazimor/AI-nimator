"""Full motion generation pipeline combining CLIP and diffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.shared.model.clip.core import ClipModel
from src.shared.model.generation.ddim import DDIM
from src.shared.model.generation.denoiser import MotionDenoiser
from src.shared.model.generation.losses import (
    DEFAULT_GEODESIC_WEIGHT,
    GEODESIC_SCHEDULE_NONE,
)
from src.shared.model.layers.correction import (
    Renormalization,
    Smoothing,
    VelocityRegularization,
)
from src.shared.constants.rotation import (
    DEFAULT_ROTATION_REPR,
    ROTATION_CHANNELS_ROT6D,
    ROTATION_KIND_AXIS_ANGLE,
    ROTATION_REPR_AXIS_ANGLE,
    ROTATION_REPR_ROT6D,
    normalizeRotationRepr,
)
from src.shared.quaternion import Rotation
from src.shared.types.network import DEFAULT_SPATIOTEMPORAL_MODE


class MotionGenerator(nn.Module):
    """
    Complete motion generation pipeline.

    Combines frozen CLIP text encoder with trainable diffusion denoiser
    and post-processing correction layers.
    """

    def __init__(
        self,
        embedDim: int = 64,
        numHeads: int = 4,
        numLayers: int = 6,
        numSpatialLayers: int = 1,
        motionChannels: int = ROTATION_CHANNELS_ROT6D,
        rotationRepr: str = DEFAULT_ROTATION_REPR,
        spatiotemporalMode: str = DEFAULT_SPATIOTEMPORAL_MODE,
        numBones: int = 65,
        diffusionSteps: int = 1000,
        modelName: str = "xlm-roberta-base",
        clipCheckpoint: Optional[Path] = None,
        smoothingKernel: int = 3,
        maxVelocity: Optional[float] = None,
        geodesicWeight: float = DEFAULT_GEODESIC_WEIGHT,
        geodesicWeightSchedule: str = GEODESIC_SCHEDULE_NONE,
    ) -> None:
        """
        Initialize MotionGenerator.

        Parameters
        ----------
        embedDim : int, optional
            Embedding dimension (must match CLIP), by default 64.
        numHeads : int, optional
            Number of attention heads, by default 4.
        numLayers : int, optional
            Number of denoising layers, by default 6.
        numSpatialLayers : int, optional
            Number of spatial GCN blocks, by default 1.
        motionChannels : int, optional
            Motion channels per bone, by default 6.
        rotationRepr : str, optional
            Rotation representation, by default "rot6d".
        spatiotemporalMode : str, optional
            Spatio-temporal strategy, by default "flat".
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
        geodesicWeight : float, optional
            Base geodesic loss weight, by default 0.1.
        geodesicWeightSchedule : str, optional
            Schedule for geodesic weighting, by default "none".
        """
        super().__init__()
        self.embedDim = embedDim
        self.numBones = numBones
        self.motionChannels = motionChannels
        rotationRepr = normalizeRotationRepr(rotationRepr)
        self.rotationRepr = rotationRepr
        self.spatiotemporalMode = spatiotemporalMode
        self.diffusionSteps = diffusionSteps
        self.geodesicWeight = geodesicWeight
        self.geodesicWeightSchedule = geodesicWeightSchedule

        # CLIP text encoder (frozen)
        self.clip = ClipModel(
            modelName=modelName,
            embedDim=embedDim,
            freezeTextEncoder=True,
        )
        if clipCheckpoint is not None:
            self._loadClipCheckpoint(clipCheckpoint)
        self._freezeClip()

        # Diffusion components
        self.ddim = DDIM(num_timesteps=diffusionSteps)
        self.denoiser = MotionDenoiser(
            embedDim=embedDim,
            numHeads=numHeads,
            numLayers=numLayers,
            numSpatialLayers=numSpatialLayers,
            motionChannels=motionChannels,
            spatiotemporalMode=spatiotemporalMode,
            numBones=numBones,
        )

        # Post-processing (inference only)
        self.renorm = Renormalization()
        self.smoothing = Smoothing(
            channels=numBones * motionChannels,
            kernel_size=smoothingKernel,
        )
        self.velocityReg = VelocityRegularization(max_velocity=maxVelocity)

    def forward(
        self,
        textInputIds: torch.Tensor,
        textAttentionMask: torch.Tensor,
        tags: Optional[list[Optional[str]]],
        noisyMotion: torch.Tensor,
        timesteps: torch.Tensor,
        targetNoise: Optional[torch.Tensor] = None,
        targetMotion: Optional[torch.Tensor] = None,
        motionMask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Parameters
        ----------
        textInputIds : torch.Tensor
            Tokenized text input IDs.
        textAttentionMask : torch.Tensor
            Text attention mask.
        tags : Optional[list[Optional[str]]]
            Batch of tag strings. Can be None or contain None elements.
        noisyMotion : torch.Tensor
            Noisy motion shaped (batch, frames, bones, channels).
        timesteps : torch.Tensor
            Diffusion timesteps shaped (batch,).
        targetNoise : Optional[torch.Tensor], optional
            Ground truth noise for loss computation.
        targetMotion : Optional[torch.Tensor], optional
            Ground truth clean motion for auxiliary losses.
        motionMask : Optional[torch.Tensor], optional
            Boolean mask indicating valid (non-padded) frames.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with predicted noise and optional loss.
        """
        # Get frozen text embedding from CLIP
        with torch.no_grad():
            textEmbeds, _ = self.clip.encodeText(
                inputIds=textInputIds,
                attentionMask=textAttentionMask,
            )

        # Predict noise
        padMask = None
        if motionMask is not None:
            padMask = ~motionMask.bool()

        predictedNoise = self.denoiser(
            noisyMotion=noisyMotion,
            textEmbedding=textEmbeds,
            tags=tags,
            timesteps=timesteps,
            mask=padMask,
        )

        result = {"predicted_noise": predictedNoise}

        if targetNoise is not None:
            if targetMotion is None:
                from src.shared.model.generation.losses import diffusionLoss

                loss = diffusionLoss(
                    predictedNoise,
                    targetNoise,
                    motionMask=motionMask,
                )
                result["loss"] = loss
            else:
                from src.shared.model.generation.losses import combinedGenerationLoss

                predictedMotion = self.ddim.predict_start_from_noise(
                    noisyMotion,
                    timesteps,
                    predictedNoise,
                )
                loss, components = combinedGenerationLoss(
                    predictedNoise=predictedNoise,
                    targetNoise=targetNoise,
                    predictedMotion=predictedMotion,
                    targetMotion=targetMotion,
                    geodesicWeight=self.geodesicWeight,
                    geodesicWeightSchedule=self.geodesicWeightSchedule,
                    rotationRepr=self.rotationRepr,
                    timesteps=timesteps,
                    numTimesteps=self.ddim.num_timesteps,
                    motionMask=motionMask,
                )
                result["loss"] = loss
                result.update(components)

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tag: Optional[str],
        numFrames: int,
        ddimSteps: int = 50,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate motion from text prompt using DDIM sampling.

        Parameters
        ----------
        prompt : str
            Text description of the motion.
        tag : Optional[str]
            Categorical tag for the motion. Can be None.
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
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Tokenize prompt
        encoded = self.clip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputIds = encoded["input_ids"].to(device)
        attentionMask = encoded["attention_mask"].to(device)

        # Get text embedding
        textEmbeds, _ = self.clip.encodeText(inputIds, attentionMask)

        # Initialize random noise
        x = torch.randn(
            1,
            numFrames,
            self.numBones,
            self.motionChannels,
            device=device,
        )

        # DDIM sampling with fewer steps
        stepRatio = self.diffusionSteps // ddimSteps
        timestepSequence = list(range(0, self.diffusionSteps, stepRatio))[::-1]

        for i, t in enumerate(timestepSequence):
            tBatch = torch.full((1,), t, device=device, dtype=torch.long)

            # Predict noise
            predictedNoise = self.denoiser(
                noisyMotion=x,
                textEmbedding=textEmbeds,
                tags=[tag],
                timesteps=tBatch,
            )

            # DDIM step
            x = self._ddimStep(x, predictedNoise, t, timestepSequence, i)

        # Post-process: apply corrections
        motion = x

        # Reshape for smoothing: (batch, frames, bones * channels)
        batch, frames, bones, channels = motion.shape
        motionFlat = motion.view(batch, frames, bones * channels)
        smoothed = self.smoothing(motionFlat)
        motion = smoothed.view(batch, frames, bones, channels)

        # Convert to quaternions
        motionQuat = self._rotationToQuaternion(motion)

        # Apply velocity regularization
        motionQuat = self.velocityReg(motionQuat)

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

    def _rotationToQuaternion(self, rotation: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation representation to quaternion.

        Parameters
        ----------
        rotation : torch.Tensor
            Rotation representation shaped (..., C).

        Returns
        -------
        torch.Tensor
            Quaternion shaped (..., 4) in (w, x, y, z) order.
        """
        if self.rotationRepr == ROTATION_REPR_ROT6D:
            return self._sixdToQuaternion(rotation)
        if self.rotationRepr == ROTATION_REPR_AXIS_ANGLE:
            return self._axisAngleToQuaternion(rotation)
        raise ValueError(f"Unknown rotation repr: {self.rotationRepr}")

    def _axisAngleToQuaternion(self, axisAngle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle rotations to quaternions.

        Parameters
        ----------
        axisAngle : torch.Tensor
            Axis-angle rotations shaped (..., 3).

        Returns
        -------
        torch.Tensor
            Quaternion shaped (..., 4) in (w, x, y, z) order.
        """
        quatXyzw = Rotation(
            axisAngle,
            kind=ROTATION_KIND_AXIS_ANGLE,
        ).quat
        return torch.cat((quatXyzw[..., 3:], quatXyzw[..., :3]), dim=-1)

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
        self.clip.load_state_dict(checkpoint["model_state_dict"])

    def _freezeClip(self) -> None:
        """Freeze all CLIP parameters."""
        for param in self.clip.parameters():
            param.requires_grad = False
