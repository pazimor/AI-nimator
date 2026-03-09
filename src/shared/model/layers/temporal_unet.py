import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLayer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences (computed on-the-fly)."""

    def __init__(self, embedDim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedDim = embedDim
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input (computed dynamically for any length)."""
        seqLen = x.size(1)
        device = x.device
        
        position = torch.arange(seqLen, device=device).unsqueeze(1)
        divTerm = torch.exp(
            torch.arange(0, self.embedDim, 2, device=device) * (-math.log(10000.0) / self.embedDim)
        )
        pe = torch.zeros(1, seqLen, self.embedDim, device=device)
        pe[0, :, 0::2] = torch.sin(position * divTerm)
        pe[0, :, 1::2] = torch.cos(position * divTerm)
        
        x = x + pe
        return self.dropout(x)


class TemporalUNet(BaseLayer):
    """
    Temporal encoder for motion sequences using Transformer.

    Bone-scoped channels and global channels are encoded by separate temporal
    branches, then fused late into a single motion embedding.
    """

    def __init__(
        self,
        embedDim: int,
        numHeads: int = 4,
        numLayers: int = 2,
        dropout: float = 0.1,
        numBones: int = 22,
        numChannels: int = 6,
        globalChannels: int = 0,
        maxFrames: int = 128,  # Max frames after downsampling
    ) -> None:
        super().__init__(name="TemporalUNet")
        self.embedDim = embedDim
        self.numBones = numBones
        self.numChannels = numChannels
        self.globalChannels = globalChannels
        self.globalBranchDim = (
            self._resolveGlobalBranchDim(embedDim, numHeads)
            if self.globalChannels > 0
            else 0
        )
        self.maxFrames = maxFrames

        if self.numChannels <= 0 and self.globalChannels <= 0:
            raise ValueError(
                "TemporalUNet requires at least one bone or global channel."
            )

        self.inputProj = (
            nn.Linear(numBones * numChannels, embedDim)
            if self.numChannels > 0
            else None
        )
        self.temporalConv = (
            self._buildTemporalConv(embedDim)
            if self.numChannels > 0
            else None
        )
        self.posEncoding = (
            PositionalEncoding(embedDim, dropout=dropout)
            if self.numChannels > 0
            else None
        )
        self.boneTransformer = (
            self._buildTransformer(embedDim, numHeads, numLayers, dropout)
            if self.numChannels > 0
            else None
        )

        self.globalInputProj = (
            nn.Linear(globalChannels, self.globalBranchDim)
            if self.globalChannels > 0
            else None
        )
        self.globalTemporalConv = (
            self._buildTemporalConv(self.globalBranchDim)
            if self.globalChannels > 0
            else None
        )
        self.globalPosEncoding = (
            PositionalEncoding(self.globalBranchDim, dropout=dropout)
            if self.globalChannels > 0
            else None
        )
        self.globalTransformer = (
            self._buildTransformer(
                self.globalBranchDim,
                numHeads,
                numLayers,
                dropout,
            )
            if self.globalChannels > 0
            else None
        )
        self.globalFeatureProj = (
            nn.Linear(self.globalBranchDim, embedDim)
            if self.globalChannels > 0
            else None
        )

        if self.numChannels > 0 and self.globalChannels > 0:
            self.fusionProj = nn.Linear(embedDim * 2, embedDim)
            self.fusionAct = nn.GELU()
            self.fusionNorm = nn.LayerNorm(embedDim)
        else:
            self.fusionProj = None
            self.fusionAct = None
            self.fusionNorm = None

        # Preserve the legacy attribute for tests and compatibility.
        self.transformer = (
            self.boneTransformer
            if self.boneTransformer is not None
            else self.globalTransformer
        )

        self.outputProj = nn.Linear(embedDim, embedDim)
        self.layerNorm = nn.LayerNorm(embedDim)

    def forward(
        self,
        boneInput: torch.Tensor | None = None,
        globalInput: torch.Tensor | None = None,
        motionMask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode motion as a single embedding vector.

        Parameters
        ----------
        boneInput : torch.Tensor | None, optional
            Bone-scoped motion payload shaped (batch, frames, bones, channels).
        globalInput : torch.Tensor | None, optional
            Global motion payload shaped (batch, frames, channels).
        motionMask : torch.Tensor | None, optional
            Boolean tensor shaped (batch, frames) marking valid frames before
            temporal padding.

        Returns
        -------
        torch.Tensor
            Motion features shaped (batch, embedDim).
        """
        batchSize, frames, device = self._resolveShape(
            boneInput=boneInput,
            globalInput=globalInput,
        )
        reducedMask = self._normalizeMotionMask(
            motionMask=motionMask,
            batchSize=batchSize,
            frames=frames,
            device=device,
        )

        boneFeatures = self._encodeBoneBranch(boneInput, reducedMask)
        globalFeatures = self._encodeGlobalBranch(globalInput, reducedMask)
        x = self._fuseBranches(boneFeatures, globalFeatures)
        x = self.outputProj(x)
        x = self.layerNorm(x)
        return x

    def _buildTemporalConv(self, embedDim: int) -> nn.Sequential:
        """Return the shared temporal downsampling stack for one branch."""
        return nn.Sequential(
            nn.Conv1d(embedDim, embedDim, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(embedDim, embedDim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def _buildTransformer(
        self,
        embedDim: int,
        numHeads: int,
        numLayers: int,
        dropout: float,
    ) -> nn.TransformerEncoder:
        """Create one transformer branch with the MPS-safe configuration."""
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=embedDim,
            nhead=numHeads,
            dim_feedforward=embedDim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        return nn.TransformerEncoder(
            encoderLayer,
            num_layers=numLayers,
            enable_nested_tensor=False,
        )

    def _resolveGlobalBranchDim(
        self,
        embedDim: int,
        numHeads: int,
    ) -> int:
        """Keep the global branch compact while staying head-compatible."""
        candidate = min(embedDim, max(128, embedDim // 4))
        aligned = max(numHeads, (candidate // numHeads) * numHeads)
        return aligned

    def _resolveShape(
        self,
        boneInput: torch.Tensor | None,
        globalInput: torch.Tensor | None,
    ) -> tuple[int, int, torch.device]:
        """Validate branch shapes and return the shared batch/frame layout."""
        if boneInput is None and globalInput is None:
            raise ValueError("TemporalUNet requires boneInput or globalInput.")
        if boneInput is not None:
            if boneInput.dim() != 4:
                raise ValueError(
                    "Expected boneInput with shape (batch, frames, bones, channels), "
                    f"got {tuple(boneInput.shape)}."
                )
            if boneInput.shape[2] != self.numBones:
                raise ValueError(
                    f"Expected {self.numBones} bones in boneInput, got "
                    f"{boneInput.shape[2]}."
                )
            if boneInput.shape[3] != self.numChannels:
                raise ValueError(
                    "Unexpected boneInput channel count: expected "
                    f"{self.numChannels}, got {boneInput.shape[3]}."
                )
            if globalInput is not None:
                if globalInput.dim() != 3:
                    raise ValueError(
                        "Expected globalInput with shape (batch, frames, channels), "
                        f"got {tuple(globalInput.shape)}."
                    )
                if globalInput.shape[:2] != boneInput.shape[:2]:
                    raise ValueError(
                        "globalInput must share the same batch/frame shape as "
                        f"boneInput: expected {tuple(boneInput.shape[:2])}, got "
                        f"{tuple(globalInput.shape[:2])}."
                    )
                if globalInput.shape[2] != self.globalChannels:
                    raise ValueError(
                        "Unexpected globalInput channel count: expected "
                        f"{self.globalChannels}, got {globalInput.shape[2]}."
                    )
            return (
                int(boneInput.shape[0]),
                int(boneInput.shape[1]),
                boneInput.device,
            )
        assert globalInput is not None
        if globalInput.dim() != 3:
            raise ValueError(
                "Expected globalInput with shape (batch, frames, channels), got "
                f"{tuple(globalInput.shape)}."
            )
        if globalInput.shape[2] != self.globalChannels:
            raise ValueError(
                "Unexpected globalInput channel count: expected "
                f"{self.globalChannels}, got {globalInput.shape[2]}."
            )
        return (
            int(globalInput.shape[0]),
            int(globalInput.shape[1]),
            globalInput.device,
        )

    def _normalizeMotionMask(
        self,
        motionMask: torch.Tensor | None,
        batchSize: int,
        frames: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Validate the optional mask against the resolved branch layout."""
        if motionMask is None:
            return None
        if motionMask.dim() != 2:
            raise ValueError(
                "Expected motionMask with shape (batch, frames), got "
                f"{tuple(motionMask.shape)}."
            )
        if motionMask.shape != (batchSize, frames):
            raise ValueError(
                "motionMask must match the temporal shape of the motion input: "
                f"expected {(batchSize, frames)}, got {tuple(motionMask.shape)}."
            )
        return motionMask.to(device=device, dtype=torch.bool)

    def _encodeBoneBranch(
        self,
        boneInput: torch.Tensor | None,
        motionMask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Encode bone-scoped channels through the dedicated branch."""
        if boneInput is None:
            return None
        if (
            self.inputProj is None
            or self.temporalConv is None
            or self.posEncoding is None
            or self.boneTransformer is None
        ):
            raise RuntimeError("Bone branch is not initialized.")
        batchSize, frames, bones, channels = boneInput.shape
        flatInput = boneInput.view(batchSize, frames, bones * channels)
        return self._encodeTemporalSequence(
            sequenceInput=flatInput,
            motionMask=motionMask,
            inputProj=self.inputProj,
            temporalConv=self.temporalConv,
            posEncoding=self.posEncoding,
            transformer=self.boneTransformer,
        )

    def _encodeGlobalBranch(
        self,
        globalInput: torch.Tensor | None,
        motionMask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Encode global channels through the dedicated branch."""
        if globalInput is None:
            return None
        if (
            self.globalInputProj is None
            or self.globalTemporalConv is None
            or self.globalPosEncoding is None
            or self.globalTransformer is None
            or self.globalFeatureProj is None
        ):
            raise RuntimeError("Global branch is not initialized.")
        pooled = self._encodeTemporalSequence(
            sequenceInput=globalInput,
            motionMask=motionMask,
            inputProj=self.globalInputProj,
            temporalConv=self.globalTemporalConv,
            posEncoding=self.globalPosEncoding,
            transformer=self.globalTransformer,
        )
        return self.globalFeatureProj(pooled)

    def _encodeTemporalSequence(
        self,
        sequenceInput: torch.Tensor,
        motionMask: torch.Tensor | None,
        inputProj: nn.Linear,
        temporalConv: nn.Sequential,
        posEncoding: PositionalEncoding,
        transformer: nn.TransformerEncoder,
    ) -> torch.Tensor:
        """Run one branch from per-frame inputs to a pooled embedding."""
        reducedMask = motionMask

        x = inputProj(sequenceInput)
        if reducedMask is not None:
            x = x * reducedMask.unsqueeze(-1).to(dtype=x.dtype)

        x = x.transpose(1, 2)
        x = temporalConv[0](x)
        x = temporalConv[1](x)
        if reducedMask is not None:
            reducedMask = self._downsampleMask(
                reducedMask,
                kernelSize=5,
                stride=4,
                padding=2,
            )
            x = x * reducedMask.unsqueeze(1).to(dtype=x.dtype)

        x = temporalConv[2](x)
        x = temporalConv[3](x)
        if reducedMask is not None:
            reducedMask = self._downsampleMask(
                reducedMask,
                kernelSize=3,
                stride=2,
                padding=1,
            )
            x = x * reducedMask.unsqueeze(1).to(dtype=x.dtype)

        if x.size(2) > self.maxFrames:
            x = F.interpolate(
                x,
                size=self.maxFrames,
                mode="linear",
                align_corners=False,
            )
            if reducedMask is not None:
                reducedMask = (
                    F.interpolate(
                        reducedMask.unsqueeze(1).float(),
                        size=self.maxFrames,
                        mode="nearest",
                    ).squeeze(1)
                    > 0.5
                )

        x = x.transpose(1, 2)
        x = posEncoding(x)
        padMask = None
        if reducedMask is not None:
            padMask = ~reducedMask
            x = x.masked_fill(padMask.unsqueeze(-1), 0.0)

        x = transformer(x, src_key_padding_mask=padMask)
        if padMask is not None:
            x = x.masked_fill(padMask.unsqueeze(-1), 0.0)

        if reducedMask is not None:
            valid = reducedMask.unsqueeze(-1).to(dtype=x.dtype)
            denom = valid.sum(dim=1).clamp(min=1.0)
            return (x * valid).sum(dim=1) / denom
        return x.mean(dim=1)

    def _fuseBranches(
        self,
        boneFeatures: torch.Tensor | None,
        globalFeatures: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine branch embeddings only after temporal encoding."""
        if boneFeatures is None:
            if globalFeatures is None:
                raise RuntimeError("TemporalUNet produced no branch features.")
            return globalFeatures
        if globalFeatures is None:
            return boneFeatures
        if (
            self.fusionProj is None
            or self.fusionAct is None
            or self.fusionNorm is None
        ):
            raise RuntimeError("Fusion layers are not initialized.")
        fused = torch.cat([boneFeatures, globalFeatures], dim=-1)
        fused = self.fusionProj(fused)
        fused = self.fusionAct(fused)
        return self.fusionNorm(fused)

    def _downsampleMask(
        self,
        motionMask: torch.Tensor,
        kernelSize: int,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        """Project a frame-validity mask through the temporal conv strides."""
        pooled = F.max_pool1d(
            motionMask.unsqueeze(1).float(),
            kernel_size=kernelSize,
            stride=stride,
            padding=padding,
        )
        return pooled.squeeze(1) > 0.0


class TemporalUNetSimple(BaseLayer):
    """Simple fallback encoder (original implementation)."""

    def __init__(self, embedDim: int, numBones: int = 22, numChannels: int = 6) -> None:
        super().__init__(name="TemporalUNetSimple")
        inputDim = numBones * numChannels
        self.projection = nn.Linear(inputDim, embedDim)

    def forward(self, motionInput: torch.Tensor) -> torch.Tensor:
        """Encode motion as a single embedding vector."""
        batchSize, frames, bones, channels = motionInput.shape
        pooled = motionInput.mean(dim=1).view(batchSize, bones * channels)
        return self.projection(pooled)
