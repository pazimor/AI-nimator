import math
import torch
import torch.nn as nn

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
    
    Encodes (batch, frames, bones, 6) motion data into (batch, embedDim) features.
    Uses temporal downsampling to handle long sequences efficiently.
    """

    def __init__(
        self,
        embedDim: int,
        numHeads: int = 4,
        numLayers: int = 2,
        dropout: float = 0.1,
        numBones: int = 22,
        numChannels: int = 6,
        maxFrames: int = 128,  # Max frames after downsampling
    ) -> None:
        super().__init__(name="TemporalUNet")
        self.embedDim = embedDim
        self.numBones = numBones
        self.numChannels = numChannels
        self.maxFrames = maxFrames
        
        # Input projection: (bones * 6) -> embedDim
        inputDim = numBones * numChannels  # 22 * 6 = 132
        self.inputProj = nn.Linear(inputDim, embedDim)
        
        # 1D Conv for temporal downsampling (stride=4 reduces 2400->600, then pool to 128)
        self.temporalConv = nn.Sequential(
            nn.Conv1d(embedDim, embedDim, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(embedDim, embedDim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        
        # Positional encoding for frames (computed dynamically)
        self.posEncoding = PositionalEncoding(embedDim, dropout=dropout)
        
        # Transformer encoder for temporal modeling
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=embedDim,
            nhead=numHeads,
            dim_feedforward=embedDim * 2,  # Reduced from 4x to 2x
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        
        # Output projection
        self.outputProj = nn.Linear(embedDim, embedDim)
        self.layerNorm = nn.LayerNorm(embedDim)

    def forward(self, motionInput: torch.Tensor) -> torch.Tensor:
        """
        Encode motion as a single embedding vector.

        Parameters
        ----------
        motionInput : torch.Tensor
            Motion payload shaped (batch, frames, bones, 6).

        Returns
        -------
        torch.Tensor
            Motion features shaped (batch, embedDim).
        """
        batchSize, frames, bones, channels = motionInput.shape
        
        # Flatten bones and channels: (batch, frames, bones*6)
        x = motionInput.view(batchSize, frames, bones * channels)
        
        # Project to embedding dimension: (batch, frames, embedDim)
        x = self.inputProj(x)
        
        # Temporal downsampling via Conv1d: (batch, embedDim, frames) -> (batch, embedDim, reduced_frames)
        x = x.transpose(1, 2)  # (batch, embedDim, frames)
        x = self.temporalConv(x)  # Reduces by ~8x (stride 4 * stride 2)
        
        # If still too long, downsample to maxFrames using interpolation (MPS compatible)
        if x.size(2) > self.maxFrames:
            x = nn.functional.interpolate(x, size=self.maxFrames, mode="linear", align_corners=False)
        
        x = x.transpose(1, 2)  # (batch, reduced_frames, embedDim)
        
        # Add positional encoding
        x = self.posEncoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling over frames: (batch, embedDim)
        x = x.mean(dim=1)
        
        # Final projection and normalization
        x = self.outputProj(x)
        x = self.layerNorm(x)
        
        return x


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
