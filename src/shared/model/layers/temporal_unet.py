import torch
import torch.nn as nn

from .base import BaseLayer


class TemporalUNet(BaseLayer):
    """Placeholder temporal encoder producing motion features."""

    def __init__(self, embedDim: int) -> None:
        super().__init__(name="TemporalUNet")
        self.projection = nn.LazyLinear(embedDim)

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
        pooled = motionInput.mean(dim=1).view(batchSize, bones * channels)
        return self.projection(pooled)
