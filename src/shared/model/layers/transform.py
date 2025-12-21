import torch
import torch.nn as nn
from .base import BaseLayer

class TransformLayer(BaseLayer):
    """
    Transform Layer (Residual 6D/Skip).
    
    Applies a transformation with residual connections, suitable for 6D rotation representations.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        """
        Initialize TransformLayer.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        hidden_dim : int
            Hidden dimension.
        output_dim : int
            Output dimension.
        dropout : float, optional
            Dropout probability, by default 0.0.
        """
        super().__init__(name="TransformLayer")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # If input and output dims match, we can use a direct skip connection.
        # Otherwise, we might need a projection or just rely on the net.
        # Given "Residual 6D/Skip", it implies a residual connection.
        self.use_residual = (input_dim == output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TransformLayer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        out = self.net(x)
        
        if self.use_residual:
            out = out + x
            
        return out
