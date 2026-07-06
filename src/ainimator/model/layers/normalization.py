import torch
import torch.nn as nn
from .base import BaseLayer

class AdaLN(BaseLayer):
    """
    Adaptive Layer Normalization (AdaLN).
    
    Modulates the layer normalization parameters (scale and shift) based on a conditioning embedding.
    """

    def __init__(self, num_features: int, cond_dim: int, eps: float = 1e-5):
        """
        Initialize AdaLN.

        Parameters
        ----------
        num_features : int
            Number of features in the input.
        cond_dim : int
            Dimension of the conditioning embedding.
        eps : float, optional
            Epsilon for numerical stability, by default 1e-5.
        """
        super().__init__(name="AdaLN")
        self.num_features = num_features
        self.eps = eps
        
        # Linear layer to project conditioning embedding to scale and shift
        self.cond_proj = nn.Linear(cond_dim, 2 * num_features)
        
        # Initialize weights to zero for identity start
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of AdaLN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., num_features).
        cond : torch.Tensor
            Conditioning tensor of shape (batch_size, cond_dim).

        Returns
        -------
        torch.Tensor
            Normalized and modulated input.
        """
        # Calculate mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Project conditioning to scale (gamma) and shift (beta)
        # Shape: (batch_size, 2 * num_features)
        style = self.cond_proj(cond)
        
        # Reshape for broadcasting if necessary, assuming cond is (B, D) and x is (B, ..., D)
        # We need to unsqueeze style to match x's dimensions except the last one
        while style.dim() < x.dim():
            style = style.unsqueeze(1)
            
        gamma, beta = style.chunk(2, dim=-1)
        
        # Modulate
        return x_norm * (1 + gamma) + beta


class FiLM(BaseLayer):
    """
    Feature-wise Linear Modulation (FiLM).
    
    Applies an affine transformation to the input features based on conditioning information.
    """

    def __init__(self, num_features: int, cond_dim: int):
        """
        Initialize FiLM.

        Parameters
        ----------
        num_features : int
            Number of features in the input.
        cond_dim : int
            Dimension of the conditioning embedding.
        """
        super().__init__(name="FiLM")
        self.cond_proj = nn.Linear(cond_dim, 2 * num_features)
        
        # Initialize weights
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FiLM.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., num_features).
        cond : torch.Tensor
            Conditioning tensor of shape (batch_size, cond_dim).

        Returns
        -------
        torch.Tensor
            Modulated input.
        """
        # Project conditioning to scale (gamma) and shift (beta)
        style = self.cond_proj(cond)
        
        # Reshape for broadcasting
        while style.dim() < x.dim():
            style = style.unsqueeze(1)
            
        gamma, beta = style.chunk(2, dim=-1)
        
        # Apply affine transformation: x * gamma + beta
        # Note: In some FiLM implementations it is x * (1 + gamma) + beta. 
        # Standard FiLM is often x * gamma + beta. 
        # Given the graph shows "note_layers2[FiLM] -.-> temporal", it's likely used for conditioning.
        # I will use the standard affine definition here, but initialize to identity-like behavior if needed.
        # However, for stable training, often gamma is initialized to 1 and beta to 0.
        # Since I initialized weights to 0, the projection gives 0. 
        # So x * 0 + 0 = 0 which is bad.
        # Let's change the calculation to x * (1 + gamma) + beta to be safe with 0 init, 
        # or add 1 to gamma.
        
        return x * (1 + gamma) + beta
