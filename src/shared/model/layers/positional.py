import torch
import torch.nn as nn
from .base import BaseLayer

class RoPE(BaseLayer):
    """
    Rotary Positional Embedding (RoPE).
    
    Applies rotary positional embeddings to the input tensor.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        """
        Initialize RoPE.

        Parameters
        ----------
        dim : int
            Dimension of the embeddings.
        max_position_embeddings : int, optional
            Maximum number of positions, by default 2048.
        base : int, optional
            Base for the geometric progression, by default 10000.
        """
        super().__init__(name="RoPE")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for cosine and sine
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x: torch.Tensor, seq_len: int):
        """
        Update cached cosine and sine tables.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to get device and dtype.
        seq_len : int
            Sequence length.
        """
        if (
            self._cos_cached is not None
            and seq_len <= self._cos_cached.shape[0]
            and self._cos_cached.device == x.device
            and self._cos_cached.dtype == x.dtype
        ):
            return

        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self._cos_cached = emb.cos().to(dtype=x.dtype)
        self._sin_cached = emb.sin().to(dtype=x.dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the hidden dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Rotated tensor.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RoPE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, ..., dim).

        Returns
        -------
        torch.Tensor
            Tensor with positional embeddings applied.
        """
        # Assuming x is (batch_size, seq_len, dim) or similar where dim is last
        # and seq_len is second dimension.
        # If x has more dimensions (e.g. batch, seq, heads, dim), we need to handle broadcasting.
        
        seq_len = x.shape[1]
        self._update_cos_sin_tables(x, seq_len)
        
        cos = self._cos_cached[:seq_len, ...]
        sin = self._sin_cached[:seq_len, ...]
        
        # Reshape cos and sin for broadcasting
        # Assuming x: (B, T, ..., D)
        # cos, sin: (T, D) -> (1, T, 1, ..., D)
        
        # We need to unsqueeze to match x's rank
        while cos.dim() < x.dim():
            if cos.dim() == 1: # (D) -> (T, D) handled by slicing above, wait.
                 # _update_cos_sin_tables creates (T, D)
                 pass
            elif cos.dim() == 2: # (T, D)
                 # If x is (B, T, D), we need (1, T, D)
                 cos = cos.unsqueeze(0)
                 sin = sin.unsqueeze(0)
            else:
                 # If x is (B, T, H, D), we need (1, T, 1, D)
                 cos = cos.unsqueeze(2)
                 sin = sin.unsqueeze(2)
                 
        # Apply rotation
        return (x * cos) + (self._rotate_half(x) * sin)
