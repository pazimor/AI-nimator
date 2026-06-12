import torch
import torch.nn as nn
from .base import BaseLayer

class TemporalLayer(BaseLayer):
    """
    Temporal Layer.
    
    Handles temporal aspects of the data, potentially using adaptive masking (DAEDAL-like).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize TemporalLayer.

        Parameters
        ----------
        embed_dim : int
            Embedding dimension.
        num_heads : int
            Number of attention heads.
        dropout : float, optional
            Dropout rate, by default 0.1.
        """
        super().__init__(name="TemporalLayer")
        # Using a Transformer Encoder Layer as a base for temporal processing
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of TemporalLayer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).
        mask : torch.Tensor, optional
            Temporal mask, by default None.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # Self-Attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed Forward Network
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return x
