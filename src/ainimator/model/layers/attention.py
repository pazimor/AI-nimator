import torch
import torch.nn as nn
from .base import BaseLayer

class MultiHeadAttention(BaseLayer):
    """
    Multi-Head Attention (MHA) factorized on bones.
    
    Standard MHA implementation that can be used for bone-factorized attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        """
        Initialize MultiHeadAttention.

        Parameters
        ----------
        embed_dim : int
            Total dimension of the model.
        num_heads : int
            Number of parallel attention heads.
        dropout : float, optional
            Dropout probability, by default 0.0.
        bias : bool, optional
            Whether to add bias to query, key, value projections, by default True.
        """
        super().__init__(name="MultiHeadAttention")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor = None, 
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of MHA.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).
        mask : torch.Tensor, optional
            Attention mask, by default None.
        is_causal : bool, optional
            Whether to apply causal masking, by default False.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: (B, T, H, D_head) -> (B, H, T, D_head)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # mask should be broadcastable to (B, H, T, T)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate heads
        out = torch.matmul(attn_weights, v)
        
        # Reshape back: (B, H, T, D_head) -> (B, T, H, D_head) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(out)
