"""Custom text encoder for AI-nimator v2.

A small bidirectional transformer trained jointly with the motion
denoiser.  It consumes the output of :class:`CustomTokenizer` and
produces the per-token embedding sequence consumed by the cross-attention
in the denoiser blocks.

Design choices (reflecting the v2 plan)
---------------------------------------
* **Pre-norm** transformer blocks (LayerNorm before attention/FFN) — more
  stable than post-norm at small depth, and standard for modern
  transformers since GPT-2.
* **Learned absolute positional embeddings** — sinusoidal would also
  work, but learned embeddings on a fixed ``maxLength=64`` add only
  ~16k params and tend to converge faster on small corpora.
* **GELU** activations and an FFN expansion ratio of 4 (256 → 1024 →
  256), matching the existing :class:`TemporalLayer` pattern in this
  codebase for visual consistency.
* **Output = full sequence** ``(B, T, hidden)`` — the cross-attention in
  the denoiser consumes per-token K/V; we never pool here (FiLM is
  removed in v2).
* **Padding mask** is a key-padding mask (``True`` on padding positions)
  derived from the tokenizer's ``attentionMask``; ``MultiheadAttention``
  expects this convention.
* **No dropout by default in inference** — handled by ``model.eval()``.

Parameter budget (default config: vocab=8000, hidden=256, heads=8, layers=4)
* token embed:           8000 × 256              ≈ 2.05M
* positional embed:        64 × 256              ≈ 16k
* per layer (4 layers):
  * QKV projection:      3 × 256 × 256           ≈ 197k
  * output projection:       256 × 256           ≈ 66k
  * FFN (256→1024→256):  ~525k
  * 2 LayerNorms:        ~1k
  * Total per layer:     ~789k
* 4 layers stack:        ~3.16M
* output LayerNorm:        512
* projection (optional):  hidden × outDim
* **Total ≈ 5.2M params** (without external projection).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from src.shared.model.text.custom_tokenizer import (
    CustomTokenizer,
    EncodedBatch,
)


@dataclass(frozen=True)
class CustomTextEncoderConfig:
    """Configuration for :class:`CustomTextEncoder`.

    Parameters
    ----------
    vocabSize : int
        Vocabulary size of the tokenizer the encoder is paired with.
        Must match :attr:`CustomTokenizer.vocabSize`.
    maxLength : int
        Maximum number of tokens per sequence.  Defines the size of the
        learned positional embedding table.
    hiddenDim : int
        Width of the encoder.  ``256`` keeps the encoder ~5M params.
    numLayers : int
        Number of stacked transformer blocks.  ``4`` is the v2 default.
    numHeads : int
        Number of attention heads.  Must divide ``hiddenDim``.  ``8``
        gives 32 dim/head with hidden=256.
    ffnDim : int
        Hidden dimension of the FFN.  Defaults to ``4 * hiddenDim`` when
        zero.
    dropout : float
        Dropout probability applied inside attention and FFN sublayers.
    padTokenId : int
        Token id that should be masked out from self-attention
        contributions.  Defaults to ``0`` (the canonical pad id from
        :class:`CustomTokenizer`).
    outputDim : int
        Optional projection of the per-token embeddings to a different
        width.  Set to ``hiddenDim`` (or 0) to disable.  This is the dim
        the downstream cross-attention will see — usually we set this
        equal to the denoiser's ``embedDim`` to avoid an extra projection
        on the consumer side.
    """

    vocabSize: int
    maxLength: int = 64
    hiddenDim: int = 256
    numLayers: int = 4
    numHeads: int = 8
    ffnDim: int = 0  # 0 → use 4 × hiddenDim
    dropout: float = 0.1
    padTokenId: int = 0
    outputDim: int = 0  # 0 → use hiddenDim (no extra projection)

    def __post_init__(self) -> None:
        if self.hiddenDim % self.numHeads != 0:
            raise ValueError(
                f"hiddenDim ({self.hiddenDim}) must be divisible by "
                f"numHeads ({self.numHeads})."
            )
        if self.numLayers < 1:
            raise ValueError("numLayers must be >= 1.")
        if self.maxLength < 1:
            raise ValueError("maxLength must be >= 1.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")

    @property
    def effectiveFfnDim(self) -> int:
        """FFN inner dimension after applying defaults."""
        return self.ffnDim if self.ffnDim > 0 else 4 * self.hiddenDim

    @property
    def effectiveOutputDim(self) -> int:
        """Output dimension exposed to consumers."""
        return self.outputDim if self.outputDim > 0 else self.hiddenDim


class _PreNormSelfAttention(nn.Module):
    """Pre-norm bidirectional self-attention sublayer."""

    def __init__(
        self,
        hiddenDim: int,
        numHeads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hiddenDim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hiddenDim,
            num_heads=numHeads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        keyPaddingMask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply pre-norm self-attention with residual connection."""
        normalized = self.norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=keyPaddingMask,
            need_weights=False,
        )
        return x + self.dropout(attended)


class _PreNormFeedForward(nn.Module):
    """Pre-norm position-wise FFN sublayer (GELU, expansion 4)."""

    def __init__(
        self,
        hiddenDim: int,
        ffnDim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hiddenDim)
        self.ffn = nn.Sequential(
            nn.Linear(hiddenDim, ffnDim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffnDim, hiddenDim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm FFN with residual connection."""
        return x + self.dropout(self.ffn(self.norm(x)))


class _TextEncoderBlock(nn.Module):
    """Pre-norm transformer block: self-attn + FFN."""

    def __init__(
        self,
        hiddenDim: int,
        numHeads: int,
        ffnDim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = _PreNormSelfAttention(
            hiddenDim=hiddenDim,
            numHeads=numHeads,
            dropout=dropout,
        )
        self.feedForward = _PreNormFeedForward(
            hiddenDim=hiddenDim,
            ffnDim=ffnDim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        keyPaddingMask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.attention(x, keyPaddingMask=keyPaddingMask)
        x = self.feedForward(x)
        return x


@dataclass(frozen=True)
class TextEncoderOutput:
    """Result of :meth:`CustomTextEncoder.forward`.

    Attributes
    ----------
    hiddenStates : torch.Tensor
        Per-token contextualised embeddings of shape ``(B, T, D_out)``
        where ``D_out`` matches :attr:`CustomTextEncoderConfig.effectiveOutputDim`.
        This is the tensor consumed by the cross-attention in the
        denoiser (Q from motion, K/V from this).
    keyPaddingMask : torch.Tensor
        Bool tensor of shape ``(B, T)`` with ``True`` on padding positions
        and ``False`` on real tokens.  Pass it directly to a downstream
        ``nn.MultiheadAttention`` call as ``key_padding_mask``.
    """

    hiddenStates: torch.Tensor
    keyPaddingMask: torch.Tensor


class CustomTextEncoder(nn.Module):
    """Compact bidirectional transformer for motion-prompt encoding.

    Pair this module with :class:`CustomTokenizer`.  The typical training
    flow is::

        tokenizer = CustomTokenizer.load("output/text/custom_tokenizer")
        encoder = CustomTextEncoder.fromTokenizer(tokenizer)
        batch = tokenizer.encode(["a person walks forward."])
        output = encoder(batch.inputIds, batch.attentionMask)
        # output.hiddenStates: (B, T, D)  →  cross-attn K/V
    """

    def __init__(self, config: CustomTextEncoderConfig) -> None:
        super().__init__()
        self._config = config

        self.tokenEmbedding = nn.Embedding(
            num_embeddings=config.vocabSize,
            embedding_dim=config.hiddenDim,
            padding_idx=config.padTokenId,
        )
        self.positionEmbedding = nn.Embedding(
            num_embeddings=config.maxLength,
            embedding_dim=config.hiddenDim,
        )
        self.embeddingDropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                _TextEncoderBlock(
                    hiddenDim=config.hiddenDim,
                    numHeads=config.numHeads,
                    ffnDim=config.effectiveFfnDim,
                    dropout=config.dropout,
                )
                for _ in range(config.numLayers)
            ]
        )
        self.finalNorm = nn.LayerNorm(config.hiddenDim)

        if config.effectiveOutputDim != config.hiddenDim:
            self.outputProjection: nn.Module = nn.Linear(
                config.hiddenDim,
                config.effectiveOutputDim,
                bias=True,
            )
        else:
            self.outputProjection = nn.Identity()

        self._initWeights()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def fromTokenizer(
        cls,
        tokenizer: CustomTokenizer,
        hiddenDim: int = 256,
        numLayers: int = 4,
        numHeads: int = 8,
        ffnDim: int = 0,
        dropout: float = 0.1,
        outputDim: int = 0,
    ) -> "CustomTextEncoder":
        """Build an encoder whose vocab/maxLength/padId match ``tokenizer``.

        This avoids the most common configuration mistake (mismatched
        vocab sizes between tokenizer and encoder).
        """
        config = CustomTextEncoderConfig(
            vocabSize=tokenizer.vocabSize,
            maxLength=tokenizer.config.maxLength,
            hiddenDim=hiddenDim,
            numLayers=numLayers,
            numHeads=numHeads,
            ffnDim=ffnDim,
            dropout=dropout,
            padTokenId=tokenizer.padTokenId,
            outputDim=outputDim,
        )
        return cls(config=config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> CustomTextEncoderConfig:
        return self._config

    @property
    def outputDim(self) -> int:
        """Effective output dimension exposed to the cross-attention."""
        return self._config.effectiveOutputDim

    def numParameters(self, trainableOnly: bool = True) -> int:
        """Return the total number of (trainable) parameters."""
        if trainableOnly:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        inputIds: torch.Tensor,
        attentionMask: torch.Tensor,
    ) -> TextEncoderOutput:
        """Encode a batch of token sequences.

        Parameters
        ----------
        inputIds : torch.Tensor
            Long tensor of shape ``(B, T)`` from
            :meth:`CustomTokenizer.encode`.
        attentionMask : torch.Tensor
            Float tensor of shape ``(B, T)`` with ``1.0`` on real tokens
            and ``0.0`` on padding positions (the convention used by the
            tokenizer).  ``None`` is **not** accepted on purpose to keep
            the contract explicit.

        Returns
        -------
        TextEncoderOutput
            Hidden states of shape ``(B, T, D_out)`` and the key-padding
            mask suitable for downstream cross-attention.
        """
        if inputIds.ndim != 2:
            raise ValueError(
                f"inputIds must be 2-D (B, T); got shape "
                f"{tuple(inputIds.shape)}."
            )
        if attentionMask.shape != inputIds.shape:
            raise ValueError(
                f"attentionMask shape {tuple(attentionMask.shape)} does "
                f"not match inputIds shape {tuple(inputIds.shape)}."
            )
        batchSize, sequenceLength = inputIds.shape
        if sequenceLength > self._config.maxLength:
            raise ValueError(
                f"Sequence length {sequenceLength} exceeds configured "
                f"maxLength {self._config.maxLength}."
            )

        # --- Embeddings ------------------------------------------------
        positions = torch.arange(
            sequenceLength,
            device=inputIds.device,
        ).unsqueeze(0).expand(batchSize, -1)
        hidden = self.tokenEmbedding(inputIds) + self.positionEmbedding(
            positions
        )
        hidden = self.embeddingDropout(hidden)

        # --- Build key-padding mask (True on padding) ------------------
        # nn.MultiheadAttention expects True == ignore.  Tokenizer gives
        # 1.0 on real tokens and 0.0 on padding, so invert.
        keyPaddingMask = attentionMask <= 0

        # --- Transformer stack ----------------------------------------
        for block in self.blocks:
            hidden = block(hidden, keyPaddingMask=keyPaddingMask)
        hidden = self.finalNorm(hidden)
        hidden = self.outputProjection(hidden)

        return TextEncoderOutput(
            hiddenStates=hidden,
            keyPaddingMask=keyPaddingMask,
        )

    # ------------------------------------------------------------------
    # Convenience: encode raw texts in one call
    # ------------------------------------------------------------------
    def encodeTexts(
        self,
        tokenizer: CustomTokenizer,
        texts: Sequence[str] | str,
    ) -> TextEncoderOutput:
        """Tokenize and encode a batch of strings on the encoder's device.

        Useful for inference and quick smoke tests; training code should
        keep tokenization out of the model's hot path.
        """
        encoded: EncodedBatch = tokenizer.encode(texts)
        device = next(self.parameters()).device
        return self.forward(
            inputIds=encoded.inputIds.to(device),
            attentionMask=encoded.attentionMask.to(device),
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initWeights(self) -> None:
        """Initialise weights with the truncated-normal scheme used by BERT/GPT."""
        std = 1.0 / math.sqrt(self._config.hiddenDim)
        nn.init.normal_(self.tokenEmbedding.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.tokenEmbedding.weight[self._config.padTokenId].zero_()
        nn.init.normal_(self.positionEmbedding.weight, mean=0.0, std=std)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
