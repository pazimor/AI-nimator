"""Frozen CLIP text encoder for AI-nimator v2 (Phase 2).

The custom 8K-BPE encoder (:mod:`custom_text_encoder`) is trained from
scratch on ~56k motion captions.  At that corpus size it cannot build a
sharp semantic space — "jumping", "stepping" and "walking" stay close
together, so the diffusion model receives a blurry conditioning signal.

This module swaps in the text tower of OpenAI CLIP (ViT-B/32) which was
pretrained contrastively on 400M image-text pairs.  CLIP is kept fully
**frozen**; only a small trainable projection (512 → ``outputDim``) and
the learnable CFG null embedding receive gradients.  The public surface
mirrors :class:`CustomTextEncoder` / :class:`CustomTokenizer` exactly so
the rest of the v2 stack treats it as a drop-in replacement:

* :class:`ClipTokenizer` exposes ``encode(texts) -> EncodedBatch`` and
  the ``config.maxLength`` / ``vocabSize`` / ``padTokenId`` attributes.
* :class:`ClipTextEncoder` exposes
  ``forward(inputIds, attentionMask) -> TextEncoderOutput``,
  ``forwardNull(batchSize, ...)`` and the ``config`` / ``outputDim`` /
  ``nullEmbedding`` attributes.

Both the CLIP weights and the CLIP BPE tokenizer are loaded from the
HuggingFace hub via ``transformers``.  The first call downloads
``openai/clip-vit-base-patch32`` (~600 MB) and caches it; offline
environments must pre-populate the HF cache.

Trainable parameter budget (outputDim=384):
* projection Linear 512→384 + LayerNorm  ≈ 197k
* null embedding 1×1×384                 ≈ 384
* **Total trainable ≈ 0.2M** (CLIP's ~63M stay frozen).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.model.text.custom_text_encoder import TextEncoderOutput
from src.shared.model.text.custom_tokenizer import EncodedBatch

# Default CLIP checkpoint — the ViT-B/32 variant whose text tower is a
# 12-layer / 512-dim transformer with a 77-token context window and a
# 49408-entry BPE vocabulary.
DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_HIDDEN_DIM = 512
CLIP_CONTEXT_LENGTH = 77


# =====================================================================
# Tokenizer
# =====================================================================
@dataclass(frozen=True)
class ClipTokenizerConfig:
    """Configuration mirror for :class:`ClipTokenizer`.

    Kept as a tiny dataclass so callers can read ``tokenizer.config.
    maxLength`` exactly like they do for :class:`CustomTokenizer`.
    """

    modelName: str = DEFAULT_CLIP_MODEL_NAME
    maxLength: int = 32

    def __post_init__(self) -> None:
        if not (1 <= self.maxLength <= CLIP_CONTEXT_LENGTH):
            raise ValueError(
                f"maxLength must be in [1, {CLIP_CONTEXT_LENGTH}]; got "
                f"{self.maxLength}."
            )


class ClipTokenizer:
    """Drop-in replacement for :class:`CustomTokenizer` backed by CLIP BPE.

    Motion prompts are short (a handful of words), so the default
    ``maxLength`` of 32 is comfortably above the typical caption length
    while keeping the per-step text cost low.  Sequences are padded /
    truncated to a fixed length so every batch has a uniform shape.
    """

    def __init__(
        self,
        modelName: str = DEFAULT_CLIP_MODEL_NAME,
        maxLength: int = 32,
    ) -> None:
        from transformers import CLIPTokenizerFast

        self._config = ClipTokenizerConfig(
            modelName=modelName, maxLength=maxLength
        )
        self._backend = CLIPTokenizerFast.from_pretrained(modelName)

    @property
    def config(self) -> ClipTokenizerConfig:
        return self._config

    @property
    def vocabSize(self) -> int:
        return int(self._backend.vocab_size)

    @property
    def padTokenId(self) -> int:
        padId = self._backend.pad_token_id
        if padId is None:
            # CLIP pads with the EOS token; fall back to it explicitly.
            padId = self._backend.eos_token_id
        return int(padId)

    def encode(self, texts: Sequence[str] | str) -> EncodedBatch:
        """Tokenise a batch of strings to a fixed-length :class:`EncodedBatch`.

        Returns ``inputIds`` / ``attentionMask`` of shape
        ``(B, maxLength)`` — ``attentionMask`` is float (1.0 on real
        tokens, 0.0 on padding) to match the :class:`CustomTokenizer`
        contract consumed by the encoder.
        """
        if isinstance(texts, str):
            texts = [texts]
        encoded = self._backend(
            list(texts),
            padding="max_length",
            max_length=self._config.maxLength,
            truncation=True,
            return_tensors="pt",
        )
        inputIds = encoded["input_ids"].to(torch.long)
        attentionMask = encoded["attention_mask"].to(torch.float32)
        lengths = attentionMask.sum(dim=-1).to(torch.long)
        return EncodedBatch(
            inputIds=inputIds,
            attentionMask=attentionMask,
            lengths=lengths,
        )


# =====================================================================
# Encoder
# =====================================================================
@dataclass(frozen=True)
class ClipTextEncoderConfig:
    """Configuration for :class:`ClipTextEncoder`.

    Parameters
    ----------
    modelName : str
        HuggingFace identifier of the CLIP checkpoint whose text tower
        is loaded and frozen.
    maxLength : int
        Token budget — must match the paired :class:`ClipTokenizer`.
        Recorded only for round-trip / sanity checks.
    outputDim : int
        Width the per-token embeddings are projected to before they are
        exposed to the denoiser cross-attention.  Set this equal to the
        denoiser ``embedDim`` so no extra projection is needed on the
        consumer side (mirrors ``CustomTextEncoderConfig.outputDim``).
    clipHiddenDim : int
        Hidden width of the frozen CLIP text tower (512 for ViT-B/32).
    dropout : float
        Dropout applied inside the trainable projection head.
    useNullEmbedding : bool
        Build a learnable null embedding for the CFG unconditional
        branch (see :meth:`ClipTextEncoder.forwardNull`).
    l2NormalizeOutput : bool
        L2-normalise the per-token output — stabilises the magnitude of
        the cross-attention K/V, matching the custom encoder.
    """

    modelName: str = DEFAULT_CLIP_MODEL_NAME
    maxLength: int = 32
    outputDim: int = 384
    clipHiddenDim: int = CLIP_HIDDEN_DIM
    dropout: float = 0.0
    useNullEmbedding: bool = True
    l2NormalizeOutput: bool = True

    def __post_init__(self) -> None:
        if self.outputDim < 1:
            raise ValueError("outputDim must be >= 1.")
        if self.clipHiddenDim < 1:
            raise ValueError("clipHiddenDim must be >= 1.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if not (1 <= self.maxLength <= CLIP_CONTEXT_LENGTH):
            raise ValueError(
                f"maxLength must be in [1, {CLIP_CONTEXT_LENGTH}]; got "
                f"{self.maxLength}."
            )

    @property
    def effectiveOutputDim(self) -> int:
        """Output dimension exposed to consumers (API parity with custom)."""
        return self.outputDim


class ClipTextEncoder(nn.Module):
    """Frozen CLIP text tower + trainable projection head.

    The CLIP weights never receive gradient and are pinned to ``eval``
    mode even when the parent module is switched to ``train`` — frozen
    submodules must not run dropout / update running stats.  Only
    :attr:`outputProjection` and :attr:`nullEmbedding` are trainable.
    """

    def __init__(self, config: ClipTextEncoderConfig) -> None:
        super().__init__()
        self._config = config

        from transformers import CLIPTextModel

        self.clip = CLIPTextModel.from_pretrained(config.modelName)
        # Freeze the entire CLIP text tower.
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.clip.eval()

        # Trainable projection: CLIP hidden width → denoiser embed dim.
        # A LayerNorm after the Linear keeps the projected magnitude
        # stable before the optional L2-normalisation.
        projectionLayers: list[nn.Module] = [
            nn.Linear(config.clipHiddenDim, config.outputDim, bias=True),
        ]
        if config.dropout > 0.0:
            projectionLayers.append(nn.Dropout(config.dropout))
        projectionLayers.append(nn.LayerNorm(config.outputDim))
        self.outputProjection = nn.Sequential(*projectionLayers)

        # Learnable CFG null embedding — identical role to the custom
        # encoder's: a single token in a distinct region of the
        # embedding space, used for the unconditional branch.
        if config.useNullEmbedding:
            self.nullEmbedding = nn.Parameter(
                torch.randn(1, 1, config.outputDim)
                * (1.0 / math.sqrt(config.outputDim))
            )
        else:
            self.register_parameter("nullEmbedding", None)

        self._initProjection()

    # ------------------------------------------------------------------
    # Frozen-submodule discipline
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "ClipTextEncoder":
        """Switch to train mode but keep the frozen CLIP tower in eval."""
        super().train(mode)
        self.clip.eval()
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> ClipTextEncoderConfig:
        return self._config

    @property
    def outputDim(self) -> int:
        """Effective output dimension exposed to the cross-attention."""
        return self._config.outputDim

    def numParameters(self, trainableOnly: bool = True) -> int:
        """Return the (trainable) parameter count.

        With ``trainableOnly=True`` (default) only the projection head
        and null embedding are counted — the ~63M frozen CLIP params
        are excluded.
        """
        if trainableOnly:
            return sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        inputIds: torch.Tensor,
        attentionMask: torch.Tensor,
    ) -> TextEncoderOutput:
        """Encode a batch of CLIP-tokenised sequences.

        Parameters
        ----------
        inputIds : torch.Tensor
            Long tensor ``(B, T)`` from :meth:`ClipTokenizer.encode`.
        attentionMask : torch.Tensor
            Float tensor ``(B, T)`` — 1.0 on real tokens, 0.0 on padding.

        Returns
        -------
        TextEncoderOutput
            Per-token hidden states ``(B, T, outputDim)`` and the
            key-padding mask ``(B, T)`` (True on padding).
        """
        if inputIds.ndim != 2:
            raise ValueError(
                f"inputIds must be 2-D (B, T); got {tuple(inputIds.shape)}."
            )
        if attentionMask.shape != inputIds.shape:
            raise ValueError(
                f"attentionMask shape {tuple(attentionMask.shape)} does "
                f"not match inputIds shape {tuple(inputIds.shape)}."
            )

        # CLIP is frozen — no graph is built through it.  The trainable
        # projection downstream still receives gradient normally.
        with torch.no_grad():
            clipOutput = self.clip(
                input_ids=inputIds,
                attention_mask=attentionMask,
            )
        hidden = clipOutput.last_hidden_state  # (B, T, clipHiddenDim)

        hidden = self.outputProjection(hidden)  # (B, T, outputDim)
        if self._config.l2NormalizeOutput:
            hidden = F.normalize(hidden, p=2.0, dim=-1, eps=1e-8)

        keyPaddingMask = attentionMask <= 0
        return TextEncoderOutput(
            hiddenStates=hidden,
            keyPaddingMask=keyPaddingMask,
        )

    def forwardNull(
        self,
        batchSize: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TextEncoderOutput:
        """Return the learnable null embedding for CFG's unconditional branch.

        Mirrors :meth:`CustomTextEncoder.forwardNull` exactly so the
        training / inference CFG plumbing is encoder-agnostic.
        """
        if self.nullEmbedding is None:
            raise RuntimeError(
                "ClipTextEncoder.forwardNull called but the encoder was "
                "configured with useNullEmbedding=False."
            )
        if batchSize < 1:
            raise ValueError(f"batchSize must be >= 1, got {batchSize}.")

        targetDevice = (
            torch.device(device)
            if device is not None
            else self.nullEmbedding.device
        )
        targetDtype = (
            dtype if dtype is not None else self.nullEmbedding.dtype
        )

        nullToken = self.nullEmbedding.to(
            device=targetDevice, dtype=targetDtype
        )
        if self._config.l2NormalizeOutput:
            nullToken = F.normalize(nullToken, p=2.0, dim=-1, eps=1e-8)
        hiddenStates = nullToken.expand(batchSize, 1, -1).contiguous()
        keyPaddingMask = torch.zeros(
            batchSize, 1, dtype=torch.bool, device=targetDevice
        )
        return TextEncoderOutput(
            hiddenStates=hiddenStates,
            keyPaddingMask=keyPaddingMask,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def encodeTexts(
        self,
        tokenizer: ClipTokenizer,
        texts: Sequence[str] | str,
    ) -> TextEncoderOutput:
        """Tokenise and encode raw strings on the encoder's device."""
        encoded = tokenizer.encode(texts)
        device = next(self.parameters()).device
        return self.forward(
            inputIds=encoded.inputIds.to(device),
            attentionMask=encoded.attentionMask.to(device),
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initProjection(self) -> None:
        """Xavier-init the trainable projection head."""
        for module in self.outputProjection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
