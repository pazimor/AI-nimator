"""Custom text encoding stack for AI-nimator v2.

This package replaces the frozen XLM-RoBERTa pipeline (``src.shared.model.clip``)
with a compact, domain-specific text encoder trained jointly with the motion
denoiser.  The tokenizer is a BPE built on the prompt corpus (AMASS-Babel +
KIT-ML), and the encoder is a small transformer (target ~5–10M params).

Phase A7 additions
------------------
* :class:`TextEncoderProtocol` — the stable interface both
  :class:`CustomTextEncoder` and :class:`ClipTextEncoder` satisfy.
  Swap encoders by config alone; call site never changes.
* :mod:`ainimator.text.artifact` — save / load / hash an encoder as a
  self-contained artifact directory.
"""

from ainimator.text.clip_text_encoder import (
    ClipTextEncoder,
    ClipTextEncoderConfig,
    ClipTokenizer,
    ClipTokenizerConfig,
)
from ainimator.text.custom_text_encoder import (
    CustomTextEncoder,
    CustomTextEncoderConfig,
    TextEncoderOutput,
)
from ainimator.text.custom_tokenizer import (
    CustomTokenizer,
    CustomTokenizerConfig,
    EncodedBatch,
)
from ainimator.text.protocol import TextEncoderProtocol

__all__ = [
    "ClipTextEncoder",
    "ClipTextEncoderConfig",
    "ClipTokenizer",
    "ClipTokenizerConfig",
    "CustomTextEncoder",
    "CustomTextEncoderConfig",
    "CustomTokenizer",
    "CustomTokenizerConfig",
    "EncodedBatch",
    "TextEncoderOutput",
    "TextEncoderProtocol",
]
