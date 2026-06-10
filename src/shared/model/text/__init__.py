"""Custom text encoding stack for AI-nimator v2.

This package replaces the frozen XLM-RoBERTa pipeline (``src.shared.model.clip``)
with a compact, domain-specific text encoder trained jointly with the motion
denoiser.  The tokenizer is a BPE built on the prompt corpus (AMASS-Babel +
KIT-ML), and the encoder is a small transformer (target ~5–10M params).
"""

from src.shared.model.text.clip_text_encoder import (
    ClipTextEncoder,
    ClipTextEncoderConfig,
    ClipTokenizer,
    ClipTokenizerConfig,
)
from src.shared.model.text.custom_text_encoder import (
    CustomTextEncoder,
    CustomTextEncoderConfig,
    TextEncoderOutput,
)
from src.shared.model.text.custom_tokenizer import (
    CustomTokenizer,
    CustomTokenizerConfig,
    EncodedBatch,
)

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
]
