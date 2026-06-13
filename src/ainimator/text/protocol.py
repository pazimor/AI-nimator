"""Stable interface contract for text encoders.

Both :class:`ainimator.text.custom_text_encoder.CustomTextEncoder` and
:class:`ainimator.text.clip_text_encoder.ClipTextEncoder` implement this
Protocol so the generation and training stacks are encoder-agnostic: the
swap from custom BPE to frozen CLIP (and back) is a CONFIG change with
ZERO code change at any call site.

Design rules (§2.9, §2.10)
---------------------------
* ``encode()`` is the only method that takes token tensors — keep it
  ONNX-traceable (no data-dependent control flow, no ``.item()``).
* ``forwardNull()`` bypasses the encoder for the CFG unconditional branch.
* ``outputDim`` exposes the channel width so consumers can build
  projections without inspecting the concrete class.
* ``nullEmbedding`` may be ``None`` when the encoder was built without
  the learnable null token (``useNullEmbedding=False``); callers must
  guard accordingly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from ainimator.text.custom_text_encoder import TextEncoderOutput


@runtime_checkable
class TextEncoderProtocol(Protocol):
    """Interface satisfied by every text encoder in the v2 stack.

    Both :class:`CustomTextEncoder` and :class:`ClipTextEncoder` are
    structurally compatible with this Protocol — no inheritance required.
    The ``@runtime_checkable`` decorator allows ``isinstance`` checks in
    tests and assertion guards.

    Methods / properties
    --------------------
    encode(inputIds, attentionMask) -> TextEncoderOutput
        Encode a tokenised batch.  Output shape:
        ``hiddenStates (B, T, outputDim)``, ``keyPaddingMask (B, T)``.
    forwardNull(batchSize, device, dtype) -> TextEncoderOutput
        Return the learnable null embedding for CFG.  Raises
        ``RuntimeError`` when ``useNullEmbedding`` was False.
    outputDim -> int
        Width of the per-token hidden states exposed to consumers.

    Note: ``nullEmbedding`` (an ``nn.Parameter`` on both concrete classes)
    is NOT part of this Protocol because Python's @runtime_checkable
    isinstance() check cannot verify nn.Parameter attributes registered
    through nn.Module.__setattr__.  Callers that need the null embedding
    directly should use ``forwardNull()`` or access the attribute from the
    concrete class.
    """

    @property
    def outputDim(self) -> int:
        """Width of the per-token hidden states exposed to consumers."""
        ...

    def encode(
        self,
        inputIds: torch.Tensor,
        attentionMask: torch.Tensor,
    ) -> TextEncoderOutput:
        """Encode a batch of token sequences.

        Parameters
        ----------
        inputIds : torch.Tensor
            Long tensor ``(B, T)`` — token ids from the paired tokenizer.
        attentionMask : torch.Tensor
            Float tensor ``(B, T)`` — 1.0 on real tokens, 0.0 on padding.

        Returns
        -------
        TextEncoderOutput
            ``hiddenStates (B, T, outputDim)`` and
            ``keyPaddingMask (B, T)`` (True on padding positions).
        """
        ...

    def forwardNull(
        self,
        batchSize: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TextEncoderOutput:
        """Return the learnable null embedding for the CFG branch.

        Parameters
        ----------
        batchSize : int
            Number of unconditional samples (batch size).
        device : optional
            Target device.  Defaults to the encoder's device.
        dtype : optional
            Target dtype.  Defaults to the null embedding's dtype.

        Returns
        -------
        TextEncoderOutput
            ``hiddenStates (B, 1, outputDim)`` and
            ``keyPaddingMask (B, 1)`` (all False = real token).
        """
        ...
