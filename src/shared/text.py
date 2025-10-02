"""Text encoder helpers."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

try:  # pragma: no cover - import side effects
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - handled at runtime
    AutoModel = None
    AutoTokenizer = None


class PretrainedTextEncoder(nn.Module):
    """Wrapper around HuggingFace models to produce sentence embeddings."""

    def __init__(
        self,
        modelName: str,
        device: torch.device,
        trainable: bool = False,
    ) -> None:
        """Load a transformer encoder for deterministic prompt embeddings."""
        super().__init__()
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                "Le package transformers est requis pour PretrainedTextEncoder."
            )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.model = AutoModel.from_pretrained(modelName)
        self.model.to(device)
        self.trainable = trainable
        if not trainable:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        hiddenSize = getattr(self.model.config, "hidden_size", None)
        if hiddenSize is None:
            raise ValueError("Le modèle texte ne fournit pas de hidden_size.")
        self.outDimension = hiddenSize

    def forward(self, texts: Sequence[str]) -> Tensor:  # type: ignore[override]
        """Embed a batch of text prompts into a dense vector space."""
        if len(texts) == 0:
            raise ValueError(
                "PretrainedTextEncoder requiert une liste non vide."
            )
        tokenBatch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenBatch = {
            key: value.to(self.device) for key, value in tokenBatch.items()
        }
        outputs = self.model(**tokenBatch)
        hasPooler = hasattr(outputs, "pooler_output")
        if hasPooler and outputs.pooler_output is not None:
            return outputs.pooler_output
        hiddenStates = outputs.last_hidden_state
        attentionMask = tokenBatch["attention_mask"].unsqueeze(-1)
        maskedStates = hiddenStates * attentionMask
        summedStates = maskedStates.sum(dim=1)
        maskSum = attentionMask.sum(dim=1).clamp(min=1)
        return summedStates / maskSum
