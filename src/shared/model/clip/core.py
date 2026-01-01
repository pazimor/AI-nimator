"""CLIP text<->motion alignment model."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedTokenizerBase,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
)
from src.shared.constants.clip import DEFAULT_LOGIT_SCALE, EPSILON, LOGIT_SCALE_MAX
from src.shared.model.layers.temporal_unet import TemporalUNet


class ClipModel(nn.Module):
    """CLIP-like module for text and motion alignment."""

    def __init__(
        self,
        modelName: str = "xlm-roberta-base",
        embedDim: int = 512,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        textEncoder: Optional[XLMRobertaModel] = None,
        freezeTextEncoder: bool = True,
        motionNumHeads: int = 4,
        motionNumLayers: int = 2,
    ) -> None:
        """
        Initialize ClipModel.

        Parameters
        ----------
        modelName : str, optional
            Name of the XLM-Roberta model, by default "xlm-roberta-base".
        embedDim : int, optional
            Dimension of the shared embedding space, by default 512.
        tokenizer : Optional[PreTrainedTokenizerBase], optional
            Tokenizer used to prepare textual inputs.
        textEncoder : Optional[XLMRobertaModel], optional
            Pre-initialized text encoder for testing or specialization.
        freezeTextEncoder : bool, optional
            When True the XLM-R encoder is frozen.
        motionNumHeads : int, optional
            Number of attention heads in motion encoder, by default 4.
        motionNumLayers : int, optional
            Number of transformer layers in motion encoder, by default 2.
        """
        super().__init__()
        self.modelName = modelName
        self.textEncoder = textEncoder or XLMRobertaModel.from_pretrained(
            modelName,
            low_cpu_mem_usage=True,  # Reduce memory during loading
        )
        if freezeTextEncoder:
            self._freezeTextEncoder()
            self.textEncoder.eval()  # Set to eval mode to save memory (no dropout)
        self.tokenizer = tokenizer or XLMRobertaTokenizerFast.from_pretrained(
            modelName,
        )

        hiddenSize = self.textEncoder.config.hidden_size
        self.textProj = nn.Linear(hiddenSize, embedDim)
        self.motionBackbone = TemporalUNet(
            embedDim=embedDim,
            numHeads=motionNumHeads,
            numLayers=motionNumLayers,
        )
        self.motionProj = nn.Linear(embedDim, embedDim)
        self.logitScale = nn.Parameter(torch.ones([]) * DEFAULT_LOGIT_SCALE)

    def forward(
        self,
        textInputIds: torch.Tensor,
        textAttentionMask: torch.Tensor,
        motionInput: torch.Tensor,
        computeLoss: bool = False,
    ) -> Dict[str, object]:
        """
        Forward pass orchestrating text and motion encoders.

        Parameters
        ----------
        textInputIds : torch.Tensor
            Token IDs shaped (batch, sequenceLength).
        textAttentionMask : torch.Tensor
            Attention mask aligned with `textInputIds`.
        motionInput : torch.Tensor
            Motion payload shaped (batch, frames, bones, 6).
        computeLoss : bool, optional
            When True the contrastive loss is returned.

        Returns
        -------
        Dict[str, object]
            Embeddings, logits and optional contrastive loss.
        """
        textEmbeds, textHidden = self.encodeText(
            inputIds=textInputIds,
            attentionMask=textAttentionMask,
        )
        motionEmbeds = self.encodeMotion(motionInput)
        logitScale = self._clampedLogitScale()
        logitsPerText, logitsPerMotion = self.computeLogits(
            textEmbeds=textEmbeds,
            motionEmbeds=motionEmbeds,
            logitScale=logitScale,
        )
        output: Dict[str, object] = {
            "text_embeds": textEmbeds,
            "motion_embeds": motionEmbeds,
            "logits_per_text": logitsPerText,
            "logits_per_motion": logitsPerMotion,
            "text_hidden": textHidden,
            "logit_scale": logitScale,
        }
        if computeLoss:
            loss, components = self.clipLoss(
                logitsPerText=logitsPerText,
                logitsPerMotion=logitsPerMotion,
                textEmbeds=textEmbeds,
                motionEmbeds=motionEmbeds,
            )
            output["clip_loss"] = loss
            output.update(components)
        return output

    def encodeText(
        self,
        inputIds: torch.Tensor,
        attentionMask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode tokenized text with a frozen XLM-R encoder.

        Parameters
        ----------
        inputIds : torch.Tensor
            Token IDs shaped (batch, sequenceLength).
        attentionMask : torch.Tensor
            Attention mask aligned with `inputIds`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Normalized text embeddings and last hidden state.
        """
        with torch.no_grad():
            outputs = self.textEncoder(
                input_ids=inputIds,
                attention_mask=attentionMask,
            )
        pooled = self._maskedMean(outputs.last_hidden_state, attentionMask)
        projected = self.textProj(pooled)
        return self._normalize(projected), outputs.last_hidden_state

    def encodeMotion(self, motionInput: torch.Tensor) -> torch.Tensor:
        """
        Encode motion inputs into the shared embedding space.

        Parameters
        ----------
        motionInput : torch.Tensor
            Motion payload shaped (batch, frames, bones, 6).

        Returns
        -------
        torch.Tensor
            Normalized motion embeddings shaped (batch, embedDim).
        """
        features = self.motionBackbone(motionInput)
        projected = self.motionProj(features)
        return self._normalize(projected)

    def computeLogits(
        self,
        textEmbeds: torch.Tensor,
        motionEmbeds: torch.Tensor,
        logitScale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contrastive logits for text and motion pairs.

        Parameters
        ----------
        textEmbeds : torch.Tensor
            Normalized text embeddings shaped (batch, embedDim).
        motionEmbeds : torch.Tensor
            Normalized motion embeddings shaped (batch, embedDim).
        logitScale : Optional[torch.Tensor], optional
            Precomputed scale factor. When None the internal parameter is used.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Pair of similarity matrices (text->motion, motion->text).
        """
        scale = logitScale if logitScale is not None else self._clampedLogitScale()
        logitsPerText = scale * torch.matmul(textEmbeds, motionEmbeds.t())
        return logitsPerText, logitsPerText.t()

    def clipLoss(
        self,
        logitsPerText: torch.Tensor,
        logitsPerMotion: torch.Tensor,
        textEmbeds: torch.Tensor,
        motionEmbeds: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute contrastive + cosine losses following the graph spec.

        Parameters
        ----------
        logitsPerText : torch.Tensor
            Similarity scores from text to motion.
        logitsPerMotion : torch.Tensor
            Similarity scores from motion to text.
        textEmbeds : torch.Tensor
            Normalized text embeddings.
        motionEmbeds : torch.Tensor
            Normalized motion embeddings.

        Returns
        -------
        tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Total loss and detailed components.
        """
        labels = torch.arange(
            logitsPerText.size(0),
            device=logitsPerText.device,
        )
        lossText = F.cross_entropy(logitsPerText, labels)
        cosineDiag = torch.sum(textEmbeds * motionEmbeds, dim=-1)
        lossMotion = 1.0 - cosineDiag.mean()
        totalLoss = (lossText + lossMotion) / 2.0
        components = {
            "loss_text_contrastive": lossText.detach(),
            "loss_motion_cosine": lossMotion.detach(),
        }
        return totalLoss, components

    def _maskedMean(
        self,
        sequenceOutput: torch.Tensor,
        attentionMask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a masked mean pooling over the sequence dimension.

        Parameters
        ----------
        sequenceOutput : torch.Tensor
            Sequence output shaped (batch, sequenceLength, hiddenSize).
        attentionMask : torch.Tensor
            Attention mask aligned with `sequenceOutput`.

        Returns
        -------
        torch.Tensor
            Pooled representation shaped (batch, hiddenSize).
        """
        expandedMask = attentionMask.unsqueeze(-1).expand_as(
            sequenceOutput,
        ).float()
        safeDenominator = expandedMask.sum(dim=1).clamp(min=EPSILON)
        maskedSum = (sequenceOutput * expandedMask).sum(dim=1)
        return maskedSum / safeDenominator

    def _normalize(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        L2-normalize embedding vectors.

        Parameters
        ----------
        embeds : torch.Tensor
            Embeddings shaped (..., embedDim).

        Returns
        -------
        torch.Tensor
            Normalized embeddings.
        """
        return embeds / embeds.norm(dim=-1, keepdim=True).clamp(min=EPSILON)

    def _clampedLogitScale(self) -> torch.Tensor:
        """
        Return a safe exponential of the learnable logit scale.

        Returns
        -------
        torch.Tensor
            Positive scale factor applied to similarity matrices.
        """
        return torch.clamp(self.logitScale, max=LOGIT_SCALE_MAX).exp()

    def _freezeTextEncoder(self) -> None:
        """
        Freeze every parameter of the text encoder.

        Returns
        -------
        None
            The method updates parameters in-place.
        """
        for parameter in self.textEncoder.parameters():
            parameter.requires_grad = False
