"""CLIP text<->motion alignment model."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from transformers import (
    PreTrainedTokenizerBase,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
)
from src.shared.constants.clip import DEFAULT_LOGIT_SCALE, EPSILON, LOGIT_SCALE_MAX
from src.shared.model.components.base import MotionComponent
from src.shared.model.layers.temporal_unet import TemporalUNet
from src.shared.model.clip.motion_input import (
    buildMotionInputFromBatch,
    buildMotionInputFromMotion,
    extractMotionContext,
    motionInputChannels,
    motionInputScopeChannels,
    requiredComponentKeys,
    splitMotionInput,
)

DEFAULT_COSINE_LOSS_WEIGHT = 0.25


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
        motionComponents: Optional[Sequence[MotionComponent]] = None,
        numBones: int = 22,
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
        motionComponents : Optional[Sequence[MotionComponent]], optional
            Optional motion feature layout consumed by the motion encoder.
            When omitted, the model falls back to the legacy rotation-only
            input tensor.
        numBones : int, optional
            Number of skeleton bones in the motion tensors.
        """
        super().__init__()
        self.modelName = modelName
        self.numBones = int(numBones)
        self.motionComponents = tuple(motionComponents or ())
        self.motionInputChannels = motionInputChannels(self.motionComponents)
        (
            self.motionBoneChannels,
            self.motionGlobalChannels,
        ) = motionInputScopeChannels(self.motionComponents)
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
            numBones=self.numBones,
            numChannels=self.motionBoneChannels,
            globalChannels=self.motionGlobalChannels,
        )
        self.motionProj = nn.Linear(embedDim, embedDim)
        self.logitScale = nn.Parameter(torch.ones([]) * DEFAULT_LOGIT_SCALE)

    def forward(
        self,
        textInputIds: Optional[torch.Tensor] = None,
        textAttentionMask: Optional[torch.Tensor] = None,
        motionInput: Optional[torch.Tensor] = None,
        motionMask: Optional[torch.Tensor] = None,
        positiveMask: Optional[torch.Tensor] = None,
        computeLoss: bool = False,
        pooledText: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """
        Forward pass orchestrating text and motion encoders.

        Parameters
        ----------
        textInputIds : Optional[torch.Tensor]
            Token IDs shaped (batch, sequenceLength).
        textAttentionMask : Optional[torch.Tensor]
            Attention mask aligned with `textInputIds`.
        motionInput : Optional[torch.Tensor]
            Motion payload shaped (batch, frames, bones, channels).
        motionMask : Optional[torch.Tensor], optional
            Boolean tensor shaped (batch, frames) marking valid motion frames.
        positiveMask : Optional[torch.Tensor], optional
            Boolean matrix marking every valid text<->motion positive pair
            inside the batch. When omitted, only the batch diagonal is treated
            as positive.
        computeLoss : bool, optional
            When True the contrastive loss is returned.
        pooledText : Optional[torch.Tensor], optional
            Precomputed pooled text representation shaped (batch, hiddenSize).

        Returns
        -------
        Dict[str, object]
            Embeddings, logits and optional contrastive loss.
        """
        if motionInput is None:
            raise ValueError("motionInput is required.")
        if pooledText is not None:
            if textInputIds is not None or textAttentionMask is not None:
                raise ValueError(
                    "Provide either pooledText or tokenized text inputs, not both."
                )
            textEmbeds, textHidden = self.encodePooledText(pooledText)
        else:
            if textInputIds is None or textAttentionMask is None:
                raise ValueError(
                    "textInputIds and textAttentionMask are required when "
                    "pooledText is not provided."
                )
            textEmbeds, textHidden = self.encodeText(
                inputIds=textInputIds,
                attentionMask=textAttentionMask,
            )
        motionEmbeds = self.encodeMotion(motionInput, motionMask=motionMask)
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
                positiveMask=positiveMask,
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

    def encodePooledText(
        self,
        pooledText: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Project a precomputed pooled text representation into CLIP space.

        Parameters
        ----------
        pooledText : torch.Tensor
            Pooled XLM-R representation shaped (batch, hiddenSize).

        Returns
        -------
        tuple[torch.Tensor, Optional[torch.Tensor]]
            Normalized text embeddings and no hidden state payload.
        """
        projected = self.textProj(pooledText)
        return self._normalize(projected), None

    def encodeMotion(
        self,
        motionInput: torch.Tensor,
        motionMask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode motion inputs into the shared embedding space.

        Parameters
        ----------
        motionInput : torch.Tensor
            Motion payload shaped (batch, frames, bones, channels).
        motionMask : Optional[torch.Tensor], optional
            Boolean tensor shaped (batch, frames) marking valid motion frames.

        Returns
        -------
        torch.Tensor
            Normalized motion embeddings shaped (batch, embedDim).
        """
        if motionInput.dim() != 4:
            raise ValueError(
                "Expected CLIP motion input with shape "
                "(batch, frames, bones, channels), got "
                f"{tuple(motionInput.shape)}."
            )
        if motionInput.shape[-2] != self.numBones:
            raise ValueError(
                f"Expected {self.numBones} bones in CLIP motion input, got "
                f"{motionInput.shape[-2]}."
            )
        if motionInput.shape[-1] != self.motionInputChannels:
            raise ValueError(
                "Unexpected CLIP motion channel count: expected "
                f"{self.motionInputChannels}, got {motionInput.shape[-1]}."
            )
        if motionMask is not None:
            if motionMask.dim() != 2:
                raise ValueError(
                    "Expected motionMask with shape (batch, frames), got "
                    f"{tuple(motionMask.shape)}."
                )
            if motionMask.shape != motionInput.shape[:2]:
                raise ValueError(
                    "motionMask must match motionInput temporal axes: "
                    f"expected {tuple(motionInput.shape[:2])}, got "
                    f"{tuple(motionMask.shape)}."
                )
            motionMask = motionMask.to(
                device=motionInput.device,
                dtype=torch.bool,
            )
        boneInput, globalInput = splitMotionInput(
            motionInput,
            self.motionComponents,
        )
        features = self.motionBackbone(
            boneInput=boneInput,
            globalInput=globalInput,
            motionMask=motionMask,
        )
        projected = self.motionProj(features)
        return self._normalize(projected)

    def buildMotionInput(self, batch: Mapping[str, object]) -> torch.Tensor:
        """Assemble the motion tensor expected by ``encodeMotion``."""
        return buildMotionInputFromBatch(
            batch=batch,
            components=self.motionComponents,
            numBones=self.numBones,
        )

    def buildMotionInputFromMotion(
        self,
        motion: torch.Tensor,
        context: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        """Build CLIP motion input from predicted rotations and batch context."""
        return buildMotionInputFromMotion(
            motion=motion,
            components=self.motionComponents,
            numBones=self.numBones,
            context=context,
        )

    def extractMotionContext(
        self,
        batch: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        """Return batch tensors needed to rebuild CLIP motion inputs later."""
        return extractMotionContext(batch, self.motionComponents)

    def requiredMotionComponentKeys(self) -> tuple[str, ...]:
        """Return manifest component keys required by this model."""
        return requiredComponentKeys(self.motionComponents)

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
        positiveMask: Optional[torch.Tensor] = None,
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
        positiveMask : Optional[torch.Tensor], optional
            Boolean matrix marking every valid text<->motion positive pair.

        Returns
        -------
        tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Total loss and detailed components.
        """
        if positiveMask is None:
            positiveMask = torch.eye(
                logitsPerText.size(0),
                dtype=torch.bool,
                device=logitsPerText.device,
            )
        else:
            positiveMask = positiveMask.to(
                device=logitsPerText.device,
                dtype=torch.bool,
            )
        if positiveMask.shape != logitsPerText.shape:
            raise ValueError(
                "positiveMask must match logits shape "
                f"{tuple(logitsPerText.shape)}, got "
                f"{tuple(positiveMask.shape)}."
            )

        lossText = self._contrastiveLoss(
            logits=logitsPerText,
            positiveMask=positiveMask,
        )
        lossMotionContrastive = self._contrastiveLoss(
            logits=logitsPerMotion,
            positiveMask=positiveMask.t(),
        )
        cosineDiag = torch.sum(textEmbeds * motionEmbeds, dim=-1)
        lossMotion = 1.0 - cosineDiag.mean()
        totalLoss = (
            lossText
            + lossMotionContrastive
            + (DEFAULT_COSINE_LOSS_WEIGHT * lossMotion)
        ) / (2.0 + DEFAULT_COSINE_LOSS_WEIGHT)
        components = {
            "loss_text_contrastive": lossText.detach(),
            "loss_motion_contrastive": lossMotionContrastive.detach(),
            "loss_motion_cosine": lossMotion.detach(),
        }
        return totalLoss, components

    def _contrastiveLoss(
        self,
        logits: torch.Tensor,
        positiveMask: torch.Tensor,
    ) -> torch.Tensor:
        """Average negative log-probability over one-or-more positives."""
        if positiveMask.shape != logits.shape:
            raise ValueError(
                "positiveMask must match logits for contrastive loss."
            )
        positiveCounts = positiveMask.sum(dim=1)
        if torch.any(positiveCounts <= 0):
            raise ValueError(
                "Each batch item must keep at least one positive pair."
            )
        logProb = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positiveLogProb = torch.where(
            positiveMask,
            logProb,
            torch.zeros_like(logProb),
        )
        lossPerSample = -positiveLogProb.sum(dim=1) / positiveCounts.clamp(min=1)
        return lossPerSample.mean()

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
