"""Temporal diffusion backbone and MoE components."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FFNMoE(nn.Module):
    """Feed-forward block with Mixture-of-Experts routing."""

    def __init__(
        self,
        modelDimension: int,
        ffDimension: int,
        expertCount: int = 8,
        topK: int = 2,
    ) -> None:
        super().__init__()
        self.expertCount = expertCount
        self.topK = topK
        self.experts = self._buildExperts(
            modelDimension,
            ffDimension,
            expertCount,
        )
        self.routerTokens = nn.Linear(modelDimension, expertCount)
        self.routerCondition = nn.Linear(
            modelDimension,
            expertCount,
            bias=False,
        )

    def forward(
        self,
        tokenHidden: Tensor,
        conditionHidden: Tensor,
    ) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        if conditionHidden is None:
            raise ValueError("FFNMoE requiert un conditionnement explicite.")
        flatTokens, flatCondition = self._reshapeInputs(
            tokenHidden,
            conditionHidden,
        )
        routing, topProbabilities, topIndices = self._computeRouting(
            flatTokens,
            flatCondition,
        )
        mixed = self._applyExperts(flatTokens, topIndices, topProbabilities)
        auxLoss = self._computeAuxiliaryLoss(
            routing,
            topIndices,
            topProbabilities,
        )
        batchSize, tokenCount, hiddenDim = tokenHidden.shape
        reshaped = mixed.reshape(batchSize, tokenCount, hiddenDim)
        return reshaped, auxLoss

    def _buildExperts(
        self,
        modelDimension: int,
        ffDimension: int,
        expertCount: int,
    ) -> nn.ModuleList:
        experts = nn.ModuleList()
        for _ in range(expertCount):
            experts.append(
                nn.Sequential(
                    nn.Linear(modelDimension, ffDimension),
                    nn.GELU(),
                    nn.Linear(ffDimension, modelDimension),
                )
            )
        return experts

    def _reshapeInputs(
        self,
        tokenHidden: Tensor,
        conditionHidden: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batchSize, tokenCount, hiddenDim = tokenHidden.shape
        flatTokens = tokenHidden.reshape(batchSize * tokenCount, hiddenDim)
        repeatedCondition = conditionHidden.unsqueeze(1).expand(
            batchSize,
            tokenCount,
            hiddenDim,
        )
        flatCondition = repeatedCondition.reshape(
            batchSize * tokenCount,
            hiddenDim,
        )
        return flatTokens, flatCondition

    def _computeRouting(
        self,
        flatTokens: Tensor,
        flatCondition: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.routerTokens(flatTokens)
        logits = logits + self.routerCondition(flatCondition)
        routingProbabilities = torch.softmax(logits, dim=-1)
        effectiveTopK = min(self.topK, self.expertCount)
        topProbabilities, topIndices = torch.topk(
            routingProbabilities,
            k=effectiveTopK,
            dim=-1,
        )
        normalizedWeights = topProbabilities / topProbabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        return routingProbabilities, normalizedWeights, topIndices

    def _applyExperts(
        self,
        flatTokens: Tensor,
        topIndices: Tensor,
        normalizedWeights: Tensor,
    ) -> Tensor:
        mixedOutput = torch.zeros_like(flatTokens)
        for expertRank in range(topIndices.shape[-1]):
            expertIds = topIndices[:, expertRank]
            weights = normalizedWeights[:, expertRank].unsqueeze(-1)
            for expertIndex in range(self.expertCount):
                mask = expertIds == expertIndex
                if not mask.any().item():
                    continue
                expertInput = flatTokens[mask]
                expertOutput = self.experts[expertIndex](expertInput)
                mixedOutput[mask] += weights[mask] * expertOutput
        return mixedOutput

    def _computeAuxiliaryLoss(
        self,
        routingProbabilities: Tensor,
        topIndices: Tensor,
        normalizedWeights: Tensor,
    ) -> Tensor:
        importance = routingProbabilities.mean(dim=0)
        load = torch.zeros_like(importance)
        flattenedIndices = topIndices.reshape(-1)
        flattenedWeights = normalizedWeights.reshape(-1)
        load.scatter_add_(0, flattenedIndices, flattenedWeights)
        importance = importance / importance.sum().clamp_min(1e-6)
        load = load / load.sum().clamp_min(1e-6)
        uniform = torch.full_like(importance, 1.0 / self.expertCount)
        return torch.nn.functional.mse_loss(importance, uniform) + torch.nn.functional.mse_loss(load, uniform)


class TransformerBlock(nn.Module):
    """Transformer block combining self-attention and MoE feed-forward."""

    def __init__(
        self,
        hiddenDim: int,
        headCount: int,
        moeConfiguration: Dict[str, int],
    ) -> None:
        super().__init__()
        self.layerNormAttention = nn.LayerNorm(hiddenDim)
        self.selfAttention = nn.MultiheadAttention(
            hiddenDim,
            headCount,
            batch_first=True,
        )
        self.layerNormMoE = nn.LayerNorm(hiddenDim)
        self.mixtureOfExperts = FFNMoE(
            hiddenDim,
            4 * hiddenDim,
            **moeConfiguration,
        )

    def forward(  # type: ignore[override]
        self,
        hiddenStates: Tensor,
        conditionHidden: Tensor,
        attentionMask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        normalizedAttention = self.layerNormAttention(hiddenStates)
        attentionOutput, _ = self.selfAttention(
            normalizedAttention,
            normalizedAttention,
            normalizedAttention,
            attn_mask=attentionMask,
            need_weights=False,
        )
        residualAttention = hiddenStates + attentionOutput
        normalizedMoE = self.layerNormMoE(residualAttention)
        moeOutput, auxLoss = self.mixtureOfExperts(
            normalizedMoE,
            conditionHidden,
        )
        residualOutput = residualAttention + moeOutput
        return residualOutput, auxLoss


class CrossAttention(nn.Module):
    """Simple cross-attention module with learnable projections."""

    def __init__(self, hiddenDim: int, contextDim: int) -> None:
        super().__init__()
        self.layerNorm = nn.LayerNorm(hiddenDim)
        self.queryProjection = nn.Linear(hiddenDim, hiddenDim)
        self.keyProjection = nn.Linear(contextDim, hiddenDim)
        self.valueProjection = nn.Linear(contextDim, hiddenDim)
        self.outputProjection = nn.Linear(hiddenDim, hiddenDim)
        self.headCount = 4
        self.headDim = hiddenDim // self.headCount
        if hiddenDim % self.headCount != 0:
            raise ValueError(
                "Le nombre de têtes doit diviser la dimension cachée."
            )

    def forward(  # type: ignore[override]
        self,
        hiddenStates: Tensor,
        contextEmbeddings: Tensor,
    ) -> Tensor:
        batchSize, tokenCount, hiddenDim = hiddenStates.shape
        normalized = self.layerNorm(hiddenStates)
        queries = self.queryProjection(normalized).view(
            batchSize,
            tokenCount,
            self.headCount,
            self.headDim,
        ).transpose(1, 2)
        if contextEmbeddings.dim() == 2:
            contextEmbeddings = contextEmbeddings.unsqueeze(1)
        keys = self.keyProjection(contextEmbeddings).view(
            batchSize,
            -1,
            self.headCount,
            self.headDim,
        ).transpose(1, 2)
        values = self.valueProjection(contextEmbeddings).view(
            batchSize,
            -1,
            self.headCount,
            self.headDim,
        ).transpose(1, 2)
        attentionScores = torch.matmul(queries, keys.transpose(-2, -1))
        attentionScores *= self.headDim ** -0.5
        attentionWeights = torch.softmax(attentionScores, dim=-1)
        attentionOutput = torch.matmul(attentionWeights, values)
        attentionOutput = attentionOutput.transpose(1, 2).contiguous().view(
            batchSize,
            tokenCount,
            hiddenDim,
        )
        return hiddenStates + self.outputProjection(attentionOutput)


class PoseEncoder(nn.Module):
    """Encode pose sequences into a compact descriptor."""

    def __init__(self, inputDim: int = 6, hiddenDim: int = 256) -> None:
        super().__init__()
        self.inputDim = inputDim
        self.featureExtractor = nn.Sequential(
            nn.Conv1d(inputDim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(256, hiddenDim)

    def forward(  # type: ignore[override]
        self,
        rotation6dSequence: Tensor,
    ) -> Tensor:
        batchSize, frameCount, boneCount, channelCount = (
            rotation6dSequence.shape
        )
        if channelCount != self.inputDim:
            raise ValueError(
                f"PoseEncoder attend {self.inputDim} canaux; "
                f"reçu {channelCount}."
            )
        reshaped = rotation6dSequence.permute(0, 2, 3, 1).reshape(
            batchSize * boneCount,
            channelCount,
            frameCount,
        )
        features = self.featureExtractor(reshaped).squeeze(-1)
        features = features.reshape(batchSize, boneCount, -1)
        pooled = features.mean(dim=1)
        return self.projection(pooled)


class TemporalUNetMoE(nn.Module):
    """Temporal U-Net with cross-attention and Mixture-of-Experts blocks."""

    def __init__(
        self,
        rotationInputDim: int,
        hiddenDim: int,
        layerCount: int,
        moeConfiguration: Dict[str, int],
        textDim: int,
    ) -> None:
        super().__init__()
        if layerCount % 2 != 0:
            raise ValueError("layerCount doit être pair pour ce U-Net.")
        self.inputProjection = nn.Linear(rotationInputDim, hiddenDim)
        self.positionalEncoding = nn.Parameter(
            torch.randn(4096, hiddenDim) * 0.01
        )
        halfLayerCount = layerCount // 2
        self.downBlocks = nn.ModuleList(
            [
                TransformerBlock(hiddenDim, 4, moeConfiguration)
                for _ in range(halfLayerCount)
            ]
        )
        self.crossAttention = CrossAttention(hiddenDim, textDim)
        self.upBlocks = nn.ModuleList(
            [
                TransformerBlock(hiddenDim, 4, moeConfiguration)
                for _ in range(halfLayerCount)
            ]
        )
        self.outputProjection = nn.Linear(hiddenDim, rotationInputDim)
        self.poseEncoder = PoseEncoder(inputDim=6, hiddenDim=hiddenDim)
        self.conditionProjection = nn.Linear(textDim, hiddenDim)

    def forward(  # type: ignore[override]
        self,
        rotation6d: Tensor,
        timeVector: Tensor,
        textEmbedding: Tensor,
        tagEmbedding: Tensor,
        contextSequence: Optional[Tensor] = None,
        causalMask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        encoded, batchSize, frameCount, boneCount = self._encodeInputs(
            rotation6d,
            timeVector,
        )
        condition = self._buildCondition(
            textEmbedding,
            tagEmbedding,
            contextSequence,
        )
        auxiliaryLosses: List[Tensor] = []
        hidden = self._applyDownBlocks(
            encoded,
            condition,
            causalMask,
            auxiliaryLosses,
        )
        hidden = self.crossAttention(hidden, textEmbedding + tagEmbedding)
        hidden = self._applyUpBlocks(
            hidden,
            condition,
            causalMask,
            auxiliaryLosses,
        )
        residual = self.outputProjection(hidden)
        residual = residual.reshape(batchSize, frameCount, boneCount, 6)
        auxiliaryLoss = torch.stack(auxiliaryLosses).mean()
        return residual, auxiliaryLoss

    def _encodeInputs(
        self,
        rotation6d: Tensor,
        timeVector: Tensor,
    ) -> Tuple[Tensor, int, int, int]:
        batchSize, frameCount, boneCount, _ = rotation6d.shape
        flattened = rotation6d.reshape(batchSize, frameCount, boneCount * 6)
        positional = self.positionalEncoding[:frameCount, :].unsqueeze(0)
        hidden = self.inputProjection(flattened) + positional
        hidden = hidden + (timeVector * 2 - 1.0) * 0.1
        return hidden, batchSize, frameCount, boneCount

    def _buildCondition(
        self,
        textEmbedding: Tensor,
        tagEmbedding: Tensor,
        contextSequence: Optional[Tensor],
    ) -> Tensor:
        baseCondition = (textEmbedding + tagEmbedding) / 2.0
        condition = self.conditionProjection(baseCondition)
        if contextSequence is None:
            return condition
        contextEncoding = self.poseEncoder(contextSequence)
        return condition + 0.5 * contextEncoding

    def _applyDownBlocks(
        self,
        hidden: Tensor,
        condition: Tensor,
        causalMask: Optional[Tensor],
        auxiliaryLosses: List[Tensor],
    ) -> Tensor:
        for block in self.downBlocks:
            hidden, auxLoss = block(hidden, condition, causalMask)
            auxiliaryLosses.append(auxLoss)
        return hidden

    def _applyUpBlocks(
        self,
        hidden: Tensor,
        condition: Tensor,
        causalMask: Optional[Tensor],
        auxiliaryLosses: List[Tensor],
    ) -> Tensor:
        for block in self.upBlocks:
            hidden, auxLoss = block(hidden, condition, causalMask)
            auxiliaryLosses.append(auxLoss)
        return hidden


class CausalDiffusion:
    """Diffusion process with causal sampling for motion sequences."""

    def __init__(self, model: TemporalUNetMoE, trainingSteps: int = 1000) -> None:
        if trainingSteps <= 0:
            raise ValueError("trainingSteps doit être strictement positif.")
        self.model = model
        self.trainingSteps = trainingSteps
        self.trainingTimeGrid = torch.linspace(
            0.0,
            1.0,
            trainingSteps,
            dtype=torch.float32,
        )

    def qSample(
        self,
        cleanRotations: Tensor,
        timeValues: Tensor,
        noise: Tensor,
    ) -> Tensor:
        alphaBar = torch.cos((timeValues * math.pi / 2)).pow(2)
        alphaSqrt = alphaBar.sqrt().unsqueeze(-1)
        sigma = (1 - alphaBar).sqrt().unsqueeze(-1)
        return cleanRotations * alphaSqrt + noise * sigma

    def loss(
        self,
        cleanRotations: Tensor,
        textEmbedding: Tensor,
        tagEmbedding: Tensor,
        contextSequence: Optional[Tensor] = None,
        causalMask: Optional[Tensor] = None,
    ) -> Tensor:
        batchSize, frameCount, _, _ = cleanRotations.shape
        timeGrid = self.trainingTimeGrid.to(cleanRotations.device)
        randomIndices = torch.randint(
            0,
            timeGrid.shape[0],
            (batchSize, frameCount),
            device=cleanRotations.device,
        )
        timeValues = timeGrid[randomIndices].unsqueeze(-1)
        noise = torch.randn_like(cleanRotations)
        noisyRotations = self.qSample(cleanRotations, timeValues, noise)
        predictedNoise, auxiliaryLoss = self.model(
            noisyRotations,
            timeValues,
            textEmbedding,
            tagEmbedding,
            contextSequence,
            causalMask,
        )
        predictionLoss = torch.nn.functional.mse_loss(predictedNoise, noise)
        return predictionLoss + 1e-4 * auxiliaryLoss

    @torch.no_grad()
    def sample(
        self,
        frameCount: int,
        boneCount: int,
        textEmbedding: Tensor,
        tagEmbedding: Tensor,
        contextSequence: Optional[Tensor] = None,
        steps: int = 12,
        guidanceScale: float = 2.0,
        causalMask: Optional[Tensor] = None,
        device: str | torch.device = "cuda",
        eta: float = 0.0,
    ) -> Tensor:
        if steps < 2:
            raise ValueError("steps doit être supérieur ou égal à deux.")
        inferenceDevice = torch.device(device)
        batchSize = textEmbedding.shape[0]
        timeGrid = torch.linspace(1.0, 0.0, steps, device=inferenceDevice)
        currentSample = torch.randn(
            batchSize,
            frameCount,
            boneCount,
            6,
            device=inferenceDevice,
        )
        zeroText = torch.zeros_like(textEmbedding)
        zeroTag = torch.zeros_like(tagEmbedding)
        for previousTime, nextTime in zip(timeGrid[:-1], timeGrid[1:]):
            timeEmbedding = previousTime.clamp(max=1.0 - 1e-4)
            expandedTime = timeEmbedding.view(1, 1, 1).expand(
                batchSize,
                frameCount,
                1,
            )
            conditioned, _ = self.model(
                currentSample,
                expandedTime,
                textEmbedding,
                tagEmbedding,
                contextSequence,
                causalMask,
            )
            unconditioned, _ = self.model(
                currentSample,
                expandedTime,
                zeroText,
                zeroTag,
                contextSequence,
                causalMask,
            )
            guidanceDelta = conditioned - unconditioned
            guided = unconditioned + guidanceScale * guidanceDelta
            currentSample = self._stepEuler(
                currentSample,
                guided,
                previousTime,
                nextTime,
                eta,
            )
        return currentSample

    def _stepEuler(
        self,
        currentSample: Tensor,
        guidedNoise: Tensor,
        previousTime: Tensor,
        nextTime: Tensor,
        eta: float,
    ) -> Tensor:
        previousAlpha = torch.cos(previousTime * math.pi / 2).pow(2)
        nextAlpha = torch.cos(nextTime * math.pi / 2).pow(2)
        previousAlpha = previousAlpha.clamp(min=1e-4, max=0.9999)
        nextAlpha = nextAlpha.clamp(min=1e-4, max=1.0)
        sqrtPreviousAlpha = previousAlpha.sqrt()
        sqrtOneMinusPrevious = (1 - previousAlpha).clamp(min=0).sqrt()
        sqrtNextAlpha = nextAlpha.sqrt()
        sqrtOneMinusNext = (1 - nextAlpha).clamp(min=0).sqrt()
        estimatedClean = currentSample - sqrtOneMinusPrevious * guidedNoise
        estimatedClean = estimatedClean / sqrtPreviousAlpha
        estimatedClean = torch.clamp(estimatedClean, -10.0, 10.0)
        if eta > 0.0 and nextTime > 0:
            ratio = (1 - nextAlpha) / (1 - previousAlpha)
            sigma = eta * torch.sqrt(ratio.clamp(min=0))
            sigma = sigma * torch.sqrt(
                (1 - previousAlpha / nextAlpha).clamp(min=0)
            )
            noise = torch.randn_like(currentSample)
        else:
            sigma = 0.0
            noise = torch.zeros_like(currentSample)
        updated = (
            sqrtNextAlpha * estimatedClean
            + sqrtOneMinusNext * guidedNoise
            + sigma * noise
        )
        return updated
