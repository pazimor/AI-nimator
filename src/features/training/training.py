"""Training pipeline for the prompt-to-animation diffusion model."""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from src.shared.device import DeviceSelector, RandomnessController
from src.shared.animation_io import convertRotationsToTensor, loadAnimationFile, loadPromptsFile
from src.shared.quaternion import QuaternionConverter, QuaternionMetrics
from src.shared.temporal_diffusion import CausalDiffusion, TemporalUNetMoE
from src.shared.text import PretrainedTextEncoder
from src.shared.types import (
    AnimationPromptSample,
    ClipRecord,
    DatasetCache,
    TrainingConfiguration,
)

from src.shared.constants import LOCAL_CLIP_RECORD_LIMIT

class AnimationPromptDataset(Dataset[AnimationPromptSample]):
    """Dataset pairing motion clips with textual descriptions."""

    def __init__(
        self,
        dataDirectory: Path,
        sequenceFrames: int = 240,
        contextHistory: int = 0,
    ) -> None:
        super().__init__()
        self.dataDirectory = Path(dataDirectory)
        self.sequenceFrames = sequenceFrames
        self.contextHistory = contextHistory
        self.clipRecords = self._loadClipRecords()
        if not self.clipRecords:
            raise FileNotFoundError(
                "Aucune paire animation/prompt détectée dans la base."
            )
        self.boneNames = convertRotationsToTensor(
            self.clipRecords[0].rotations
        )[1]

    def __len__(self) -> int:
        return len(self.clipRecords)

    def __getitem__(self, index: int) -> AnimationPromptSample:
        clipRecord = self.clipRecords[index]
        quaternionTensor, _ = convertRotationsToTensor(clipRecord.rotations)
        segment = self._extractSegment(quaternionTensor)
        rotation6d = QuaternionConverter.rotation6dFromQuaternion(segment)
        promptEntry = clipRecord.prompts[index % len(clipRecord.prompts)]
        textPrompt, tagLabel = self._composePromptText(promptEntry)
        contextSequence = self._buildContextSequence(index)
        return AnimationPromptSample(
            rotation6d=rotation6d,
            textPrompt=textPrompt,
            tagLabel=tagLabel,
            boneNames=self.boneNames,
            contextSequence=contextSequence,
        )

    def fetchContextSequence(
        self,
        index: int,
        historyCount: int,
    ) -> Optional[Tensor]:
        if historyCount <= 0:
            return None
        historyIndices = [
            max(0, index - offset - 1) for offset in range(historyCount)
        ]
        return self._gatherContexts(historyIndices)

    def _loadClipRecords(self) -> List[ClipRecord]:
        records: List[ClipRecord] = []
        for animationPath, promptPath in self._discoverPairs(self.dataDirectory):
            rotations = loadAnimationFile(animationPath)
            prompts = loadPromptsFile(promptPath)
            records.append(ClipRecord(animationPath, rotations, prompts))
            if (
                LOCAL_CLIP_RECORD_LIMIT > 0
                and len(records) >= LOCAL_CLIP_RECORD_LIMIT
            ):
                break
        return records

    def _discoverPairs(
        self,
        rootDirectory: Path,
    ) -> Iterable[Tuple[Path, Path]]:
        for directory in rootDirectory.rglob("*"):
            if not directory.is_dir():
                continue
            animationPath = directory / "animation.json"
            promptPath = directory / "prompts.json"
            if animationPath.exists() and promptPath.exists():
                yield animationPath, promptPath

    def _extractSegment(self, quaternionTensor: Tensor) -> Tensor:
        frameCount = quaternionTensor.shape[0]
        if frameCount >= self.sequenceFrames:
            startFrame = random.randint(0, frameCount - self.sequenceFrames)
            endFrame = startFrame + self.sequenceFrames
            return quaternionTensor[startFrame:endFrame]
        requiredRepeats = math.ceil(self.sequenceFrames / frameCount)
        tiled = quaternionTensor.repeat(requiredRepeats, 1, 1)
        return tiled[: self.sequenceFrames]

    def _composePromptText(self, promptEntry: Dict[str, str]) -> Tuple[str, str]:
        """Return the textual prompt and its tag extracted from ``promptEntry``.

        Parameters
        ----------
        promptEntry:
            Mapping coming from ``prompt.json`` files. Older datasets may use
            different casing for keys, therefore we accept both upper and
            lower-case variants.

        Returns
        -------
        tuple[str, str]
            Text prompt (concatenation of ``Simple``/``advanced``/``expert``)
            and the associated tag label (empty string when missing).
        """

        components = [
            promptEntry.get("Simple") or promptEntry.get("simple") or "",
            promptEntry.get("advanced") or promptEntry.get("Advanced") or "",
            promptEntry.get("expert") or promptEntry.get("Expert") or "",
        ]
        joined = " ".join(component.strip() for component in components)
        tag = (
            promptEntry.get("tag")
            or promptEntry.get("Tag")
            or ""
        )
        return joined.strip(), str(tag).strip()

    def _buildContextSequence(self, index: int) -> Optional[Tensor]:
        if self.contextHistory <= 0:
            return None
        historyIndices = [
            max(0, index - offset - 1) for offset in range(self.contextHistory)
        ]
        return self._gatherContexts(historyIndices)

    def _gatherContexts(self, indices: Sequence[int]) -> Optional[Tensor]:
        sequences: List[Tensor] = []
        for historyIndex in indices:
            rotationTensor, bones = convertRotationsToTensor(
                self.clipRecords[historyIndex].rotations
            )
            if bones != self.boneNames:
                raise ValueError(
                    "Les squelettes doivent rester identiques dans le dataset."
                )
            sequences.append(
                QuaternionConverter.rotation6dFromQuaternion(rotationTensor)
            )
        if not sequences:
            return None
        concatenated = torch.cat(sequences, dim=0)
        desiredLength = min(self.sequenceFrames, concatenated.shape[0])
        if concatenated.shape[0] >= desiredLength:
            contextSequence = concatenated[-desiredLength:]
        else:
            padding = concatenated.new_zeros(
                desiredLength - concatenated.shape[0],
                concatenated.shape[1],
                concatenated.shape[2],
            )
            contextSequence = torch.cat([padding, concatenated], dim=0)
        return contextSequence.unsqueeze(0)
class DatasetCacheBuilder:
    """Factory responsible for building dataset caches."""

    @staticmethod
    def buildCache(
        dataset: AnimationPromptDataset,
        device: torch.device,
        textEncoder: PretrainedTextEncoder,
        cacheOnDevice: bool,
    ) -> DatasetCache:
        if textEncoder.trainable:
            raise ValueError(
                "Impossible de pré-charger un encodeur texte entraînable."
            )
        rotationSequences: List[Tensor] = []
        textEmbeddings: List[Tensor] = []
        tagEmbeddings: List[Tensor] = []
        print("[cache] Pré-chargement du dataset...", flush=True)
        for index in range(len(dataset)):
            sample = dataset[index]
            rotationSequence = sample.rotation6d.unsqueeze(0)
            textEmbedding = DatasetCacheBuilder._encodeText(
                textEncoder,
                sample.textPrompt,
            )
            tagEmbedding = DatasetCacheBuilder._encodeTag(
                textEncoder,
                sample.tagLabel,
            )
            if cacheOnDevice:
                rotationSequence = rotationSequence.to(
                    device,
                    non_blocking=True,
                )
                textEmbedding = textEmbedding.to(
                    device,
                    non_blocking=True,
                )
                tagEmbedding = tagEmbedding.to(
                    device,
                    non_blocking=True,
                )
            rotationSequences.append(rotationSequence)
            textEmbeddings.append(textEmbedding.squeeze(0))
            tagEmbeddings.append(tagEmbedding.squeeze(0))
        location = "VRAM" if cacheOnDevice else "RAM"
        print(
            "[cache] "
            f"{len(rotationSequences)} éléments pré-chargés ({location}).",
            flush=True,
        )
        return DatasetCache(rotationSequences, textEmbeddings, tagEmbeddings)

    @staticmethod
    def _encodeText(
        textEncoder: PretrainedTextEncoder,
        textPrompt: str,
    ) -> Tensor:
        with torch.no_grad():
            return textEncoder([textPrompt]).detach()

    @staticmethod
    def _encodeTag(
        textEncoder: PretrainedTextEncoder,
        tagLabel: str,
    ) -> Tensor:
        cleanedLabel = tagLabel.strip()
        if not cleanedLabel:
            return torch.zeros(
                1,
                textEncoder.outDimension,
                device=textEncoder.device,
                dtype=torch.float32,
            )
        with torch.no_grad():
            return textEncoder([cleanedLabel]).detach()


class SuccessRateEvaluator:
    """Utility to measure validation success rate."""

    @staticmethod
    @torch.no_grad()
    def computeSuccessRate(
        model: TemporalUNetMoE,
        diffusion: CausalDiffusion,
        textEncoder: PretrainedTextEncoder,
        dataset: AnimationPromptDataset,
        validationIndices: Sequence[int],
        cache: DatasetCache,
        successThresholdDegrees: float,
        maximumSamples: int,
        steps: int,
        guidance: float,
        device: torch.device,
    ) -> float:
        model.eval()
        textEncoder.eval()
        selectedIndices = list(validationIndices)[:maximumSamples]
        successCount = 0
        totalCount = 0
        for index in selectedIndices:
            rotationSequence = cache.rotationSequences[index].to(device)
            textEmbedding = cache.textEmbeddings[index].unsqueeze(0).to(device)
            tagEmbedding = cache.tagEmbeddings[index].unsqueeze(0).to(device)
            frameCount = rotationSequence.shape[1]
            boneCount = rotationSequence.shape[2]
            causalMask = torch.triu(
                torch.ones(
                    frameCount,
                    frameCount,
                    device=device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            generated = diffusion.sample(
                frameCount=frameCount,
                boneCount=boneCount,
                textEmbedding=textEmbedding,
                tagEmbedding=tagEmbedding,
                contextSequence=None,
                steps=steps,
                guidanceScale=guidance,
                causalMask=causalMask,
                device=device,
            )
            generatedQuaternion = QuaternionConverter.quaternionFromRotation6d(
                generated[0]
            )
            targetQuaternion = QuaternionConverter.quaternionFromRotation6d(
                rotationSequence[0]
            )
            errorDegrees = QuaternionMetrics.geodesicDistanceDegrees(
                generatedQuaternion,
                targetQuaternion,
            ).mean()
            totalCount += 1
            if float(errorDegrees) <= successThresholdDegrees:
                successCount += 1
        if totalCount == 0:
            return 0.0
        return successCount / totalCount


class CheckpointManager:
    """Handle serialization of training checkpoints."""

    @staticmethod
    def save(
        path: Path,
        model: TemporalUNetMoE,
        textEncoder: PretrainedTextEncoder,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler],
        step: int,
        epoch: int,
        configuration: Dict[str, Any],
    ) -> None:
        """Persist a full training checkpoint and report its size.

        Parameters
        ----------
        path : Path
            Destination file for the checkpoint payload.
        model : TemporalUNetMoE
            Model instance whose parameters must be saved.
        textEncoder : PretrainedTextEncoder
            Frozen text encoder used during training.
        optimizer : torch.optim.Optimizer
            Optimizer whose state enables seamless resume.
        scaler : Optional[GradScaler]
            AMP gradient scaler (``torch.amp.GradScaler``) state when mixed
            precision is enabled.
        step : int
            Training step index to encode in the checkpoint.
        epoch : int
            Epoch index to encode in the checkpoint.
        configuration : Dict[str, Any]
            Additional configuration metadata associated with the run.
        """
        checkpoint = {
            "model": model.state_dict(),
            "text_encoder": textEncoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "epoch": epoch,
            "configuration": configuration,
        }
        torch.save(checkpoint, path)
        sizeBytes = path.stat().st_size
        sizeMegabytes = sizeBytes / (1024 ** 2)
        message = (
            "[checkpoint] "
            f"saved='{path.name}' size={sizeMegabytes:.2f} MB "
            f"({sizeBytes} bytes)"
        )
        print(message, flush=True)

    @staticmethod
    def load(
        path: Path,
        model: TemporalUNetMoE,
        textEncoder: PretrainedTextEncoder,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[GradScaler] = None,
        mapLocation: str | torch.device = "cpu",
    ) -> Tuple[int, int, Optional[Dict[str, Any]]]:
        checkpoint = torch.load(
            path,
            map_location=mapLocation,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])
        textEncoder.load_state_dict(checkpoint["text_encoder"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        configuration = checkpoint.get("configuration")
        return (
            checkpoint.get("step", 0),
            checkpoint.get("epoch", 0),
            configuration,
        )
class Prompt2AnimDiffusionTrainer:
    """Coordinate the end-to-end training pipeline."""

    def __init__(self, configuration: TrainingConfiguration) -> None:
        self.configuration = configuration
        self.device: Optional[torch.device] = None
        self.deviceLabel: Optional[str] = None
        self.textEncoder: Optional[PretrainedTextEncoder] = None
        self.dataset: Optional[AnimationPromptDataset] = None
        self.model: Optional[TemporalUNetMoE] = None
        self.diffusion: Optional[CausalDiffusion] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler: Optional[GradScaler] = None
        self.cache: Optional[DatasetCache] = None
        self.trainIndices: List[int] = []
        self.validationIndices: List[int] = []
        self.globalStep: int = 0
        self.startEpoch: int = 0
        self.bestSuccess: float = 0.0

    def runTraining(self) -> None:
        RandomnessController.seedEverywhere(self.configuration.randomSeed)
        self._selectDevice()
        self._initializeComponents()
        self._splitDataset()
        self._buildCache()
        self._maybeResume()
        self._trainEpochs()

    def _selectDevice(self) -> None:
        options = self.configuration.deviceOptions
        device, label = DeviceSelector.selectDevice(options)
        self.device = device
        self.deviceLabel = label
        print(f"[device] backend={label}")

    def _initializeComponents(self) -> None:
        self._buildTextEncoder()
        self._buildDataset()
        self._buildModelAndDiffusion()
        self._buildOptimizer()
        self._buildScaler()

    def _buildTextEncoder(self) -> None:
        assert self.device is not None
        self.textEncoder = PretrainedTextEncoder(
            modelName=self.configuration.textModelName,
            device=self.device,
            trainable=False,
        )

    def _buildDataset(self) -> None:
        self.dataset = AnimationPromptDataset(
            dataDirectory=self.configuration.dataDirectory,
            sequenceFrames=self.configuration.sequenceFrames,
            contextHistory=self.configuration.contextHistory,
        )

    def _buildModelAndDiffusion(self) -> None:
        assert self.dataset is not None
        assert self.textEncoder is not None
        assert self.device is not None
        rotationInputDim = len(self.dataset.boneNames) * 6
        moeConfig = {
            "expertCount": self.configuration.expertCount,
            "topK": self.configuration.expertTopK,
        }
        self.model = TemporalUNetMoE(
            rotationInputDim=rotationInputDim,
            hiddenDim=self.configuration.modelDimension,
            layerCount=self.configuration.layerCount,
            moeConfiguration=moeConfig,
            textDim=self.textEncoder.outDimension,
        ).to(self.device)
        self.diffusion = CausalDiffusion(self.model)

    def _buildOptimizer(self) -> None:
        assert self.model is not None
        assert self.textEncoder is not None
        parameters = list(self.model.parameters())
        if self.textEncoder.trainable:
            parameters += list(self.textEncoder.parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.configuration.learningRate,
            weight_decay=1e-2,
        )

    def _buildScaler(self) -> None:
        assert self.deviceLabel is not None
        useAmp = self.deviceLabel in {"cuda", "mps"}
        enableGradScaler = useAmp and self.deviceLabel == "cuda"
        self.scaler = GradScaler("cuda", enabled=enableGradScaler)

    def _splitDataset(self) -> None:
        assert self.dataset is not None
        datasetSize = len(self.dataset)
        indices = np.arange(datasetSize)
        np.random.shuffle(indices)
        validationSize = max(
            1,
            int(self.configuration.validationSplit * datasetSize),
        )
        self.validationIndices = indices[:validationSize].tolist()
        self.trainIndices = indices[validationSize:].tolist()
        print(
            f"[data] total={datasetSize} train={len(self.trainIndices)} "
            f"val={len(self.validationIndices)}"
        )

    def _buildCache(self) -> None:
        assert self.dataset is not None
        assert self.device is not None
        assert self.textEncoder is not None
        self.cache = DatasetCacheBuilder.buildCache(
            dataset=self.dataset,
            device=self.device,
            textEncoder=self.textEncoder,
            cacheOnDevice=self.configuration.cacheOnDevice,
        )

    def _maybeResume(self) -> None:
        resumePath = self.configuration.resumePath
        if resumePath is None or not resumePath.exists():
            return
        assert self.model is not None
        assert self.textEncoder is not None
        assert self.optimizer is not None
        checkpointStep, checkpointEpoch, _ = CheckpointManager.load(
            resumePath,
            self.model,
            self.textEncoder,
            optimizer=self.optimizer,
            scaler=self.scaler,
            mapLocation=self.device or "cpu",
        )
        self.globalStep = checkpointStep
        self.startEpoch = checkpointEpoch
        print(f"[resume] step={self.globalStep} epoch={self.startEpoch}")

    def _trainEpochs(self) -> None:
        assert self.model is not None
        assert self.diffusion is not None
        assert self.textEncoder is not None
        assert self.optimizer is not None
        assert self.dataset is not None
        assert self.cache is not None
        assert self.device is not None
        for epoch in range(self.startEpoch, self.configuration.epochs):
            self._trainSingleEpoch(epoch)
            self._saveCheckpoint(epoch)

    def _trainSingleEpoch(self, epoch: int) -> None:
        assert self.model is not None
        assert self.diffusion is not None
        assert self.optimizer is not None
        assert self.textEncoder is not None
        assert self.cache is not None
        assert self.device is not None
        self._maybeRecacheDataset()
        self._setModelTrainMode()
        iterations = max(
            1,
            len(self.trainIndices) // max(1, self.configuration.batchSize),
        )
        progressBar = tqdm(
            range(iterations),
            desc=f"epoch {epoch + 1}/{self.configuration.epochs}",
        )
        for _ in progressBar:
            lossValue = self._executeTrainingStep(epoch)
            progressBar.set_postfix({"loss": float(lossValue)})
            if self.bestSuccess >= self.configuration.targetSuccessRate:
                print(
                    "[early-stop] target success "
                    f"{self.configuration.targetSuccessRate:.2f} atteint."
                )
                return

    def _maybeRecacheDataset(self) -> None:
        if not self.configuration.recacheEveryEpoch:
            return
        assert self.device is not None
        self._buildCache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _setModelTrainMode(self) -> None:
        assert self.model is not None
        assert self.textEncoder is not None
        self.model.train()
        if self.textEncoder.trainable:
            self.textEncoder.train()
        else:
            self.textEncoder.eval()

    def _executeTrainingStep(self, epoch: int) -> Tensor:
        useContext = self._shouldUseContext(epoch)
        batch = self._sampleMinibatch(useContext)
        lossValue = self._optimizeBatch(batch, useContext)
        self.globalStep += 1
        self._handlePeriodicActions(epoch)
        return lossValue

    def _handlePeriodicActions(self, epoch: int) -> None:
        if self.globalStep % self.configuration.checkpointInterval == 0:
            self._saveCheckpoint(epoch)
        if self.globalStep % self.configuration.validationInterval == 0:
            self._runValidation(epoch)

    def _saveCheckpoint(self, epoch: int) -> None:
        assert self.model is not None
        assert self.textEncoder is not None
        assert self.optimizer is not None
        saveDirectory = self.configuration.saveDirectory
        saveDirectory.mkdir(parents=True, exist_ok=True)
        lastPath = saveDirectory / (
            f"{self.configuration.experimentName}_last.pt"
        )
        CheckpointManager.save(
            lastPath,
            self.model,
            self.textEncoder,
            self.optimizer,
            self.scaler,
            self.globalStep,
            epoch,
            asdict(self.configuration),
        )
        if self.bestSuccess > 0:
            bestPath = saveDirectory / (
                f"{self.configuration.experimentName}_best.pt"
            )
            CheckpointManager.save(
                bestPath,
                self.model,
                self.textEncoder,
                self.optimizer,
                self.scaler,
                self.globalStep,
                epoch,
                asdict(self.configuration),
            )

    def _shouldUseContext(self, epoch: int) -> bool:
        if self.configuration.contextHistory <= 0:
            return False
        mode = self.configuration.contextTrainMode
        if mode == "off":
            return False
        if mode == "alt":
            return epoch % 2 == 1
        if mode == "ratio":
            return random.random() < self.configuration.contextTrainRatio
        raise ValueError(f"Mode de contexte inconnu: {mode}")

    def _sampleMinibatch(self, useContext: bool) -> Dict[str, Tensor]:
        assert self.cache is not None
        assert self.device is not None
        selection = random.sample(
            self.trainIndices,
            k=min(self.configuration.batchSize, len(self.trainIndices)),
        )
        rotations = torch.cat(
            [self.cache.rotationSequences[i] for i in selection],
            dim=0,
        )
        textEmbeddings = torch.stack(
            [self.cache.textEmbeddings[i] for i in selection], dim=0
        )
        tagEmbeddings = torch.stack(
            [self.cache.tagEmbeddings[i] for i in selection], dim=0
        )
        contextSequence = None
        if useContext:
            contextSequence = self._buildBatchContext(selection)
        frameCount = rotations.shape[1]
        attentionMask = torch.triu(
            torch.ones(
                frameCount,
                frameCount,
                device=self.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        return {
            "rotation6d": rotations.to(self.device),
            "text": textEmbeddings.to(self.device),
            "tag": tagEmbeddings.to(self.device),
            "context": contextSequence,
            "mask": attentionMask,
        }

    def _buildBatchContext(self, selection: Sequence[int]) -> Tensor:
        assert self.dataset is not None
        contextList: List[Tensor] = []
        for index in selection:
            contextSequence = self.dataset.fetchContextSequence(
                index,
                self.configuration.contextHistory,
            )
            if contextSequence is None:
                frames = self.configuration.sequenceFrames
                bones = len(self.dataset.boneNames)
                placeholder = torch.zeros(
                    1, frames, bones, 6, device=self.device
                )
                contextList.append(placeholder)
            else:
                contextList.append(contextSequence.to(self.device))
        return torch.cat(contextList, dim=0)

    def _optimizeBatch(
        self,
        batch: Dict[str, Tensor],
        useContext: bool,
    ) -> Tensor:
        assert self.model is not None
        assert self.diffusion is not None
        assert self.optimizer is not None
        assert self.scaler is not None
        rotation6d = batch["rotation6d"]
        textEmbedding = batch["text"]
        tagEmbedding = batch["tag"]
        contextSequence = batch["context"] if useContext else None
        attentionMask = batch["mask"]
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            with torch.cuda.amp.autocast():
                loss = self.diffusion.loss(
                    rotation6d,
                    textEmbedding,
                    tagEmbedding,
                    contextSequence,
                    attentionMask,
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.diffusion.loss(
                rotation6d,
                textEmbedding,
                tagEmbedding,
                contextSequence,
                attentionMask,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return loss.detach()

    def _runValidation(self, epoch: int) -> None:
        assert self.model is not None
        assert self.diffusion is not None
        assert self.textEncoder is not None
        assert self.cache is not None
        assert self.device is not None
        successRate = SuccessRateEvaluator.computeSuccessRate(
            model=self.model,
            diffusion=self.diffusion,
            textEncoder=self.textEncoder,
            dataset=self.dataset,
            validationIndices=self.validationIndices,
            cache=self.cache,
            successThresholdDegrees=self.configuration.successThresholdDegrees,
            maximumSamples=self.configuration.maximumValidationSamples,
            steps=8,
            guidance=2.0,
            device=self.device,
        )
        print(
            "[val] step="
            f"{self.globalStep} epoch={epoch} success_rate={successRate:.3f}"
        )
        if successRate > self.bestSuccess:
            self.bestSuccess = successRate
            self._saveCheckpoint(epoch)
