"""Training pipeline for the prompt-to-animation diffusion model."""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
        """Initialise the dataset from a directory of clip pairs.

        Parameters
        ----------
        dataDirectory : Path
            Root directory containing ``animation.json``/``prompts.json`` pairs.
        sequenceFrames : int, optional
            Target frame count for each sampled segment, by default ``240``.
        contextHistory : int, optional
            Number of previous clips to use as context, by default ``0``.

        Raises
        ------
        FileNotFoundError
            If no valid animation/prompt pairs are discovered in the directory.
        """

        super().__init__()
        self.dataDirectory = Path(dataDirectory)
        self.sequenceFrames = sequenceFrames
        self.contextHistory = contextHistory
        self.clipRecords = self._loadClipRecords()
        if not self.clipRecords:
            raise FileNotFoundError(
                "Aucune paire animation/prompt détectée dans la base."
            )
        self.sampleDescriptors = self._buildSampleDescriptors()
        if not self.sampleDescriptors:
            raise ValueError(
                "Aucun prompt exploitable détecté dans les fichiers prompts.json."
            )
        self.boneNames = convertRotationsToTensor(
            self.clipRecords[0].rotations
        )[1]

    def __len__(self) -> int:
        """Return the total number of prompt variants available.

        Returns
        -------
        int
            Number of ``(clip, prompt)`` combinations discovered on disk.
        """

        return len(self.sampleDescriptors)

    def __getitem__(self, index: int) -> AnimationPromptSample:
        """Return the sample at ``index`` with optional context.

        Parameters
        ----------
        index : int
            Dataset index to fetch.

        Returns
        -------
        AnimationPromptSample
            Structured sample containing rotations, prompt, tag and context.
        """

        clipIndex, textPrompt, tagLabel = self.sampleDescriptors[index]
        clipRecord = self.clipRecords[clipIndex]
        quaternionTensor, _ = convertRotationsToTensor(clipRecord.rotations)
        segment = self._extractSegment(quaternionTensor)
        rotation6d = QuaternionConverter.rotation6dFromQuaternion(segment)
        contextSequence = self._buildContextSequence(clipIndex)
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
        """Aggregate prior clips to serve as context for the current item.

        Parameters
        ----------
        index : int
            Dataset sample index whose underlying clip defines the context.
        historyCount : int
            Number of previous samples to include in the context tensor.

        Returns
        -------
        Optional[Tensor]
            Batched context tensor or ``None`` when no context is available.
        """

        if historyCount <= 0:
            return None
        clipIndex = self.sampleDescriptors[index][0]
        historyIndices = [
            max(0, clipIndex - offset - 1) for offset in range(historyCount)
        ]
        return self._gatherContexts(historyIndices)

    def _loadClipRecords(self) -> List[ClipRecord]:
        """Load animation/prompt pairs from the dataset directory.

        Returns
        -------
        List[ClipRecord]
            In-memory representation of all discovered clip pairs.
        """

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

    def _buildSampleDescriptors(self) -> List[Tuple[int, str, str]]:
        """Flatten prompt variants into dataset-level descriptors."""

        descriptors: List[Tuple[int, str, str]] = []
        for clipIndex, clipRecord in enumerate(self.clipRecords):
            for promptEntry in clipRecord.prompts:
                variants = self._composePromptText(promptEntry)
                for promptText, tagLabel in variants:
                    descriptors.append((clipIndex, promptText, tagLabel))
        return descriptors

    def _discoverPairs(
        self,
        rootDirectory: Path,
    ) -> Iterable[Tuple[Path, Path]]:
        """Yield animation/prompt file pairs below ``rootDirectory``.

        Parameters
        ----------
        rootDirectory : Path
            Directory tree scanned to locate valid clip pairs.

        Yields
        ------
        Iterable[Tuple[Path, Path]]
            Tuples containing paths to ``animation.json`` and ``prompts.json``.
        """

        for directory in rootDirectory.rglob("*"):
            if not directory.is_dir():
                continue
            animationPath = directory / "animation.json"
            promptPath = directory / "prompts.json"
            if animationPath.exists() and promptPath.exists():
                yield animationPath, promptPath

    def _extractSegment(self, quaternionTensor: Tensor) -> Tensor:
        """Slice or tile ``quaternionTensor`` to match ``sequenceFrames``.

        Parameters
        ----------
        quaternionTensor : Tensor
            Quaternion rotations with shape ``(frames, joints, 4)``.

        Returns
        -------
        Tensor
            Segment of length ``sequenceFrames`` ready for conversion to 6D.
        """

        frameCount = quaternionTensor.shape[0]
        if frameCount >= self.sequenceFrames:
            startFrame = random.randint(0, frameCount - self.sequenceFrames)
            endFrame = startFrame + self.sequenceFrames
            return quaternionTensor[startFrame:endFrame]
        requiredRepeats = math.ceil(self.sequenceFrames / frameCount)
        tiled = quaternionTensor.repeat(requiredRepeats, 1, 1)
        return tiled[: self.sequenceFrames]

    def _composePromptText(self, promptEntry: Dict[str, str]) -> List[Tuple[str, str]]:
        """Return all textual prompt variants present in ``promptEntry``.

        Parameters
        ----------
        promptEntry : Dict[str, str]
            Mapping coming from ``prompts.json`` files.

        Returns
        -------
        List[Tuple[str, str]]
            Collection of `(prompt, tag)` pairs. Each non-empty variant among
            `simple`, `advanced` yields its own entry. When
            none of these fields are available, a single fallback entry is
            produced from the remaining textual values.
        """

        def _extract_value(*keys: str) -> Optional[str]:
            for key in keys:
                value = promptEntry.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    return text
            return None

        tagRaw = (
            promptEntry.get("tag") or promptEntry.get("Tag") or ""
        )
        tagLabel = str(tagRaw).strip()
        variants: List[Tuple[str, str]] = []
        seenTexts: Set[str] = set()
        for lowerKey, upperKey in (
            ("simple", "Simple"),
            ("advanced", "Advanced")
        ):
            textValue = _extract_value(lowerKey, upperKey)
            if textValue and textValue not in seenTexts:
                variants.append((textValue, tagLabel))
                seenTexts.add(textValue)

        assert variants is not None
        return variants

    def _buildContextSequence(self, clipIndex: int) -> Optional[Tensor]:
        """Create the context tensor for a specific clip index.

        Parameters
        ----------
        clipIndex : int
            Index of the clip whose context should be assembled.

        Returns
        -------
        Optional[Tensor]
            Context tensor on success, otherwise ``None``.
        """

        if self.contextHistory <= 0:
            return None
        historyIndices = [
            max(0, clipIndex - offset - 1)
            for offset in range(self.contextHistory)
        ]
        return self._gatherContexts(historyIndices)

    def _gatherContexts(self, indices: Sequence[int]) -> Optional[Tensor]:
        """Collect rotation sequences for the provided indices.

        Parameters
        ----------
        indices : Sequence[int]
            Clip indices used to retrieve context samples.

        Returns
        -------
        Optional[Tensor]
            Concatenated context tensor or ``None`` when no indices provided.
        """

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

class SuccessRateEvaluator:
    """Utility to measure validation success rate."""

    @staticmethod
    @torch.no_grad()
    def computeSuccessRate(
        model: TemporalUNetMoE,
        diffusion: CausalDiffusion,
        textEncoder: PretrainedTextEncoder,
        validationIndices: Sequence[int],
        cache: DatasetCache,
        successThresholdDegrees: float,
        maximumSamples: int,
        steps: int,
        guidance: float,
        device: torch.device,
    ) -> float:
        """Estimate validation success rate under the configured metric.

        Parameters
        ----------
        model : TemporalUNetMoE
            Model under evaluation.
        diffusion : CausalDiffusion
            Diffusion process used to generate samples.
        textEncoder : PretrainedTextEncoder
            Text encoder required to embed prompts and tags.
        validationIndices : Sequence[int]
            Indices designating validation samples.
        cache : DatasetCache
            Cached tensors that avoid redundant encoding work.
        successThresholdDegrees : float
            Threshold for the geodesic distance criterion.
        maximumSamples : int
            Maximum number of validation samples to evaluate.
        steps : int
            Diffusion sampling steps.
        guidance : float
            Guidance scale applied during sampling.
        device : torch.device
            Device where evaluation should run.

        Returns
        -------
        float
            Fraction of samples whose error is below the threshold.
        """

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
        """Restore training artefacts from ``path``.

        Parameters
        ----------
        path : Path
            Checkpoint file to load.
        model : TemporalUNetMoE
            Model instance whose state should be restored.
        textEncoder : PretrainedTextEncoder
            Text encoder receiving restored weights.
        optimizer : Optional[torch.optim.Optimizer], optional
            Optimizer receiving its state_dict when provided.
        scaler : Optional[GradScaler], optional
            Gradient scaler updated when present.
        mapLocation : str or torch.device, optional
            Device mapping for ``torch.load``.

        Returns
        -------
        tuple[int, int, Optional[Dict[str, Any]]]
            Restored ``(step, epoch, configuration)`` triple.
        """

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


class NeuralLayerInitializer:
    """Centralise la construction des couches du ``TemporalUNetMoE``."""

    def __init__(
        self,
        configuration: TrainingConfiguration,
        textEncoder: PretrainedTextEncoder,
    ) -> None:
        self.configuration = configuration
        self.textEncoder = textEncoder

    def createModel(
        self,
        boneNames: Sequence[str],
        device: torch.device,
    ) -> Tuple[TemporalUNetMoE, CausalDiffusion]:
        """Instancie le modèle et son wrapper diffusion."""

        rotationInputDim = self._rotationInputDim(boneNames)
        moeConfig = {
            "expertCount": self.configuration.expertCount,
            "topK": self.configuration.expertTopK,
        }
        model = TemporalUNetMoE(
            rotationInputDim=rotationInputDim,
            hiddenDim=self.configuration.modelDimension,
            layerCount=self.configuration.layerCount,
            moeConfiguration=moeConfig,
            textDim=self.textEncoder.outDimension,
        ).to(device)
        diffusion = CausalDiffusion(model)
        return model, diffusion

    def trainableParameters(self, model: TemporalUNetMoE) -> List[torch.nn.Parameter]:
        """Retourne les paramètres à optimiser pour l'entraînement."""

        parameters: List[torch.nn.Parameter] = list(model.parameters())
        if self.textEncoder.trainable:
            parameters += list(self.textEncoder.parameters())
        return parameters

    @staticmethod
    def _rotationInputDim(boneNames: Sequence[str]) -> int:
        return len(boneNames) * 6
    
class DatasetCacheBuilder:
    """Factory responsible for building dataset caches."""

    @staticmethod
    def buildCache(
        dataset: AnimationPromptDataset,
        device: torch.device,
        textEncoder: PretrainedTextEncoder,
        cacheOnDevice: bool,
    ) -> DatasetCache:
        """Pre-compute tensors required for fast training access.

        Parameters
        ----------
        dataset : AnimationPromptDataset
            Source dataset that yields samples.
        device : torch.device
            Target device where tensors should ultimately live.
        textEncoder : PretrainedTextEncoder
            Encoder used to transform textual prompts and tags to embeddings.
        cacheOnDevice : bool
            When ``True`` tensors are moved to ``device`` as they are cached.

        Returns
        -------
        DatasetCache
            Container holding cached rotations and text embeddings.
        """

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
        """Generate a text embedding for ``textPrompt`` without gradients.

        Parameters
        ----------
        textEncoder : PretrainedTextEncoder
            Encoder responsible for computing embeddings.
        textPrompt : str
            Prompt text to embed.

        Returns
        -------
        Tensor
            Encoded text representation detached from autograd.
        """

        with torch.no_grad():
            return textEncoder([textPrompt]).detach()

    @staticmethod
    def _encodeTag(
        textEncoder: PretrainedTextEncoder,
        tagLabel: str,
    ) -> Tensor:
        """Return embedding for ``tagLabel`` or a zero-vector fallback.

        Parameters
        ----------
        textEncoder : PretrainedTextEncoder
            Encoder responsible for computing embeddings.
        tagLabel : str
            Tag string gathered from the prompt metadata.

        Returns
        -------
        Tensor
            Tag embedding tensor, zeroed if no tag was provided.
        """

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

class Prompt2AnimDiffusionTrainer:
    """Coordinate the end-to-end training pipeline."""

    def __init__(self, configuration: TrainingConfiguration) -> None:
        """Create a trainer bound to ``configuration``."""

        self.configuration = configuration
        self.device: Optional[torch.device] = None
        self.deviceLabel: Optional[str] = None
        self.textEncoder: Optional[PretrainedTextEncoder] = None
        self.dataset: Optional[AnimationPromptDataset] = None
        self.model: Optional[TemporalUNetMoE] = None
        self.diffusion: Optional[CausalDiffusion] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler: Optional[GradScaler] = None
        self.modelInitializer: Optional[NeuralLayerInitializer] = None
        self.cache: Optional[DatasetCache] = None
        self.trainIndices: List[int] = []
        self.validationIndices: List[int] = []
        self.globalStep: int = 0
        self.startEpoch: int = 0
        self.bestSuccess: float = 0.0

    def runTraining(self) -> None:
        """Execute the full training lifecycle."""

        RandomnessController.seedEverywhere(self.configuration.randomSeed)
        self._selectDevice()
        self._initializeComponents()
        self._splitDataset()
        self._buildCache()
        self._maybeResume()
        self._trainEpochs()

    def _selectDevice(self) -> None:
        """Choose computation device according to configuration."""

        options = self.configuration.deviceOptions
        device, label = DeviceSelector.selectDevice(options)
        self.device = device
        self.deviceLabel = label
        print(f"[device] backend={label}")

    def _initializeComponents(self) -> None:
        """Instantiate encoder, dataset, model, optimizer and scaler."""

        self._buildTextEncoder()
        self._buildDataset()
        self._buildModelAndDiffusion()
        self._buildOptimizer()
        self._buildScaler()

    def _buildTextEncoder(self) -> None:
        """Initialise the pretrained text encoder on the chosen device."""

        assert self.device is not None
        self.textEncoder = PretrainedTextEncoder(
            modelName=self.configuration.textModelName,
            device=self.device,
            trainable=False,
        )

    def _buildDataset(self) -> None:
        """Construct the animation prompt dataset from disk assets."""

        self.dataset = AnimationPromptDataset(
            dataDirectory=self.configuration.dataDirectory,
            sequenceFrames=self.configuration.sequenceFrames,
            contextHistory=self.configuration.contextHistory,
        )

    def _buildModelAndDiffusion(self) -> None:
        """Create the temporal diffusion model and wrapper."""

        assert self.dataset is not None
        assert self.textEncoder is not None
        assert self.device is not None
        self.modelInitializer = NeuralLayerInitializer(
            self.configuration,
            self.textEncoder,
        )
        self.model, self.diffusion = self.modelInitializer.createModel(
            self.dataset.boneNames,
            self.device,
        )

    def _buildOptimizer(self) -> None:
        """Instantiate the optimizer with model and encoder parameters."""

        assert self.model is not None
        assert self.modelInitializer is not None
        parameters = self.modelInitializer.trainableParameters(self.model)
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.configuration.learningRate,
            weight_decay=1e-2,
        )

    def _buildScaler(self) -> None:
        """Configure automatic mixed precision scaler when appropriate."""

        assert self.deviceLabel is not None
        useAmp = self.deviceLabel in {"cuda", "mps"}
        enableGradScaler = useAmp and self.deviceLabel == "cuda"
        self.scaler = GradScaler("cuda", enabled=enableGradScaler)

    def _splitDataset(self) -> None:
        """Partition dataset indices into train and validation splits."""

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
        """Construct the dataset cache for efficient batch sampling."""

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
        """Load checkpoint when a resume path is supplied."""

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
        """Iterate over epochs handling training and checkpointing."""

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
        """Run a single training epoch with optional validation."""


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
        """Refresh cached tensors at the beginning of each epoch when needed."""

        if not self.configuration.recacheEveryEpoch:
            return
        assert self.device is not None
        self._buildCache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _setModelTrainMode(self) -> None:
        """Set model and encoder to the appropriate training/eval modes."""

        assert self.model is not None
        assert self.textEncoder is not None
        self.model.train()
        if self.textEncoder.trainable:
            self.textEncoder.train()
        else:
            self.textEncoder.eval()

    def _executeTrainingStep(self, epoch: int) -> Tensor:
        """Run forward/backward pass for a mini-batch and update counters."""

        useContext = self._shouldUseContext(epoch)
        batch = self._sampleMinibatch(useContext)
        lossValue = self._optimizeBatch(batch, useContext)
        self.globalStep += 1
        self._handlePeriodicActions(epoch)
        return lossValue

    def _handlePeriodicActions(self, epoch: int) -> None:
        """Invoke checkpointing and validation at configured intervals."""

        if self.globalStep % self.configuration.checkpointInterval == 0:
            self._saveCheckpoint(epoch)
        if self.globalStep % self.configuration.validationInterval == 0:
            self._runValidation(epoch)

    def _saveCheckpoint(self, epoch: int) -> None:
        """Persist the current training state and best-performing model."""

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
        """Return ``True`` when context conditioning should be applied."""

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
        """Sample a mini-batch of cached tensors, optionally with context.

        Parameters
        ----------
        useContext : bool
            Whether contextual sequences should be included in the batch.

        Returns
        -------
        Dict[str, Tensor]
            Dictionary containing rotations, embeddings, context and mask.
        """

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
        """Construct a batch-level context tensor for ``selection`` indices.

        Parameters
        ----------
        selection : Sequence[int]
            Dataset indices forming the current mini-batch.

        Returns
        -------
        Tensor
            Context tensor aligned with the sampled batch.
        """

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
        """Compute diffusion loss and apply optimisation step.

        Parameters
        ----------
        batch : Dict[str, Tensor]
            Mini-batch dictionary produced by :meth:`_sampleMinibatch`.
        useContext : bool
            Indicates whether contextual conditioning is active.

        Returns
        -------
        Tensor
            Detached loss tensor for logging.
        """

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
        """Evaluate current model on the validation subset.

        Parameters
        ----------
        epoch : int
            Current epoch number used for logging and checkpointing.
        """

        assert self.model is not None
        assert self.diffusion is not None
        assert self.textEncoder is not None
        assert self.cache is not None
        assert self.device is not None
        successRate = SuccessRateEvaluator.computeSuccessRate(
            model=self.model,
            diffusion=self.diffusion,
            textEncoder=self.textEncoder,
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
