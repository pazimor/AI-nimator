"""Inference pipeline for the prompt-to-animation diffusion model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from src.shared.device import DeviceSelector
from src.shared.animation_io import convertRotationsToTensor, loadAnimationFile, loadPromptsFile
from src.shared.quaternion import QuaternionConverter
from src.shared.temporal_diffusion import CausalDiffusion, TemporalUNetMoE
from src.shared.text import PretrainedTextEncoder
from src.shared.types import SamplingConfiguration

class Prompt2AnimDiffusionSampler:
    """Run inference sampling using a trained checkpoint."""

    def __init__(self, configuration: SamplingConfiguration) -> None:
        """Initialise the sampler with the runtime configuration."""

        self.configuration = configuration
        self.device: Optional[torch.device] = None
        self.deviceLabel: Optional[str] = None
        self.checkpoint: Optional[Dict[str, Any]] = None
        self.model: Optional[TemporalUNetMoE] = None
        self.textEncoder: Optional[PretrainedTextEncoder] = None
        self.diffusion: Optional[CausalDiffusion] = None
        self.boneNames: List[str] = []

    def runSampling(self) -> None:
        """Execute the full sampling pipeline."""

        self._selectDevice()
        self._loadCheckpointMetadata()
        self._buildTextEncoder()
        self._determineBoneNames()
        self._buildModelAndDiffusion()
        self._loadModelWeights()
        promptEntry = self._loadPromptEntry()
        textEmbedding, tagEmbedding = self._encodePrompts(promptEntry)
        contextSequence = self._loadContextSequence()
        rotation6d = self._generateSequence(
            textEmbedding,
            tagEmbedding,
            contextSequence,
        )
        self._writeOutput(promptEntry, rotation6d)

    def _selectDevice(self) -> None:
        """Select an execution device based on configuration."""

        options = self.configuration.deviceOptions
        device, label = DeviceSelector.selectDevice(options)
        self.device = device
        self.deviceLabel = label
        print(f"[device] backend={label}")

    def _loadCheckpointMetadata(self) -> None:
        """Load the checkpoint dictionary from disk."""

        self.checkpoint = torch.load(
            self.configuration.checkpointPath,
            map_location=self.device or "cpu",
        )

    def _buildTextEncoder(self) -> None:
        """Instantiate the pretrained text encoder for inference."""

        assert self.device is not None
        assert self.checkpoint is not None
        checkpointConfig = self.checkpoint.get("configuration", {})
        modelName = self.configuration.textModelName or checkpointConfig.get(
            "textModelName",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.textEncoder = PretrainedTextEncoder(
            modelName=modelName,
            device=self.device,
            trainable=False,
        )

    def _determineBoneNames(self) -> None:
        """Determine the bone ordering used for generation."""

        boneNames = list(self.configuration.boneNames)
        if boneNames:
            self.boneNames = boneNames
            return
        if self.configuration.contextJsons:
            firstContext = self.configuration.contextJsons[0]
            contextRotations = loadAnimationFile(firstContext)
            self.boneNames = list(contextRotations.keys())
            return
        if self.checkpoint is None:
            raise ValueError(
                "Impossible de déterminer les bones sans checkpoint."
            )
        weightMatrix = self.checkpoint["model"]["inputProjection.weight"]
        rotationInputDim = weightMatrix.shape[0]
        inferredBoneCount = rotationInputDim // 6
        raise ValueError(
            "Fournir --bones ou --context-jsons pour fixer l'ordre des bones "
            f"(dimension détectée: {inferredBoneCount})."
        )

    def _buildModelAndDiffusion(self) -> None:
        """Create the model and diffusion process for inference."""

        assert self.checkpoint is not None
        assert self.textEncoder is not None
        assert self.device is not None
        checkpointConfig = self.checkpoint.get("configuration", {})
        modelDimension = checkpointConfig.get("modelDimension")
        layerCount = checkpointConfig.get("layerCount")
        expertCount = checkpointConfig.get("expertCount")
        expertTopK = checkpointConfig.get("expertTopK")
        if None in {modelDimension, layerCount, expertCount, expertTopK}:
            raise ValueError(
                "Le checkpoint ne fournit pas les métadonnées du modèle."
            )
        rotationInputDim = len(self.boneNames) * 6
        moeConfig = {"expertCount": expertCount, "topK": expertTopK}
        self.model = TemporalUNetMoE(
            rotationInputDim=rotationInputDim,
            hiddenDim=modelDimension,
            layerCount=layerCount,
            moeConfiguration=moeConfig,
            textDim=self.textEncoder.outDimension,
        ).to(self.device)
        self.diffusion = CausalDiffusion(self.model)

    def _loadModelWeights(self) -> None:
        """Load model and text encoder weights from checkpoint."""

        assert self.model is not None
        assert self.textEncoder is not None
        assert self.checkpoint is not None
        self.model.load_state_dict(self.checkpoint["model"])
        self.textEncoder.load_state_dict(self.checkpoint["text_encoder"])
        self.model.eval()
        self.textEncoder.eval()

    def _loadPromptEntry(self) -> Dict[str, str]:
        """Load the first prompt entry from the prompts file."""

        prompts = loadPromptsFile(self.configuration.promptsPath)
        return prompts[0]

    def _encodePrompts(
        self,
        promptEntry: Dict[str, str],
    ) -> Tuple[Tensor, Tensor]:
        """Encode textual prompt and tag using the text encoder.

        Parameters
        ----------
        promptEntry : Dict[str, str]
            Mapping representing a single prompt description.

        Returns
        -------
        tuple[Tensor, Tensor]
            Pair containing text and tag embeddings on the target device.
        """

        assert self.textEncoder is not None
        textPrompt = " ".join(
            [
                promptEntry.get("Simple", ""),
                promptEntry.get("advanced", "")
            ]
        ).strip()
        tagLabel = (promptEntry.get("tag", "") or "").strip()
        with torch.no_grad():
            textEmbedding = self.textEncoder([textPrompt])
        if tagLabel:
            with torch.no_grad():
                tagEmbedding = self.textEncoder([tagLabel])
        else:
            tagEmbedding = torch.zeros(
                1,
                self.textEncoder.outDimension,
                device=self.textEncoder.device,
                dtype=torch.float32,
            )
        return textEmbedding.to(self.device), tagEmbedding.to(self.device)

    def _loadContextSequence(self) -> Optional[Tensor]:
        """Load optional context sequences for conditioned sampling.

        Returns
        -------
        Optional[Tensor]
            Context tensor matching the requested frame count when available.
        """

        if not self.configuration.contextJsons:
            return None
        sequences: List[Tensor] = []
        for path in self.configuration.contextJsons:
            rotations = loadAnimationFile(path)
            if list(rotations.keys()) != self.boneNames:
                raise ValueError(
                    "Les bones du contexte diffèrent du modèle entraîné."
                )
            quaternionTensor, _ = convertRotationsToTensor(rotations)
            rotation6d = QuaternionConverter.rotation6dFromQuaternion(
                quaternionTensor
            )
            sequences.append(rotation6d)
        concatenated = torch.cat(sequences, dim=0)
        desiredLength = min(
            self.configuration.frameCount,
            concatenated.shape[0],
        )
        if concatenated.shape[0] >= desiredLength:
            context = concatenated[-desiredLength:]
        else:
            padding = concatenated.new_zeros(
                desiredLength - concatenated.shape[0],
                concatenated.shape[1],
                concatenated.shape[2],
            )
            context = torch.cat([padding, concatenated], dim=0)
        return context.unsqueeze(0).to(self.device)

    def _generateSequence(
        self,
        textEmbedding: Tensor,
        tagEmbedding: Tensor,
        contextSequence: Optional[Tensor],
    ) -> Tensor:
        """Generate a rotation sequence using the diffusion model.

        Parameters
        ----------
        textEmbedding : Tensor
            Conditioning embedding derived from the prompt text.
        tagEmbedding : Tensor
            Conditioning embedding derived from the tag metadata.
        contextSequence : Optional[Tensor]
            Optional context tensor providing preceding motion.

        Returns
        -------
        Tensor
            Predicted rotation sequence in 6D representation.
        """

        assert self.diffusion is not None
        frameCount = self.configuration.frameCount
        boneCount = len(self.boneNames)
        causalMask = torch.triu(
            torch.ones(
                frameCount,
                frameCount,
                device=self.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        return self.diffusion.sample(
            frameCount=frameCount,
            boneCount=boneCount,
            textEmbedding=textEmbedding,
            tagEmbedding=tagEmbedding,
            contextSequence=contextSequence,
            steps=self.configuration.steps,
            guidanceScale=self.configuration.guidance,
            causalMask=causalMask,
            device=self.device or "cpu",
        )

    def _writeOutput(
        self,
        promptEntry: Dict[str, str],
        rotation6d: Tensor,
    ) -> None:
        """Persist generated rotations to JSON on disk.

        Parameters
        ----------
        promptEntry : Dict[str, str]
            Prompt metadata used for generation.
        rotation6d : Tensor
            Generated rotation sequence in 6D representation.
        """

        quaternionSequence = QuaternionConverter.quaternionFromRotation6d(
            rotation6d[0]
        )
        output = {"rotations": {}}
        for boneIndex, boneName in enumerate(self.boneNames):
            sequence = [
                QuaternionConverter.formatQuaternionPipeString(
                    quaternionSequence[frameIndex, boneIndex]
                )
                for frameIndex in range(self.configuration.frameCount)
            ]
            output["rotations"][boneName] = sequence
        if not self.configuration.omitMetadata:
            output["fps"] = 60
            output["prompt"] = promptEntry
        with open(
            self.configuration.outputPath,
            "w",
            encoding="utf-8",
        ) as handle:
            import json

            json.dump(output, handle, ensure_ascii=False, separators=(",", ":"))
        print(f"[ok] Wrote {self.configuration.outputPath}")
