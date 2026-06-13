"""Data utilities for the text<->motion CLIP pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ainimator.data.builder.dataset_reader import (
    DetailedMotionPayload,
    loadAnimationPayloadWithExtras,
    loadPromptSegments,
)
from ainimator.geometry.components import (
    buildComponentRegistry,
    buildMotionFeatureTensors,
)
from ainimator.geometry.components.base import MotionComponent
from ainimator.geometry.skeleton import SkeletonNormalizer
from ainimator.core.types import (
    ClipDatasetRecord,
    ClipPromptSegment,
    MotionTextSample,
)

LOGGER = logging.getLogger("shared.clip.data")
OPTIONAL_COMPONENTS = tuple(
    component
    for component in buildComponentRegistry()
    if component.sampleKey != "motion"
)
OPTIONAL_COMPONENT_SAMPLE_KEYS = tuple(
    component.sampleKey for component in OPTIONAL_COMPONENTS
)
EXTRAS_DEPENDENT_COMPONENT_KEYS = frozenset(
    {
        "root_translation",
        "root_velocity",
    }
)


def loadPromptFile(path: str | Path) -> List[ClipPromptSegment]:
    """
    Return prompt segments listed in a prompt.json file.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the prompt JSON file.

    Returns
    -------
    List[ClipPromptSegment]
        Parsed segments enriched with prompt metadata.
    """
    metadata, segments = loadPromptSegments(path)
    return [
        ClipPromptSegment(
            startFrame=segment.startFrame,
            endFrame=segment.endFrame,
            text=segment.text,
            sourceFile=segment.sourceFile,
            metadata=metadata,
        )
        for segment in segments
    ]


def sliceMotion(
    motion: torch.Tensor,
    startFrame: int,
    endFrame: int,
) -> torch.Tensor:
    """
    Return a cropped view of the motion tensor.

    Parameters
    ----------
    motion : torch.Tensor
        Full motion payload shaped (frames, bones, 6).
    startFrame : int
        First frame included in the slice (inclusive).
    endFrame : int
        First frame excluded from the slice (exclusive).

    Returns
    -------
    torch.Tensor
        Cropped motion tensor shaped (T, bones, 6).
    """
    return motion[startFrame:endFrame]


class MotionTextClipDataset(Dataset[MotionTextSample]):
    """Dataset pairing prompt segments with sliced motions."""

    def __init__(
        self,
        rootPrompts: Path,
        rootAnimations: Path,
        tokenizer: PreTrainedTokenizerBase,
        maxLength: int,
        cacheMotion: bool = True,
        skeletonNormalizer: SkeletonNormalizer | None = None,
    ) -> None:
        """
        Build the dataset index without loading motions into memory.

        Parameters
        ----------
        rootPrompts : Path
            Directory containing prompt.json files.
        rootAnimations : Path
            Directory containing animation files mirroring prompt layout.
        tokenizer : PreTrainedTokenizerBase
            Tokenizer used for XLM-Roberta inputs.
        maxLength : int
            Maximum token length used for padding and truncation.
        cacheMotion : bool, optional
            When True motions are cached after the first load.
        skeletonNormalizer : SkeletonNormalizer | None, optional
            Optional normalizer applied when loading animation bones.
        """
        self.rootPrompts = rootPrompts
        self.rootAnimations = rootAnimations
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.cacheMotion = cacheMotion
        self.skeletonNormalizer = skeletonNormalizer
        # LRU cache with size 1: keeps only the last loaded file
        # Since prompts from the same file are consecutive, this avoids reloading
        self._cachedPath: Optional[Path] = None
        self._cachedMotion: Optional[DetailedMotionPayload] = None
        self._cacheHits = 0
        self._cacheMisses = 0
        self.records: List[ClipDatasetRecord] = self._buildIndex()

    def __len__(self) -> int:
        """
        Return the number of indexed prompt segments.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """
        Return a single tokenized text and motion slice pair.

        Parameters
        ----------
        index : int
            Index of the sample in the internal record list.

        Returns
        -------
        Dict[str, object]
            Tokenized text fields, the base rotation tensor, and any derived
            motion features available for downstream consumers.
        """
        record = self.records[index]
        try:
            motionSlice, meta, featureTensors = self._sliceMotion(record)
        except Exception:
            LOGGER.error(
                "Failed to load sample %d from file: %s (frames %d-%d)",
                index,
                record.animationPath,
                record.startFrame,
                record.endFrame,
            )
            raise
        encoded = self._tokenize(record.promptText)
        sample = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "motion": motionSlice,
            "time": torch.tensor([record.startFrame, record.endFrame]),
            "meta": meta,
        }
        sample.update(featureTensors)
        return sample

    def _buildIndex(self) -> List[ClipDatasetRecord]:
        """
        Pre-index prompt segments across every prompt file.

        Returns
        -------
        List[ClipDatasetRecord]
            Indexed segments mapped to animation files.
        """
        records: List[ClipDatasetRecord] = []
        for promptPath in self._listPromptFiles():
            segments = loadPromptFile(promptPath)
            animationPath = self._resolveAnimationPath(promptPath)
            for segment in segments:
                records.append(
                    ClipDatasetRecord(
                        promptText=segment.text,
                        animationPath=animationPath,
                        startFrame=segment.startFrame,
                        endFrame=segment.endFrame,
                        sourceFile=segment.sourceFile,
                        metadata=segment.metadata,
                    ),
                )
        return records

    def _listPromptFiles(self) -> Sequence[Path]:
        """
        Return every prompt.json found under the root directory.

        Returns
        -------
        Sequence[Path]
            Sorted list of prompt file paths.
        """
        return sorted(self.rootPrompts.rglob("prompt.json"))

    def _resolveAnimationPath(self, promptPath: Path) -> Path:
        """
        Infer the animation path based on the prompt location.

        Parameters
        ----------
        promptPath : Path
            Location of the prompt.json file.

        Returns
        -------
        Path
            Resolved animation path living next to the prompt file.

        Raises
        ------
        FileNotFoundError
            Raised when no candidate animation file is present.
        """
        for candidate in self._candidateAnimationPaths(promptPath):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No animation file found next to {promptPath}")

    def _candidateAnimationPaths(self, promptPath: Path) -> Sequence[Path]:
        """
        Return every possible animation path derived from a prompt file.

        Parameters
        ----------
        promptPath : Path
            Prompt file path used as reference.

        Returns
        -------
        Sequence[Path]
            Ordered list of animation path candidates.
        """
        try:
            relativeDirectory = promptPath.parent.relative_to(self.rootPrompts)
        except ValueError:
            relativeDirectory = promptPath.parent
        targetDirectory = self.rootAnimations / relativeDirectory
        promptStem = promptPath.stem.replace("prompt", "animation")
        return [
            targetDirectory / "animation.js",
            targetDirectory / "animation.json",
            targetDirectory / f"{promptStem}.js",
            targetDirectory / f"{promptStem}.json",
        ]

    def _loadMotion(self, path: Path) -> DetailedMotionPayload:
        """
        Load motion, metadata, and extras from disk or LRU cache.

        Uses a single-entry LRU cache. Since prompts from the same animation
        file are indexed consecutively, this avoids reloading the same file
        multiple times within a batch or consecutive samples.

        Parameters
        ----------
        path : Path
            Motion payload location.

        Returns
        -------
        DetailedMotionPayload
            Motion tensor, metadata, and top-level extras.
        """
        # Check if this is the cached file
        if (
            self.cacheMotion
            and self._cachedPath == path
            and self._cachedMotion is not None
        ):
            self._cacheHits += 1
            return self._cachedMotion

        # Cache miss - need to load from disk
        self._cacheMisses += 1
        LOGGER.debug(
            "Loading motion file: %s (cache hits: %d, misses: %d)",
            path,
            self._cacheHits,
            self._cacheMisses,
        )
        motion, meta, extras = loadAnimationPayloadWithExtras(
            path,
            skeletonNormalizer=self.skeletonNormalizer,
        )

        # Update LRU cache (replace previous entry)
        if self.cacheMotion:
            self._cachedPath = path
            self._cachedMotion = (motion, meta, extras)

        return motion, meta, extras

    def clearCache(self) -> None:
        """
        Clear the motion cache.

        Call this between dataset rotations or when memory needs to be freed.
        """
        self._cachedPath = None
        self._cachedMotion = None
        LOGGER.debug(
            "Motion cache cleared (total hits: %d, misses: %d)",
            self._cacheHits,
            self._cacheMisses,
        )

    def getCacheStats(self) -> Dict[str, int]:
        """Return cache hit/miss statistics."""
        return {
            "hits": self._cacheHits,
            "misses": self._cacheMisses,
            "hitRate": (
                self._cacheHits / max(1, self._cacheHits + self._cacheMisses)
            ),
        }

    def _sliceMotion(
        self,
        record: ClipDatasetRecord,
    ) -> tuple[torch.Tensor, Dict[str, object], Dict[str, torch.Tensor]]:
        """
        Return the cropped motion slice, merged metadata, and derived features.

        Parameters
        ----------
        record : ClipDatasetRecord
            Index entry describing the desired motion slice.

        Returns
        -------
        tuple[torch.Tensor, Dict[str, object], Dict[str, torch.Tensor]]
            Cropped motion, merged metadata, and optional feature tensors.
        """
        motion, meta, extras = self._loadMotion(record.animationPath)
        motionSlice = sliceMotion(motion, record.startFrame, record.endFrame)
        slicedExtras = _sliceTemporalExtras(
            extras=extras,
            startFrame=record.startFrame,
            endFrame=record.endFrame,
        )
        featureTensors = buildMotionFeatureTensors(
            motion=motionSlice,
            extras=slicedExtras,
            enabledComponents=_availableOptionalComponents(slicedExtras),
        )
        if "trans" not in slicedExtras:
            zeroTranslation = torch.zeros(
                motionSlice.shape[0],
                3,
                dtype=motionSlice.dtype,
            )
            featureTensors["root_translation"] = zeroTranslation
            featureTensors["root_velocity"] = torch.zeros_like(zeroTranslation)
        mergedMeta = meta | record.metadata
        return motionSlice, mergedMeta, featureTensors

    def _tokenize(self, promptText: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize the prompt text.

        Parameters
        ----------
        promptText : str
            Raw prompt text.

        Returns
        -------
        Dict[str, torch.Tensor]
            Tokenized tensors squeezed on the batch dimension.
        """
        encoded = self.tokenizer(
            promptText,
            padding="max_length",
            truncation=True,
            max_length=self.maxLength,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def motionTextCollate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """
    Collate function handling variable-length motion sequences.

    Parameters
    ----------
    batch : Sequence[Dict[str, object]]
        Items emitted by MotionTextClipDataset.__getitem__.

    Returns
    -------
    Dict[str, object]
        Batched tensors padded on the temporal dimension.
    """
    lengths = [item["motion"].shape[0] for item in batch]
    maxTime = max(lengths)
    motionBatch = torch.stack(
        [_padTemporalTensor(item["motion"], maxTime) for item in batch],
    )
    motionMask = torch.stack(
        [_buildMotionMask(length, maxTime, motionBatch.device) for length in lengths],
    )
    payload = {
        "motion": motionBatch,
        "motion_mask": motionMask,
        "time": torch.stack([item["time"] for item in batch]),
        "meta": [item["meta"] for item in batch],
    }
    if all("input_ids" in item for item in batch):
        payload["input_ids"] = torch.stack([item["input_ids"] for item in batch])
    if all("attention_mask" in item for item in batch):
        payload["attention_mask"] = torch.stack(
            [item["attention_mask"] for item in batch],
        )
    if all("sample_id" in item for item in batch):
        payload["sample_id"] = torch.tensor(
            [int(item["sample_id"]) for item in batch],
            dtype=torch.long,
        )
    if all("text_id" in item for item in batch):
        payload["text_id"] = torch.tensor(
            [int(item["text_id"]) for item in batch],
            dtype=torch.long,
        )
    if all("pooled_text" in item for item in batch):
        payload["pooled_text"] = torch.stack(
            [item["pooled_text"] for item in batch]
        )
    if all("generation_text_embedding" in item for item in batch):
        payload["generation_text_embedding"] = torch.stack(
            [item["generation_text_embedding"] for item in batch]
        )
    for sampleKey in OPTIONAL_COMPONENT_SAMPLE_KEYS:
        tensors = [item.get(sampleKey) for item in batch]
        if not any(isinstance(value, torch.Tensor) for value in tensors):
            continue
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise KeyError(
                f"Inconsistent batch: missing tensor component {sampleKey!r}."
            )
        payload[sampleKey] = torch.stack(
            [_padTemporalTensor(value, maxTime) for value in tensors]
        )
    return payload


def _sliceTemporalExtras(
    extras: Dict[str, object],
    startFrame: int,
    endFrame: int,
) -> Dict[str, object]:
    """Slice top-level animation extras so they stay aligned with motion."""
    sliced: Dict[str, object] = {}
    rawTranslation = extras.get("trans")
    if rawTranslation is None:
        return sliced
    translation = torch.as_tensor(rawTranslation, dtype=torch.float32)
    if translation.dim() != 2 or translation.shape[1] != 3:
        return sliced
    sliced["trans"] = translation[startFrame:endFrame]
    return sliced


def _availableOptionalComponents(
    extras: Dict[str, object],
) -> tuple[MotionComponent, ...]:
    """Return only the optional components supported by the current sample."""
    if "trans" in extras:
        return OPTIONAL_COMPONENTS
    return tuple(
        component
        for component in OPTIONAL_COMPONENTS
        if component.key not in EXTRAS_DEPENDENT_COMPONENT_KEYS
    )


def _padTemporalTensor(
    tensor: torch.Tensor,
    targetLength: int,
) -> torch.Tensor:
    """
    Pad a temporal tensor to a target temporal length.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor shaped (frames, ...).
    targetLength : int
        Desired temporal length after padding.

    Returns
    -------
    torch.Tensor
        Tensor padded with zeros at the end.
    """
    if tensor.shape[0] == targetLength:
        return tensor
    paddingShape = (targetLength - tensor.shape[0],) + tuple(tensor.shape[1:])
    padding = torch.zeros(
        paddingShape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=0)


def _buildMotionMask(
    length: int,
    targetLength: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a boolean mask indicating which frames are valid (not padding).
    """
    return torch.arange(targetLength, device=device) < length
