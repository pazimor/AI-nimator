"""Data utilities for the text<->motion CLIP pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.features.dataset_builder.dataset_reader import (
    MotionPayload,
    loadAnimationPayload,
    loadPromptSegments,
)
from src.shared.skeleton import SkeletonNormalizer
from src.shared.types import (
    ClipDatasetRecord,
    ClipPromptSegment,
    MotionTextSample,
)

LOGGER = logging.getLogger("shared.clip.data")
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
        Parsed segments enriched with the file-level tag.
    """
    tag, metadata, segments = loadPromptSegments(path)
    return [
        ClipPromptSegment(
            startFrame=segment.startFrame,
            endFrame=segment.endFrame,
            text=segment.text,
            sourceFile=segment.sourceFile,
            tag=tag,
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
        self._cachedMotion: Optional[Tuple] = None
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
            Tokenized text fields and motion slice for contrastive training.
        """
        record = self.records[index]
        try:
            motionSlice, meta = self._sliceMotion(record)
        except Exception as e:
            LOGGER.error(
                "Failed to load sample %d from file: %s (frames %d-%d)",
                index,
                record.animationPath,
                record.startFrame,
                record.endFrame,
            )
            raise
        encoded = self._tokenize(record.tag, record.promptText)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "motion": motionSlice,
            "time": torch.tensor([record.startFrame, record.endFrame]),
            "tag": record.tag,
            "meta": meta,
        }

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
                        tag=segment.tag,
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

    def _loadMotion(self, path: Path) -> MotionPayload:
        """
        Load motion from disk or LRU cache.
        
        Uses a single-entry LRU cache. Since prompts from the same animation
        file are indexed consecutively, this avoids reloading the same file
        multiple times within a batch or consecutive samples.

        Parameters
        ----------
        path : Path
            Motion payload location.

        Returns
        -------
        MotionPayload
            Motion tensor and metadata.
        """
        # Check if this is the cached file
        if self.cacheMotion and self._cachedPath == path and self._cachedMotion is not None:
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
        motion, meta = loadAnimationPayload(
            path,
            skeletonNormalizer=self.skeletonNormalizer,
        )
        
        # Update LRU cache (replace previous entry)
        if self.cacheMotion:
            self._cachedPath = path
            self._cachedMotion = (motion, meta)
        
        return motion, meta
    
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
            "hitRate": self._cacheHits / max(1, self._cacheHits + self._cacheMisses),
        }

    @staticmethod
    def _composeText(tag: str, promptText: str) -> str:
        """
        Prefix the prompt text with the dataset tag when provided.

        Parameters
        ----------
        tag : str
            Dataset-level tag.
        promptText : str
            Raw prompt text.

        Returns
        -------
        str
            Text ready to be tokenized.
        """
        tagPrefix = f"[Tag: {tag}] " if tag else ""
        return f"{tagPrefix}{promptText}"

    def _sliceMotion(
        self,
        record: ClipDatasetRecord,
    ) -> MotionPayload:
        """
        Return the cropped motion slice and merged metadata.

        Parameters
        ----------
        record : ClipDatasetRecord
            Index entry describing the desired motion slice.

        Returns
        -------
        MotionPayload
            Cropped motion and merged metadata.
        """
        motion, meta = self._loadMotion(record.animationPath)
        motionSlice = sliceMotion(motion, record.startFrame, record.endFrame)
        mergedMeta = meta | record.metadata
        return motionSlice, mergedMeta

    def _tokenize(self, tag: str, promptText: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize the composed text prompt.

        Parameters
        ----------
        tag : str
            Dataset-level tag.
        promptText : str
            Raw prompt text.

        Returns
        -------
        Dict[str, torch.Tensor]
            Tokenized tensors squeezed on the batch dimension.
        """
        textContent = self._composeText(tag, promptText)
        encoded = self.tokenizer(
            textContent,
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
    maxTime = max(item["motion"].shape[0] for item in batch)
    motionBatch = torch.stack(
        [_padMotion(item["motion"], maxTime) for item in batch],
    )
    attentionMasks = torch.stack(
        [item["attention_mask"] for item in batch],
    )
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": attentionMasks,
        "motion": motionBatch,
        "time": torch.stack([item["time"] for item in batch]),
        "tag": [item["tag"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def _padMotion(motion: torch.Tensor, targetLength: int) -> torch.Tensor:
    """
    Pad a motion tensor to a target temporal length.

    Parameters
    ----------
    motion : torch.Tensor
        Motion tensor shaped (frames, bones, channels).
    targetLength : int
        Desired temporal length after padding.

    Returns
    -------
    torch.Tensor
        Motion tensor padded with zeros at the end.
    """
    if motion.shape[0] == targetLength:
        return motion
    paddingShape = (
        targetLength - motion.shape[0],
        motion.shape[1],
        motion.shape[2],
    )
    padding = torch.zeros(
        paddingShape,
        dtype=motion.dtype,
        device=motion.device,
    )
    return torch.cat([motion, padding], dim=0)
