"""Shared dataclass definitions for configurations and data containers."""

from .config import DeviceSelectionOptions, SamplingConfiguration, TrainingConfiguration
from .data import AnimationPromptSample, ClipRecord, DatasetCache
from .uniformizer import UniformizerJob, UniformizerDirectoryJob
from .rag import RagBatchJob, RagFetchJob, RagLocalFetchJob, RagTestJob
from src.shared.quaternion import Quaternion

__all__ = [
    "DeviceSelectionOptions",
    "SamplingConfiguration",
    "TrainingConfiguration",
    "AnimationPromptSample",
    "ClipRecord",
    "DatasetCache",
    "Quaternion",
    "UniformizerJob",
    "UniformizerDirectoryJob",
    "RagTestJob",
    "RagBatchJob",
    "RagFetchJob",
    "RagLocalFetchJob",
]
