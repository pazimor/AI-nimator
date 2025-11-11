"""Dataset builder orchestrating AMASS → canonical JSON conversions."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from src.features.dataset_builder.progress import (
    ProgressReporter,
    createProgressReporter,
)
from src.shared.constants.skeletons import (
    SMPL22_BONE_ORDER,
    SMPL24_BONE_ORDER,
)
from src.shared.quaternion import QuaternionConverter
from src.shared.types import (
    DatasetBuildOptions,
    DatasetBuildReport,
    DatasetBuilderConfig,
)

LOGGER = logging.getLogger("datasetBuilder")


@dataclass(frozen=True)
class PromptRecord:
    """Single row from the HumanML3D index CSV."""

    animationRelativePath: Path
    startFrame: int
    endFrame: int
    promptFile: str


@dataclass(frozen=True)
class PromptSegment:
    """Prompt slice covering a contiguous frame range."""

    startFrame: int
    endFrame: int
    text: str
    sourceFile: str


@dataclass(frozen=True)
class PromptSample:
    """Bundled prompt records for a single animation."""

    relativePath: Path
    records: List[PromptRecord]


@dataclass(frozen=True)
class ConvertedPrompt:
    """Custom prompt exported from a converted dataset sample."""

    simple: str
    advanced: str
    tag: str
    promptIdentifier: str


@dataclass(frozen=True)
class AnimationSample:
    """Subset of fields loaded from an AMASS npz file."""

    relativePath: Path
    resolvedPath: Path
    axisAngles: np.ndarray
    fps: int
    extras: Dict[str, object]


class PromptRepository:
    """Resolve prompt files stored as text or NumPy arrays."""

    def __init__(self, roots: List[Path], preferredExtension: str) -> None:
        self.roots = roots
        self.preferredExtension = preferredExtension

    def loadText(self, promptFile: str) -> str:
        """Return the textual representation for the provided prompt file."""
        for candidate in self._candidatePaths(promptFile):
            if candidate.exists():
                return self._readPrompt(candidate)
        raise FileNotFoundError(f"Missing prompt asset: {promptFile}")

    def _candidatePaths(self, promptFile: str) -> Iterable[Path]:
        relative = Path(promptFile)
        if relative.is_absolute():
            yield from self._withAlternatives(relative)
            return
        for root in self.roots:
            base = root / relative
            yield from self._withAlternatives(base)

    def _readPrompt(self, path: Path) -> str:
        if path.suffix.lower() == ".txt":
            return self._readTextPrompt(path)
        if path.suffix.lower() == ".npy":
            return self._readNpyPrompt(path)
        return path.read_text(encoding="utf-8").strip()

    def _withAlternatives(self, base: Path) -> List[Path]:
        candidates: List[Path] = [base]
        if base.suffix.lower() == ".npy":
            candidates.append(base.with_suffix(".txt"))
        preferred = base.with_suffix(self.preferredExtension)
        if preferred not in candidates:
            candidates.append(preferred)
        return candidates

    def _readTextPrompt(self, path: Path) -> str:
        lines = []
        for rawLine in path.read_text(encoding="utf-8").splitlines():
            line = rawLine.strip()
            if not line:
                continue
            text = line.split("#", 1)[0].strip()
            if text:
                lines.append(text)
        return " ".join(lines)

    def _readNpyPrompt(self, path: Path) -> str:
        array = np.load(path, allow_pickle=True)
        if array.ndim == 0:
            return str(array.item())
        flattened = array.flatten().tolist()
        return " ".join(str(entry) for entry in flattened)


class CustomPromptRepository:
    """Load custom prompt generations from JSON/JSONL exports."""

    def __init__(self, sources: List[Path]) -> None:
        self.entries: Dict[str, Dict[str, str]] = {}
        for source in sources:
            if source.is_dir():
                for filePath in source.glob("**/*"):
                    if filePath.is_file():
                        self._ingest_file(filePath)
            elif source.is_file():
                self._ingest_file(source)

    def find(self, relativePath: Path) -> Optional[Dict[str, str]]:
        for key in self._candidate_keys(relativePath):
            normalized = self._normalize_key(key)
            if normalized in self.entries:
                return self.entries[normalized]
        return None

    def _candidate_keys(self, relativePath: Path) -> List[str]:
        candidates = [
            relativePath.as_posix(),
            relativePath.with_suffix("").as_posix(),
            relativePath.name,
            relativePath.with_suffix("").name,
        ]
        parts = relativePath.parts
        if parts:
            candidates.append(parts[-1])
        return candidates

    def _ingest_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self._ingest_record(record)
            elif suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    for record in payload:
                        self._ingest_record(record)
                elif isinstance(payload, dict):
                    self._ingest_record(payload)
        except Exception:
            LOGGER.warning("Unable to parse custom prompt file: %s", path)

    def _ingest_record(self, record: Dict[str, object]) -> None:
        customId = record.get("custom_id") or record.get("customId")
        if not customId:
            return
        textPayload = self._extract_text(record)
        if not textPayload:
            return
        try:
            parsed = json.loads(textPayload)
        except json.JSONDecodeError:
            return
        normalized = self._normalize_key(str(customId))
        prompt = {
            "simple": str(parsed.get("Simple") or parsed.get("simple") or ""),
            "advanced": str(
                parsed.get("advanced") or parsed.get("Advanced") or ""
            ),
            "tag": str(parsed.get("tag") or parsed.get("Tag") or ""),
        }
        self.entries[normalized] = prompt

    def _extract_text(self, record: Dict[str, object]) -> str:
        response = record.get("response")
        if isinstance(response, dict):
            body = response.get("body")
            if isinstance(body, dict):
                outputs = body.get("output")
                if isinstance(outputs, list):
                    for output in outputs:
                        if (
                            isinstance(output, dict)
                            and output.get("type") == "message"
                        ):
                            contents = output.get("content", [])
                            for content in contents:
                                if (
                                    isinstance(content, dict)
                                    and content.get("type") == "output_text"
                                ):
                                    text = content.get("text")
                                    if isinstance(text, str):
                                        return text
        textValue = record.get("text")
        if isinstance(textValue, str):
            return textValue
        return ""

    def _normalize_key(self, key: str) -> str:
        normalized = key.replace("\\", "/").lower()
        if "__" in normalized:
            normalized = normalized.split("__", 1)[0]
        if "." in normalized:
            normalized = normalized.split(".", 1)[0]
        return normalized.strip()


class ConvertedPromptRepository:
    """Index prompts stored inside converted dataset directories."""

    _META_PATTERN = re.compile(
        r'"meta"\s*:\s*({.*?})\s*,\s*"(?:bones|frames)"',
        re.DOTALL,
    )

    def __init__(self, root: Optional[Path]) -> None:
        self.root = root
        self.entries: Dict[str, ConvertedPrompt] = {}
        if root is None:
            return
        if not root.exists():
            LOGGER.warning("Converted dataset root missing: %s", root)
            return
        for animationPath in root.rglob("animation.json"):
            loaded = self._load_entry(animationPath)
            if not loaded:
                continue
            entry, keys = loaded
            for key in keys:
                normalized = self._normalize(key)
                if normalized and normalized not in self.entries:
                    self.entries[normalized] = entry

    def find(self, relativePath: Path) -> Optional[ConvertedPrompt]:
        if not self.entries:
            return None
        for candidate in self._relative_keys(relativePath):
            normalized = self._normalize(candidate)
            if normalized in self.entries:
                return self.entries[normalized]
        return None

    def _load_entry(
        self,
        animationPath: Path,
    ) -> Optional[tuple[ConvertedPrompt, List[str]]]:
        source = self._read_meta_source(animationPath)
        if not source:
            return None
        promptPath = animationPath.with_name("prompts.json")
        if not promptPath.exists():
            return None
        payload = self._read_prompt_payload(promptPath)
        simple = self._extract_field(payload, ["Simple", "simple"])
        advanced = self._extract_field(payload, ["advanced", "Advanced"])
        tag = self._extract_field(payload, ["tag", "Tag"])
        if not any([simple, advanced, tag]):
            return None
        entry = ConvertedPrompt(
            simple=simple,
            advanced=advanced,
            tag=tag,
            promptIdentifier=self._prompt_identifier(promptPath),
        )
        keys = self._source_keys(source)
        return entry, keys

    def _read_meta_source(self, animationPath: Path) -> Optional[str]:
        maxBytes = 262_144
        chunkSize = 8_192
        try:
            with animationPath.open("r", encoding="utf-8") as handle:
                data = ""
                while len(data) < maxBytes:
                    chunk = handle.read(chunkSize)
                    if not chunk:
                        break
                    data += chunk
                    match = self._META_PATTERN.search(data)
                    if match:
                        metaJson = match.group(1)
                        meta = json.loads(metaJson)
                        source = meta.get("source")
                        if isinstance(source, str):
                            return source.strip()
                        return None
        except Exception:
            LOGGER.debug("Unable to parse converted animation meta: %s", animationPath)
        return None

    def _read_prompt_payload(self, promptPath: Path) -> Dict[str, object]:
        try:
            return json.loads(promptPath.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.debug("Unable to parse converted prompt file: %s", promptPath)
            return {}

    def _extract_field(self, payload: Dict[str, object], keys: List[str]) -> str:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
        return ""

    def _prompt_identifier(self, promptPath: Path) -> str:
        if self.root is None:
            return promptPath.as_posix()
        try:
            relative = promptPath.relative_to(self.root)
            return relative.as_posix()
        except ValueError:
            return promptPath.as_posix()

    def _source_keys(self, source: str) -> List[str]:
        path = Path(source.strip())
        candidates = [path.as_posix()]
        if path.suffix:
            candidates.append(path.with_suffix("").as_posix())
        candidates.append(path.name)
        if path.suffix:
            candidates.append(path.with_suffix("").name)
        return list({candidate for candidate in candidates if candidate})

    def _relative_keys(self, relativePath: Path) -> List[str]:
        candidates = [relativePath.as_posix()]
        if relativePath.suffix:
            candidates.append(relativePath.with_suffix("").as_posix())
        if relativePath.name:
            candidates.append(relativePath.name)
        if relativePath.suffix:
            candidates.append(relativePath.with_suffix("").name)
        return list({candidate for candidate in candidates if candidate})

    def _normalize(self, key: str) -> str:
        return key.replace("\\", "/").lower().strip()


class PromptTimelineBuilder:
    """Expand sparse prompt spans into full-length timelines."""

    def __init__(self, repository: PromptRepository) -> None:
        self.repository = repository

    def build(
        self,
        records: List[PromptRecord],
        frameCount: int,
    ) -> List[PromptSegment]:
        """Return contiguous prompt coverage for the provided animation."""
        if not records:
            return [
                PromptSegment(
                    startFrame=0,
                    endFrame=frameCount,
                    text="",
                    sourceFile="",
                )
            ]
        sortedRecords = sorted(records, key=lambda record: record.startFrame)
        segments = [
            self._recordToSegment(record, frameCount)
            for record in sortedRecords
        ]
        segments[0] = PromptSegment(
            startFrame=0,
            endFrame=segments[0].endFrame,
            text=segments[0].text,
            sourceFile=segments[0].sourceFile,
        )
        for index in range(len(segments) - 1):
            current = segments[index]
            following = segments[index + 1]
            adjustedEnd = max(current.endFrame, following.startFrame)
            segments[index] = PromptSegment(
                startFrame=current.startFrame,
                endFrame=min(adjustedEnd, frameCount),
                text=current.text,
                sourceFile=current.sourceFile,
            )
        last = segments[-1]
        segments[-1] = PromptSegment(
            startFrame=last.startFrame,
            endFrame=frameCount,
            text=last.text,
            sourceFile=last.sourceFile,
        )
        return segments

    def _recordToSegment(
        self,
        record: PromptRecord,
        frameCount: int,
    ) -> PromptSegment:
        startFrame = max(0, min(record.startFrame, frameCount))
        provisionalEnd = (
            record.endFrame if record.endFrame > startFrame else startFrame
        )
        endFrame = max(startFrame, min(provisionalEnd, frameCount))
        text = self.repository.loadText(record.promptFile)
        return PromptSegment(
            startFrame=startFrame,
            endFrame=endFrame,
            text=text,
            sourceFile=record.promptFile,
        )


class AnimationRebuilder:
    """Convert AMASS pose parameters into the canonical JSON schema."""

    def __init__(self, config: DatasetBuilderConfig) -> None:
        self.config = config
        self.sourceBoneOrder = SMPL24_BONE_ORDER
        self.targetBoneOrder = SMPL22_BONE_ORDER
        self.boneIndex = {
            boneName: index
            for index, boneName in enumerate(self.sourceBoneOrder)
        }
        self.animationRoots = [
            root.resolve() for root in config.paths.animationRoots
        ]

    def loadSample(self, relativePath: Path) -> AnimationSample:
        """Load the NPZ file referenced by `relativePath`."""
        resolved = self._resolveAnimationPath(relativePath)
        with np.load(resolved) as raw:
            axisAngles = raw["poses"].astype(np.float32)
            extras = {
                key: self._serialize(raw[key])
                for key in raw.files
                if key != "poses"
            }
        fps = self._resolveFps(extras)
        return AnimationSample(
            relativePath=relativePath,
            resolvedPath=resolved,
            axisAngles=axisAngles,
            fps=fps,
            extras=extras,
        )

    def buildPayload(self, sample: AnimationSample) -> Dict[str, object]:
        """Return the canonical payload with metadata, bones, and extras."""
        bones = self._buildBones(sample.axisAngles)
        meta = {
            "fps": sample.fps,
            "frames": int(sample.axisAngles.shape[0]),
            "joints": "SMPL-22",
            "source": sample.relativePath.as_posix(),
        }
        return {"meta": meta, "bones": bones, "extras": sample.extras}

    def _resolveAnimationPath(self, relativePath: Path) -> Path:
        for candidate in self._animationCandidates(relativePath):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Animation file not found: {relativePath} in {self._animationCandidates(relativePath)}")

    def _animationCandidates(self, relativePath: Path) -> List[Path]:
        bases: List[Path] = []
        parts = list(relativePath.parts)
        for root in self.animationRoots:
            bases.append(root / relativePath)
            if parts:
                bases.append(root / parts[0] / relativePath)
            if len(parts) >= 2:
                bases.append(root / parts[0] / parts[1] / relativePath)
        suffixPreferences = [
            "",
            self.config.processing.animationExtension,
            ".npz",
            ".npy",
        ]
        candidates: List[Path] = []
        for base in bases:
            suffixes: List[str] = []
            if base.suffix:
                suffixes.append(base.suffix)
            for suffix in suffixPreferences:
                if suffix and suffix not in suffixes:
                    suffixes.append(suffix)
            candidates.append(base)
            for suffix in suffixes:
                candidates.append(base.with_suffix(suffix))
        unique: List[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = str(candidate.resolve(strict=False))
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(candidate)
        return unique

    def _resolveFps(self, extras: Dict[str, object]) -> int:
        value = extras.get("mocap_framerate")
        if isinstance(value, (float, int)):
            return int(round(value))
        if isinstance(value, list) and value:
            return int(round(float(value[0])))
        return self.config.processing.fallbackFps

    def _buildBones(self, axisAngles: np.ndarray) -> List[Dict[str, object]]:
        frames = axisAngles.shape[0]
        reshaped = axisAngles.reshape(frames, -1, 3)
        bones: List[Dict[str, object]] = []
        for boneName in self.targetBoneOrder:
            sourceIndex = self.boneIndex.get(boneName)
            if sourceIndex is None or sourceIndex >= reshaped.shape[1]:
                bones.append(self._emptyBone(boneName, frames))
                continue
            angles = reshaped[:, sourceIndex, :]
            rotations = self._rotation6dFromAxisAngles(angles)
            bones.append(self._boneWithFrames(boneName, rotations))
        return bones

    def _rotation6dFromAxisAngles(
        self,
        axisAngles: np.ndarray,
    ) -> List[List[float]]:
        tensor = torch.from_numpy(axisAngles.astype(np.float32))
        rotation6d = QuaternionConverter.rotation6dFromAxisAngle(tensor)
        return [
            [float(value) for value in frame.tolist()]
            for frame in rotation6d
        ]

    def _boneWithFrames(
        self,
        boneName: str,
        rotations: List[List[float]],
    ) -> Dict[str, object]:
        frames = [
            {"frameIndex": index, "rotation": rotation, "extras": {}}
            for index, rotation in enumerate(rotations)
        ]
        return {"name": boneName, "length": 0.0, "frames": frames}

    def _emptyBone(self, boneName: str, frameCount: int) -> Dict[str, object]:
        frameEntries = []
        for index in range(frameCount):
            frameEntries.append(
                {
                    "frameIndex": index,
                    "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    "extras": {},
                }
            )
        return {"name": boneName, "length": 0.0, "frames": frameEntries}

    def _serialize(self, value: object) -> object:
        return self._ensureJsonSerializable(value)

    def _ensureJsonSerializable(self, value: object) -> object:
        if isinstance(value, np.ndarray):
            return self._ensureJsonSerializable(value.tolist())
        if isinstance(value, np.generic):
            return self._ensureJsonSerializable(value.item())
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("latin1")
        if isinstance(value, dict):
            return {
                str(key): self._ensureJsonSerializable(entry)
                for key, entry in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._ensureJsonSerializable(entry) for entry in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


class DatasetBuilder:
    """Main entry-point orchestrating dataset conversions."""

    def __init__(
        self,
        config: DatasetBuilderConfig,
        options: DatasetBuildOptions | None = None,
    ) -> None:
        self.config = config
        self.options = options or DatasetBuildOptions()
        self.promptRepository = PromptRepository(
            roots=config.paths.promptSources,
            preferredExtension=config.processing.promptTextExtension,
        )
        self.timelineBuilder = PromptTimelineBuilder(repository=self.promptRepository)
        self.customPrompts = CustomPromptRepository(
            sources=config.paths.customPromptFiles,
        )
        self.convertedPrompts = ConvertedPromptRepository(
            root=config.paths.convertedRoot,
        )
        self.animationRebuilder = AnimationRebuilder(config=config)
        self.animationRoots = [
            root.resolve() for root in config.paths.animationRoots
        ]
        logging.basicConfig(level=logging.INFO)
        if self.options.debugMode:
            LOGGER.setLevel(logging.DEBUG)

    def buildDataset(self) -> DatasetBuildReport:
        """Rebuild the dataset according to the current configuration."""
        samples = self._groupPromptRecords()
        reporter = self._createProgress(len(samples))
        processed, failures = self._processSamples(samples, reporter)
        reporter.close()
        return DatasetBuildReport(
            processedSamples=processed,
            failedSamples=failures,
            outputDirectory=self.config.paths.outputRoot,
        )

    def _groupPromptRecords(self) -> List[PromptSample]:
        grouped: Dict[Path, List[PromptRecord]] = {}
        with self.config.paths.indexCsv.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                record = self._rowToPromptRecord(row)
                if record is None:
                    continue
                grouped.setdefault(
                    record.animationRelativePath,
                    [],
                ).append(record)
        ordered = [
            PromptSample(relativePath=path, records=records)
            for path, records in sorted(
                grouped.items(),
                key=lambda item: item[0].as_posix(),
            )
        ]
        return self._limitSamples(ordered)

    def _limitSamples(self, samples: List[PromptSample]) -> List[PromptSample]:
        if self.options.maxSamples is None:
            return samples
        return samples[: self.options.maxSamples]

    def _rowToPromptRecord(self, row: Sequence[str]) -> PromptRecord | None:
        if not row or row[0].startswith("#"):
            return None
        try:
            animationPath = self._normalizeAnimationPath(row[0])
            startFrame = int(float(row[1])) if len(row) > 1 and row[1] else 0
            endFrame = (
                int(float(row[2])) if len(row) > 2 and row[2] else startFrame
            )
            promptFile = row[3].strip() if len(row) > 3 else ""
        except ValueError:
            return None
        return PromptRecord(
            animationRelativePath=animationPath,
            startFrame=startFrame,
            endFrame=endFrame,
            promptFile=promptFile,
        )

    def _processSamples(
        self,
        samples: List[PromptSample],
        reporter: ProgressReporter,
    ) -> tuple[int, List[str]]:
        processed = 0
        failures: List[str] = []
        for sample in samples:
            try:
                
                self._processSingleSample(sample)
                processed += 1
                reporter.advance(sample.relativePath.as_posix())
            except Exception as error:  # noqa: PERF203
                failures.append(sample.relativePath.as_posix())
                LOGGER.error(
                    "Unable to rebuild %s: %s",
                    sample.relativePath,
                    error,
                )
                if self.options.debugMode:
                    reporter.close()
                    raise
        return processed, failures

    def _processSingleSample(
        self,
        sample: PromptSample,
    ) -> None:
        animationSample = self.animationRebuilder.loadSample(
            sample.relativePath,
        )
        payload = self.animationRebuilder.buildPayload(animationSample)
        frameCount = int(animationSample.axisAngles.shape[0])
        promptSegments = self.timelineBuilder.build(
            records=sample.records,
            frameCount=frameCount,
        )
        convertedPrompt = self.convertedPrompts.find(sample.relativePath)
        if convertedPrompt:
            promptSegments = (
                self._convertedSegments(convertedPrompt, frameCount)
                + promptSegments
            )
        targetDir = self._resolveSampleOutputDir(sample.relativePath)
        self._writeJson(targetDir / "animation.json", payload)
        segmentsPayload = [asdict(segment) for segment in promptSegments]
        promptPayload: Dict[str, object] = {"segments": segmentsPayload}
        if convertedPrompt and convertedPrompt.tag:
            promptPayload["tag"] = convertedPrompt.tag
        customPrompt = self.customPrompts.find(sample.relativePath)
        if customPrompt:
            promptPayload["customPrompts"] = customPrompt
        self._writeJson(targetDir / "prompt.json", promptPayload)

    def _convertedSegments(
        self,
        convertedPrompt: ConvertedPrompt,
        frameCount: int,
    ) -> List[PromptSegment]:
        segments: List[PromptSegment] = []
        if convertedPrompt.simple:
            segments.append(
                PromptSegment(
                    startFrame=0,
                    endFrame=frameCount,
                    text=convertedPrompt.simple,
                    sourceFile=f"{convertedPrompt.promptIdentifier}#Simple",
                )
            )
        if convertedPrompt.advanced:
            segments.append(
                PromptSegment(
                    startFrame=0,
                    endFrame=frameCount,
                    text=convertedPrompt.advanced,
                    sourceFile=f"{convertedPrompt.promptIdentifier}#Advanced",
                )
            )
        return segments

    def _resolveSampleOutputDir(self, relativePath: Path) -> Path:
        directoryName = relativePath.with_suffix("")
        targetDir = self.config.paths.outputRoot / directoryName
        targetDir.mkdir(parents=True, exist_ok=True)
        return targetDir

    def _writeJson(self, path: Path, payload: Dict[str, object]) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        path.write_text(serialized, encoding="utf-8")

    def _createProgress(self, total: int) -> ProgressReporter:
        return createProgressReporter(
            style=self.options.progressStyle,
            total=total,
            description="Dataset rebuild",
        )

    def _normalizeAnimationPath(self, rawPath: str) -> Path:
        path = Path(rawPath.strip())
        if not path.parts:
            return path
        if path.is_absolute():
            for root in self.animationRoots:
                try:
                    return path.relative_to(root)
                except ValueError:
                    continue
            return path
        cleaned = self._strip_relative_prefix(path)
        for root in self.animationRoots:
            if cleaned.parts and cleaned.parts[0] == root.name:
                cleaned = Path(*cleaned.parts[1:])
        cleaned = self._dropPoseDataPrefix(cleaned)
        return cleaned

    @staticmethod
    def _strip_relative_prefix(path: Path) -> Path:
        parts = [part for part in path.parts if part not in {".", ""}]
        return Path(*parts) if parts else Path(path.name)

    @staticmethod
    def _dropPoseDataPrefix(path: Path) -> Path:
        parts = list(path.parts)
        if parts and parts[0].lower() == "pose_data":
            parts = parts[1:]
        return Path(*parts) if parts else Path()
