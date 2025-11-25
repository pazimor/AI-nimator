"""Dataset builder orchestrating AMASS → canonical JSON conversions."""

from __future__ import annotations

import csv
import json
import logging

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

from src.features.dataset_builder.animation_rebuilder import AnimationRebuilder
from src.features.dataset_builder.converted_prompt_repository import ConvertedPromptRepository
from src.features.dataset_builder.custom_prompt_repository import CustomPromptRepository
from src.features.dataset_builder.progress import (
    ProgressReporter,
    TqdmProgressReporter
)
from src.features.dataset_builder.prompt_repository import PromptRepository
from src.features.dataset_builder.prompt_timeline_builder import PromptTimelineBuilder
from src.shared.types import (
    DatasetBuildOptions,
    DatasetBuildReport,
    DatasetBuilderConfig, PromptSample, PromptRecord, ConvertedPrompt, PromptSegment,
)

LOGGER = logging.getLogger("DatasetBuilder")


class DatasetBuilder:
    """Main entry-point orchestrating dataset conversions."""

    def __init__(
        self,
        config: DatasetBuilderConfig,
        options: DatasetBuildOptions | None = None,
    ) -> None:
        self.config = config
        logging.basicConfig(level=logging.INFO)
        self.options = options or DatasetBuildOptions()
        
        self.promptRepository = PromptRepository(
            root=config.paths.promptSources,
            preferredExtension=config.processing.promptTextExtension,
        )
        
        # self.customPrompts = CustomPromptRepository(source=config.paths.convertedRoot) # in case of data loss
        self.timelineBuilder = PromptTimelineBuilder(repository=self.promptRepository)
        self.convertedPrompts = ConvertedPromptRepository(root=config.paths.convertedRoot)
        self.animationRebuilder = AnimationRebuilder(config=config)
        self.animationRoots = config.paths.animationRoot.resolve() ##TODO: might simplify array

        if self.options.debugMode:
            LOGGER.setLevel(logging.DEBUG)

    def buildDataset(self) -> DatasetBuildReport:
        """Rebuild the dataset according to the current configuration."""
        # 1. Group samples from CSV
        samples = self._groupPromptRecords()
        
        # 2. Process them
        reporter = self._createProgress(len(samples))
        processed, failures = self._processSamples(samples, reporter)
        reporter.close()
        
        return DatasetBuildReport(
            processedSamples=processed,
            failedSamples=failures,
            outputDirectory=self.config.paths.outputRoot,
        )

    def _groupPromptRecords(self) -> List[PromptSample]:
        """Reads the index CSV and groups prompt records by animation file."""
        grouped: Dict[Path, List[PromptRecord]] = {}

        with self.config.paths.indexCsv.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                record = self._rowToPromptRecord(row)
                if record is None:
                    continue ## should be 1 here (residue from CSV file)
                grouped.setdefault(
                    record.animationRelativePath,
                    [],
                ).append(record)

        # Convert to list of PromptSample
        ordered = [
            PromptSample(relativePath=path, records=records)
            for path, records in sorted(
                grouped.items(),
                key=lambda item: item[0].as_posix(),
            )
        ]
        
        LOGGER.info(f"Total animation groups found: {len(ordered)}")
        return ordered

    def _rowToPromptRecord(self, row: Sequence[str]) -> PromptRecord | None:
        try:
            if len(row) >= 3:
                animationPath = self._normalizeAnimationPath(row[0])
                startFrame = int(row[1])
                endFrame = int(row[2])
                promptFile = row[3].strip()
            else:
                animationPath = None
                startFrame = 0
                endFrame = 0
                promptFile = ""
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
            except Exception as error:
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
        return TqdmProgressReporter(total=total, description="Dataset rebuild")

    def _normalizeAnimationPath(self, rawPath: str) -> Path:
        path = Path(rawPath.strip())
        parts = [p for p in path.parts if p not in {".", ""}]
        # Remove common prefix if present (legacy dataset artifact)
        if parts and parts[0].lower() == "pose_data":
            parts = parts[1:]

        return Path(*parts) if parts else Path()
