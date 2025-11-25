import logging
from typing import List

from src.features.dataset_builder.prompt_repository import PromptRepository
from src.shared.types import PromptSegment, PromptRecord

LOGGER = logging.getLogger("timeline builder")

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
        provisionalEnd = (record.endFrame if record.endFrame > startFrame else startFrame) # should be useless
        endFrame = max(startFrame, min(provisionalEnd, frameCount))
        text = self.repository.loadText(record.promptFile)
        return PromptSegment(
            startFrame=startFrame,
            endFrame=endFrame,
            text=text,
            sourceFile=record.promptFile,
        )

