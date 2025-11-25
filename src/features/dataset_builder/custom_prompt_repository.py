
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional


LOGGER = logging.getLogger("Custom Prompts from OpenAI api") #to use in case of emergency (takes time)

class CustomPromptRepository:
    """Load custom prompt generations from JSON/JSONL exports."""

    def __init__(self, source: Path) -> None:
        self.entries: Dict[str, Dict[str, str]] = {}
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

