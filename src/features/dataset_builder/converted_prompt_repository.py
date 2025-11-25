
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.shared.types import ConvertedPrompt

LOGGER = logging.getLogger("converted Prompt Repository")

class ConvertedPromptRepository:
    """Index prompts stored inside converted dataset directories."""

    _META_PATTERN = re.compile(
        r'"meta"\s*:\s*({.*?})\s*,\s*"(?:bones|frames)"',
        re.DOTALL,
    )

    def __init__(self, root: Optional[Path]) -> None:
        self.root = root
        self.entries: Dict[str, ConvertedPrompt] = {}
        if root is None or not root.exists():
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
        LOGGER.info("totale entry fetched: %s", len(self.entries))

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