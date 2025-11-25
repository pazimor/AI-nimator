
import logging
import numpy
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger("HumanML3D-Prompts")

class PromptRepository:
    """Resolve prompt files stored as text or NumPy arrays."""

    """read prompts from HumanML3D repository - i suppose ... """
    ## TODO: evaluate if Numpy Read is useless or not (should be yes)

    def __init__(self, root: Path, preferredExtension: str) -> None:
        self.root = root
        self.preferredExtension = preferredExtension
        LOGGER.info("%s %s", root, preferredExtension)

    def loadText(self, promptFile: str) -> str:
        """Return the textual representation for the provided prompt file."""

        for candidate in self._candidatePaths(promptFile):
            if candidate.exists():
                return self._readTextPrompt(candidate)
        raise FileNotFoundError(f"Missing prompt asset: {promptFile}")

    def _candidatePaths(self, promptFile: str) -> Iterable[Path]:
        relative = Path(promptFile)
        base = self.root / relative
        yield from self._withAlternatives(base)

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
        array = numpy.load(path, allow_pickle=True)
        if array.ndim == 0:
            return str(array.item())
        flattened = array.flatten().tolist()
        return " ".join(str(entry) for entry in flattened)

