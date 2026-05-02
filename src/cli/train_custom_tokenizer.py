"""CLI entry-point for training a custom BPE tokenizer on the prompt corpus.

This script walks one or more dataset roots (e.g. AMASS-Babel, KIT-ML),
collects every motion prompt found in ``prompt.json`` files, and trains
a fresh BPE tokenizer that will replace XLM-RoBERTa in the AI-nimator v2
text encoder.

Usage example
-------------
.. code-block:: bash

    poetry run python -m src.cli.train_custom_tokenizer \\
        --dataset-root /Users/pazimor/dataset \\
        --output-dir output/text/custom_tokenizer \\
        --vocab-size 8000 \\
        --max-length 64

The output directory will contain ``tokenizer.json`` and ``config.json``
ready to be loaded with :meth:`CustomTokenizer.load`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator, Sequence

from src.shared.model.text.custom_tokenizer import (
    CustomTokenizer,
    CustomTokenizerConfig,
    iterPromptCorpus,
    iterPromptCorpusFromRoots,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("output/text/custom_tokenizer")
DEFAULT_PROMPT_FILENAME = "prompt.json"


def buildArgumentParser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a custom BPE tokenizer on the motion prompt corpus."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        dest="datasetRoots",
        type=Path,
        action="append",
        required=True,
        help=(
            "Path to a dataset root containing prompt.json files. "
            "Repeat the flag to combine multiple roots "
            "(e.g. AMASS + KIT-ML)."
        ),
    )
    parser.add_argument(
        "--include-folders",
        dest="includeFolders",
        type=str,
        default=None,
        help=(
            "Optional comma-separated whitelist of top-level folders to "
            "include (e.g. ACCAD,BMLmovi,CMU,KIT). Applied to each "
            "dataset root.  Default: include everything."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="outputDir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where tokenizer.json and config.json will be "
            "written. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--vocab-size",
        dest="vocabSize",
        type=int,
        default=8000,
        help="Target BPE vocabulary size. (default: %(default)s)",
    )
    parser.add_argument(
        "--min-frequency",
        dest="minFrequency",
        type=int,
        default=2,
        help="Minimum BPE merge frequency. (default: %(default)s)",
    )
    parser.add_argument(
        "--max-length",
        dest="maxLength",
        type=int,
        default=64,
        help=(
            "Default maximum sequence length used at encode() time. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--prompt-filename",
        dest="promptFilename",
        type=str,
        default=DEFAULT_PROMPT_FILENAME,
        help=(
            "Filename to look for inside each subdirectory. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. (default: %(default)s)",
    )
    return parser


def _parseIncludeFolders(raw: str | None) -> tuple[str, ...] | None:
    """Parse the ``--include-folders`` CSV into a tuple of folder names."""
    if raw is None:
        return None
    folders = tuple(
        folder.strip() for folder in raw.split(",") if folder.strip()
    )
    return folders or None


def _filteredCorpusFromRoots(
    datasetRoots: Sequence[Path],
    includeFolders: tuple[str, ...] | None,
    promptFilename: str,
) -> Iterator[str]:
    """Yield prompts from ``datasetRoots`` honouring ``includeFolders``."""
    if includeFolders is None:
        yield from iterPromptCorpusFromRoots(
            datasetRoots, promptFilename=promptFilename
        )
        return

    allowed = set(includeFolders)
    for root in datasetRoots:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}.")
        for childName in allowed:
            child = root / childName
            if not child.exists():
                LOGGER.warning(
                    "Skipping missing folder %s under root %s",
                    childName,
                    root,
                )
                continue
            yield from iterPromptCorpus(
                child, promptFilename=promptFilename
            )


def _runTraining(arguments: argparse.Namespace) -> Path:
    """Execute the training pipeline and return the output directory."""
    includeFolders = _parseIncludeFolders(arguments.includeFolders)
    config = CustomTokenizerConfig(
        vocabSize=arguments.vocabSize,
        minFrequency=arguments.minFrequency,
        maxLength=arguments.maxLength,
    )

    LOGGER.info(
        "Training BPE tokenizer (vocab=%d, min_freq=%d, max_len=%d) "
        "on %d root(s); folders=%s.",
        config.vocabSize,
        config.minFrequency,
        config.maxLength,
        len(arguments.datasetRoots),
        includeFolders or "ALL",
    )

    # Stream the corpus twice: once to count (for visibility) and once
    # for training.  Training is the slow path so the count is cheap by
    # comparison and helps catch empty-corpus configuration mistakes
    # early.
    counter = 0
    for _ in _filteredCorpusFromRoots(
        arguments.datasetRoots,
        includeFolders=includeFolders,
        promptFilename=arguments.promptFilename,
    ):
        counter += 1
    if counter == 0:
        raise RuntimeError(
            "No prompts found.  Check --dataset-root and "
            "--include-folders values."
        )
    LOGGER.info("Collected %d prompt segments for training.", counter)

    corpus = _filteredCorpusFromRoots(
        arguments.datasetRoots,
        includeFolders=includeFolders,
        promptFilename=arguments.promptFilename,
    )
    tokenizer = CustomTokenizer.train(corpus, config=config)
    LOGGER.info(
        "Training complete; effective vocab size = %d.",
        tokenizer.vocabSize,
    )

    outputDir = tokenizer.save(arguments.outputDir)
    return outputDir


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry-point.  Returns a process exit code."""
    parser = buildArgumentParser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.logLevel),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        outputDir = _runTraining(arguments)
    except (FileNotFoundError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1
    LOGGER.info("Tokenizer saved under %s.", outputDir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
