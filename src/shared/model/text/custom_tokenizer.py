"""Custom BPE tokenizer for the motion-prompt corpus.

This module wraps :mod:`tokenizers` (HuggingFace's Rust-backed library) to
produce a small BPE tokenizer specialised on the motion-description corpus
that drives AI-nimator (AMASS-Babel + KIT-ML, English-only, short
sentences).  Compared to XLM-RoBERTa-base (250k tokens, 768-d hidden,
multilingual, frozen) this tokenizer keeps a ~5–8k vocabulary closed on
the actual motion vocabulary, freeing capacity for the joint motion
encoder and removing distribution shift between training and inference.

Design choices
--------------
* **Algorithm** — Byte-Pair Encoding (BPE).  Better than WordPiece for
  short open-domain English; mature support in HuggingFace tokenizers.
* **Pre-tokenizer** — Whitespace + Punctuation.  No ByteLevel: motion
  prompts are pure ASCII English, so byte-level encoding only inflates
  the vocabulary with low-frequency byte sequences.
* **Special tokens** — ``<pad>``, ``<unk>``, ``<bos>``, ``<eos>``.  The
  ``<bos>`` / ``<eos>`` pair lets the downstream encoder add a sentence
  boundary signal cheaply (template ``<bos> A <eos>``).
* **Lower-casing** — Always lowercase.  Motion captions exhibit no
  case-bearing semantics ("WALKS" and "walks" are identical), and
  lowercasing roughly halves the vocabulary footprint at fixed coverage.

Public surface
--------------
* :class:`CustomTokenizerConfig` — frozen dataclass with all knobs.
* :class:`CustomTokenizer` — train / encode / decode / save / load.
* :class:`EncodedBatch` — typed return for :meth:`CustomTokenizer.encode`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import Lowercase, NFKC, Sequence as NormSequence
from tokenizers.processors import TemplateProcessing

LOGGER = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_BOS_TOKEN = "<bos>"
DEFAULT_EOS_TOKEN = "<eos>"


@dataclass(frozen=True)
class CustomTokenizerConfig:
    """Configuration for :class:`CustomTokenizer`.

    Parameters
    ----------
    vocabSize : int
        Target BPE vocabulary size **including** the four special tokens.
        ~5–8k is a good range for the motion corpus (~37k unique prompts,
        ~3k unique words).
    minFrequency : int
        Minimum number of occurrences for a merge to be considered.
        ``2`` keeps rare-but-real motion terms; ``1`` adds typo noise.
    maxLength : int
        Maximum number of tokens kept after truncation in
        :meth:`CustomTokenizer.encode`.  Matches the existing
        ``maxPromptLength`` (64 in the legacy XLM-R pipeline) so the
        downstream cross-attention sequence length is unchanged.
    padToken, unkToken, bosToken, eosToken : str
        Reserved special tokens.  IDs are deterministic in the order they
        are declared (pad=0, unk=1, bos=2, eos=3).
    """

    vocabSize: int = 8000
    minFrequency: int = 2
    maxLength: int = 64
    padToken: str = DEFAULT_PAD_TOKEN
    unkToken: str = DEFAULT_UNK_TOKEN
    bosToken: str = DEFAULT_BOS_TOKEN
    eosToken: str = DEFAULT_EOS_TOKEN

    @property
    def specialTokens(self) -> tuple[str, str, str, str]:
        """Return the special tokens in canonical order (pad, unk, bos, eos)."""
        return (self.padToken, self.unkToken, self.bosToken, self.eosToken)


@dataclass(frozen=True)
class EncodedBatch:
    """Result of :meth:`CustomTokenizer.encode` on a batch of texts.

    Attributes
    ----------
    inputIds : torch.Tensor
        Integer tensor of shape ``(B, T)`` with token IDs.  Padded with
        the pad-token id.
    attentionMask : torch.Tensor
        Float tensor of shape ``(B, T)`` with 1.0 on real tokens and 0.0
        on padding positions.  Float on purpose — most cross-attention
        implementations expect a float key-padding mask.
    lengths : torch.Tensor
        Long tensor of shape ``(B,)`` with the per-sample token count
        (excluding padding).  Useful for logging / debugging.
    """

    inputIds: torch.Tensor
    attentionMask: torch.Tensor
    lengths: torch.Tensor


class CustomTokenizer:
    """BPE tokenizer trained on the motion-prompt corpus.

    The class wraps a :class:`tokenizers.Tokenizer` instance and exposes a
    minimal, type-stable API tuned for the AI-nimator training and
    inference pipelines.  It is *not* a drop-in replacement for the
    HuggingFace ``PreTrainedTokenizerFast`` interface — only the methods
    actually used by the downstream encoder are implemented.

    Use :meth:`train` (class method) to fit a fresh tokenizer on a
    corpus, then :meth:`save` / :meth:`load` for persistence.
    """

    # --- Filename conventions for save() / load() round trips. --------
    TOKENIZER_FILENAME: str = "tokenizer.json"
    CONFIG_FILENAME: str = "config.json"

    def __init__(
        self,
        backend: Tokenizer,
        config: CustomTokenizerConfig,
    ) -> None:
        """Wrap an already-built tokenizer backend with our config."""
        self._backend = backend
        self._config = config
        self._padTokenId = backend.token_to_id(config.padToken)
        self._unkTokenId = backend.token_to_id(config.unkToken)
        self._bosTokenId = backend.token_to_id(config.bosToken)
        self._eosTokenId = backend.token_to_id(config.eosToken)
        if any(
            tokenId is None
            for tokenId in (
                self._padTokenId,
                self._unkTokenId,
                self._bosTokenId,
                self._eosTokenId,
            )
        ):
            raise ValueError(
                "Backend tokenizer is missing one or more special tokens."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> CustomTokenizerConfig:
        return self._config

    @property
    def vocabSize(self) -> int:
        """Effective vocabulary size of the trained tokenizer."""
        return self._backend.get_vocab_size()

    @property
    def padTokenId(self) -> int:
        return int(self._padTokenId)

    @property
    def unkTokenId(self) -> int:
        return int(self._unkTokenId)

    @property
    def bosTokenId(self) -> int:
        return int(self._bosTokenId)

    @property
    def eosTokenId(self) -> int:
        return int(self._eosTokenId)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        config: CustomTokenizerConfig | None = None,
    ) -> "CustomTokenizer":
        """Train a fresh BPE tokenizer on ``corpus``.

        Parameters
        ----------
        corpus : Iterable[str]
            Iterable of training sentences.  Consumed lazily — pass a
            generator if memory matters.  Lowercasing and normalisation
            are applied by the tokenizer pipeline; raw input is fine.
        config : CustomTokenizerConfig, optional
            Tokenizer configuration.  Defaults to :class:`CustomTokenizerConfig`
            with stock parameters (8000-token vocab, max-length 64).

        Returns
        -------
        CustomTokenizer
            Fully-initialised tokenizer ready for :meth:`encode`.
        """
        config = config or CustomTokenizerConfig()
        backend = cls._buildBackend(config)
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocabSize,
            min_frequency=config.minFrequency,
            special_tokens=list(config.specialTokens),
            show_progress=False,
        )
        backend.train_from_iterator(corpus, trainer=trainer)
        cls._installPostProcessor(backend, config)
        cls._installPaddingAndTruncation(backend, config)
        return cls(backend=backend, config=config)

    @staticmethod
    def _buildBackend(config: CustomTokenizerConfig) -> Tokenizer:
        """Build an empty BPE backend with the standard pre-pipeline."""
        backend = Tokenizer(models.BPE(unk_token=config.unkToken))
        backend.normalizer = NormSequence([NFKC(), Lowercase()])
        backend.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Punctuation(),
            ]
        )
        backend.decoder = decoders.BPEDecoder()
        return backend

    @staticmethod
    def _installPostProcessor(
        backend: Tokenizer,
        config: CustomTokenizerConfig,
    ) -> None:
        """Wrap each encoded sequence with ``<bos> ... <eos>``."""
        bosId = backend.token_to_id(config.bosToken)
        eosId = backend.token_to_id(config.eosToken)
        backend.post_processor = TemplateProcessing(
            single=f"{config.bosToken} $A {config.eosToken}",
            pair=(
                f"{config.bosToken} $A {config.eosToken} "
                f"$B:1 {config.eosToken}:1"
            ),
            special_tokens=[
                (config.bosToken, bosId),
                (config.eosToken, eosId),
            ],
        )

    @staticmethod
    def _installPaddingAndTruncation(
        backend: Tokenizer,
        config: CustomTokenizerConfig,
    ) -> None:
        """Configure deterministic right-padding and truncation."""
        padId = backend.token_to_id(config.padToken)
        backend.enable_padding(
            pad_id=padId,
            pad_token=config.padToken,
            length=config.maxLength,
        )
        backend.enable_truncation(max_length=config.maxLength)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode(
        self,
        texts: Sequence[str] | str,
    ) -> EncodedBatch:
        """Encode one or more strings into a padded batch.

        Parameters
        ----------
        texts : str or Sequence[str]
            Single string or batch.  A single string is wrapped into a
            batch of size 1 for shape consistency.

        Returns
        -------
        EncodedBatch
            ``inputIds`` and ``attentionMask`` are always 2-D tensors
            (batch dimension preserved even for a single input).
        """
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            raise ValueError("encode() received an empty batch.")
        encodings = self._backend.encode_batch(list(texts))
        inputIds = torch.tensor(
            [encoding.ids for encoding in encodings],
            dtype=torch.long,
        )
        attentionMask = torch.tensor(
            [encoding.attention_mask for encoding in encodings],
            dtype=torch.float32,
        )
        lengths = attentionMask.sum(dim=-1).to(torch.long)
        return EncodedBatch(
            inputIds=inputIds,
            attentionMask=attentionMask,
            lengths=lengths,
        )

    def decode(
        self,
        ids: Sequence[int] | torch.Tensor,
        skipSpecialTokens: bool = True,
    ) -> str:
        """Decode a single sequence of token IDs back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return self._backend.decode(
            list(ids),
            skip_special_tokens=skipSpecialTokens,
        )

    def decodeBatch(
        self,
        ids: torch.Tensor,
        skipSpecialTokens: bool = True,
    ) -> list[str]:
        """Decode a batched ID tensor of shape ``(B, T)`` back to texts."""
        if ids.ndim != 2:
            raise ValueError(
                f"decodeBatch expects a 2-D tensor (B, T); got shape "
                f"{tuple(ids.shape)}."
            )
        rows = ids.detach().cpu().tolist()
        return self._backend.decode_batch(
            rows,
            skip_special_tokens=skipSpecialTokens,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist the tokenizer + config under ``directory``.

        The directory is created if needed.  Two files are written:
        ``tokenizer.json`` (HuggingFace canonical format) and
        ``config.json`` (our :class:`CustomTokenizerConfig`).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tokenizerPath = directory / self.TOKENIZER_FILENAME
        configPath = directory / self.CONFIG_FILENAME
        self._backend.save(str(tokenizerPath))
        configPath.write_text(
            json.dumps(self._configToDict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        LOGGER.info(
            "Saved CustomTokenizer (vocab=%d) to %s",
            self.vocabSize,
            directory,
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "CustomTokenizer":
        """Load a tokenizer previously saved by :meth:`save`."""
        directory = Path(directory)
        tokenizerPath = directory / cls.TOKENIZER_FILENAME
        configPath = directory / cls.CONFIG_FILENAME
        if not tokenizerPath.exists():
            raise FileNotFoundError(
                f"Missing tokenizer file: {tokenizerPath}."
            )
        if not configPath.exists():
            raise FileNotFoundError(
                f"Missing config file: {configPath}."
            )
        backend = Tokenizer.from_file(str(tokenizerPath))
        config = cls._configFromDict(
            json.loads(configPath.read_text(encoding="utf-8"))
        )
        cls._installPaddingAndTruncation(backend, config)
        return cls(backend=backend, config=config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _configToDict(self) -> dict[str, object]:
        return {
            "vocabSize": self._config.vocabSize,
            "minFrequency": self._config.minFrequency,
            "maxLength": self._config.maxLength,
            "padToken": self._config.padToken,
            "unkToken": self._config.unkToken,
            "bosToken": self._config.bosToken,
            "eosToken": self._config.eosToken,
        }

    @staticmethod
    def _configFromDict(payload: dict[str, object]) -> CustomTokenizerConfig:
        return CustomTokenizerConfig(
            vocabSize=int(payload["vocabSize"]),
            minFrequency=int(payload["minFrequency"]),
            maxLength=int(payload["maxLength"]),
            padToken=str(payload["padToken"]),
            unkToken=str(payload["unkToken"]),
            bosToken=str(payload["bosToken"]),
            eosToken=str(payload["eosToken"]),
        )


# ---------------------------------------------------------------------
# Corpus helpers — used by the CLI training script.
# ---------------------------------------------------------------------
def iterPromptCorpus(
    datasetRoot: str | Path,
    promptFilename: str = "prompt.json",
) -> Iterator[str]:
    """Yield every textual prompt found under ``datasetRoot``.

    The function walks ``datasetRoot`` recursively, opens every file
    named ``promptFilename`` and yields the ``text`` field of each
    ``segments[]`` entry.  Empty texts are skipped.

    Parameters
    ----------
    datasetRoot : str or Path
        Root directory of the source dataset (e.g. AMASS root containing
        ACCAD/, BMLmovi/, KIT/, ...).
    promptFilename : str
        Filename to look for inside each subdirectory.  Defaults to
        ``prompt.json`` matching the existing AMASS-Babel layout.

    Yields
    ------
    str
        One prompt text per yield.  Order follows ``Path.rglob``.
    """
    root = Path(datasetRoot)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}.")
    for promptFile in root.rglob(promptFilename):
        try:
            payload = json.loads(promptFile.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            LOGGER.warning(
                "Skipping unreadable prompt file %s: %s",
                promptFile,
                error,
            )
            continue
        for segment in payload.get("segments", []) or []:
            text = (segment or {}).get("text")
            if isinstance(text, str):
                stripped = text.strip()
                if stripped:
                    yield stripped


def iterPromptCorpusFromRoots(
    datasetRoots: Iterable[str | Path],
    promptFilename: str = "prompt.json",
) -> Iterator[str]:
    """Chain :func:`iterPromptCorpus` over multiple roots."""
    for root in datasetRoots:
        yield from iterPromptCorpus(root, promptFilename=promptFilename)
