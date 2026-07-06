"""Reference implementation of the B7 in-engine CLIP BPE tokenizer.

Normative counterpart of ``text_encoding.md`` §2: the engine plugins
(Unity C# / Unreal C++) mirror this algorithm value-for-value.  Parity
is locked by ``text_encoding_parity.json`` (canonical prompts → token
ids), shared by the pytest suite and both engine test suites.

The algorithm reproduces the HuggingFace ``CLIPTokenizer`` for the
``openai/clip-vit-base-patch32`` vocabulary, with ONE documented
simplification (§2.1 of the spec): the "letter" character class is
``[a-z]`` plus **any non-ASCII codepoint**, instead of Unicode
``\\p{L}``.  For the FR/EN motion-prompt domain the outputs are
identical; non-ASCII punctuation/emoji may split differently (they
still tokenize — byte-level BPE never fails).

Inputs: ``vocab.json`` + ``merges.txt`` shipped in the controller
bundle under ``tokenizer/`` (verbatim HuggingFace files).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|endoftext|>"
END_OF_WORD_SUFFIX = "</w>"

# Contraction suffixes split off before word tokenization (CLIP regex).
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


@dataclass(frozen=True)
class EncodedPrompt:
    """Fixed-length encoding of one prompt.

    Attributes
    ----------
    inputIds : list[int]
        ``maxLength`` token ids: ``bos + tokens + eos`` padded with the
        pad id (== eos id for CLIP).
    attentionMask : list[float]
        ``maxLength`` floats — 1.0 on real tokens (bos/tokens/eos),
        0.0 on padding.
    """

    inputIds: list[int]
    attentionMask: list[float]


def _bytesToUnicode() -> dict[int, str]:
    """GPT-2/CLIP byte → printable-unicode mapping (bijective).

    Printable Latin-1 bytes map to themselves; the remaining 68 bytes
    map to U+0100.. so every byte has a visible, dict-safe character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapped = printable[:]
    offset = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + offset)
            offset += 1
    return {b: chr(c) for b, c in zip(printable, mapped)}


def _isAsciiSpace(char: str) -> bool:
    """ASCII whitespace test (spec §2.1 — the only separator class)."""
    return char in " \t\n\r\x0b\x0c"


def _isLetter(char: str) -> bool:
    """Spec letter class: ``[a-z]`` (post-lowercase) or non-ASCII."""
    return ("a" <= char <= "z") or ord(char) > 0x7F


def _isDigit(char: str) -> bool:
    """Spec digit class: ASCII ``[0-9]`` (one token per digit)."""
    return "0" <= char <= "9"


def _cleanText(text: str) -> str:
    """Lowercase (ASCII) and collapse whitespace runs to single spaces."""
    lowered = "".join(
        chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text
    )
    words = [w for w in _splitOnSpace(lowered) if w]
    return " ".join(words)


def _splitOnSpace(text: str) -> list[str]:
    """Split on ASCII whitespace characters."""
    words: list[str] = []
    current = ""
    for char in text:
        if _isAsciiSpace(char):
            words.append(current)
            current = ""
        else:
            current += char
    words.append(current)
    return words


def _splitWords(text: str) -> list[str]:
    """Split cleaned text into CLIP pre-tokenization words (spec §2.2).

    Order per position: contraction suffix > letter run > single digit
    > punctuation run.  Spaces separate, never emitted.
    """
    words: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if _isAsciiSpace(char):
            index += 1
            continue
        contraction = _matchContraction(text, index)
        if contraction is not None:
            words.append(contraction)
            index += len(contraction)
        elif _isLetter(char):
            index = _consumeRun(text, index, words, _isLetter)
        elif _isDigit(char):
            words.append(char)
            index += 1
        else:
            index = _consumeRun(text, index, words, _isOther)
    return words


def _isOther(char: str) -> bool:
    """Punctuation class: not space, not letter, not digit."""
    return not (
        _isAsciiSpace(char) or _isLetter(char) or _isDigit(char)
    )


def _matchContraction(text: str, index: int) -> str | None:
    """Return the contraction starting at ``index``, if any."""
    for contraction in _CONTRACTIONS:
        if text.startswith(contraction, index):
            return contraction
    return None


def _consumeRun(
    text: str,
    start: int,
    words: list[str],
    predicate,
) -> int:
    """Append the maximal run satisfying ``predicate``; return new index.

    A contraction suffix terminates a punctuation run (mirrors the CLIP
    regex alternation order where contractions match first).
    """
    end = start
    while end < len(text) and predicate(text[end]):
        if end > start and _matchContraction(text, end) is not None:
            break
        end += 1
    words.append(text[start:end])
    return end


class ClipBpeReference:
    """CLIP byte-level BPE over bundle ``vocab.json`` / ``merges.txt``."""

    def __init__(self, vocabPath: Path, mergesPath: Path) -> None:
        vocab: dict[str, int] = json.loads(
            vocabPath.read_text(encoding="utf-8")
        )
        self._vocab = vocab
        self._byteEncoder = _bytesToUnicode()
        self._mergeRanks = self._loadMerges(mergesPath)
        self._bosId = vocab[BOS_TOKEN]
        self._eosId = vocab[EOS_TOKEN]

    @property
    def bosId(self) -> int:
        return self._bosId

    @property
    def eosId(self) -> int:
        return self._eosId

    @staticmethod
    def _loadMerges(mergesPath: Path) -> dict[tuple[str, str], int]:
        """Parse merges.txt: one ``left right`` pair per line, ranked."""
        ranks: dict[tuple[str, str], int] = {}
        lines = mergesPath.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            left, right = line.split(" ")
            ranks[(left, right)] = len(ranks)
        return ranks

    def _bpe(self, word: str) -> list[str]:
        """Apply BPE merges to one pre-tokenization word (spec §2.4)."""
        mapped = "".join(
            self._byteEncoder[b] for b in word.encode("utf-8")
        )
        if not mapped:
            return []
        parts = list(mapped[:-1]) + [mapped[-1] + END_OF_WORD_SUFFIX]
        while len(parts) > 1:
            best = min(
                zip(parts[:-1], parts[1:]),
                key=lambda p: self._mergeRanks.get(p, 1 << 30),
            )
            if best not in self._mergeRanks:
                break
            parts = _mergePair(parts, best)
        return parts

    def encode(self, text: str, maxLength: int) -> EncodedPrompt:
        """Encode one prompt to fixed-length ids + attention mask.

        Mirrors HF ``padding="max_length", truncation=True``: the BPE
        token stream is truncated to ``maxLength - 2``, then wrapped in
        ``bos``/``eos`` and padded with the eos id.
        """
        tokens: list[int] = []
        for word in _splitWords(_cleanText(text)):
            for piece in self._bpe(word):
                tokens.append(self._vocab[piece])
        tokens = tokens[: maxLength - 2]
        inputIds = [self._bosId] + tokens + [self._eosId]
        realCount = len(inputIds)
        inputIds += [self._eosId] * (maxLength - realCount)
        attentionMask = [1.0] * realCount + [0.0] * (maxLength - realCount)
        return EncodedPrompt(
            inputIds=inputIds, attentionMask=attentionMask
        )


def _mergePair(
    parts: list[str], pair: tuple[str, str]
) -> list[str]:
    """Merge every occurrence of ``pair`` in ``parts`` (left to right)."""
    merged: list[str] = []
    index = 0
    while index < len(parts):
        if (
            index + 1 < len(parts)
            and (parts[index], parts[index + 1]) == pair
        ):
            merged.append(parts[index] + parts[index + 1])
            index += 2
        else:
            merged.append(parts[index])
            index += 1
    return merged
