"""Unit tests for :mod:`src.shared.model.text.custom_tokenizer`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ainimator.text.custom_tokenizer import (
    CustomTokenizer,
    CustomTokenizerConfig,
    iterPromptCorpus,
    iterPromptCorpusFromRoots,
)


# A small but realistic motion corpus — repeated tokens ensure BPE has
# enough frequency signal to learn merges with min_frequency=2.
MOTION_CORPUS: list[str] = [
    "a person walks forward.",
    "a person walks backward.",
    "the person walks slowly.",
    "the person walks quickly.",
    "a man crawls on the floor.",
    "a man crawls forward slowly.",
    "the woman dances in a circle.",
    "the woman dances gracefully.",
    "a person jumps high.",
    "a person jumps over an obstacle.",
    "the person sits down on a chair.",
    "the person sits and stands up.",
    "a person waves their right hand.",
    "a person waves their left hand.",
    "someone runs forward.",
    "someone runs backward then stops.",
]


def _train(maxLength: int = 16, vocabSize: int = 256) -> CustomTokenizer:
    """Build a small tokenizer suitable for fast tests."""
    config = CustomTokenizerConfig(
        vocabSize=vocabSize,
        minFrequency=2,
        maxLength=maxLength,
    )
    return CustomTokenizer.train(MOTION_CORPUS, config=config)


# ---------------------------------------------------------------------
# Construction & training
# ---------------------------------------------------------------------
def test_train_produces_tokenizer_with_special_tokens() -> None:
    tokenizer = _train()

    assert tokenizer.padTokenId == 0
    assert tokenizer.unkTokenId == 1
    assert tokenizer.bosTokenId == 2
    assert tokenizer.eosTokenId == 3
    assert tokenizer.vocabSize <= 256
    assert tokenizer.vocabSize >= 4  # at minimum the four specials


def test_train_with_default_config_uses_8000_vocab_target() -> None:
    config = CustomTokenizerConfig()
    assert config.vocabSize == 8000
    assert config.maxLength == 64
    assert config.specialTokens == ("<pad>", "<unk>", "<bos>", "<eos>")


# ---------------------------------------------------------------------
# Encoding shape & padding
# ---------------------------------------------------------------------
def test_encode_single_string_returns_2d_batch() -> None:
    tokenizer = _train(maxLength=16)
    batch = tokenizer.encode("a person walks forward.")

    assert batch.inputIds.ndim == 2
    assert batch.inputIds.shape == (1, 16)
    assert batch.attentionMask.shape == (1, 16)
    assert batch.lengths.shape == (1,)
    assert batch.lengths.item() <= 16


def test_encode_batch_pads_to_max_length() -> None:
    tokenizer = _train(maxLength=16)
    batch = tokenizer.encode(
        [
            "a person walks forward.",
            "the woman dances in a circle gracefully.",
        ]
    )

    assert batch.inputIds.shape == (2, 16)
    assert batch.attentionMask.shape == (2, 16)
    # Padding rows must end with pad-token IDs.
    paddedRow = batch.inputIds[0]
    assert paddedRow[-1].item() == tokenizer.padTokenId
    # Attention mask is 1 on real tokens, 0 on padding.
    assert (batch.attentionMask[0] == 0.0).any()
    assert (batch.attentionMask[1] >= batch.attentionMask[0]).all()


def test_encode_starts_with_bos_and_ends_with_eos() -> None:
    tokenizer = _train(maxLength=16)
    batch = tokenizer.encode("a person walks forward.")
    ids = batch.inputIds[0].tolist()
    length = int(batch.lengths[0].item())

    assert ids[0] == tokenizer.bosTokenId
    assert ids[length - 1] == tokenizer.eosTokenId


def test_encode_truncates_to_max_length() -> None:
    tokenizer = _train(maxLength=8)
    longText = " ".join(["walks forward"] * 50)
    batch = tokenizer.encode(longText)

    assert batch.inputIds.shape == (1, 8)
    assert int(batch.lengths[0].item()) == 8


def test_encode_empty_batch_raises() -> None:
    tokenizer = _train()
    with pytest.raises(ValueError):
        tokenizer.encode([])


# ---------------------------------------------------------------------
# Decoding round-trip
# ---------------------------------------------------------------------
def test_decode_skips_special_tokens_by_default() -> None:
    tokenizer = _train(maxLength=16)
    batch = tokenizer.encode("a person walks forward.")
    text = tokenizer.decode(batch.inputIds[0])

    # The decoded text should not contain any of the special token markers.
    for special in tokenizer.config.specialTokens:
        assert special not in text


def test_decode_batch_returns_one_string_per_row() -> None:
    tokenizer = _train(maxLength=16)
    inputs = ["a person walks forward.", "someone runs backward then stops."]
    batch = tokenizer.encode(inputs)
    decoded = tokenizer.decodeBatch(batch.inputIds)

    assert len(decoded) == 2
    assert all(isinstance(text, str) for text in decoded)


def test_decode_rejects_non_2d_tensors() -> None:
    tokenizer = _train()
    with pytest.raises(ValueError):
        tokenizer.decodeBatch(torch.tensor([0, 1, 2]))


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------
def test_save_then_load_round_trips(tmp_path: Path) -> None:
    tokenizer = _train(maxLength=16, vocabSize=200)
    tokenizer.save(tmp_path)

    assert (tmp_path / CustomTokenizer.TOKENIZER_FILENAME).exists()
    assert (tmp_path / CustomTokenizer.CONFIG_FILENAME).exists()

    reloaded = CustomTokenizer.load(tmp_path)
    assert reloaded.vocabSize == tokenizer.vocabSize
    assert reloaded.config == tokenizer.config

    # Encoding the same text on both instances must match exactly.
    text = "a person walks forward."
    original = tokenizer.encode(text)
    restored = reloaded.encode(text)
    assert torch.equal(original.inputIds, restored.inputIds)
    assert torch.equal(original.attentionMask, restored.attentionMask)


def test_load_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CustomTokenizer.load(tmp_path)


# ---------------------------------------------------------------------
# Corpus iterators
# ---------------------------------------------------------------------
def _writePromptFile(directory: Path, segments: list[dict[str, object]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prompt.json"
    path.write_text(
        json.dumps({"meta": {}, "segments": segments}),
        encoding="utf-8",
    )
    return path


def test_iter_prompt_corpus_yields_text_from_segments(tmp_path: Path) -> None:
    _writePromptFile(
        tmp_path / "subjectA" / "motion1",
        [
            {"text": "a person walks forward.", "startFrame": 0},
            {"text": "  ", "startFrame": 10},  # empty after strip → skipped
            {"text": "the woman dances.", "startFrame": 20},
        ],
    )
    _writePromptFile(
        tmp_path / "subjectB" / "motion2",
        [{"text": "someone runs.", "startFrame": 0}],
    )

    yielded = sorted(iterPromptCorpus(tmp_path))
    assert yielded == [
        "a person walks forward.",
        "someone runs.",
        "the woman dances.",
    ]


def test_iter_prompt_corpus_handles_corrupted_files(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good"
    good.mkdir()
    (good / "prompt.json").write_text(
        json.dumps({"segments": [{"text": "valid prompt"}]}),
        encoding="utf-8",
    )
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "prompt.json").write_text(
        "this is not json {{{",
        encoding="utf-8",
    )

    # Bad file must be skipped silently (with a warning) and not abort.
    yielded = list(iterPromptCorpus(tmp_path))
    assert yielded == ["valid prompt"]


def test_iter_prompt_corpus_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iterPromptCorpus(tmp_path / "does_not_exist"))


def test_iter_prompt_corpus_from_roots_chains_multiple(
    tmp_path: Path,
) -> None:
    rootA = tmp_path / "rootA"
    rootB = tmp_path / "rootB"
    _writePromptFile(rootA / "x", [{"text": "from A"}])
    _writePromptFile(rootB / "y", [{"text": "from B"}])

    yielded = sorted(iterPromptCorpusFromRoots([rootA, rootB]))
    assert yielded == ["from A", "from B"]
