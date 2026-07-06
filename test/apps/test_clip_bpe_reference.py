"""Parity tests for the B7 CLIP BPE reference tokenizer.

Two locks (text_encoding.md §2 / §4):

1. **Reference vs canonical vectors** — ``clip_bpe_reference.py``
   reproduces every case of ``text_encoding_parity.json`` (the file
   the engine suites also consume). Runs offline from the HuggingFace
   cache (vocab/merges saved once per session).
2. **Reference vs HuggingFace** — the reference matches
   ``CLIPTokenizerFast`` on a broader adversarial set (accents,
   contractions, truncation, empty input). Skipped when transformers
   or the cached model is unavailable.

Regenerating the parity file: adjust the prompt list in this module's
``_PARITY_PATH`` payload only through the reference implementation —
never hand-edit ids.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SPEC_DIR = Path(__file__).parents[2] / "apps" / "spec"
sys.path.insert(0, str(_SPEC_DIR))
from clip_bpe_reference import ClipBpeReference  # noqa: E402

_PARITY_PATH = _SPEC_DIR / "text_encoding_parity.json"
_MODEL_NAME = "openai/clip-vit-base-patch32"


@pytest.fixture(scope="module")
def vocabFiles(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export vocab.json/merges.txt from the HF tokenizer cache."""
    transformers = pytest.importorskip("transformers")
    directory = tmp_path_factory.mktemp("clip_tokenizer")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            _MODEL_NAME
        )
    except OSError:
        pytest.skip(f"{_MODEL_NAME} tokenizer not cached/downloadable")
    tokenizer.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def reference(vocabFiles: Path) -> ClipBpeReference:
    """Reference tokenizer built from the exported vocabulary."""
    return ClipBpeReference(
        vocabFiles / "vocab.json", vocabFiles / "merges.txt"
    )


def test_parity_file_matches_reference(
    reference: ClipBpeReference,
) -> None:
    """Every canonical parity case is reproduced bit-exact."""
    payload = json.loads(_PARITY_PATH.read_text(encoding="utf-8"))
    assert payload["cases"], "parity file has no cases"
    maxLength = payload["max_length"]
    for case in payload["cases"]:
        encoded = reference.encode(case["prompt"], maxLength)
        assert encoded.inputIds == case["input_ids"], case["prompt"]
        assert [int(m) for m in encoded.attentionMask] == (
            case["attention_mask"]
        ), case["prompt"]


def test_parity_file_special_ids(reference: ClipBpeReference) -> None:
    """bos/eos ids in the parity file match the vocabulary."""
    payload = json.loads(_PARITY_PATH.read_text(encoding="utf-8"))
    assert payload["bos_id"] == reference.bosId
    assert payload["eos_id"] == reference.eosId
    assert payload["pad_id"] == reference.eosId


@pytest.mark.parametrize("text", [
    "a person dances",
    "A person walks FORWARD, slowly...",
    "the man   jumps\thigh",
    "someone's arm-wave (energetic!)",
    "un danseur pivote sur lui-meme",
    "une personne marche vers l'avant",
    "danse énergique avec pirouette",
    "il court a 10 km/h pendant 30 minutes",
    "jumping-jacks x25!!",
    "T-pose then squat; repeat 3x",
    "he can't stop, won't stop",
    "",
    "   ",
    "a",
    "supercalifragilisticexpialidocious motion",
    "l'homme s'assoit et se releve",
    "quick brown fox jumps over the lazy dog 0123456789",
    "a " * 60 + "end",
])
def test_reference_matches_huggingface(
    text: str,
    reference: ClipBpeReference,
) -> None:
    """Reference output equals CLIPTokenizerFast, padding included."""
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(_MODEL_NAME)
    expected = tokenizer(
        text, padding="max_length", max_length=32, truncation=True
    )
    encoded = reference.encode(text, 32)
    assert encoded.inputIds == expected["input_ids"], text
    assert [int(m) for m in encoded.attentionMask] == (
        expected["attention_mask"]
    ), text
