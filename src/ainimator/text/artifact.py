"""Encoder artifact: save / load / hash a standalone text encoder.

An encoder artifact is a directory on disk:

    <artifact_dir>/
        encoder_config.yaml   # encoder type + architecture knobs
        tokenizer/            # BPE tokenizer saved by CustomTokenizer.save()
                              # (omitted for CLIP — tokenizer is from HF hub)
        weights.pt            # torch.save'd state_dict (trainable only)
        hash.txt              # SHA-256 of weights.pt (content hash)

Format decisions
----------------
* YAML for the config so it is human-readable and diff-able.
* ``weights.pt`` stores only trainable parameters (frozen CLIP tower
  excluded, same policy as A4 checkpoint hygiene).
* The hash covers ``weights.pt`` only — tokenizer files and the config
  are metadata and may be regenerated cheaply; the weights are what
  uniquely identify a trained encoder.
* ``loadEncoderArtifact`` returns the encoder AND the tokenizer (custom
  BPE or CLIP stub) so callers never need to know the encoder type.

Imports: only ``core`` and ``text`` — no ``model``, ``training``, or
``data`` (L3 dependency rules, §3.2).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Union

import torch
import yaml

from ainimator.text.custom_text_encoder import (
    CustomTextEncoder,
    CustomTextEncoderConfig,
)
from ainimator.text.clip_text_encoder import (
    ClipTextEncoder,
    ClipTextEncoderConfig,
    ClipTokenizer,
)
from ainimator.text.custom_tokenizer import (
    CustomTokenizer,
)

# Type alias for the encoders the artifact system supports.
AnyEncoder = Union[CustomTextEncoder, ClipTextEncoder]
AnyTokenizer = Union[CustomTokenizer, ClipTokenizer]

# Key written to encoder_config.yaml to identify the encoder type.
_KEY_ENCODER_TYPE = "encoderType"
_ENCODER_TYPE_CUSTOM = "custom"
_ENCODER_TYPE_CLIP = "clip"

_CONFIG_FILENAME = "encoder_config.yaml"
_WEIGHTS_FILENAME = "weights.pt"
_HASH_FILENAME = "hash.txt"
_TOKENIZER_SUBDIR = "tokenizer"


# =====================================================================
# Hash
# =====================================================================

def computeWeightsHash(weightsPath: Path) -> str:
    """Return the SHA-256 hex digest of the weights file.

    Parameters
    ----------
    weightsPath : Path
        Path to the ``weights.pt`` file.

    Returns
    -------
    str
        Lowercase hex SHA-256 digest string.
    """
    sha = hashlib.sha256()
    with weightsPath.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


# =====================================================================
# Save
# =====================================================================

def saveEncoderArtifact(
    encoder: AnyEncoder,
    artifactDir: Path,
    tokenizer: AnyTokenizer | None = None,
) -> str:
    """Persist an encoder artifact to *artifactDir*.

    Creates the directory (and parents) if it does not exist.

    Parameters
    ----------
    encoder : AnyEncoder
        The encoder to persist.  Frozen CLIP tower parameters are
        excluded from ``weights.pt`` (same policy as A4).
    artifactDir : Path
        Target directory.  Created if missing.
    tokenizer : AnyTokenizer | None
        Must be provided for ``CustomTextEncoder`` (the BPE tokenizer
        is saved in ``tokenizer/``).  Ignored for ``ClipTextEncoder``
        (the CLIP tokenizer is reconstructed from the HF hub).

    Returns
    -------
    str
        SHA-256 hex digest of the saved ``weights.pt``.

    Raises
    ------
    ValueError
        When a custom encoder is provided without a tokenizer.
    """
    artifactDir = Path(artifactDir)
    artifactDir.mkdir(parents=True, exist_ok=True)

    isClip = isinstance(encoder, ClipTextEncoder)

    if not isClip and tokenizer is None:
        raise ValueError(
            "saveEncoderArtifact: tokenizer must be provided for "
            "CustomTextEncoder."
        )

    config = _buildConfigDict(encoder)
    configPath = artifactDir / _CONFIG_FILENAME
    configPath.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=True),
        encoding="utf-8",
    )

    if not isClip and isinstance(tokenizer, CustomTokenizer):
        tokenizer.save(artifactDir / _TOKENIZER_SUBDIR)

    state = _trainableState(encoder)
    weightsPath = artifactDir / _WEIGHTS_FILENAME
    _saveTensorsDeterministically(state, weightsPath)

    digest = computeWeightsHash(weightsPath)
    (artifactDir / _HASH_FILENAME).write_text(digest + "\n", encoding="utf-8")

    return digest


def _buildConfigDict(encoder: AnyEncoder) -> dict[str, object]:
    """Serialise encoder config to a plain Python dict."""
    if isinstance(encoder, ClipTextEncoder):
        cfg = encoder.config
        return {
            _KEY_ENCODER_TYPE: _ENCODER_TYPE_CLIP,
            "modelName": cfg.modelName,
            "maxLength": cfg.maxLength,
            "outputDim": cfg.outputDim,
            "clipHiddenDim": cfg.clipHiddenDim,
            "dropout": cfg.dropout,
            "useNullEmbedding": cfg.useNullEmbedding,
            "l2NormalizeOutput": cfg.l2NormalizeOutput,
        }
    cfg = encoder.config  # type: ignore[union-attr]
    return {
        _KEY_ENCODER_TYPE: _ENCODER_TYPE_CUSTOM,
        "vocabSize": cfg.vocabSize,
        "maxLength": cfg.maxLength,
        "hiddenDim": cfg.hiddenDim,
        "numLayers": cfg.numLayers,
        "numHeads": cfg.numHeads,
        "ffnDim": cfg.ffnDim,
        "dropout": cfg.dropout,
        "padTokenId": cfg.padTokenId,
        "outputDim": cfg.outputDim,
        "useNullEmbedding": cfg.useNullEmbedding,
        "l2NormalizeOutput": cfg.l2NormalizeOutput,
    }


def _trainableState(
    encoder: AnyEncoder,
) -> dict[str, torch.Tensor]:
    """Return only trainable (non-frozen) parameters."""
    if isinstance(encoder, ClipTextEncoder):
        return {
            k: v for k, v in encoder.state_dict().items()
            if not k.startswith("clip.")
        }
    return dict(encoder.state_dict())


def _saveTensorsDeterministically(
    state: dict[str, torch.Tensor],
    path: Path,
) -> None:
    """Save a state-dict atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpRaw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    import os
    tmpPath = Path(tmpRaw)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmpPath, path)
    except Exception:
        try:
            tmpPath.unlink()
        except FileNotFoundError:
            pass
        raise


# =====================================================================
# Load
# =====================================================================

def loadEncoderArtifact(
    artifactDir: Path,
    device: torch.device | str = "cpu",
) -> tuple[AnyEncoder, AnyTokenizer]:
    """Load an encoder artifact from *artifactDir*.

    Parameters
    ----------
    artifactDir : Path
        Directory produced by :func:`saveEncoderArtifact`.
    device : torch.device | str
        Device to place the encoder on.

    Returns
    -------
    tuple[AnyEncoder, AnyTokenizer]
        ``(encoder, tokenizer)`` fully constructed and placed on
        *device*.

    Raises
    ------
    FileNotFoundError
        When the artifact directory or required files are missing.
    ValueError
        When the hash in ``hash.txt`` does not match the loaded
        ``weights.pt`` (file corruption guard).
    """
    artifactDir = Path(artifactDir)
    _assertArtifactExists(artifactDir)

    configPath = artifactDir / _CONFIG_FILENAME
    config = yaml.safe_load(configPath.read_text(encoding="utf-8"))

    weightsPath = artifactDir / _WEIGHTS_FILENAME
    _assertHashMatches(artifactDir, weightsPath)

    targetDevice = torch.device(device)
    state: dict[str, torch.Tensor] = torch.load(
        weightsPath, map_location=targetDevice, weights_only=True
    )

    encoderType = config.get(_KEY_ENCODER_TYPE, _ENCODER_TYPE_CUSTOM)
    if encoderType == _ENCODER_TYPE_CLIP:
        return _loadClipArtifact(config, state, targetDevice)
    return _loadCustomArtifact(artifactDir, config, state, targetDevice)


def _assertArtifactExists(artifactDir: Path) -> None:
    """Raise FileNotFoundError when required artifact files are missing."""
    for name in (_CONFIG_FILENAME, _WEIGHTS_FILENAME, _HASH_FILENAME):
        candidate = artifactDir / name
        if not candidate.exists():
            raise FileNotFoundError(
                f"Encoder artifact file missing: {candidate}"
            )


def _assertHashMatches(
    artifactDir: Path, weightsPath: Path
) -> None:
    """Raise ValueError when the stored hash does not match weights.pt."""
    storedHash = (
        (artifactDir / _HASH_FILENAME)
        .read_text(encoding="utf-8")
        .strip()
    )
    computedHash = computeWeightsHash(weightsPath)
    if storedHash != computedHash:
        raise ValueError(
            f"Encoder artifact hash mismatch in {artifactDir}: "
            f"stored={storedHash[:12]}… "
            f"computed={computedHash[:12]}…"
        )


def _loadCustomArtifact(
    artifactDir: Path,
    config: dict[str, object],
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[CustomTextEncoder, CustomTokenizer]:
    """Reconstruct a CustomTextEncoder from its artifact."""
    tokenizerDir = artifactDir / _TOKENIZER_SUBDIR
    if not tokenizerDir.exists():
        raise FileNotFoundError(
            f"Custom encoder artifact missing tokenizer/: {tokenizerDir}"
        )
    tokenizer = CustomTokenizer.load(tokenizerDir)
    outputDimRaw = config.get("outputDim", 0)
    encoderCfg = CustomTextEncoderConfig(
        vocabSize=int(config["vocabSize"]),  # type: ignore[arg-type]
        maxLength=int(config["maxLength"]),  # type: ignore[arg-type]
        hiddenDim=int(config.get("hiddenDim", 256)),  # type: ignore[arg-type]
        numLayers=int(config.get("numLayers", 4)),  # type: ignore[arg-type]
        numHeads=int(config.get("numHeads", 8)),  # type: ignore[arg-type]
        ffnDim=int(config.get("ffnDim", 0)),  # type: ignore[arg-type]
        dropout=float(config.get("dropout", 0.1)),  # type: ignore[arg-type]
        padTokenId=int(config.get("padTokenId", 0)),  # type: ignore[arg-type]
        outputDim=int(outputDimRaw) if outputDimRaw else 0,  # type: ignore[arg-type]
        useNullEmbedding=bool(config.get("useNullEmbedding", True)),
        l2NormalizeOutput=bool(config.get("l2NormalizeOutput", True)),
    )
    encoder = CustomTextEncoder(encoderCfg).to(device)
    encoder.load_state_dict(state, strict=True)
    return encoder, tokenizer


def _loadClipArtifact(
    config: dict[str, object],
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[ClipTextEncoder, ClipTokenizer]:
    """Reconstruct a ClipTextEncoder from its artifact."""
    modelName = str(config.get("modelName", "openai/clip-vit-base-patch32"))
    maxLength = int(config.get("maxLength", 32))  # type: ignore[arg-type]
    encoderCfg = ClipTextEncoderConfig(
        modelName=modelName,
        maxLength=maxLength,
        outputDim=int(config.get("outputDim", 384)),  # type: ignore[arg-type]
        clipHiddenDim=int(config.get("clipHiddenDim", 512)),  # type: ignore[arg-type]
        dropout=float(config.get("dropout", 0.0)),  # type: ignore[arg-type]
        useNullEmbedding=bool(config.get("useNullEmbedding", True)),
        l2NormalizeOutput=bool(config.get("l2NormalizeOutput", True)),
    )
    encoder = ClipTextEncoder(encoderCfg).to(device)
    encoder.load_state_dict(state, strict=False)
    tokenizer = ClipTokenizer(modelName=modelName, maxLength=maxLength)
    return encoder, tokenizer


def readArtifactHash(artifactDir: Path) -> str:
    """Return the stored hash string from an artifact directory.

    Parameters
    ----------
    artifactDir : Path
        Directory produced by :func:`saveEncoderArtifact`.

    Returns
    -------
    str
        The SHA-256 hex string recorded in ``hash.txt``.
    """
    hashPath = Path(artifactDir) / _HASH_FILENAME
    if not hashPath.exists():
        raise FileNotFoundError(
            f"Encoder artifact hash file missing: {hashPath}"
        )
    return hashPath.read_text(encoding="utf-8").strip()
