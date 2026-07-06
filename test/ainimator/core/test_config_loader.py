"""Tests for shared configuration loading."""

from __future__ import annotations

from pathlib import Path

from ainimator.core.config_loader import loadNetworkConfig


def test_load_network_config_parses_bone_data(tmp_path) -> None:
    configPath = tmp_path / "network.yaml"
    configPath.write_text(
        "\n".join(
            [
                "default:",
                "  embed-dim: 128",
                "  clip:",
                "    bone-data:",
                "      rotation6d: false",
                "      joint-xyz: true",
                "  generation:",
                "    embed-dim: 256",
                "    num-heads: 4",
                "    bone-data:",
                "      rotation6d: true",
                "      foot-contact: true",
                "      hand-contact: true",
                "      root-velocity: true",
                "      joint-xyz: true",
            ]
        ),
        encoding="utf-8",
    )

    config = loadNetworkConfig(configPath)

    assert config.clip.boneData is not None
    assert config.clip.boneData.rotation6d is False
    assert config.clip.boneData.jointXyz is True
    assert config.generation.boneData.rotation6d is True
    assert config.generation.boneData.footContact is True
    assert config.generation.boneData.handContact is True
    assert config.generation.boneData.rootVelocity is True
    assert config.generation.boneData.jointXyz is True
    assert config.generation.boneData.pelvisHeight is False


# ---------------------------------------------------------------------
# defaultPreprocessedDatasetRoot (Goal A controller CLIs)
# ---------------------------------------------------------------------
def test_default_preprocessed_root_reads_output_root(tmp_path) -> None:
    from ainimator.core.config_loader import defaultPreprocessedDatasetRoot

    configPath = tmp_path / "preprocess_dataset.yaml"
    configPath.write_text(
        "paths:\n  input-root: /in\n  output-root: /data/prep\n",
        encoding="utf-8",
    )
    assert defaultPreprocessedDatasetRoot(configPath) == Path("/data/prep")


def test_default_preprocessed_root_missing_config_returns_none(
    tmp_path,
) -> None:
    from ainimator.core.config_loader import defaultPreprocessedDatasetRoot

    assert defaultPreprocessedDatasetRoot(tmp_path / "absent.yaml") is None


def test_default_preprocessed_root_no_side_effect(tmp_path) -> None:
    """The helper must NOT create the output-root directory."""
    from ainimator.core.config_loader import defaultPreprocessedDatasetRoot

    target = tmp_path / "prep_out"
    configPath = tmp_path / "preprocess_dataset.yaml"
    configPath.write_text(
        f"paths:\n  output-root: {target}\n", encoding="utf-8"
    )
    defaultPreprocessedDatasetRoot(configPath)
    assert not target.exists()
