"""Tests for shared configuration loading."""

from __future__ import annotations

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
