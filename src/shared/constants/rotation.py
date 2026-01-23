"""Rotation representation constants and helpers."""

from __future__ import annotations

from typing import Final


ROTATION_REPR_ROT6D: Final[str] = "rot6d"
ROTATION_REPR_AXIS_ANGLE: Final[str] = "axis-angle"
ROTATION_KIND_AXIS_ANGLE: Final[str] = "axis_angle"
DEFAULT_ROTATION_REPR: Final[str] = ROTATION_REPR_ROT6D

ROTATION_CHANNELS_ROT6D: Final[int] = 6
ROTATION_CHANNELS_AXIS_ANGLE: Final[int] = 3

ROTATION_REPR_CHANNELS: Final[dict[str, int]] = {
    ROTATION_REPR_ROT6D: ROTATION_CHANNELS_ROT6D,
    ROTATION_REPR_AXIS_ANGLE: ROTATION_CHANNELS_AXIS_ANGLE,
}
ROTATION_REPR_ALIASES: Final[dict[str, str]] = {
    ROTATION_KIND_AXIS_ANGLE: ROTATION_REPR_AXIS_ANGLE,
}


def normalizeRotationRepr(rotationRepr: str | None) -> str:
    """Normalize rotation representation string."""
    if rotationRepr is None:
        return DEFAULT_ROTATION_REPR
    normalized = rotationRepr.strip().lower()
    if normalized in ROTATION_REPR_ALIASES:
        return ROTATION_REPR_ALIASES[normalized]
    if normalized in ROTATION_REPR_CHANNELS:
        return normalized
    return DEFAULT_ROTATION_REPR


def resolveRotationChannels(rotationRepr: str, fallback: int) -> int:
    """Return channels for rotation representation."""
    return ROTATION_REPR_CHANNELS.get(rotationRepr, fallback)
