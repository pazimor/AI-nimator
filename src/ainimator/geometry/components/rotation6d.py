"""Rotation-based motion component definitions."""

from __future__ import annotations

from ainimator.geometry.components.base import (
    MotionComponent,
    MotionComponentDescriptor,
    SCOPE_BONE,
)


class Rotation6DComponent(MotionComponent):
    """Canonical local 6D joint rotations."""

    descriptor = MotionComponentDescriptor(
        key="rotation6d",
        configKey="rotation6d",
        sampleKey="motion",
        scope=SCOPE_BONE,
        channels=6,
        defaultEnabled=True,
        description="Local joint rotations in continuous 6D form.",
    )
