"""Root-motion component definitions."""

from __future__ import annotations

from src.shared.model.components.base import (
    MotionComponent,
    MotionComponentDescriptor,
    SCOPE_GLOBAL,
)


class RootTranslationComponent(MotionComponent):
    """Root translation (typically pelvis world-space translation)."""

    descriptor = MotionComponentDescriptor(
        key="root_translation",
        configKey="rootTranslation",
        sampleKey="root_translation",
        scope=SCOPE_GLOBAL,
        channels=3,
        defaultEnabled=False,
        description="Root translation in XYZ space.",
    )


class RootVelocityComponent(MotionComponent):
    """Root linear velocity."""

    descriptor = MotionComponentDescriptor(
        key="root_velocity",
        configKey="rootVelocity",
        sampleKey="root_velocity",
        scope=SCOPE_GLOBAL,
        channels=3,
        defaultEnabled=False,
        description="Root linear velocity in XYZ space.",
    )


class RootYawComponent(MotionComponent):
    """Root heading angle around the up axis."""

    descriptor = MotionComponentDescriptor(
        key="root_yaw",
        configKey="rootYaw",
        sampleKey="root_yaw",
        scope=SCOPE_GLOBAL,
        channels=1,
        defaultEnabled=False,
        description="Root yaw / heading angle.",
    )


class RootYawVelocityComponent(MotionComponent):
    """Root angular velocity around the up axis."""

    descriptor = MotionComponentDescriptor(
        key="root_yaw_velocity",
        configKey="rootYawVelocity",
        sampleKey="root_yaw_velocity",
        scope=SCOPE_GLOBAL,
        channels=1,
        defaultEnabled=False,
        description="Root angular velocity around the up axis.",
    )


class PelvisHeightComponent(MotionComponent):
    """Root joint height above the inferred floor."""

    descriptor = MotionComponentDescriptor(
        key="pelvis_height",
        configKey="pelvisHeight",
        sampleKey="pelvis_height",
        scope=SCOPE_GLOBAL,
        channels=1,
        defaultEnabled=False,
        description="Pelvis height above the floor plane.",
    )
