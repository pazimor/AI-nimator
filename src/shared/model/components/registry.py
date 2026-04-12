"""Registry helpers for motion components."""

from __future__ import annotations

from src.shared.model.components.base import MotionComponent, SCOPE_BONE, SCOPE_GLOBAL
from src.shared.model.components.contacts import (
    FootContactComponent,
    HandContactComponent,
)
from src.shared.model.components.kinematics import (
    EndEffectorVelocityComponent,
    JointVelocityComponent,
    JointXyzComponent,
)
from src.shared.model.components.root import (
    PelvisHeightComponent,
    RootTranslationComponent,
    RootVelocityComponent,
    RootYawComponent,
    RootYawVelocityComponent,
)
from src.shared.model.components.rotation6d import Rotation6DComponent
from src.shared.types.network import BoneDataConfig

COMPONENT_TYPES: tuple[type[MotionComponent], ...] = (
    Rotation6DComponent,
    FootContactComponent,
    HandContactComponent,
    RootTranslationComponent,
    RootVelocityComponent,
    RootYawComponent,
    RootYawVelocityComponent,
    JointXyzComponent,
    JointVelocityComponent,
    EndEffectorVelocityComponent,
    PelvisHeightComponent,
)


def buildComponentRegistry() -> tuple[MotionComponent, ...]:
    """Instantiate all built-in motion components in a stable order."""
    return tuple(componentType() for componentType in COMPONENT_TYPES)


def buildEnabledComponents(
    config: BoneDataConfig,
) -> tuple[MotionComponent, ...]:
    """Return the active components for the provided config."""
    return tuple(
        component
        for component in buildComponentRegistry()
        if component.isEnabled(config)
    )


def getComponent(key: str) -> MotionComponent:
    """Return a component instance by key."""
    normalized = key.strip().lower()
    for component in buildComponentRegistry():
        if component.key == normalized:
            return component
    raise KeyError(f"Unknown motion component: {key!r}")


def computeFeatureLayout(
    components: tuple[MotionComponent, ...],
) -> tuple[int, int, tuple[MotionComponent, ...], tuple[MotionComponent, ...]]:
    """
    Compute per-bone and global channel counts from enabled components.

    Returns
    -------
    tuple[int, int, tuple[MotionComponent, ...], tuple[MotionComponent, ...]]
        (boneChannels, globalChannels, boneComponents, globalComponents)
    """
    boneComponents = tuple(c for c in components if c.scope == SCOPE_BONE)
    globalComponents = tuple(c for c in components if c.scope == SCOPE_GLOBAL)
    boneChannels = sum(c.channels for c in boneComponents)
    globalChannels = sum(c.channels for c in globalComponents)
    return boneChannels, globalChannels, boneComponents, globalComponents
