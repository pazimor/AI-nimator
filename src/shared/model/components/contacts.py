"""Contact feature component definitions."""

from __future__ import annotations

from src.shared.model.components.base import (
    MotionComponent,
    MotionComponentDescriptor,
    SCOPE_GLOBAL,
)


class FootContactComponent(MotionComponent):
    """
    MDM-style foot contact labels.

    The original HumanML3D/MDM representation stores four channels:
    two left-foot and two right-foot contact indicators.
    """

    descriptor = MotionComponentDescriptor(
        key="foot_contact",
        configKey="footContact",
        sampleKey="foot_contact",
        scope=SCOPE_GLOBAL,
        channels=4,
        defaultEnabled=False,
        description="MDM-style foot contact labels (4 channels).",
    )


class HandContactComponent(MotionComponent):
    """
    Hand contact labels for upper-body-supported motions.

    This is not part of the default MDM feature set; it is a project-specific
    extension for motions such as crawl, climb, or crawl-like locomotion.
    """

    descriptor = MotionComponentDescriptor(
        key="hand_contact",
        configKey="handContact",
        sampleKey="hand_contact",
        scope=SCOPE_GLOBAL,
        channels=2,
        defaultEnabled=False,
        description="Custom hand contact labels (left/right wrists).",
    )
