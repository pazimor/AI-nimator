"""Contact feature component definitions."""

from __future__ import annotations

import torch

from src.shared.model.components.base import (
    MotionComponent,
    MotionComponentDescriptor,
    SCOPE_GLOBAL,
)
from src.shared.model.components.ops import maskedMean


class FootContactComponent(MotionComponent):
    """
    MDM-style foot contact labels.

    The original HumanML3D/MDM representation stores four channels:
    two left-foot and two right-foot contact indicators.

    The loss uses binary cross-entropy instead of plain MSE because the
    targets are binary contact flags (0 or 1). This produces sharper
    gradients around the decision boundary and learn cleaner contacts.
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

    def loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        motionMask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Binary cross-entropy loss suited for contact indicators."""
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            predicted,
            target,
            reduction="none",
        )
        return maskedMean(bce, motionMask)


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
