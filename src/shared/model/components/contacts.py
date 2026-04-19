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
    MDM-style foot contact labels (4 channels: 2 left-foot + 2 right-foot).

    Loss is MSE, not BCE-with-logits.  Foot-contact channels travel through
    the shared z-normalization / diffusion pipeline alongside every other
    global feature: targets are scaled by (x - mean) / std before the
    denoiser sees them, and predictions are denormalized on the way out.
    BCE-with-logits would interpret the denormalized prediction as a
    *logit*, but the denoiser outputs are bounded by the learned
    distribution of normalized targets and cannot reach the ±∞ needed for
    sigmoid(logit) to hit 0 or 1 -- even on a single-sample overfit.  The
    result was a hard floor around 0.31 on ``loss_foot_contact`` that
    pinned aux and masked convergence of the rest of the auxiliary
    components.  MSE keeps the loss in the same space as every other
    global component and converges to 0 cleanly in overfit.
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
        """MSE loss on denormalized predictions vs binary 0/1 targets."""
        return maskedMean((predicted - target).pow(2), motionMask)


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
