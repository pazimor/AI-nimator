"""Contact feature component definitions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.shared.model.components.base import (
    MotionComponent,
    MotionComponentDescriptor,
    SCOPE_GLOBAL,
)
from src.shared.model.components.ops import maskedMean


class FootContactComponent(MotionComponent):
    """
    MDM-style foot contact labels (4 channels: 2 left-foot + 2 right-foot).

    Loss is **BCE-with-logits** on raw (non-z-normalized) predictions.
    Foot-contact channels are **excluded from z-normalization** in
    ``computeGlobalStatistics`` so the denoiser predicts logits directly
    and the target remains binary {0, 1}.

    History: the original implementation used MSE on z-normalized values,
    but with mean=0.997 and std=0.05-0.07 the z-norm produced a bimodal
    distribution {+0.14, -14} that caused gradient spikes at lift-off and
    contaminated ``glob_d``.  BCE treats the signal as what it is — a
    binary indicator — and avoids the pathological z-norm distribution.

    At inference, the 4 foot-contact channels are passed through sigmoid
    to recover probabilities.
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

    # Mark this component as requiring raw (non-normalized) targets.
    skipNormalization: bool = True

    def loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        motionMask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BCE-with-logits loss: predicted=logits, target=binary {0,1}."""
        bce = F.binary_cross_entropy_with_logits(
            predicted, target, reduction="none",
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
