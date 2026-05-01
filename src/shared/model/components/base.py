"""Base classes shared by motion feature components."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.shared.model.components.ops import maskedMean
from src.shared.types.network import BoneDataConfig

SCOPE_BONE = "bone"
SCOPE_GLOBAL = "global"


@dataclass(frozen=True)
class MotionComponentDescriptor:
    """Metadata describing one motion feature channel group."""

    key: str
    configKey: str
    sampleKey: str
    scope: str
    channels: int
    defaultEnabled: bool
    description: str


class MotionComponent:
    """
    Base motion component.

    Components expose enough metadata for dataset preprocessing, tensor
    validation, and loss routing. Concrete subclasses can override
    ``extract`` and ``loss`` when a component needs custom behavior.

    Set ``skipNormalization = True`` on a subclass to exclude its
    channels from the shared z-normalization pass.  The statistics for
    those channels are forced to mean=0 / std=1 (identity transform)
    after ``computeGlobalStatistics`` runs.
    """

    # Override in subclasses to skip z-normalization.
    skipNormalization: bool = False

    descriptor = MotionComponentDescriptor(
        key="component",
        configKey="rotation6d",
        sampleKey="motion",
        scope=SCOPE_GLOBAL,
        channels=0,
        defaultEnabled=False,
        description="base component",
    )

    def __init__(self) -> None:
        self._descriptor = self.descriptor

    @property
    def key(self) -> str:
        """Stable registry key for this component."""
        return self._descriptor.key

    @property
    def configKey(self) -> str:
        """Boolean attribute on :class:`BoneDataConfig` controlling this component."""
        return self._descriptor.configKey

    @property
    def sampleKey(self) -> str:
        """Default key expected in a preprocessed sample."""
        return self._descriptor.sampleKey

    @property
    def scope(self) -> str:
        """Whether this component is stored per-bone or globally."""
        return self._descriptor.scope

    @property
    def channels(self) -> int:
        """Feature channels contributed by this component."""
        return self._descriptor.channels

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self._descriptor.description

    @property
    def defaultEnabled(self) -> bool:
        """Default enabled state used when the config is omitted."""
        return self._descriptor.defaultEnabled

    def isEnabled(self, config: BoneDataConfig) -> bool:
        """Return True when this component is enabled in the config."""
        return bool(getattr(config, self.configKey))

    def outputShape(
        self,
        frames: int,
        numBones: int,
    ) -> tuple[int, ...]:
        """Return the expected unbatched tensor shape for this component."""
        if self.scope == SCOPE_BONE:
            return (frames, numBones, self.channels)
        return (frames, self.channels)

    def extract(self, sample: dict[str, object]) -> torch.Tensor:
        """
        Return the tensor associated with this component from a sample.

        Subclasses can override this when a component is synthesized from
        several tensors instead of living under a single sample key.
        """
        value = sample.get(self.sampleKey)
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                f"Sample is missing tensor component {self.sampleKey!r}."
            )
        return value

    def loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        motionMask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Default masked MSE loss used by most components."""
        squaredError = (predicted - target) ** 2
        return maskedMean(squaredError, motionMask)
