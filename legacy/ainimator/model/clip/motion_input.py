"""Helpers for building CLIP motion input tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ainimator.geometry.components import MotionComponent, buildMotionFeatureTensors
from ainimator.geometry.components.base import SCOPE_BONE

PRECOMPUTED_CONTEXT_COMPONENT_KEYS = frozenset(
    {
        "root_translation",
        "root_velocity",
    }
)


@dataclass(frozen=True)
class MotionInputSlice:
    """Contiguous channel span contributed by one motion component."""

    key: str
    scope: str
    start: int
    end: int

    @property
    def channels(self) -> int:
        """Number of channels covered by this slice."""
        return self.end - self.start


def motionInputChannels(components: Sequence[MotionComponent]) -> int:
    """Return the per-bone channel count consumed by the CLIP motion encoder."""
    resolved = tuple(components)
    if not resolved:
        return 6
    return sum(component.channels for component in resolved)


def motionInputSlices(
    components: Sequence[MotionComponent],
) -> tuple[MotionInputSlice, ...]:
    """Describe how each component maps into the concatenated channel axis."""
    resolved = tuple(components)
    if not resolved:
        return (
            MotionInputSlice(
                key="rotation6d",
                scope=SCOPE_BONE,
                start=0,
                end=6,
            ),
        )
    offset = 0
    slices: list[MotionInputSlice] = []
    for component in resolved:
        slices.append(
            MotionInputSlice(
                key=component.key,
                scope=component.scope,
                start=offset,
                end=offset + component.channels,
            )
        )
        offset += component.channels
    return tuple(slices)


def motionInputScopeChannels(
    components: Sequence[MotionComponent],
) -> tuple[int, int]:
    """Return separate channel counts for bone-scoped and global features."""
    boneChannels = 0
    globalChannels = 0
    for part in motionInputSlices(components):
        if part.scope == SCOPE_BONE:
            boneChannels += part.channels
        else:
            globalChannels += part.channels
    return boneChannels, globalChannels


def splitMotionInput(
    motionInput: torch.Tensor,
    components: Sequence[MotionComponent],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Split the concatenated CLIP motion input into per-bone and global branches.

    Global components are repeated across the bone axis during collation; the
    branch encoder therefore keeps a single copy from the first bone slot.
    """
    slices = motionInputSlices(components)
    boneParts: list[torch.Tensor] = []
    globalParts: list[torch.Tensor] = []
    for part in slices:
        values = motionInput[..., part.start:part.end]
        if part.scope == SCOPE_BONE:
            boneParts.append(values)
        else:
            globalParts.append(values[:, :, 0, :])
    boneInput = torch.cat(boneParts, dim=-1) if boneParts else None
    globalInput = torch.cat(globalParts, dim=-1) if globalParts else None
    return boneInput, globalInput


def buildMotionInputFromBatch(
    batch: Mapping[str, object],
    components: Sequence[MotionComponent],
    numBones: int,
) -> torch.Tensor:
    """
    Assemble the CLIP motion tensor from a collated batch or single sample.

    When no explicit components are configured, this falls back to the legacy
    rotation-only ``motion`` tensor.
    """
    resolved = tuple(components)
    if not resolved:
        return _normalizeRotationTensor(_requireTensor(batch, "motion"), numBones)

    parts: list[torch.Tensor] = []
    for component in resolved:
        if component.key == "rotation6d":
            part = _normalizeRotationTensor(
                _requireTensor(batch, "motion"),
                numBones,
            )
        else:
            part = _normalizeComponentTensor(
                _requireTensor(batch, component.sampleKey),
                component,
                numBones,
            )
        parts.append(part)
    return torch.cat(parts, dim=-1)


def buildMotionInputFromMotion(
    motion: torch.Tensor,
    components: Sequence[MotionComponent],
    numBones: int,
    context: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """
    Build a CLIP motion tensor from predicted rotations plus optional context.

    Components derivable from rotations are recomputed from ``motion`` so the
    guidance loss backpropagates into the generator output. Components that are
    not derivable in the current model (for example root translation) are read
    from ``context`` when requested.
    """
    resolved = tuple(components)
    normalizedMotion = _normalizeRotationTensor(motion, numBones)
    if not resolved:
        return normalizedMotion

    derivedComponents = tuple(
        component
        for component in resolved
        if component.key not in PRECOMPUTED_CONTEXT_COMPONENT_KEYS
        and component.key != "rotation6d"
    )
    batchedInputs: list[torch.Tensor] = []
    for batchIndex in range(normalizedMotion.shape[0]):
        sampleMotion = normalizedMotion[batchIndex]
        samplePayload: dict[str, object] = {"motion": sampleMotion}
        if derivedComponents:
            samplePayload.update(
                buildMotionFeatureTensors(
                    motion=sampleMotion,
                    extras={},
                    enabledComponents=derivedComponents,
                )
            )
        if context is not None:
            samplePayload.update(
                _extractContextSample(
                    context=context,
                    batchIndex=batchIndex,
                    batchSize=normalizedMotion.shape[0],
                    device=sampleMotion.device,
                )
            )
        batchedInputs.append(
            buildMotionInputFromBatch(
                batch=samplePayload,
                components=resolved,
                numBones=numBones,
            ).squeeze(0)
        )
    return torch.stack(batchedInputs, dim=0)


def extractMotionContext(
    batch: Mapping[str, object],
    components: Sequence[MotionComponent],
) -> dict[str, torch.Tensor]:
    """Return the subset of batch tensors needed to rebuild CLIP motion input."""
    requiredKeys = {
        component.sampleKey
        for component in components
        if component.key in PRECOMPUTED_CONTEXT_COMPONENT_KEYS
    }
    context: dict[str, torch.Tensor] = {}
    for sampleKey in requiredKeys:
        value = batch.get(sampleKey)
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                f"Batch is missing required CLIP motion context {sampleKey!r}."
            )
        context[sampleKey] = value
    return context


def requiredComponentKeys(components: Sequence[MotionComponent]) -> tuple[str, ...]:
    """Return the manifest component keys required by this CLIP motion layout."""
    resolved = tuple(components)
    if not resolved:
        return ("rotation6d",)
    return tuple(component.key for component in resolved)


def _requireTensor(batch: Mapping[str, object], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"Batch is missing tensor component {key!r}.")
    return value


def _normalizeRotationTensor(motion: torch.Tensor, numBones: int) -> torch.Tensor:
    if motion.dim() == 3:
        motion = motion.unsqueeze(0)
    if motion.dim() != 4:
        raise ValueError(
            "Expected rotation tensor with shape (batch, frames, bones, 6) "
            f"or (frames, bones, 6), got {tuple(motion.shape)}."
        )
    if motion.shape[-2] != numBones:
        raise ValueError(
            f"Expected {numBones} bones in CLIP motion input, got {motion.shape[-2]}."
        )
    if motion.shape[-1] != 6:
        raise ValueError(
            "Expected rotation tensor with 6 channels per bone, got "
            f"{motion.shape[-1]}."
        )
    return motion


def _normalizeComponentTensor(
    tensor: torch.Tensor,
    component: MotionComponent,
    numBones: int,
) -> torch.Tensor:
    if component.scope == SCOPE_BONE:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError(
                f"Expected bone-scoped component {component.sampleKey!r} to have "
                "shape (batch, frames, bones, channels) or "
                "(frames, bones, channels), got "
                f"{tuple(tensor.shape)}."
            )
        if tensor.shape[-2] != numBones:
            raise ValueError(
                f"Expected {numBones} bones for {component.sampleKey!r}, "
                f"got {tensor.shape[-2]}."
            )
        if tensor.shape[-1] != component.channels:
            raise ValueError(
                f"Expected {component.channels} channels for "
                f"{component.sampleKey!r}, got {tensor.shape[-1]}."
            )
        return tensor
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(
            f"Expected global component {component.sampleKey!r} to have "
            "shape (batch, frames, channels) or (frames, channels), got "
            f"{tuple(tensor.shape)}."
        )
    if tensor.shape[-1] != component.channels:
        raise ValueError(
            f"Expected {component.channels} channels for "
            f"{component.sampleKey!r}, got {tensor.shape[-1]}."
        )
    return tensor.unsqueeze(2).expand(
        tensor.shape[0],
        tensor.shape[1],
        numBones,
        tensor.shape[2],
    )


def _extractContextSample(
    context: Mapping[str, object],
    batchIndex: int,
    batchSize: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample: dict[str, torch.Tensor] = {}
    for key, value in context.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.shape and value.shape[0] == batchSize:
            sample[key] = value[batchIndex].to(device)
        else:
            sample[key] = value.to(device)
    return sample
