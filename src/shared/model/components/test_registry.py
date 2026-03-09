"""Tests for the motion component registry."""

from __future__ import annotations

import torch

from src.shared.model.components import (
    Rotation6DComponent,
    buildEnabledComponents,
    getComponent,
)
from src.shared.types import BoneDataConfig


def test_default_bone_data_enables_only_rotation6d() -> None:
    components = buildEnabledComponents(BoneDataConfig())
    assert [component.key for component in components] == ["rotation6d"]


def test_custom_bone_data_enables_requested_components() -> None:
    config = BoneDataConfig(
        footContact=True,
        rootVelocity=True,
        jointXyz=True,
    )
    components = buildEnabledComponents(config)
    assert [component.key for component in components] == [
        "rotation6d",
        "foot_contact",
        "root_velocity",
        "joint_xyz",
    ]


def test_component_metadata_exposes_expected_shapes() -> None:
    rotation = Rotation6DComponent()
    rootVelocity = getComponent("root_velocity")
    assert rotation.outputShape(frames=8, numBones=22) == (8, 22, 6)
    assert rootVelocity.outputShape(frames=8, numBones=22) == (8, 3)


def test_default_component_loss_honors_motion_mask() -> None:
    component = Rotation6DComponent()
    predicted = torch.tensor(
        [
            [[1.0, 0.0], [5.0, 0.0]],
            [[3.0, 0.0], [9.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    target = torch.zeros_like(predicted)
    mask = torch.tensor([True, False])
    loss = component.loss(predicted, target, motionMask=mask)
    assert torch.isclose(loss, torch.tensor(6.5), atol=1e-6)
