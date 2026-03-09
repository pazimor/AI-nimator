"""Tests for CLIP motion input assembly helpers."""

from __future__ import annotations

import torch

from src.shared.model.clip.motion_input import (
    buildMotionInputFromBatch,
    buildMotionInputFromMotion,
    motionInputScopeChannels,
    splitMotionInput,
)
from src.shared.model.components.kinematics import JointXyzComponent
from src.shared.model.components.root import PelvisHeightComponent, RootTranslationComponent


def test_build_motion_input_from_batch_broadcasts_global_features() -> None:
    components = (JointXyzComponent(), RootTranslationComponent())
    jointXyz = torch.randn(2, 4, 3, 3)
    rootTranslation = torch.randn(2, 4, 3)

    motionInput = buildMotionInputFromBatch(
        batch={
            "joint_xyz": jointXyz,
            "root_translation": rootTranslation,
        },
        components=components,
        numBones=3,
    )

    assert motionInput.shape == (2, 4, 3, 6)
    expectedRoot = rootTranslation.unsqueeze(2).expand(2, 4, 3, 3)
    assert torch.allclose(motionInput[..., 3:], expectedRoot)


def test_build_motion_input_from_motion_uses_context_for_non_predicted_terms() -> None:
    components = (JointXyzComponent(), RootTranslationComponent())
    motion = torch.randn(1, 4, 3, 6)
    rootTranslation = torch.randn(1, 4, 3)

    motionInput = buildMotionInputFromMotion(
        motion=motion,
        components=components,
        numBones=3,
        context={"root_translation": rootTranslation},
    )

    assert motionInput.shape == (1, 4, 3, 6)
    expectedRoot = rootTranslation.unsqueeze(2).expand(1, 4, 3, 3)
    assert torch.allclose(motionInput[..., 3:], expectedRoot)


def test_build_motion_input_from_motion_can_derive_pelvis_height_without_extras() -> None:
    motionInput = buildMotionInputFromMotion(
        motion=torch.randn(1, 5, 3, 6),
        components=(PelvisHeightComponent(),),
        numBones=3,
    )

    assert motionInput.shape == (1, 5, 3, 1)


def test_split_motion_input_separates_bone_and_global_channels() -> None:
    components = (JointXyzComponent(), RootTranslationComponent())
    jointXyz = torch.randn(2, 4, 3, 3)
    rootTranslation = torch.randn(2, 4, 3)

    motionInput = buildMotionInputFromBatch(
        batch={
            "joint_xyz": jointXyz,
            "root_translation": rootTranslation,
        },
        components=components,
        numBones=3,
    )
    boneChannels, globalChannels = motionInputScopeChannels(components)
    boneInput, globalInput = splitMotionInput(motionInput, components)

    assert boneChannels == 3
    assert globalChannels == 3
    assert boneInput is not None
    assert globalInput is not None
    assert boneInput.shape == (2, 4, 3, 3)
    assert globalInput.shape == (2, 4, 3)
    assert torch.allclose(boneInput, jointXyz)
    assert torch.allclose(globalInput, rootTranslation)
