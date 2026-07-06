"""Motion feature components shared across preprocessing and training."""

from __future__ import annotations

from .base import (
    MotionComponent,
    MotionComponentDescriptor,
    SCOPE_BONE,
    SCOPE_GLOBAL,
)
from .contacts import FootContactComponent, HandContactComponent
from .feature_builder import buildMotionFeatureTensors
from .kinematics import (
    EndEffectorVelocityComponent,
    JointVelocityComponent,
    JointXyzComponent,
)
from .ops import (
    maskedMean,
    orthonormalizeRot6d,
    rot6dToJointXYZ,
    sixdToRotationMatrix,
    temporalAngleDifference,
    temporalDifference,
)
from .registry import (
    COMPONENT_TYPES,
    buildComponentRegistry,
    buildEnabledComponents,
    computeFeatureLayout,
    getComponent,
)
from .root import (
    PelvisHeightComponent,
    RootTranslationComponent,
    RootVelocityComponent,
    RootYawComponent,
    RootYawVelocityComponent,
)
from .rotation6d import Rotation6DComponent

__all__ = [
    "COMPONENT_TYPES",
    "MotionComponent",
    "MotionComponentDescriptor",
    "SCOPE_BONE",
    "SCOPE_GLOBAL",
    "maskedMean",
    "Rotation6DComponent",
    "FootContactComponent",
    "HandContactComponent",
    "buildMotionFeatureTensors",
    "orthonormalizeRot6d",
    "rot6dToJointXYZ",
    "RootTranslationComponent",
    "RootVelocityComponent",
    "RootYawComponent",
    "RootYawVelocityComponent",
    "JointXyzComponent",
    "JointVelocityComponent",
    "EndEffectorVelocityComponent",
    "PelvisHeightComponent",
    "buildComponentRegistry",
    "buildEnabledComponents",
    "computeFeatureLayout",
    "getComponent",
    "sixdToRotationMatrix",
    "temporalAngleDifference",
    "temporalDifference",
]
