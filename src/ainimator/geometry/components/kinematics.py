"""Kinematics-derived motion component definitions."""

from __future__ import annotations

from ainimator.geometry.components.base import (
    MotionComponent,
    MotionComponentDescriptor,
    SCOPE_BONE,
    SCOPE_GLOBAL,
)


class JointXyzComponent(MotionComponent):
    """Global joint positions obtained from forward kinematics."""

    descriptor = MotionComponentDescriptor(
        key="joint_xyz",
        configKey="jointXyz",
        sampleKey="joint_xyz",
        scope=SCOPE_BONE,
        channels=3,
        defaultEnabled=False,
        description="Global joint XYZ positions from forward kinematics.",
    )


class JointVelocityComponent(MotionComponent):
    """Global joint velocities."""

    descriptor = MotionComponentDescriptor(
        key="joint_velocity",
        configKey="jointVelocity",
        sampleKey="joint_velocity",
        scope=SCOPE_BONE,
        channels=3,
        defaultEnabled=False,
        description="Global joint XYZ velocities.",
    )


class EndEffectorVelocityComponent(MotionComponent):
    """
    Velocities for key end-effectors.

    The default layout assumes wrists + ankles (4 joints x 3 channels).
    """

    descriptor = MotionComponentDescriptor(
        key="end_effector_velocity",
        configKey="endEffectorVelocity",
        sampleKey="end_effector_velocity",
        scope=SCOPE_GLOBAL,
        channels=12,
        defaultEnabled=False,
        description="End-effector velocities for wrists and ankles.",
    )
