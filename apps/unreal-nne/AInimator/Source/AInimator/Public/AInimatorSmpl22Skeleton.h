// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * SMPL-22 kinematic constants (bone order, parent hierarchy, T-pose
 * bone-local offsets) ported to C++ for engine-side forward kinematics
 * (B4, `apps/spec/footlock_blending.md` §2/§3).
 *
 * Source of truth (never edit these values independently — regenerate
 * from the Python side if the skeleton constants ever change):
 * `src/ainimator/core/constants/skeletons.py`
 * (`SMPL22_BONE_ORDER`, `SMPL22_HIERARCHY`, `SMPL22_DEFAULT_OFFSETS`),
 * consumed by
 * `src/ainimator/geometry/components/ops.py::smpl22KinematicParams`
 * / `rot6dToJointXYZ`. Offsets are approximate T-pose bone-local
 * translations for the mean SMPL shape, in meters, `(x, y, z)` Y-up.
 *
 * Mirrors `apps/unity-sentis/.../PostProcess/Smpl22Skeleton.cs`
 * bone-for-bone, value-for-value (Goal B vérité #7 parity) — this is
 * a straight port, not a reinterpretation.
 */
namespace AInimatorSmpl22Skeleton
{
	// NumBones/RotationChannelsPerBone intentionally duplicate
	// AInimatorContract::NumBones/RotationChannelsPerBone
	// (AInimatorContractConstants.h): that namespace documents the
	// *inference contract*'s frozen dims, this one documents the
	// *skeleton definition*'s bone count/rotation width. They are the
	// same values by construction (SMPL-22 is the contract's
	// skeleton), not two sources of truth that could silently diverge
	// — if the contract ever changes num_bones, both must be updated
	// together (both point back to the same manifest.schema.json
	// `const` fields).
	constexpr int32 NumBones = 22;
	constexpr int32 RotationChannelsPerBone = 6;

	constexpr int32 Pelvis = 0;
	constexpr int32 LeftHip = 1;
	constexpr int32 RightHip = 2;
	constexpr int32 Spine1 = 3;
	constexpr int32 LeftKnee = 4;
	constexpr int32 RightKnee = 5;
	constexpr int32 Spine2 = 6;
	constexpr int32 LeftAnkle = 7;
	constexpr int32 RightAnkle = 8;
	constexpr int32 Spine3 = 9;
	constexpr int32 LeftFoot = 10;
	constexpr int32 RightFoot = 11;
	constexpr int32 Neck = 12;
	constexpr int32 LeftCollar = 13;
	constexpr int32 RightCollar = 14;
	constexpr int32 Head = 15;
	constexpr int32 LeftShoulder = 16;
	constexpr int32 RightShoulder = 17;
	constexpr int32 LeftElbow = 18;
	constexpr int32 RightElbow = 19;
	constexpr int32 LeftWrist = 20;
	constexpr int32 RightWrist = 21;

	/** Foot-contact joints (footlock_blending.md §2): left, right. */
	inline const int32 FootJointIndices[2] = {LeftFoot, RightFoot};

	/** Left-leg two-bone IK chain (footlock_blending.md §3):
	 *  hip -> knee -> ankle. */
	inline const int32 LeftLegChain[3] = {LeftHip, LeftKnee, LeftAnkle};

	/** Right-leg two-bone IK chain (footlock_blending.md §3):
	 *  hip -> knee -> ankle. */
	inline const int32 RightLegChain[3] = {RightHip, RightKnee, RightAnkle};

	/** Parent index per bone, -1 for the root (pelvis). */
	inline const int32 ParentIndices[NumBones] = {
		/* pelvis        */ -1,
		/* leftHip       */ Pelvis,
		/* rightHip      */ Pelvis,
		/* spine1        */ Pelvis,
		/* leftKnee      */ LeftHip,
		/* rightKnee     */ RightHip,
		/* spine2        */ Spine1,
		/* leftAnkle     */ LeftKnee,
		/* rightAnkle    */ RightKnee,
		/* spine3        */ Spine2,
		/* leftFoot      */ LeftAnkle,
		/* rightFoot     */ RightAnkle,
		/* neck          */ Spine3,
		/* leftCollar    */ Spine3,
		/* rightCollar   */ Spine3,
		/* head          */ Neck,
		/* leftShoulder  */ LeftCollar,
		/* rightShoulder */ RightCollar,
		/* leftElbow     */ LeftShoulder,
		/* rightElbow    */ RightShoulder,
		/* leftWrist     */ LeftElbow,
		/* rightWrist    */ RightElbow,
	};

	/** Bone-local T-pose offsets (meters, Y-up), matching
	 *  SMPL22_DEFAULT_OFFSETS exactly (same order, same values). Uses
	 *  Unreal's FVector (X, Y, Z) with the same numeric values as the
	 *  Python/Unity (x, y, z) tuples — no axis remapping, since the
	 *  contract's coord_system ("Y-up right-handed") already matches
	 *  Unreal's own convention for this data. */
	inline const FVector BoneOffsets[NumBones] = {
		/* pelvis        */ FVector(0.0f, 0.0f, 0.0f),
		/* leftHip       */ FVector(0.07f, -0.04f, 0.0f),
		/* rightHip      */ FVector(-0.07f, -0.04f, 0.0f),
		/* spine1        */ FVector(0.0f, 0.1f, 0.02f),
		/* leftKnee      */ FVector(0.0f, -0.40f, 0.0f),
		/* rightKnee     */ FVector(0.0f, -0.40f, 0.0f),
		/* spine2        */ FVector(0.0f, 0.15f, -0.02f),
		/* leftAnkle     */ FVector(0.0f, -0.42f, 0.0f),
		/* rightAnkle    */ FVector(0.0f, -0.42f, 0.0f),
		/* spine3        */ FVector(0.0f, 0.15f, 0.0f),
		/* leftFoot      */ FVector(0.0f, -0.06f, 0.12f),
		/* rightFoot     */ FVector(0.0f, -0.06f, 0.12f),
		/* neck          */ FVector(0.0f, 0.12f, 0.0f),
		/* leftCollar    */ FVector(0.06f, 0.08f, -0.02f),
		/* rightCollar   */ FVector(-0.06f, 0.08f, -0.02f),
		/* head          */ FVector(0.0f, 0.12f, 0.04f),
		/* leftShoulder  */ FVector(0.12f, 0.0f, 0.0f),
		/* rightShoulder */ FVector(-0.12f, 0.0f, 0.0f),
		/* leftElbow     */ FVector(0.26f, 0.0f, 0.0f),
		/* rightElbow    */ FVector(-0.26f, 0.0f, 0.0f),
		/* leftWrist     */ FVector(0.24f, 0.0f, 0.0f),
		/* rightWrist    */ FVector(-0.24f, 0.0f, 0.0f),
	};

	/** Length of the left thigh (hip -> knee), meters. */
	inline float LeftThighLength() { return BoneOffsets[LeftKnee].Size(); }
	/** Length of the left shin (knee -> ankle), meters. */
	inline float LeftShinLength() { return BoneOffsets[LeftAnkle].Size(); }
	/** Length of the right thigh (hip -> knee), meters. */
	inline float RightThighLength() { return BoneOffsets[RightKnee].Size(); }
	/** Length of the right shin (knee -> ankle), meters. */
	inline float RightShinLength() { return BoneOffsets[RightAnkle].Size(); }
}
