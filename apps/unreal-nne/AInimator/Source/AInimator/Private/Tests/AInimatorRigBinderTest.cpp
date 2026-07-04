// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorRigBinder.h"
#include "AInimatorRigMap.h"
#include "AInimatorSmpl22Skeleton.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace AInimatorRigBinderTestHelpers
{
	/**
	 * Builds a synthetic 3-bone mini-rig mapping (pelvis -> "root",
	 * leftHip -> "hip_l", leftKnee -> "knee_l") with everything else
	 * left unmapped, matching rig_binding.md §2.1's "unmapped bone,
	 * rotation ignored" rule.
	 */
	UAInimatorRigMap* MakeMiniRigMap()
	{
		UAInimatorRigMap* RigMap = NewObject<UAInimatorRigMap>();
		RigMap->Entries[AInimatorSmpl22Skeleton::Pelvis].TargetBoneName = TEXT("root");
		RigMap->Entries[AInimatorSmpl22Skeleton::LeftHip].TargetBoneName = TEXT("hip_l");
		RigMap->Entries[AInimatorSmpl22Skeleton::LeftKnee].TargetBoneName = TEXT("knee_l");
		return RigMap;
	}

	/** Rest-pose world rotations for the synthetic rig: root at identity,
	 *  hip_l with a known +30 deg offset around Y (yaw), knee_l with a
	 *  known -20 deg offset around X (pitch) — deliberately NOT identity
	 *  so the test actually exercises the offset-correction formula. */
	FQuat RestWorldRotationForBone(FName BoneName)
	{
		if (BoneName == TEXT("root"))
		{
			return FQuat::Identity;
		}
		if (BoneName == TEXT("hip_l"))
		{
			return FQuat(FRotator(0.0f, 30.0f, 0.0f));
		}
		if (BoneName == TEXT("knee_l"))
		{
			return FQuat(FRotator(-20.0f, 0.0f, 0.0f));
		}
		return FQuat::Identity;
	}
}

/**
 * Retarget math on a synthetic mini-rig with KNOWN offsets: when the
 * SMPL world rotation for every mapped bone equals identity (i.e. the
 * controller's current pose exactly matches the SMPL rest pose), the
 * spec §2.2 formula
 *   worldRot_target(bone) = R_smpl_world(bone) * worldRot_rig_rest(bone)
 *   localRot_target(bone) = worldRot_target(parent)^-1 * worldRot_target(bone)
 * must reproduce the rig's OWN rest pose exactly: localRot_target(root)
 * == identity (root has no mapped parent, so local == world == its own
 * rest rotation... wait: identity SMPL means worldRot_target(root) ==
 * rest rotation of root == identity here) and localRot_target(hip_l) ==
 * rest offset composed appropriately. This test locks in the exact
 * expected quaternions given the known offsets above.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorRigBinderReproducesRestPoseTest,
	"AInimator.RigBinder.IdentitySmplPoseReproducesRigRestPose",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorRigBinderReproducesRestPoseTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorRigBinderTestHelpers;

	UAInimatorRigMap* RigMap = MakeMiniRigMap();
	TArray<FRigBinder::FCalibratedBone> Calibration;
	const bool bCalibrated = FRigBinder::Calibrate(
		*RigMap, &RestWorldRotationForBone, Calibration);
	TestTrue(TEXT("Calibration succeeds with pelvis mapped"), bCalibrated);
	TestEqual(TEXT("3 bones calibrated (pelvis, leftHip, leftKnee)"), Calibration.Num(), 3);

	// SMPL world rotation is IDENTITY for every bone this frame (the
	// controller output exactly matches the SMPL rest pose).
	TArray<FQuat> SmplWorldRotations;
	SmplWorldRotations.SetNumZeroed(AInimatorSmpl22Skeleton::NumBones);
	for (FQuat& Rotation : SmplWorldRotations)
	{
		Rotation = FQuat::Identity;
	}

	TArray<FQuat> TargetWorldScratch;
	TArray<FName> BoneNames;
	TArray<FQuat> BoneLocalRotations;
	FRigBinder::RetargetFrame(
		SmplWorldRotations, Calibration, TargetWorldScratch, BoneNames, BoneLocalRotations);

	// Index 0 = pelvis/root (no mapped parent): localRot_target ==
	// worldRot_target == identity * identity == identity.
	TestTrue(TEXT("root local rotation is identity"),
		BoneLocalRotations[0].Equals(FQuat::Identity, 1e-4f));

	// Index 1 = leftHip/hip_l: worldRot_target = identity * rest(hip_l)
	// = rest(hip_l); parent (root) worldRot_target = identity; so
	// localRot_target(hip_l) = identity^-1 * rest(hip_l) = rest(hip_l).
	const FQuat ExpectedHipLocal = RestWorldRotationForBone(TEXT("hip_l"));
	TestTrue(TEXT("hip_l local rotation equals its own rest offset (root is identity)"),
		BoneLocalRotations[1].Equals(ExpectedHipLocal, 1e-4f));

	// Index 2 = leftKnee/knee_l: worldRot_target(knee_l) = rest(knee_l);
	// parent is hip_l (nearest MAPPED ancestor), worldRot_target(hip_l)
	// = rest(hip_l); so localRot_target(knee_l) = rest(hip_l)^-1 * rest(knee_l).
	const FQuat ExpectedKneeLocal =
		RestWorldRotationForBone(TEXT("hip_l")).Inverse() * RestWorldRotationForBone(TEXT("knee_l"));
	TestTrue(TEXT("knee_l local rotation composes against its mapped parent's (hip_l) rest offset"),
		BoneLocalRotations[2].Equals(ExpectedKneeLocal, 1e-4f));

	return true;
}

/**
 * When the controller's current SMPL pose diverges from rest by a known
 * rotation, that SAME divergence must show up, un-attenuated, in the
 * retargeted local rotation of a mapped bone with no mapped parent
 * (root): localRot_target(root) == SmplWorldRotation(root) * rest(root)
 * == SmplWorldRotation(root) since rest(root) is identity here.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorRigBinderAppliesSmplDeltaTest,
	"AInimator.RigBinder.SmplRotationDeltaPropagatesToRootLocalRotation",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorRigBinderAppliesSmplDeltaTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorRigBinderTestHelpers;

	UAInimatorRigMap* RigMap = MakeMiniRigMap();
	TArray<FRigBinder::FCalibratedBone> Calibration;
	FRigBinder::Calibrate(*RigMap, &RestWorldRotationForBone, Calibration);

	TArray<FQuat> SmplWorldRotations;
	SmplWorldRotations.SetNumZeroed(AInimatorSmpl22Skeleton::NumBones);
	for (FQuat& Rotation : SmplWorldRotations)
	{
		Rotation = FQuat::Identity;
	}
	const FQuat PelvisTurn(FRotator(0.0f, 45.0f, 0.0f));
	SmplWorldRotations[AInimatorSmpl22Skeleton::Pelvis] = PelvisTurn;

	TArray<FQuat> TargetWorldScratch;
	TArray<FName> BoneNames;
	TArray<FQuat> BoneLocalRotations;
	FRigBinder::RetargetFrame(
		SmplWorldRotations, Calibration, TargetWorldScratch, BoneNames, BoneLocalRotations);

	TestTrue(TEXT("root local rotation equals the SMPL pelvis delta (rest(root) is identity)"),
		BoneLocalRotations[0].Equals(PelvisTurn, 1e-4f));
	return true;
}

/**
 * Unmapped SMPL bones (spine1 etc. in the mini-rig) contribute zero
 * calibrated entries — Calibrate must only ever return entries for
 * bones RigMap actually maps (spec §2.1 v1 rule).
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorRigBinderSkipsUnmappedBonesTest,
	"AInimator.RigBinder.UnmappedBonesAreSkippedNotSubstituted",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorRigBinderSkipsUnmappedBonesTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorRigBinderTestHelpers;

	UAInimatorRigMap* RigMap = MakeMiniRigMap();
	TArray<FRigBinder::FCalibratedBone> Calibration;
	FRigBinder::Calibrate(*RigMap, &RestWorldRotationForBone, Calibration);

	for (const FRigBinder::FCalibratedBone& Entry : Calibration)
	{
		const bool bIsOneOfTheThreeMapped =
			Entry.SmplBoneIndex == AInimatorSmpl22Skeleton::Pelvis ||
			Entry.SmplBoneIndex == AInimatorSmpl22Skeleton::LeftHip ||
			Entry.SmplBoneIndex == AInimatorSmpl22Skeleton::LeftKnee;
		TestTrue(TEXT("Every calibrated entry is one of the 3 explicitly mapped bones"),
			bIsOneOfTheThreeMapped);
	}
	TestEqual(TEXT("Exactly 3 calibrated entries, no substitution for unmapped bones"),
		Calibration.Num(), 3);
	return true;
}

/** Calibration must fail loudly (never silently) when the pelvis itself
 *  is unmapped — the retarget parent chain has no root frame to divide
 *  out against otherwise. */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorRigBinderRejectsUnmappedPelvisTest,
	"AInimator.RigBinder.RejectsCalibrationWithUnmappedPelvis",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorRigBinderRejectsUnmappedPelvisTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorRigBinderTestHelpers;

	UAInimatorRigMap* RigMap = NewObject<UAInimatorRigMap>();
	RigMap->Entries[AInimatorSmpl22Skeleton::LeftHip].TargetBoneName = TEXT("hip_l");
	// Pelvis (bone 0) deliberately left unmapped.

	TArray<FRigBinder::FCalibratedBone> Calibration;
	const bool bCalibrated = FRigBinder::Calibrate(
		*RigMap, &RestWorldRotationForBone, Calibration);
	TestFalse(TEXT("Calibration fails fast when the pelvis is unmapped"), bCalibrated);
	return true;
}

/** Root-motion scale (spec §2.3): rig rest pelvis height / canonical
 *  SMPL rest pelvis height (~0.91m), with the documented cm->m
 *  conversion for measurements taken directly in Unreal's native unit. */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorRigBinderRigScaleUnitsTest,
	"AInimator.RigBinder.RigScaleHandlesCentimeterToMeterConversion",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorRigBinderRigScaleUnitsTest::RunTest(const FString& Parameters)
{
	// A rig whose rest pelvis sits exactly at the canonical SMPL height
	// (0.91m == 91cm) must scale root motion by exactly 1.0.
	const float ScaleAtCanonicalHeight = FRigBinder::ComputeRigScaleFromCentimeters(91.0f);
	TestTrue(TEXT("Rig at the canonical SMPL pelvis height scales root motion by 1.0"),
		FMath::IsNearlyEqual(ScaleAtCanonicalHeight, 1.0f, 1e-3f));

	// A taller rig (e.g. 182cm pelvis height, twice the canonical 91cm)
	// must scale root motion by exactly 2.0 — root motion authored for
	// the canonical SMPL proportions is doubled for a rig twice as tall.
	const float ScaleAtDoubleHeight = FRigBinder::ComputeRigScaleFromCentimeters(182.0f);
	TestTrue(TEXT("A rig twice as tall scales root motion by 2.0"),
		FMath::IsNearlyEqual(ScaleAtDoubleHeight, 2.0f, 1e-3f));

	// The meters-native overload must agree with the centimeters overload
	// after manual conversion (no double conversion / off-by-100 bug).
	const float ScaleFromMeters = FRigBinder::ComputeRigScale(0.91f);
	TestTrue(TEXT("ComputeRigScale (meters) agrees with the centimeters overload"),
		FMath::IsNearlyEqual(ScaleFromMeters, ScaleAtCanonicalHeight, 1e-3f));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
