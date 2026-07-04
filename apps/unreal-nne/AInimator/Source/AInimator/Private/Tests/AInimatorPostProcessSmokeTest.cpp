// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorForwardKinematics.h"
#include "AInimatorSmpl22Skeleton.h"
#include "AInimatorFootContactDetector.h"
#include "AInimatorFootLockIK.h"
#include "AInimatorIdleMoveBlender.h"
#include "AInimatorFootSlidingMetric.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace AInimatorPostProcessSmokeTestHelpers
{
	using namespace AInimatorSmpl22Skeleton;

	/**
	 * Synthesizes one frame's SMPL-22 rotation6d bone frame for a
	 * simple walk cycle: legs swing (hip rotation about X), pelvis
	 * stays identity, everything else stays at rest. Not a realistic
	 * gait, only enough kinematic variety to exercise FK + foot
	 * contact/IK over a sustained rollout without NaN.
	 */
	void BuildWalkCycleFrame(int32 Step, TArray<float>& OutBoneFrame)
	{
		OutBoneFrame.SetNumUninitialized(NumBones * RotationChannelsPerBone);
		for (int32 Bone = 0; Bone < NumBones; ++Bone)
		{
			OutBoneFrame[Bone * 6 + 0] = 1.0f;
			OutBoneFrame[Bone * 6 + 1] = 0.0f;
			OutBoneFrame[Bone * 6 + 2] = 0.0f;
			OutBoneFrame[Bone * 6 + 3] = 0.0f;
			OutBoneFrame[Bone * 6 + 4] = 1.0f;
			OutBoneFrame[Bone * 6 + 5] = 0.0f;
		}

		// Alternate a small swing on the hips/knees to make the feet
		// touch down out of phase, like a coarse walk cycle.
		const float Phase = static_cast<float>(Step) * 0.2f;
		const float LeftSwing = FMath::Sin(Phase);
		const float RightSwing = FMath::Sin(Phase + PI);

		auto ApplySwing = [&OutBoneFrame](int32 Bone, float Swing)
		{
			// Small rotation about X in the (col0, col1) plane: tilt
			// column0 toward -Z, column1 stays close to Y (small-angle,
			// bounded, never fully degenerate).
			const float Angle = Swing * 0.3f;
			OutBoneFrame[Bone * 6 + 0] = FMath::Cos(Angle);
			OutBoneFrame[Bone * 6 + 2] = -FMath::Sin(Angle);
			OutBoneFrame[Bone * 6 + 4] = FMath::Cos(Angle);
			OutBoneFrame[Bone * 6 + 5] = FMath::Sin(Angle);
		};
		ApplySwing(LeftHip, LeftSwing);
		ApplySwing(RightHip, RightSwing);
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorPostProcessSixtySecondsNoExplosionTest,
	"AInimator.PostProcess.SixtySecondsWalkCycleNoNaNNoExplosion",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorPostProcessSixtySecondsNoExplosionTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorPostProcessSmokeTestHelpers;
	using namespace AInimatorSmpl22Skeleton;

	constexpr float TargetFps = 30.0f;
	constexpr float SimulatedSeconds = 60.0f;
	constexpr int32 NumSteps = static_cast<int32>(TargetFps * SimulatedSeconds);
	constexpr float DeltaSeconds = 1.0f / TargetFps;

	FFootContactDetector ContactDetector(/*FootCount=*/2);
	FFootLockIK FootLockIK;
	FIdleMoveBlender IdleMoveBlender;
	FFootSlidingMetric SlidingMetricAfter;
	FFootSlidingMetric SlidingMetricBefore;

	FFootLockIK::FLegChain LeftChain;
	LeftChain.UpperLength = LeftThighLength();
	LeftChain.LowerLength = LeftShinLength();
	FFootLockIK::FLegChain RightChain;
	RightChain.UpperLength = RightThighLength();
	RightChain.LowerLength = RightShinLength();

	FVector WorldRoot = FVector::ZeroVector;

	for (int32 Step = 0; Step < NumSteps; ++Step)
	{
		TArray<float> BoneFrame;
		BuildWalkCycleFrame(Step, BoneFrame);

		TArray<FMatrix> GlobalRotations;
		TArray<FVector> RootLocalPositions;
		GlobalRotations.SetNum(NumBones);
		RootLocalPositions.SetNum(NumBones);
		AInimatorForwardKinematics::ComputeJointPositions(
			BoneFrame, NumBones, GlobalRotations, RootLocalPositions);

		WorldRoot += FVector(0.01f, 0.0f, 0.0f); // steady forward drift

		const FVector WorldHip[2] = {
			AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[LeftHip], WorldRoot),
			AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[RightHip], WorldRoot),
		};
		const FVector WorldKnee[2] = {
			AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[LeftKnee], WorldRoot),
			AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[RightKnee], WorldRoot),
		};
		const FVector WorldAnkle[2] = {
			AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[LeftAnkle], WorldRoot),
			AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[RightAnkle], WorldRoot),
		};
		const FVector WorldFoot[2] = {
			AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[LeftFoot], WorldRoot),
			AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[RightFoot], WorldRoot),
		};

		bool bIsLocked[2];
		for (int32 FootSlot = 0; FootSlot < 2; ++FootSlot)
		{
			bIsLocked[FootSlot] = ContactDetector.Update(FootSlot, WorldFoot[FootSlot]);
			SlidingMetricBefore.AccumulateSample(FootSlot, WorldFoot[FootSlot], bIsLocked[FootSlot]);
		}

		FVector CorrectedKnee[2];
		FVector CorrectedAnkle[2];
		FootLockIK.Solve(ContactDetector, LeftChain, RightChain, WorldHip, WorldKnee, WorldAnkle,
			DeltaSeconds, CorrectedKnee, CorrectedAnkle);

		for (int32 FootSlot = 0; FootSlot < 2; ++FootSlot)
		{
			SlidingMetricAfter.AccumulateSample(FootSlot, CorrectedAnkle[FootSlot], bIsLocked[FootSlot]);

			if (CorrectedAnkle[FootSlot].ContainsNaN() || CorrectedKnee[FootSlot].ContainsNaN())
			{
				AddError(FString::Printf(
					TEXT("NaN detected at step %d, foot %d (B4 post-process smoke test)."),
					Step, FootSlot));
				return false;
			}
		}

		IdleMoveBlender.Tick(0.0f, 0.033f, DeltaSeconds);
		if (!FMath::IsFinite(IdleMoveBlender.GetMoveWeight()))
		{
			AddError(FString::Printf(
				TEXT("Non-finite move blend weight at step %d."), Step));
			return false;
		}
	}

	TestTrue(TEXT("60s rollout completes without NaN/Inf"), true);

	// Foot-sliding should be measurably lower after IK than the raw
	// FK-only pose's ankle "sliding" measured at the same contact
	// frames (spec §5 acceptance evidence — this smoke test provides a
	// synthetic lower bound; real numbers come from the reference
	// bundle rollout, see README).
	if (SlidingMetricBefore.GetSampleCount() > 0 && SlidingMetricAfter.GetSampleCount() > 0)
	{
		TestTrue(TEXT("Foot-lock IK reduces measured foot-sliding vs raw FK pose"),
			SlidingMetricAfter.ComputeAverageSlidingMeters()
				<= SlidingMetricBefore.ComputeAverageSlidingMeters());
	}
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
