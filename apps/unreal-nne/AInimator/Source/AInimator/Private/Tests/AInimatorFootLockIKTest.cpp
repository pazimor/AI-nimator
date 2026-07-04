// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorFootLockIK.h"
#include "AInimatorFootContactDetector.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace AInimatorFootLockIkTestHelpers
{
	FFootLockIK::FLegChain MakeChain(float UpperLength, float LowerLength)
	{
		FFootLockIK::FLegChain Chain;
		Chain.HipBone = 0;
		Chain.KneeBone = 1;
		Chain.AnkleBone = 2;
		Chain.UpperLength = UpperLength;
		Chain.LowerLength = LowerLength;
		return Chain;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootLockIkHoldsCapturedPositionTest,
	"AInimator.FootLockIK.HoldsCapturedPositionWhileLocked",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootLockIkHoldsCapturedPositionTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorFootLockIkTestHelpers;

	FFootContactDetector ContactDetector(/*FootCount=*/2);
	FFootLockIK FootLockIK;

	const FFootLockIK::FLegChain LeftChain = MakeChain(0.4f, 0.4f);
	const FFootLockIK::FLegChain RightChain = MakeChain(0.4f, 0.4f);

	// Frame 0: left foot touches down at (0, 0.02, 0) -> locks and
	// captures that ankle position.
	const FVector Hip[2] = {FVector(0.0f, 0.8f, 0.0f), FVector(0.1f, 0.8f, 0.0f)};
	const FVector Knee[2] = {FVector(0.0f, 0.4f, 0.0f), FVector(0.1f, 0.4f, 0.0f)};
	FVector Ankle[2] = {FVector(0.0f, 0.02f, 0.0f), FVector(1.0f, 0.5f, 0.0f)};

	ContactDetector.Update(0, Ankle[0]);
	ContactDetector.Update(1, Ankle[1]);

	FVector CorrectedKnee[2];
	FVector CorrectedAnkle[2];
	FootLockIK.Solve(ContactDetector, LeftChain, RightChain, Hip, Knee, Ankle,
		/*DeltaSeconds=*/0.033f, CorrectedKnee, CorrectedAnkle);

	const FVector CapturedLeftAnkle = CorrectedAnkle[0];
	TestTrue(TEXT("Left ankle holds near its captured position"),
		(CapturedLeftAnkle - Ankle[0]).Size() < 1e-2f);

	// Frame 1: controller pose drifts the raw ankle slightly (small
	// noise within the leg's articulation) -- corrected ankle should
	// still track close to the ORIGINAL captured position, not the new
	// raw pose.
	Ankle[0] = FVector(0.01f, 0.03f, 0.0f);
	ContactDetector.Update(0, Ankle[0]);
	ContactDetector.Update(1, Ankle[1]);
	FootLockIK.Solve(ContactDetector, LeftChain, RightChain, Hip, Knee, Ankle,
		0.033f, CorrectedKnee, CorrectedAnkle);

	TestTrue(TEXT("Locked foot stays near the originally captured position"),
		(CorrectedAnkle[0] - CapturedLeftAnkle).Size() < 5e-2f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootLockIkClampsLargeDisplacementTest,
	"AInimator.FootLockIK.ReleasesLockWhenDisplacementExceedsClamp",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootLockIkClampsLargeDisplacementTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorFootLockIkTestHelpers;

	FFootContactDetector ContactDetector(
		/*FootCount=*/2,
		/*HeightThreshold=*/0.05f,
		/*SpeedThreshold=*/100.0f); // disable the speed criterion for this test
	FFootLockIK FootLockIK(/*MaxCorrectionMeters=*/0.3f);

	const FFootLockIK::FLegChain LeftChain = MakeChain(0.4f, 0.4f);
	const FFootLockIK::FLegChain RightChain = MakeChain(0.4f, 0.4f);

	const FVector Hip[2] = {FVector(0.0f, 0.8f, 0.0f), FVector(0.1f, 0.8f, 0.0f)};
	const FVector Knee[2] = {FVector(0.0f, 0.4f, 0.0f), FVector(0.1f, 0.4f, 0.0f)};
	FVector Ankle[2] = {FVector(0.0f, 0.02f, 0.0f), FVector(1.0f, 0.5f, 0.0f)};

	ContactDetector.Update(0, Ankle[0]);
	ContactDetector.Update(1, Ankle[1]);
	FVector CorrectedKnee[2];
	FVector CorrectedAnkle[2];
	FootLockIK.Solve(ContactDetector, LeftChain, RightChain, Hip, Knee, Ankle,
		0.033f, CorrectedKnee, CorrectedAnkle);

	// The controller's own pose now moves the raw ankle 0.5m away
	// (> 0.3m clamp) while height stays low -- the lock must release,
	// so the corrected ankle should match the RAW pose, not be pulled
	// back to the stale captured position.
	Ankle[0] = FVector(0.5f, 0.02f, 0.0f);
	ContactDetector.Update(0, Ankle[0]);
	ContactDetector.Update(1, Ankle[1]);
	FootLockIK.Solve(ContactDetector, LeftChain, RightChain, Hip, Knee, Ankle,
		0.033f, CorrectedKnee, CorrectedAnkle);

	TestTrue(TEXT("Clamp releases the lock: corrected ankle matches the raw pose"),
		(CorrectedAnkle[0] - Ankle[0]).Size() < 1e-2f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootLockIkReleaseFadesNotSnapsTest,
	"AInimator.FootLockIK.ReleaseFadesTowardRawPoseOverTime",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootLockIkReleaseFadesNotSnapsTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorFootLockIkTestHelpers;

	FFootContactDetector ContactDetector(/*FootCount=*/2);
	FFootLockIK FootLockIK(
		/*MaxCorrectionMeters=*/0.3f,
		/*ReleaseFadeSeconds=*/0.1f);

	const FFootLockIK::FLegChain LeftChain = MakeChain(0.4f, 0.4f);
	const FFootLockIK::FLegChain RightChain = MakeChain(0.4f, 0.4f);

	const FVector Hip[2] = {FVector(0.0f, 0.8f, 0.0f), FVector(0.1f, 0.8f, 0.0f)};
	const FVector Knee[2] = {FVector(0.0f, 0.4f, 0.0f), FVector(0.1f, 0.4f, 0.0f)};
	FVector Ankle[2] = {FVector(0.0f, 0.02f, 0.0f), FVector(1.0f, 0.5f, 0.0f)};

	// Frame 0: lock the left foot.
	ContactDetector.Update(0, Ankle[0]);
	ContactDetector.Update(1, Ankle[1]);
	FVector CorrectedKnee[2];
	FVector CorrectedAnkle[2];
	FootLockIK.Solve(ContactDetector, LeftChain, RightChain, Hip, Knee, Ankle,
		0.033f, CorrectedKnee, CorrectedAnkle);
	const FVector LockedPosition = CorrectedAnkle[0];

	// Frame 1: raw pose lifts the foot well above the release-height
	// threshold (airborne) -> contact releases; the corrected pose
	// must NOT snap immediately to the raw (now airborne) position —
	// it should still be blended toward the locked position at
	// fade-weight ~1 immediately after release, given DeltaSeconds is
	// small relative to ReleaseFadeSeconds (0.1s).
	Ankle[0] = FVector(0.0f, 0.5f, 0.0f);
	ContactDetector.Update(0, Ankle[0]);
	ContactDetector.Update(1, Ankle[1]);
	FootLockIK.Solve(ContactDetector, LeftChain, RightChain, Hip, Knee, Ankle,
		/*DeltaSeconds=*/0.01f, CorrectedKnee, CorrectedAnkle);

	TestFalse(TEXT("No longer locked after release"), ContactDetector.IsLocked(0));
	TestTrue(TEXT("Immediately after release, corrected pose is still close to the locked position (no snap)"),
		(CorrectedAnkle[0] - LockedPosition).Size() < (Ankle[0] - LockedPosition).Size());

	// After several more frames covering > ReleaseFadeSeconds total,
	// the corrected pose should have fully faded to the raw pose.
	for (int32 Frame = 0; Frame < 20; ++Frame)
	{
		ContactDetector.Update(0, Ankle[0]);
		ContactDetector.Update(1, Ankle[1]);
		FootLockIK.Solve(ContactDetector, LeftChain, RightChain, Hip, Knee, Ankle,
			0.033f, CorrectedKnee, CorrectedAnkle);
	}
	TestTrue(TEXT("Fully faded to the raw pose after the fade duration elapses"),
		(CorrectedAnkle[0] - Ankle[0]).Size() < 1e-3f);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
