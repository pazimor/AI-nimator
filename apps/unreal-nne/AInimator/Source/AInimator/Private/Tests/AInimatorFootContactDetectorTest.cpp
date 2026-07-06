// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorFootContactDetector.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootContactDetectorEntersContactImmediatelyTest,
	"AInimator.FootContactDetector.EntersContactImmediatelyWhenBelowThresholds",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootContactDetectorEntersContactImmediatelyTest::RunTest(const FString& Parameters)
{
	FFootContactDetector Detector(/*FootCount=*/1);

	// First sample: height below threshold, no previous position yet
	// so speed defaults to 0 (also below threshold) -> contact.
	const bool bContact = Detector.Update(0, FVector(0.0f, 0.02f, 0.0f));
	TestTrue(TEXT("Enters contact on the very first qualifying frame"), bContact);
	TestTrue(TEXT("IsLocked reflects Update's result"), Detector.IsLocked(0));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootContactDetectorRejectsHighOrFastFootTest,
	"AInimator.FootContactDetector.RejectsHighOrFastFoot",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootContactDetectorRejectsHighOrFastFootTest::RunTest(const FString& Parameters)
{
	FFootContactDetector Detector(/*FootCount=*/1);

	// Above the height threshold (0.05m default) -> no contact.
	TestFalse(TEXT("Airborne foot never enters contact"),
		Detector.Update(0, FVector(0.0f, 0.5f, 0.0f)));

	// Low but moving fast planar-wise between two low samples -> no lock.
	Detector.Reset();
	Detector.Update(0, FVector(0.0f, 0.01f, 0.0f));
	const bool bFastContact = Detector.Update(0, FVector(1.0f, 0.01f, 0.0f));
	TestFalse(TEXT("Fast-moving low foot never enters contact"), bFastContact);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootContactDetectorHysteresisExitFramesTest,
	"AInimator.FootContactDetector.HysteresisRequiresConsecutiveExitFrames",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootContactDetectorHysteresisExitFramesTest::RunTest(const FString& Parameters)
{
	FFootContactDetector Detector(
		/*FootCount=*/1,
		/*HeightThreshold=*/0.05f,
		/*SpeedThreshold=*/0.01f,
		/*ExitFrames=*/2,
		/*ReleaseHeightMultiplier=*/1.5f);

	// Frame 0: enter contact (stationary, low).
	TestTrue(TEXT("Frame 0 enters contact"),
		Detector.Update(0, FVector(0.0f, 0.0f, 0.0f)));

	// Frame 1: still low height (< 0.075 release threshold) but now
	// moving fast enough to violate the raw criterion -- one violation,
	// not yet enough to release (ExitFrames=2).
	TestTrue(TEXT("First violating frame stays locked (hysteresis)"),
		Detector.Update(0, FVector(1.0f, 0.02f, 0.0f)));

	// Frame 2: second consecutive violation at the same low height ->
	// releases now.
	TestFalse(TEXT("Second consecutive violating frame releases the lock"),
		Detector.Update(0, FVector(2.0f, 0.02f, 0.0f)));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootContactDetectorImmediateReleaseOnHighHeightTest,
	"AInimator.FootContactDetector.ImmediateReleaseAboveReleaseHeightThreshold",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootContactDetectorImmediateReleaseOnHighHeightTest::RunTest(const FString& Parameters)
{
	FFootContactDetector Detector(
		/*FootCount=*/1,
		/*HeightThreshold=*/0.05f,
		/*SpeedThreshold=*/0.01f,
		/*ExitFrames=*/2,
		/*ReleaseHeightMultiplier=*/1.5f);

	Detector.Update(0, FVector(0.0f, 0.0f, 0.0f));
	TestTrue(TEXT("Locked after first frame"), Detector.IsLocked(0));

	// Height jumps well above 0.075 (1.5x default 0.05) -> releases
	// immediately regardless of ExitFrames.
	const bool bStillLocked = Detector.Update(0, FVector(0.0f, 0.2f, 0.0f));
	TestFalse(TEXT("Releases immediately when clearly airborne"), bStillLocked);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
