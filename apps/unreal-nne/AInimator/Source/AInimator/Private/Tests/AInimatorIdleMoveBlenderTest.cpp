// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorIdleMoveBlender.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorIdleMoveBlenderStaysMoveWhileAboveThresholdTest,
	"AInimator.IdleMoveBlender.StaysAtFullMoveWeightWhileAboveThreshold",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorIdleMoveBlenderStaysMoveWhileAboveThresholdTest::RunTest(const FString& Parameters)
{
	FIdleMoveBlender Blender;
	// Starts at full move weight by construction (no time has passed
	// below threshold yet).
	TestEqual(TEXT("Initial weight is full move"), Blender.GetMoveWeight(), 1.0f);

	const float Weight = Blender.Tick(/*Vx=*/0.0f, /*Vz=*/0.033f, /*DeltaSeconds=*/0.033f);
	TestEqual(TEXT("Stays at full move weight while above threshold"), Weight, 1.0f);
	TestFalse(TEXT("Target is not idle while moving"), Blender.IsTargetIdle());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorIdleMoveBlenderDelaysIdleEntryTest,
	"AInimator.IdleMoveBlender.DelaysIdleEntryUntilIdleDelayElapses",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorIdleMoveBlenderDelaysIdleEntryTest::RunTest(const FString& Parameters)
{
	FIdleMoveBlender Blender(
		/*MoveSpeedThreshold=*/0.005f,
		/*IdleDelaySeconds=*/0.25f,
		/*CrossFadeSeconds=*/0.2f);

	// 0.2s of below-threshold control: not yet 0.25s -> still "move"
	// target, cross-fade has not started.
	for (int32 Step = 0; Step < 6; ++Step) // 6 * 0.033 ~= 0.2s
	{
		Blender.Tick(0.0f, 0.0f, 0.033f);
	}
	TestFalse(TEXT("Target still not idle before the 0.25s delay elapses"),
		Blender.IsTargetIdle());
	TestEqual(TEXT("Move weight unchanged before idle delay elapses"),
		Blender.GetMoveWeight(), 1.0f);

	// A few more frames push past 0.25s total -> now targets idle and
	// the cross-fade begins (weight starts dropping below 1).
	for (int32 Step = 0; Step < 3; ++Step)
	{
		Blender.Tick(0.0f, 0.0f, 0.033f);
	}
	TestTrue(TEXT("Target becomes idle once the delay elapses"),
		Blender.IsTargetIdle());
	TestTrue(TEXT("Move weight has started dropping"), Blender.GetMoveWeight() < 1.0f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorIdleMoveBlenderCrossFadeCompletesTest,
	"AInimator.IdleMoveBlender.CrossFadeReachesZeroAfterFullDuration",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorIdleMoveBlenderCrossFadeCompletesTest::RunTest(const FString& Parameters)
{
	FIdleMoveBlender Blender(
		/*MoveSpeedThreshold=*/0.005f,
		/*IdleDelaySeconds=*/0.0f, // no delay, isolate the cross-fade itself
		/*CrossFadeSeconds=*/0.2f);

	// Drive well past the cross-fade duration.
	float Weight = 1.0f;
	for (int32 Step = 0; Step < 20; ++Step) // 20 * 0.033 ~= 0.66s > 0.2s
	{
		Weight = Blender.Tick(0.0f, 0.0f, 0.033f);
	}
	TestEqual(TEXT("Move weight fully settles to idle (0)"), Weight, 0.0f);
	TestEqual(TEXT("Idle weight is the complement"), Blender.GetIdleWeight(), 1.0f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorIdleMoveBlenderResumesMoveImmediatelyTest,
	"AInimator.IdleMoveBlender.ResumingMoveResetsIdleDelayImmediately",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorIdleMoveBlenderResumesMoveImmediatelyTest::RunTest(const FString& Parameters)
{
	FIdleMoveBlender Blender(0.005f, 0.25f, 0.2f);

	for (int32 Step = 0; Step < 10; ++Step)
	{
		Blender.Tick(0.0f, 0.0f, 0.033f);
	}
	TestTrue(TEXT("Idle after enough below-threshold time"), Blender.IsTargetIdle());

	Blender.Tick(0.0f, 0.5f, 0.033f);
	TestFalse(TEXT("A single above-threshold frame resets target to move"),
		Blender.IsTargetIdle());
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
