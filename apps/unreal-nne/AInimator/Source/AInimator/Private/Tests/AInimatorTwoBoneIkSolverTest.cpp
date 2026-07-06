// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorTwoBoneIkSolver.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorTwoBoneIkSolverReachableTargetTest,
	"AInimator.TwoBoneIkSolver.ReachesReachableTargetExactly",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorTwoBoneIkSolverReachableTargetTest::RunTest(const FString& Parameters)
{
	const FVector Root(0.0f, 0.0f, 0.0f);
	const FVector Mid(0.0f, -0.4f, 0.0f);   // current knee, straight down.
	const FVector Effector(0.0f, -0.8f, 0.0f); // current ankle.
	const float UpperLength = 0.4f;
	const float LowerLength = 0.4f;

	// A target well within reach (bent leg): distance 0.6 < 0.8 max reach.
	const FVector Target(0.3f, -0.5f, 0.0f);

	const AInimatorTwoBoneIkSolver::FResult Result = AInimatorTwoBoneIkSolver::Solve(
		Root, Mid, Effector, Target, UpperLength, LowerLength);

	TestTrue(TEXT("Reachable target reports TargetReachable=true"), Result.bTargetReachable);
	TestTrue(TEXT("Effector reaches the target within tolerance"),
		(Result.EffectorPosition - Target).Size() < 1e-3f);

	// Triangle inequality sanity: root->mid and mid->effector segment
	// lengths match the chain's bone lengths.
	TestTrue(TEXT("Upper segment length preserved"),
		FMath::IsNearlyEqual((Result.MidPosition - Root).Size(), UpperLength, 1e-3f));
	TestTrue(TEXT("Lower segment length preserved"),
		FMath::IsNearlyEqual((Result.EffectorPosition - Result.MidPosition).Size(), LowerLength, 1e-3f));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorTwoBoneIkSolverUnreachableTargetTest,
	"AInimator.TwoBoneIkSolver.ClampsUnreachableTargetAndReportsUnreachable",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorTwoBoneIkSolverUnreachableTargetTest::RunTest(const FString& Parameters)
{
	const FVector Root(0.0f, 0.0f, 0.0f);
	const FVector Mid(0.0f, -0.4f, 0.0f);
	const FVector Effector(0.0f, -0.8f, 0.0f);
	const float UpperLength = 0.4f;
	const float LowerLength = 0.4f;

	// Target far beyond max reach (0.8m): fully extends toward it
	// instead of teleporting exactly onto it.
	const FVector Target(0.0f, -5.0f, 0.0f);

	const AInimatorTwoBoneIkSolver::FResult Result = AInimatorTwoBoneIkSolver::Solve(
		Root, Mid, Effector, Target, UpperLength, LowerLength);

	TestFalse(TEXT("Unreachable target reports TargetReachable=false"), Result.bTargetReachable);

	const float DistanceFromRoot = (Result.EffectorPosition - Root).Size();
	TestTrue(TEXT("Chain fully extends (does not overreach past max reach)"),
		DistanceFromRoot <= UpperLength + LowerLength + 1e-3f);
	TestTrue(TEXT("Chain extends close to its maximum reach toward the target"),
		DistanceFromRoot > (UpperLength + LowerLength) * 0.9f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorTwoBoneIkSolverDegenerateTargetAtRootTest,
	"AInimator.TwoBoneIkSolver.DegenerateTargetAtRootFallsBackToCurrentPose",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorTwoBoneIkSolverDegenerateTargetAtRootTest::RunTest(const FString& Parameters)
{
	const FVector Root(1.0f, 2.0f, 3.0f);
	const FVector Mid(1.0f, 1.6f, 3.0f);
	const FVector Effector(1.0f, 1.2f, 3.0f);

	const AInimatorTwoBoneIkSolver::FResult Result = AInimatorTwoBoneIkSolver::Solve(
		Root, Mid, Effector, /*Target=*/Root, 0.4f, 0.4f);

	TestFalse(TEXT("Degenerate (target==root) reports unreachable"), Result.bTargetReachable);
	TestEqual(TEXT("Degenerate case preserves current mid position"),
		Result.MidPosition, Mid);
	TestEqual(TEXT("Degenerate case preserves current effector position"),
		Result.EffectorPosition, Effector);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
