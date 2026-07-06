// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorTextToControlResolver.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace AInimatorTextToControlTestHelpers
{
	constexpr float Walk = 0.033f;
	constexpr float Run = 0.1f;
	const float Diag = FMath::Sqrt(0.5f); // 1/sqrt(2)
	constexpr float Kinda = 1e-4f; // parity tolerance; reference test uses abs=1e-9 in double

	/** Mirrors test_resolves_expected_vector's parametrize table verbatim
	 *  (test/apps/test_text_to_control.py) — same phrases, same expected
	 *  (vx, vz). */
	struct FCase
	{
		const TCHAR* Text;
		float Vx;
		float Vz;
	};
}

// --- test_resolves_expected_vector (9 parametrized cases). ---
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorTextToControlParityTest,
	"AInimator.TextToControl.ParityWithPythonReference",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorTextToControlParityTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorTextToControlTestHelpers;

	const FCase Cases[] = {
		{TEXT("cours vers la gauche"), -Run, 0.0f},
		{TEXT("run left"), -Run, 0.0f},
		{TEXT("walk forward"), 0.0f, Walk},
		{TEXT("avance"), 0.0f, Walk},
		{TEXT("recule lentement"), 0.0f, -0.015f},
		{TEXT("sprint"), 0.0f, 0.15f},
		{TEXT("marche a droite"), Walk, 0.0f},
		{TEXT("run forward and left"), -Run * Diag, Run * Diag},
		{TEXT("cours lentement"), 0.0f, Run},
	};

	for (const FCase& Case : Cases)
	{
		const TOptional<FAInimatorResolvedControl> Result = FTextToControlResolver::Resolve(Case.Text);
		if (!TestTrue(FString::Printf(TEXT("'%s' should resolve"), Case.Text), Result.IsSet()))
		{
			continue;
		}
		TestEqual(FString::Printf(TEXT("'%s' vx"), Case.Text), Result->Vx, Case.Vx, Kinda);
		TestEqual(FString::Printf(TEXT("'%s' vz"), Case.Text), Result->Vz, Case.Vz, Kinda);
	}
	return true;
}

// --- test_stop_always_wins_and_zeroes_control. ---
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorTextToControlStopWinsTest,
	"AInimator.TextToControl.StopAlwaysWinsAndZeroesControl",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorTextToControlStopWinsTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorTextToControlTestHelpers;

	const TOptional<FAInimatorResolvedControl> Result =
		FTextToControlResolver::Resolve(TEXT("cours vite et arrete a gauche"));
	if (!TestTrue(TEXT("stop-family phrase should resolve"), Result.IsSet()))
	{
		return false;
	}
	TestEqual(TEXT("stop wins: vx"), Result->Vx, 0.0f, Kinda);
	TestEqual(TEXT("stop wins: vz"), Result->Vz, 0.0f, Kinda);
	TestEqual(TEXT("stop wins: aimX"), Result->AimX, 0.0f, Kinda);
	TestEqual(TEXT("stop wins: aimZ"), Result->AimZ, 1.0f, Kinda);
	return true;
}

// --- test_aim_follows_movement_direction. ---
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorTextToControlAimFollowsMovementTest,
	"AInimator.TextToControl.AimFollowsMovementDirection",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorTextToControlAimFollowsMovementTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorTextToControlTestHelpers;

	const TOptional<FAInimatorResolvedControl> Result = FTextToControlResolver::Resolve(TEXT("run left"));
	if (!TestTrue(TEXT("'run left' should resolve"), Result.IsSet()))
	{
		return false;
	}
	TestEqual(TEXT("aim follows movement: aimX"), Result->AimX, -1.0f, Kinda);
	TestEqual(TEXT("aim follows movement: aimZ"), Result->AimZ, 0.0f, Kinda);
	return true;
}

// --- test_unresolvable_returns_none (3 parametrized cases). ---
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorTextToControlUnresolvableTest,
	"AInimator.TextToControl.UnresolvableReturnsNone",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorTextToControlUnresolvableTest::RunTest(const FString& Parameters)
{
	const TCHAR* UnresolvableCases[] = {
		TEXT("bonjour tout le monde"),
		TEXT(""),
		TEXT("gauche droite"),
	};
	for (const TCHAR* Text : UnresolvableCases)
	{
		const TOptional<FAInimatorResolvedControl> Result = FTextToControlResolver::Resolve(Text);
		TestFalse(FString::Printf(TEXT("'%s' should be unresolvable"), Text), Result.IsSet());
	}
	return true;
}

// --- test_accents_are_stripped. ---
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorTextToControlAccentsStrippedTest,
	"AInimator.TextToControl.AccentsAreStripped",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorTextToControlAccentsStrippedTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorTextToControlTestHelpers;

	// "arrête-toi" resolves like "arrete" (accent-insensitive).
	const TOptional<FAInimatorResolvedControl> Result = FTextToControlResolver::Resolve(TEXT("arrête-toi"));
	if (!TestTrue(TEXT("'arrête-toi' should resolve"), Result.IsSet()))
	{
		return false;
	}
	TestEqual(TEXT("accented stop: vx"), Result->Vx, 0.0f, Kinda);
	TestEqual(TEXT("accented stop: vz"), Result->Vz, 0.0f, Kinda);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
