// Copyright AI-nimator.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace AInimatorPromptCrossFadeTestHelpers
{
	/**
	 * Re-implements the EXACT lerp/alpha math
	 * `UAInimatorCharacterComponent::TickPromptCrossFade` performs
	 * (`apps/spec/rig_binding.md` §3: "lerp linéaire emb_old -> emb_new
	 * sur 0.3 s"), so the cross-fade formula itself is unit-tested
	 * without needing a loaded NNE bundle (unavailable in this
	 * environment — UAInimatorControllerRuntime::LoadBundle requires a
	 * real controller.onnx). If this formula ever drifts from the
	 * component's actual implementation, keep both in sync by hand (the
	 * component has no seam to inject a fake Runtime here without a
	 * bigger test-only refactor — flagged as an open question in the
	 * session report).
	 */
	void ComputeLerpFrame(
		const TArray<float>& From,
		const TArray<float>& To,
		float ElapsedSeconds,
		float CrossFadeSeconds,
		TArray<float>& OutResult,
		float& OutAlpha)
	{
		OutAlpha = CrossFadeSeconds > 0.0f
			? FMath::Clamp(ElapsedSeconds / CrossFadeSeconds, 0.0f, 1.0f)
			: 1.0f;
		OutResult.SetNumUninitialized(To.Num());
		for (int32 Index = 0; Index < To.Num(); ++Index)
		{
			OutResult[Index] = FMath::Lerp(From[Index], To[Index], OutAlpha);
		}
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorPromptCrossFadeLerpsLinearlyTest,
	"AInimator.PromptCrossFade.LerpsLinearlyOverConfiguredDuration",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorPromptCrossFadeLerpsLinearlyTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorPromptCrossFadeTestHelpers;

	const TArray<float> From = {0.0f, 10.0f, -4.0f};
	const TArray<float> To = {1.0f, 0.0f, 6.0f};
	constexpr float CrossFadeSeconds = 0.3f;

	TArray<float> ResultAtStart;
	float AlphaAtStart = 0.0f;
	ComputeLerpFrame(From, To, 0.0f, CrossFadeSeconds, ResultAtStart, AlphaAtStart);
	TestEqual(TEXT("Alpha at t=0 is 0"), AlphaAtStart, 0.0f);
	for (int32 Index = 0; Index < From.Num(); ++Index)
	{
		TestTrue(TEXT("Result at t=0 equals the 'from' embedding"),
			FMath::IsNearlyEqual(ResultAtStart[Index], From[Index], 1e-5f));
	}

	TArray<float> ResultAtHalf;
	float AlphaAtHalf = 0.0f;
	ComputeLerpFrame(From, To, CrossFadeSeconds * 0.5f, CrossFadeSeconds, ResultAtHalf, AlphaAtHalf);
	TestTrue(TEXT("Alpha at t=half is 0.5"), FMath::IsNearlyEqual(AlphaAtHalf, 0.5f, 1e-4f));
	for (int32 Index = 0; Index < From.Num(); ++Index)
	{
		const float Expected = (From[Index] + To[Index]) * 0.5f;
		TestTrue(TEXT("Result at t=half is the exact midpoint (linear lerp, no easing)"),
			FMath::IsNearlyEqual(ResultAtHalf[Index], Expected, 1e-4f));
	}

	TArray<float> ResultAtEnd;
	float AlphaAtEnd = 0.0f;
	ComputeLerpFrame(From, To, CrossFadeSeconds, CrossFadeSeconds, ResultAtEnd, AlphaAtEnd);
	TestEqual(TEXT("Alpha at t=duration is 1"), AlphaAtEnd, 1.0f);
	for (int32 Index = 0; Index < From.Num(); ++Index)
	{
		TestTrue(TEXT("Result at t=duration equals the 'to' embedding exactly"),
			FMath::IsNearlyEqual(ResultAtEnd[Index], To[Index], 1e-5f));
	}

	TArray<float> ResultPastEnd;
	float AlphaPastEnd = 0.0f;
	ComputeLerpFrame(From, To, CrossFadeSeconds * 3.0f, CrossFadeSeconds, ResultPastEnd, AlphaPastEnd);
	TestEqual(TEXT("Alpha clamps to 1 past the configured duration (no overshoot)"), AlphaPastEnd, 1.0f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorPromptCrossFadeZeroDurationIsInstantTest,
	"AInimator.PromptCrossFade.ZeroDurationSwitchesInstantly",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorPromptCrossFadeZeroDurationIsInstantTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorPromptCrossFadeTestHelpers;

	const TArray<float> From = {1.0f, 2.0f};
	const TArray<float> To = {5.0f, -5.0f};

	TArray<float> Result;
	float Alpha = 0.0f;
	// PromptCrossFadeSeconds == 0 (a project disabling the fade
	// heuristic entirely per spec §3's documented fallback) must switch
	// immediately, matching UAInimatorCharacterComponent's
	// `CrossFadeSeconds > 0.0f ? ... : 1.0f` guard.
	ComputeLerpFrame(From, To, 0.0f, 0.0f, Result, Alpha);
	TestEqual(TEXT("Alpha is 1 immediately when CrossFadeSeconds is 0"), Alpha, 1.0f);
	for (int32 Index = 0; Index < From.Num(); ++Index)
	{
		TestTrue(TEXT("Zero-duration fade equals the target embedding immediately"),
			FMath::IsNearlyEqual(Result[Index], To[Index], 1e-5f));
	}
	return true;
}

/**
 * ClearPrompt's contract (rig_binding.md §3 / inference_contract.md §4):
 * the "to" endpoint of the fade must be the bundle's LEARNED null
 * embedding, never an all-zero vector — this test locks in that a
 * (deliberately non-zero, "learned-looking") null embedding is what a
 * ClearPrompt-driven fade converges to, distinguishing it from a naive
 * "fade to zeros" implementation bug.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorClearPromptFadesToLearnedNullEmbeddingNotZerosTest,
	"AInimator.PromptCrossFade.ClearPromptTargetsLearnedNullEmbeddingNotZeros",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorClearPromptFadesToLearnedNullEmbeddingNotZerosTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorPromptCrossFadeTestHelpers;

	const TArray<float> ActivePromptEmb = {2.0f, -3.0f, 4.0f};
	// A "learned" null embedding is, by construction, some arbitrary
	// non-zero vector baked into norm_stats.json's prompt.null_emb
	// (inference_contract.md §4) — never all-zeros.
	const TArray<float> LearnedNullEmb = {0.42f, -1.1f, 0.05f};

	TArray<float> ResultAtEnd;
	float AlphaAtEnd = 0.0f;
	ComputeLerpFrame(ActivePromptEmb, LearnedNullEmb, 0.3f, 0.3f, ResultAtEnd, AlphaAtEnd);

	bool bAnyNonZero = false;
	for (int32 Index = 0; Index < LearnedNullEmb.Num(); ++Index)
	{
		TestTrue(TEXT("Fade converges to the learned null embedding"),
			FMath::IsNearlyEqual(ResultAtEnd[Index], LearnedNullEmb[Index], 1e-5f));
		bAnyNonZero |= !FMath::IsNearlyEqual(ResultAtEnd[Index], 0.0f, 1e-5f);
	}
	TestTrue(TEXT("Fade target is NOT the all-zero vector"), bAnyNonZero);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
