// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorNormalizer.h"
#include "AInimatorManifest.h"
#include "AInimatorNormStats.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace AInimatorNormalizerTestHelpers
{
	/** Builds a minimal 1-bone manifest + matching stats so the test
	 *  focuses purely on the normalization arithmetic (parity-critical
	 *  path), not on full 22-bone bookkeeping. */
	FAInimatorManifest MakeOneBoneManifest()
	{
		FAInimatorManifest Manifest;
		Manifest.BundleVersion = TEXT("A7.0");
		Manifest.StateChannels = 136;
		Manifest.NumBones = 1;
		Manifest.RotationChannelsPerBone = 6;
		Manifest.RootLocalMotionChannels = 4;
		Manifest.ControlChannels = 2;
		Manifest.ControlLayout = {TEXT("vx"), TEXT("vz")};
		Manifest.PhaseChannels = 0;
		Manifest.PromptEmbChannels = 0;
		Manifest.ContextFrames = 1;
		Manifest.OutputLayout = TEXT("bone_delta|global_delta");
		Manifest.CoordSystem = TEXT("Y-up right-handed");
		Manifest.bIsFullyParsed = true;
		return Manifest;
	}

	FAInimatorNormStats MakeStatsForOneBoneManifest()
	{
		FAInimatorNormStats Stats;
		Stats.BoneMean = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
		Stats.BoneStd = {2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
		Stats.GlobalMean = {0.f, 0.f, 0.f, 0.f};
		Stats.GlobalStd = {1.f, 1.f, 1.f, 1.f};
		Stats.DeltaBoneMean = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
		Stats.DeltaBoneStd = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
		Stats.DeltaGlobalMean = {1.f, 1.f, 1.f, 1.f};
		Stats.DeltaGlobalStd = {1.f, 1.f, 1.f, 1.f};
		Stats.ControlMean = {0.f, 0.f};
		Stats.ControlStd = {0.1f, 0.2f};
		Stats.ControlStatChannels = {TEXT("vx"), TEXT("vz")};
		Stats.bIsFullyParsed = true;
		return Stats;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorNormalizerBoneRoundTripTest,
	"AInimator.Normalizer.BoneNormalizeDenormalizeRoundTrip",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorNormalizerBoneRoundTripTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorNormalizerTestHelpers;
	FAInimatorManifest Manifest = MakeOneBoneManifest();
	FAInimatorNormStats Stats = MakeStatsForOneBoneManifest();
	FNormalizer Normalizer(Manifest, Stats);

	const TArray<float> RawBoneFrame = {2.f, 4.f, -2.f, 0.f, 1.f, 3.f};
	TArray<float> Normalized;
	Normalizer.NormalizeBoneFrame(RawBoneFrame, Normalized);

	// (x - 0) / 2 for every channel with mean=0, std=2.
	TestEqual(TEXT("Normalized[0]"), Normalized[0], 1.0f);
	TestEqual(TEXT("Normalized[1]"), Normalized[1], 2.0f);
	TestEqual(TEXT("Normalized[2]"), Normalized[2], -1.0f);
	TestEqual(TEXT("Normalized[5]"), Normalized[5], 1.5f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorNormalizerControlAimPassthroughTest,
	"AInimator.Normalizer.ControlAimPassthroughUnchanged",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorNormalizerControlAimPassthroughTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorNormalizerTestHelpers;
	FAInimatorManifest Manifest = MakeOneBoneManifest();
	Manifest.ControlChannels = 4;
	Manifest.ControlLayout = {TEXT("vx"), TEXT("vz"), TEXT("aim_x"), TEXT("aim_z")};
	FAInimatorNormStats Stats = MakeStatsForOneBoneManifest();
	// control.channels only ever covers vx,vz per the contract, even
	// when the manifest declares 4 control_channels.
	FNormalizer Normalizer(Manifest, Stats);

	const TArray<float> RawControl = {0.1f, 0.4f, 0.6f, -0.8f};
	TArray<float> Normalized;
	Normalizer.NormalizeControlVector(RawControl, Normalized);

	TestEqual(TEXT("vx normalized"), Normalized[0], 1.0f);   // 0.1/0.1
	TestEqual(TEXT("vz normalized"), Normalized[1], 2.0f);   // 0.4/0.2
	TestEqual(TEXT("aim_x passthrough"), Normalized[2], 0.6f);
	TestEqual(TEXT("aim_z passthrough"), Normalized[3], -0.8f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorNormalizerDeltaDenormalizeTest,
	"AInimator.Normalizer.DeltaDenormalizeMatchesManualMath",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorNormalizerDeltaDenormalizeTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorNormalizerTestHelpers;
	FAInimatorManifest Manifest = MakeOneBoneManifest();
	FAInimatorNormStats Stats = MakeStatsForOneBoneManifest();
	FNormalizer Normalizer(Manifest, Stats);

	const TArray<float> NormalizedGlobalDelta = {1.f, 0.f, -1.f, 2.f};
	TArray<float> RawGlobalDelta;
	Normalizer.DenormalizeGlobalDelta(NormalizedGlobalDelta, RawGlobalDelta);

	// value * std + mean, with DeltaGlobalStd=1, DeltaGlobalMean=1.
	TestEqual(TEXT("dForward"), RawGlobalDelta[0], 2.0f);
	TestEqual(TEXT("dLateral"), RawGlobalDelta[1], 1.0f);
	TestEqual(TEXT("dHeight"), RawGlobalDelta[2], 0.0f);
	TestEqual(TEXT("dYaw"), RawGlobalDelta[3], 3.0f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorNormalizerEpsilonFloorTest,
	"AInimator.Normalizer.StdFlooredByEpsilon",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorNormalizerEpsilonFloorTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorNormalizerTestHelpers;
	FAInimatorManifest Manifest = MakeOneBoneManifest();
	FAInimatorNormStats Stats = MakeStatsForOneBoneManifest();
	// Simulate a near-zero-variance locked bone (e.g. a fixed foot
	// bone in the reference bundle, std ~= 1e-5).
	Stats.BoneStd = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
	FNormalizer Normalizer(Manifest, Stats);

	const TArray<float> RawBoneFrame = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
	TArray<float> Normalized;
	Normalizer.NormalizeBoneFrame(RawBoneFrame, Normalized);

	// Must not be Inf/NaN: std is floored to NormalizationEpsilon.
	for (float Value : Normalized)
	{
		TestTrue(TEXT("Finite after epsilon floor"), FMath::IsFinite(Value));
	}
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
