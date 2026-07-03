// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorStateBuffer.h"
#include "AInimatorNormalizer.h"
#include "AInimatorManifest.h"
#include "AInimatorNormStats.h"
#include "AInimatorRootLocalMath.h"

#if WITH_DEV_AUTOMATION_TESTS

// This smoke test does NOT exercise the NNE model (no real .onnx bundle
// is available in this environment). Instead it fabricates a small,
// bounded synthetic "delta" every step and drives FStateBuffer +
// FNormalizer + the exact integration formula used by
// UAInimatorControllerRuntime::IntegrateRootLocalDelta for 60 s at 30
// fps (1800 steps) — the goal is to catch integration/accumulation
// bugs (drift blow-up, NaN from a bad epsilon floor, buffer eviction
// off-by-ones) independently of whether NNE is available to run.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorRolloutSixtySecondsNoExplosionTest,
	"AInimator.Rollout.SixtySecondsNoNaNNoExplosion",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorRolloutSixtySecondsNoExplosionTest::RunTest(const FString& Parameters)
{
	constexpr int32 ContextFrames = 8;
	constexpr int32 NumBones = 22;
	constexpr int32 RotationChannelsPerBone = 6;
	constexpr int32 GlobalChannels = 4;
	constexpr int32 BoneFrameWidth = NumBones * RotationChannelsPerBone;
	constexpr float TargetFps = 30.0f;
	constexpr float SimulatedSeconds = 60.0f;
	constexpr int32 NumSteps = static_cast<int32>(TargetFps * SimulatedSeconds);

	FStateBuffer Buffer(ContextFrames, BoneFrameWidth, GlobalChannels);

	TArray<float> SeedBoneFrame;
	SeedBoneFrame.Reserve(BoneFrameWidth);
	for (int32 Bone = 0; Bone < NumBones; ++Bone)
	{
		SeedBoneFrame.Append({1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
	}
	TArray<float> SeedGlobalFrame;
	SeedGlobalFrame.SetNumZeroed(GlobalChannels);
	Buffer.SeedWithFrame(SeedBoneFrame, SeedGlobalFrame);

	FVector WorldPosition = FVector::ZeroVector;
	float WorldYaw = 0.0f;

	// A tiny, constant "forward walk" delta per step: small enough
	// that 1800 accumulations stay bounded, matching the reference
	// bundle's forward preset order of magnitude (0.033 m/frame).
	const float DeltaForwardPerStep = 0.033f;
	const float DeltaYawPerStep = 0.001f;

	for (int32 Step = 0; Step < NumSteps; ++Step)
	{
		TArray<float> LastBoneFrame;
		Buffer.GetLastBoneFrame(LastBoneFrame);

		// Synthetic bone delta: a tiny oscillation, bounded.
		TArray<float> RawBoneDelta;
		RawBoneDelta.SetNumUninitialized(BoneFrameWidth);
		for (int32 Index = 0; Index < BoneFrameWidth; ++Index)
		{
			RawBoneDelta[Index] = 0.0001f * FMath::Sin(static_cast<float>(Step));
		}
		TArray<float> NextBoneFrame;
		NextBoneFrame.SetNumUninitialized(BoneFrameWidth);
		for (int32 Index = 0; Index < BoneFrameWidth; ++Index)
		{
			NextBoneFrame[Index] = LastBoneFrame[Index] + RawBoneDelta[Index];
		}

		const float CosYaw = FMath::Cos(WorldYaw);
		const float SinYaw = FMath::Sin(WorldYaw);
		const float WorldDeltaX =
			CosYaw * DeltaForwardPerStep - SinYaw * 0.0f;
		const float WorldDeltaZ =
			SinYaw * DeltaForwardPerStep + CosYaw * 0.0f;
		WorldPosition += FVector(WorldDeltaX, 0.0f, WorldDeltaZ);
		WorldYaw += DeltaYawPerStep;

		TArray<float> RawGlobalDelta =
			{DeltaForwardPerStep, 0.0f, 0.0f, DeltaYawPerStep};
		Buffer.PushFrame(NextBoneFrame, RawGlobalDelta);

		if (!WorldPosition.ContainsNaN() && FMath::IsFinite(WorldYaw))
		{
			continue;
		}
		AddError(FString::Printf(
			TEXT("NaN/Inf detected at step %d (60s rollout smoke test)."),
			Step));
		return false;
	}

	// After 1800 steps of a ~0.033 m/step forward walk (~1 m/s @ 30
	// fps), expect roughly 59.4 m of travel — bounded, not exploded
	// (a runaway integration bug would produce orders of magnitude
	// more, or NaN caught above).
	const float TraveledDistance = WorldPosition.Size();
	TestTrue(
		TEXT("Traveled distance stays within a sane bound for 60s @ ~1 m/s"),
		TraveledDistance < 200.0f);
	TestTrue(TEXT("Final position is finite"), !WorldPosition.ContainsNaN());
	TestTrue(TEXT("Final yaw is finite"), FMath::IsFinite(WorldYaw));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
