// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorControlPreset.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorControlPresetVxVzOnlyTest,
	"AInimator.ControlPreset.BuildsVxVzVector",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorControlPresetVxVzOnlyTest::RunTest(const FString& Parameters)
{
	UAInimatorControlPreset* Preset =
		NewObject<UAInimatorControlPreset>(GetTransientPackage());
	Preset->HydrateFromRaw(
		TEXT("forward"), /*Vx=*/0.0f, /*Vz=*/0.033f,
		/*bHasAim=*/false, 0.0f, 0.0f, FString(), {});

	TArray<float> ControlVector;
	const bool bBuilt =
		Preset->BuildControlVector({TEXT("vx"), TEXT("vz")}, ControlVector);

	TestTrue(TEXT("Build succeeds for vx/vz layout"), bBuilt);
	TestEqual(TEXT("vx"), ControlVector[0], 0.0f);
	TestEqual(TEXT("vz"), ControlVector[1], 0.033f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorControlPresetMissingAimFailsTest,
	"AInimator.ControlPreset.RejectsAimLayoutWithoutAimData",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorControlPresetMissingAimFailsTest::RunTest(const FString& Parameters)
{
	UAInimatorControlPreset* Preset =
		NewObject<UAInimatorControlPreset>(GetTransientPackage());
	Preset->HydrateFromRaw(
		TEXT("forward"), 0.0f, 0.033f, /*bHasAim=*/false, 0.0f, 0.0f,
		FString(), {});

	TArray<float> ControlVector;
	const bool bBuilt = Preset->BuildControlVector(
		{TEXT("vx"), TEXT("vz"), TEXT("aim_x"), TEXT("aim_z")}, ControlVector);

	TestFalse(
		TEXT("Preset without aim data must refuse a 4-channel layout"), bBuilt);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorControlPresetWithAimTest,
	"AInimator.ControlPreset.BuildsFourChannelVectorWithAim",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorControlPresetWithAimTest::RunTest(const FString& Parameters)
{
	UAInimatorControlPreset* Preset =
		NewObject<UAInimatorControlPreset>(GetTransientPackage());
	Preset->HydrateFromRaw(
		TEXT("forward_aim_right"), 0.0f, 0.033f, /*bHasAim=*/true,
		/*AimX=*/1.0f, /*AimZ=*/0.0f, FString(), {});

	TArray<float> ControlVector;
	const bool bBuilt = Preset->BuildControlVector(
		{TEXT("vx"), TEXT("vz"), TEXT("aim_x"), TEXT("aim_z")}, ControlVector);

	TestTrue(TEXT("Build succeeds with aim data present"), bBuilt);
	TestEqual(TEXT("aim_x"), ControlVector[2], 1.0f);
	TestEqual(TEXT("aim_z"), ControlVector[3], 0.0f);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
