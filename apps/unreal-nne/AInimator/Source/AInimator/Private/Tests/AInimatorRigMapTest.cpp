// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorRigMap.h"
#include "AInimatorSmpl22Skeleton.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorRigMapConstructsWithCanonicalEntryCountTest,
	"AInimator.RigMap.ConstructsWith22CanonicalEntries",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorRigMapConstructsWithCanonicalEntryCountTest::RunTest(const FString& Parameters)
{
	UAInimatorRigMap* RigMap = NewObject<UAInimatorRigMap>();
	TestEqual(TEXT("RigMap always has exactly 22 entries"),
		RigMap->Entries.Num(), AInimatorSmpl22Skeleton::NumBones);
	for (int32 Index = 0; Index < AInimatorSmpl22Skeleton::NumBones; ++Index)
	{
		TestTrue(TEXT("Every fresh entry starts unmapped"),
			RigMap->Entries[Index].TargetBoneName == NAME_None);
	}
	TestFalse(TEXT("IsMapped is false for a fresh entry"),
		RigMap->IsMapped(AInimatorSmpl22Skeleton::Pelvis));
	return true;
}

/** Auto-map against a candidate bone list using UE5 Mannequin naming
 *  (thigh_l/calf_l/foot_l/pelvis/spine_0X/clavicle_l/upperarm_l/...)
 *  must resolve at least the core locomotion chain without any manual
 *  editing (spec deliverable: "auto-mapping best-effort par noms d'os
 *  courants, y compris le Mannequin UE5"). */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorRigMapAutoMapsUe5MannequinNamesTest,
	"AInimator.RigMap.AutoMapsUe5MannequinBoneNames",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorRigMapAutoMapsUe5MannequinNamesTest::RunTest(const FString& Parameters)
{
	UAInimatorRigMap* RigMap = NewObject<UAInimatorRigMap>();

	const TArray<FName> MannequinBoneNames = {
		TEXT("pelvis"), TEXT("spine_01"), TEXT("spine_02"), TEXT("spine_03"),
		TEXT("neck_01"), TEXT("head"),
		TEXT("thigh_l"), TEXT("calf_l"), TEXT("foot_l"), TEXT("ball_l"),
		TEXT("thigh_r"), TEXT("calf_r"), TEXT("foot_r"), TEXT("ball_r"),
		TEXT("clavicle_l"), TEXT("upperarm_l"), TEXT("lowerarm_l"), TEXT("hand_l"),
		TEXT("clavicle_r"), TEXT("upperarm_r"), TEXT("lowerarm_r"), TEXT("hand_r"),
	};

	const int32 NewlyMapped = RigMap->AutoMapByCommonNames(MannequinBoneNames);
	TestEqual(TEXT("All 22 SMPL bones resolve against the UE5 Mannequin skeleton"),
		NewlyMapped, AInimatorSmpl22Skeleton::NumBones);

	TestTrue(TEXT("pelvis -> pelvis"),
		RigMap->GetTargetBoneName(AInimatorSmpl22Skeleton::Pelvis) == FName(TEXT("pelvis")));
	TestTrue(TEXT("leftHip -> thigh_l"),
		RigMap->GetTargetBoneName(AInimatorSmpl22Skeleton::LeftHip) == FName(TEXT("thigh_l")));
	TestTrue(TEXT("rightKnee -> calf_r"),
		RigMap->GetTargetBoneName(AInimatorSmpl22Skeleton::RightKnee) == FName(TEXT("calf_r")));
	TestTrue(TEXT("leftWrist -> hand_l"),
		RigMap->GetTargetBoneName(AInimatorSmpl22Skeleton::LeftWrist) == FName(TEXT("hand_l")));
	return true;
}

/** Auto-map must never clobber an existing mapping unless explicitly
 *  told to overwrite (so re-running it after hand-tuning a few bones
 *  is safe). */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorRigMapAutoMapPreservesManualEditsTest,
	"AInimator.RigMap.AutoMapPreservesManualEditsByDefault",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorRigMapAutoMapPreservesManualEditsTest::RunTest(const FString& Parameters)
{
	UAInimatorRigMap* RigMap = NewObject<UAInimatorRigMap>();
	RigMap->Entries[AInimatorSmpl22Skeleton::Pelvis].TargetBoneName = TEXT("MyCustomPelvisBone");

	const TArray<FName> CandidateNames = {TEXT("pelvis"), TEXT("thigh_l")};
	RigMap->AutoMapByCommonNames(CandidateNames, /*bOverwriteExisting=*/false);

	TestTrue(TEXT("Manual mapping is preserved when bOverwriteExisting is false"),
		RigMap->GetTargetBoneName(AInimatorSmpl22Skeleton::Pelvis) == FName(TEXT("MyCustomPelvisBone")));
	TestTrue(TEXT("Unmapped bones still get auto-mapped in the same call"),
		RigMap->GetTargetBoneName(AInimatorSmpl22Skeleton::LeftHip) == FName(TEXT("thigh_l")));

	RigMap->AutoMapByCommonNames(CandidateNames, /*bOverwriteExisting=*/true);
	TestTrue(TEXT("bOverwriteExisting=true replaces the manual mapping"),
		RigMap->GetTargetBoneName(AInimatorSmpl22Skeleton::Pelvis) == FName(TEXT("pelvis")));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
