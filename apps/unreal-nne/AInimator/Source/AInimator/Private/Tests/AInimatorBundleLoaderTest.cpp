// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorBundleLoader.h"
#include "AInimatorManifest.h"
#include "AInimatorNormStats.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace AInimatorBundleLoaderTestHelpers
{
	/** A minimal, contract-valid 1-context-frame / no-prompt manifest +
	 *  norm_stats pair, written to a scratch directory so
	 *  FBundleLoader can be exercised without a real exported bundle. */
	FString WriteValidBundle(const FString& RootDir)
	{
		IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
		PlatformFile.CreateDirectoryTree(*RootDir);

		const FString ManifestJson = TEXT(R"JSON({
			"bundle_version": "A7.0",
			"state_channels": 136,
			"num_bones": 22,
			"rotation_channels_per_bone": 6,
			"root_local_motion_channels": 4,
			"control_channels": 2,
			"control_layout": ["vx", "vz"],
			"phase_channels": 0,
			"prompt_emb_channels": 0,
			"context_frames": 8,
			"output_layout": "bone_delta|global_delta",
			"coord_system": "Y-up right-handed",
			"normalization_note": "test fixture",
			"reserved_input_groups": []
		})JSON");
		FFileHelper::SaveStringToFile(
			ManifestJson, *FPaths::Combine(RootDir, TEXT("manifest.json")));

		FString BoneArray = TEXT("[");
		for (int32 Bone = 0; Bone < 22; ++Bone)
		{
			BoneArray += TEXT("[0.0,0.0,0.0,0.0,0.0,0.0]");
			if (Bone != 21) BoneArray += TEXT(",");
		}
		BoneArray += TEXT("]");

		const FString NormStatsJson = FString::Printf(TEXT(R"JSON({
			"state": {
				"bone_mean": [[%s]],
				"bone_std": [[%s]],
				"global_mean": [[[0.0,0.0,0.0,0.0]]],
				"global_std": [[[1.0,1.0,1.0,1.0]]]
			},
			"delta": {
				"bone_mean": [[%s]],
				"bone_std": [[%s]],
				"global_mean": [[[0.0,0.0,0.0,0.0]]],
				"global_std": [[[1.0,1.0,1.0,1.0]]]
			},
			"control": {
				"mean": [[0.0,0.0]],
				"std": [[1.0,1.0]],
				"channels": ["vx","vz"]
			}
		})JSON"), *BoneArray, *BoneArray, *BoneArray, *BoneArray);
		FFileHelper::SaveStringToFile(
			NormStatsJson, *FPaths::Combine(RootDir, TEXT("norm_stats.json")));

		return RootDir;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorBundleLoaderValidBundleTest,
	"AInimator.BundleLoader.LoadsValidBundle",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorBundleLoaderValidBundleTest::RunTest(const FString& Parameters)
{
	const FString RootDir = FPaths::Combine(
		FPaths::ProjectIntermediateDir(), TEXT("AInimatorTests/ValidBundle"));
	AInimatorBundleLoaderTestHelpers::WriteValidBundle(RootDir);

	FAInimatorManifest Manifest;
	FAInimatorNormStats Stats;
	const bool bLoaded = FBundleLoader::LoadFromDirectory(RootDir, Manifest, Stats);

	TestTrue(TEXT("Valid bundle loads"), bLoaded);
	TestEqual(TEXT("ContextFrames parsed"), Manifest.ContextFrames, 8);
	TestEqual(TEXT("BoneMean length"), Stats.BoneMean.Num(), 22 * 6);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorBundleLoaderRejectsIncompatibleVersionTest,
	"AInimator.BundleLoader.RejectsIncompatibleContractVersion",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorBundleLoaderRejectsIncompatibleVersionTest::RunTest(
	const FString& Parameters)
{
	const FString RootDir = FPaths::Combine(
		FPaths::ProjectIntermediateDir(), TEXT("AInimatorTests/BadVersion"));
	AInimatorBundleLoaderTestHelpers::WriteValidBundle(RootDir);

	// Corrupt bundle_version to an incompatible major (B99 vs the
	// plugin's supported major, currently 7) — must fail-fast, not
	// silently load with mismatched assumptions.
	FString ManifestText;
	FFileHelper::LoadFileToString(
		ManifestText, *FPaths::Combine(RootDir, TEXT("manifest.json")));
	ManifestText = ManifestText.Replace(TEXT("\"A7.0\""), TEXT("\"A99.0\""));
	FFileHelper::SaveStringToFile(
		ManifestText, *FPaths::Combine(RootDir, TEXT("manifest.json")));

	FAInimatorManifest Manifest;
	FAInimatorNormStats Stats;
	const bool bLoaded = FBundleLoader::LoadFromDirectory(RootDir, Manifest, Stats);

	TestFalse(TEXT("Incompatible version must fail, not silently load"), bLoaded);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorBundleLoaderRejectsWrongStateChannelsTest,
	"AInimator.BundleLoader.RejectsWrongStateChannels",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorBundleLoaderRejectsWrongStateChannelsTest::RunTest(
	const FString& Parameters)
{
	const FString RootDir = FPaths::Combine(
		FPaths::ProjectIntermediateDir(), TEXT("AInimatorTests/BadStateChannels"));
	AInimatorBundleLoaderTestHelpers::WriteValidBundle(RootDir);

	FString ManifestText;
	FFileHelper::LoadFileToString(
		ManifestText, *FPaths::Combine(RootDir, TEXT("manifest.json")));
	ManifestText = ManifestText.Replace(
		TEXT("\"state_channels\": 136"), TEXT("\"state_channels\": 64"));
	FFileHelper::SaveStringToFile(
		ManifestText, *FPaths::Combine(RootDir, TEXT("manifest.json")));

	FAInimatorManifest Manifest;
	FAInimatorNormStats Stats;
	const bool bLoaded = FBundleLoader::LoadFromDirectory(RootDir, Manifest, Stats);

	TestFalse(TEXT("Wrong state_channels must fail, not silently load"), bLoaded);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorBundleLoaderRejectsMissingNullEmbTest,
	"AInimator.BundleLoader.RejectsMissingNullEmbWhenPromptChannelsPositive",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorBundleLoaderRejectsMissingNullEmbTest::RunTest(
	const FString& Parameters)
{
	const FString RootDir = FPaths::Combine(
		FPaths::ProjectIntermediateDir(), TEXT("AInimatorTests/MissingNullEmb"));
	AInimatorBundleLoaderTestHelpers::WriteValidBundle(RootDir);

	// Declare prompt_emb_channels > 0 without adding a "prompt" section
	// to norm_stats.json — must fail-fast per inference_contract §4
	// ("a vector of zeros is NOT a substitute").
	FString ManifestText;
	FFileHelper::LoadFileToString(
		ManifestText, *FPaths::Combine(RootDir, TEXT("manifest.json")));
	ManifestText = ManifestText.Replace(
		TEXT("\"prompt_emb_channels\": 0"), TEXT("\"prompt_emb_channels\": 512"));
	FFileHelper::SaveStringToFile(
		ManifestText, *FPaths::Combine(RootDir, TEXT("manifest.json")));

	FAInimatorManifest Manifest;
	FAInimatorNormStats Stats;
	const bool bLoaded = FBundleLoader::LoadFromDirectory(RootDir, Manifest, Stats);

	TestFalse(
		TEXT("Missing prompt.null_emb with prompt_emb_channels>0 must fail"),
		bLoaded);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
