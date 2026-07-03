// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "AInimatorManifest.generated.h"

/**
 * Plain data mirror of manifest.json (apps/spec/manifest.schema.json).
 *
 * Deliberately NOT a UDataAsset: the manifest is parsed fresh from the
 * bundle's manifest.json every load (it is a build artifact, never a
 * versioned Unreal asset) by FBundleLoader. USTRUCT only so it can be
 * inspected/logged conveniently and, if useful, exposed read-only to
 * Blueprint.
 */
USTRUCT(BlueprintType)
struct AINIMATOR_API FAInimatorManifest
{
	GENERATED_BODY()

	/** e.g. "A7.0" — engines refuse incompatible majors (fail-fast). */
	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	FString BundleVersion;

	/** Frozen at 136 by the contract; validated, not trusted blindly. */
	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	int32 StateChannels = 0;

	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	int32 NumBones = 0;

	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	int32 RotationChannelsPerBone = 0;

	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	int32 RootLocalMotionChannels = 0;

	/** 2 = (vx, vz); 4 = (vx, vz, aim_x, aim_z). */
	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	int32 ControlChannels = 0;

	/** Ordered channel names, e.g. ["vx", "vz"] or with aim appended. */
	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	TArray<FString> ControlLayout;

	/** 0 = no phase input; 2 = (cos, sin) explicit gait phase. */
	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	int32 PhaseChannels = 0;

	/** 0 = no text conditioning; otherwise width of promptEmb input. */
	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	int32 PromptEmbChannels = 0;

	/** Autoregressive state window length seen by one forward. */
	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	int32 ContextFrames = 0;

	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	FString OutputLayout;

	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	FString CoordSystem;

	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	FString NormalizationNote;

	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Manifest")
	TArray<FString> ReservedInputGroups;

	/** True once all required fields were present and parsed. */
	bool bIsFullyParsed = false;
};
