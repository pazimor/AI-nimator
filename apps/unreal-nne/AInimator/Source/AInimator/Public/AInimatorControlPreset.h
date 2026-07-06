// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "AInimatorControlPreset.generated.h"

/**
 * A named control-signal configuration, matching
 * apps/spec/control_preset.schema.json exactly.
 *
 * Values are RAW (unnormalized): Vx/Vz in meters per frame in the
 * root-local ground frame; the runtime applies the z-norm from
 * norm_stats.json (FNormalizer). Aim is unit-norm by construction —
 * authored directly in [-1, 1], never normalized further.
 *
 * A preset is a convenience, never a mandatory entry point
 * (ROADMAP_PLUGINS.md §1.1): gameplay code may always build a raw
 * control vector directly (e.g. from WASD/stick input) without going
 * through an asset.
 */
UCLASS(BlueprintType)
class AINIMATOR_API UAInimatorControlPreset : public UDataAsset
{
	GENERATED_BODY()

public:
	/** Preset identifier (snake_case), doubles as the JSON filename stem. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AInimator|Preset")
	FString PresetName;

	/** Desired lateral velocity, meters/frame, root-local frame. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AInimator|Preset")
	float Vx = 0.0f;

	/** Desired forward velocity, meters/frame, root-local frame
	 *  (+Z = facing). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AInimator|Preset")
	float Vz = 0.0f;

	/** Whether this preset also carries an aim direction (only
	 *  meaningful when the bundle's control_channels == 4). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AInimator|Preset")
	bool bHasAim = false;

	/** Aim direction X (unit-norm 2D vector with AimZ). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AInimator|Preset",
		meta = (EditCondition = "bHasAim", ClampMin = "-1.0", ClampMax = "1.0"))
	float AimX = 0.0f;

	/** Aim direction Z (unit-norm 2D vector with AimX). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AInimator|Preset",
		meta = (EditCondition = "bHasAim", ClampMin = "-1.0", ClampMax = "1.0"))
	float AimZ = 0.0f;

	/** Optional authoring-time text prompt this preset was built for
	 *  (informational only; the runtime consumes PromptEmb, not text). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AInimator|Preset")
	FString Prompt;

	/** Whether PromptEmb below is populated. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AInimator|Preset")
	bool bHasPromptEmb = false;

	/** Optional precomputed prompt embedding; length must equal
	 *  manifest.PromptEmbChannels when bHasPromptEmb is true. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AInimator|Preset",
		meta = (EditCondition = "bHasPromptEmb"))
	TArray<float> PromptEmb;

	/**
	 * Builds the raw (unnormalized) control vector for this preset in
	 * the manifest's declared control_layout order.
	 *
	 * Parameters
	 * ----------
	 * ControlLayout : ordered channel names from the manifest (e.g.
	 *     ["vx", "vz"] or ["vx", "vz", "aim_x", "aim_z"]).
	 * OutControlVector : resized and filled to ControlLayout.Num().
	 *
	 * Returns
	 * -------
	 * bool
	 *     False if this preset cannot satisfy the requested layout
	 *     (e.g. layout requires aim but bHasAim is false) — the caller
	 *     should treat this as a configuration error, not silently
	 *     default to zero aim.
	 */
	bool BuildControlVector(
		const TArray<FString>& ControlLayout,
		TArray<float>& OutControlVector) const;

	/** Hydrates this asset's fields from one parsed control_preset.json
	 *  object (see FControlPresetJson in AInimatorControlPresetJson.h). */
	void HydrateFromRaw(
		const FString& InName,
		float InVx,
		float InVz,
		bool bInHasAim,
		float InAimX,
		float InAimZ,
		const FString& InPrompt,
		const TArray<float>& InPromptEmb);
};
