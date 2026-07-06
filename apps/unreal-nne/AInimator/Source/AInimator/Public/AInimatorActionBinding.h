// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "AInimatorActionBinding.generated.h"

class UInputAction;
class UAInimatorControlPreset;

/**
 * One authoring-time binding: an input (Enhanced Input action, or a
 * legacy FKey fallback when no `UInputAction` asset is assigned) maps
 * to a `UAInimatorControlPreset` applied to the owning
 * `UAInimatorActionComponent`'s runtime while the input is active
 * (ROADMAP_PLUGINS.md §1.1 / §4 B3: "[Input/Bouton] -> [Action]").
 *
 * This struct is pure data — no engine subsystem dependency beyond the
 * two UObject references — so binding resolution
 * (`UAInimatorActionComponent::ResolvePresetForAction` /
 * `ResolvePresetForKey`) is unit-testable without a running world.
 */
USTRUCT(BlueprintType)
struct AINIMATOR_API FAInimatorActionBinding
{
	GENERATED_BODY()

	/** Preferred binding source: an Enhanced Input action asset. Takes
	 *  priority over Key when both are set (see resolution order in
	 *  `UAInimatorActionComponent::BeginPlay`). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Binding")
	TObjectPtr<UInputAction> Action = nullptr;

	/** Fallback binding source when no Enhanced Input `UInputAction` is
	 *  available/desired (e.g. a quick prototype without an Input
	 *  Mapping Context set up yet). Ignored if Action is set. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Binding")
	FKey Key;

	/** The preset applied to the runtime while this binding is active. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Binding")
	TObjectPtr<UAInimatorControlPreset> Preset = nullptr;

	/** Human-readable label shown in the Details customization list
	 *  (purely cosmetic, never read by the runtime). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Binding")
	FString DisplayName;

	/** True if this binding has enough information to be wired at
	 *  BeginPlay: a Preset, and either an Action or a non-None Key. */
	bool IsValidBinding() const
	{
		return Preset != nullptr && (Action != nullptr || Key.IsValid());
	}

	/** True if this binding resolves via Enhanced Input rather than the
	 *  legacy FKey fallback. */
	bool UsesEnhancedInput() const
	{
		return Action != nullptr;
	}
};
