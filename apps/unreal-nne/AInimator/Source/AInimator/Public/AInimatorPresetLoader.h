// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

class UAInimatorControlPreset;

/**
 * Loads ControlPreset JSON files (control_preset.schema.json) from a
 * bundle's presets/ directory into transient
 * UAInimatorControlPreset instances.
 *
 * Presets are always parsed fresh from the bundle (a build artifact),
 * never versioned as engine content directly — mirrors FBundleLoader's
 * treatment of manifest.json/norm_stats.json. A game may still persist
 * its own edited copies as regular UAInimatorControlPreset assets; this
 * loader is only the bundle-import path.
 */
class AINIMATOR_API FPresetLoader
{
public:
	/**
	 * Parses every *.json file directly under PresetsDirectory.
	 *
	 * Parameters
	 * ----------
	 * PresetsDirectory : the bundle's presets/ folder.
	 * Outer : UObject that owns the newly created preset instances
	 *     (pass the calling UControllerRuntime or a transient package).
	 * OutPresets : appended with one preset per valid JSON file; a
	 *     malformed file is skipped with an UE_LOG(Error) — it does not
	 *     abort loading the rest of the bundle, since presets are a
	 *     convenience, not the contract itself (unlike manifest/stats).
	 *
	 * Returns
	 * -------
	 * int32
	 *     Number of presets successfully loaded.
	 */
	static int32 LoadPresetsFromDirectory(
		const FString& PresetsDirectory,
		UObject* Outer,
		TArray<UAInimatorControlPreset*>& OutPresets);

	/** Parses one control_preset.json file into an existing preset
	 *  instance. Returns false (logged) on any schema violation. */
	static bool LoadPresetFile(
		const FString& FilePath,
		UAInimatorControlPreset* OutPreset);
};
