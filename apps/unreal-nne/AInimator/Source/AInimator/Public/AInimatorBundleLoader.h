// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "AInimatorManifest.h"
#include "AInimatorNormStats.h"

/**
 * Loads and validates one AI-nimator controller bundle from disk.
 *
 * A bundle is a directory produced by
 * `python -m ainimator.cli.export_onnx bundle` (inference_contract.md
 * §1): controller.onnx + norm_stats.json + manifest.json +
 * resolved_config.yaml + presets/. This loader only parses
 * manifest.json and norm_stats.json; the .onnx itself is handed to
 * UNNEModelData / FControllerRuntime separately (NNE asset import is
 * a UObject/editor-time concern, kept out of this pure loader so it
 * stays unit-testable without engine subsystems).
 *
 * Fail-fast contract (ROADMAP_PLUGINS.md §2 vérité #4): any missing
 * required field, dimension mismatch, or incompatible bundle_version
 * major makes LoadFromDirectory() return false and log a specific
 * UE_LOG(LogAInimator, Error, ...) — never a silent partial load.
 */
class AINIMATOR_API FBundleLoader
{
public:
	/**
	 * Parse manifest.json and norm_stats.json from BundleDirectory and
	 * validate them against the contract.
	 *
	 * Parameters
	 * ----------
	 * BundleDirectory : the folder containing manifest.json,
	 *     norm_stats.json (and controller.onnx / presets, not read here).
	 * OutManifest : receives the parsed manifest on success.
	 * OutNormStats : receives the parsed stats on success, shaped to
	 *     OutManifest (bone/global/control arrays sized from its dims).
	 *
	 * Returns
	 * -------
	 * true if both files parsed and validated; false otherwise (with
	 * the reason already logged to LogAInimator).
	 */
	static bool LoadFromDirectory(
		const FString& BundleDirectory,
		FAInimatorManifest& OutManifest,
		FAInimatorNormStats& OutNormStats);

private:
	static bool LoadManifest(
		const FString& ManifestPath,
		FAInimatorManifest& OutManifest);

	static bool LoadNormStats(
		const FString& NormStatsPath,
		const FAInimatorManifest& Manifest,
		FAInimatorNormStats& OutNormStats);

	/** Validates dims/version once both structures are parsed; the
	 *  single fail-fast gate before any inference is permitted. */
	static bool ValidateManifest(const FAInimatorManifest& Manifest);

	/** Parses "A7.0"-style version strings; returns false on a
	 *  malformed string (also fail-fast). */
	static bool TryParseVersionMajor(const FString& Version, int32& OutMajor);
};
