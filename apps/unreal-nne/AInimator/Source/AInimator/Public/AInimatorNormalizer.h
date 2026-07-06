// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "AInimatorManifest.h"
#include "AInimatorNormStats.h"

/**
 * Pure z-normalization math, mirroring
 * ainimator/model/motion_normalizer.py exactly so the same bundle
 * produces the same normalized tensors in Unreal as in Python/Unity
 * (ROADMAP_PLUGINS.md §2 vérité #7, parity).
 *
 * All methods are pure functions of (value, stats) — no engine
 * dependency, no allocation beyond the caller-provided output array —
 * so this class is unit-testable in an Automation test without a
 * running world, and reusable per-frame with zero heap churn (caller
 * pre-sizes the output arrays once).
 *
 * Conventions (inference_contract.md §3):
 * 1. State (bone window + global window): z-norm per channel with the
 *    "state" stats.
 * 2. Control: vx/vz z-normed with "control" stats; aim_x/aim_z passed
 *    through unchanged (unit-norm by construction, no stats).
 * 3. Output Δstate: denormalized with the "delta" stats before the
 *    engine integrates it into the running state.
 * 4. Phase (cos, sin): passed through unchanged, already bounded.
 */
class AINIMATOR_API FNormalizer
{
public:
	explicit FNormalizer(
		const FAInimatorManifest& InManifest,
		const FAInimatorNormStats& InStats);

	/**
	 * Z-normalize one frame of bone rotations in place semantics via
	 * output param: `(bone - mean) / max(std, epsilon)` per (bone,
	 * channel), matching MotionNormalizer.normalizeBone.
	 *
	 * Parameters
	 * ----------
	 * RawBoneFrame : NumBones * RotationChannelsPerBone raw values,
	 *     bone-major / channel-minor (same order as norm_stats.json).
	 * OutNormalized : resized and filled with the normalized values.
	 */
	void NormalizeBoneFrame(
		const TArray<float>& RawBoneFrame,
		TArray<float>& OutNormalized) const;

	/** Same as NormalizeBoneFrame but for the RootLocalMotionChannels
	 *  global (root-local motion) frame, using the "state.global_*"
	 *  stats. */
	void NormalizeGlobalFrame(
		const TArray<float>& RawGlobalFrame,
		TArray<float>& OutNormalized) const;

	/**
	 * Normalize one raw control vector for ONNX input.
	 *
	 * Only the channels named in norm_stats.json's control.channels
	 * (vx, vz) are z-normed; any remaining channels (aim_x, aim_z, when
	 * manifest.ControlChannels==4) are copied through unchanged — they
	 * are unit-norm by construction and carry no stats
	 * (inference_contract.md §3.2).
	 */
	void NormalizeControlVector(
		const TArray<float>& RawControl,
		TArray<float>& OutNormalized) const;

	/**
	 * Denormalize one predicted bone delta frame:
	 * `delta * std + mean`, matching
	 * MotionNormalizer.denormalizeBone / denormalizeStepDelta.
	 */
	void DenormalizeBoneDelta(
		const TArray<float>& NormalizedDelta,
		TArray<float>& OutRawDelta) const;

	/** Denormalize one predicted global (root-local motion) delta
	 *  frame using the "delta.global_*" stats. */
	void DenormalizeGlobalDelta(
		const TArray<float>& NormalizedDelta,
		TArray<float>& OutRawDelta) const;

private:
	const FAInimatorManifest& Manifest;
	const FAInimatorNormStats& Stats;

	static void ApplyZNorm(
		const TArray<float>& Values,
		const TArray<float>& Mean,
		const TArray<float>& Std,
		TArray<float>& Out);

	static void ApplyZDenorm(
		const TArray<float>& Values,
		const TArray<float>& Mean,
		const TArray<float>& Std,
		TArray<float>& Out);
};
