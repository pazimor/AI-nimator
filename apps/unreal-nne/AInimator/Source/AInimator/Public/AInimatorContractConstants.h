// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Frozen constants from the AI-nimator inference contract
 * (apps/spec/inference_contract.md, apps/spec/manifest.schema.json).
 *
 * These are the values the schema declares as `const` — i.e. they are
 * NOT expected to vary between bundles of the same contract major
 * version, so hard-coding them here (instead of re-deriving them from
 * JSON every time) is legitimate and keeps the manifest validation
 * code readable. Everything that CAN legitimately vary between bundles
 * (control_channels, phase_channels, prompt_emb_channels,
 * context_frames, bundle_version) is read from manifest.json at load
 * time by FBundleLoader — never hard-coded.
 */
namespace AInimatorContract
{
	/** Regressed state width: rotation6d (132) + root local motion (4). */
	constexpr int32 StateChannels = 136;

	/** SMPL-22 skeleton bone count. */
	constexpr int32 NumBones = 22;

	/** 6D rotation representation channels per bone (Zhou et al. 2019). */
	constexpr int32 RotationChannelsPerBone = 6;

	/** (dForward, dLateral, dHeight, dYaw) root-local motion channels. */
	constexpr int32 RootLocalMotionChannels = 4;

	/** Explicit phase channel count when phase conditioning is active. */
	constexpr int32 PhaseChannelsWhenActive = 2;

	/** Floor added to std before dividing, mirrors EPSILON_STD in
	 *  ainimator/model/motion_normalizer.py so both runtimes clamp
	 *  identically near-zero-variance channels (e.g. locked bones). */
	constexpr float NormalizationEpsilon = 1e-5f;

	/** Major contract version this plugin build was written against.
	 *  Bundles whose `bundle_version` major differs are rejected
	 *  (fail-fast, ROADMAP_PLUGINS.md §2 vérité #4). */
	constexpr int32 SupportedBundleVersionMajor = 7;

	/** ONNX input/output tensor names (export_onnx.py `inputNames` /
	 *  `outputNames`) — the single source of truth for tensor naming
	 *  is the Python export code; these mirror it exactly. */
	inline const TCHAR* const BoneWindowInputName = TEXT("bone_window");
	inline const TCHAR* const ControlInputName = TEXT("control");
	inline const TCHAR* const GlobalWindowInputName = TEXT("global_window");
	inline const TCHAR* const PromptEmbInputName = TEXT("prompt_emb");
	inline const TCHAR* const PhaseInputName = TEXT("phase");
	inline const TCHAR* const BoneDeltaOutputName = TEXT("bone_delta");
	inline const TCHAR* const GlobalDeltaOutputName = TEXT("global_delta");
}
