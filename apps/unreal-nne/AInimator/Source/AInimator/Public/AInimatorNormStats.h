// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Plain data mirror of norm_stats.json (inference_contract.md §3/§4).
 *
 * Layout notes (matches the reference bundle exactly):
 * - BoneMean/BoneStd are flattened per-bone-per-channel arrays of
 *   length NumBones * RotationChannelsPerBone (row-major: bone-major,
 *   channel-minor), taken from the JSON's
 *   state.bone_mean[0][0][bone][channel] (the leading 1x1 "frame"
 *   dims in the JSON are a broadcasting artifact of the Python
 *   MotionNormalizer buffers, shape (1,1,numBones,channels); we drop
 *   them on load since a single frame is normalized at a time here).
 * - GlobalMean/GlobalStd have length RootLocalMotionChannels, from
 *   state.global_mean[0][0][channel].
 * - Delta* mirror the same layout for the "delta" section.
 * - ControlMean/ControlStd have length manifest.ControlChannels for
 *   the (vx, vz[, aim_x, aim_z]) channels that DO carry z-norm stats;
 *   per the contract, aim_x/aim_z are unit-norm by construction and
 *   are never z-normalized even when control_channels == 4 — the
 *   normalizer must apply control stats to the first
 *   control.channels.Num() entries only (see FNormalizer).
 * - PromptNullEmb is the learned null embedding (never zeros).
 */
struct AINIMATOR_API FAInimatorNormStats
{
	TArray<float> BoneMean;
	TArray<float> BoneStd;
	TArray<float> GlobalMean;
	TArray<float> GlobalStd;

	TArray<float> DeltaBoneMean;
	TArray<float> DeltaBoneStd;
	TArray<float> DeltaGlobalMean;
	TArray<float> DeltaGlobalStd;

	TArray<float> ControlMean;
	TArray<float> ControlStd;
	/** Channel names covered by ControlMean/ControlStd, e.g. ["vx","vz"]. */
	TArray<FString> ControlStatChannels;

	/** Learned null prompt embedding; empty when prompt_emb_channels==0. */
	TArray<float> PromptNullEmb;

	bool bIsFullyParsed = false;
};
