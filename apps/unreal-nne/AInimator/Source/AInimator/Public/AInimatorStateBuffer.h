// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Rolling autoregressive context window (bone + global) for the
 * controller runtime (ROADMAP_PLUGINS.md §3.3: "the engine carries the
 * state").
 *
 * Holds `ContextFrames` raw (unnormalized) frames of the bone rotation
 * window (NumBones * RotationChannelsPerBone floats each) and the
 * root-local motion / global window (RootLocalMotionChannels floats
 * each). PushFrame() drops the oldest frame and appends the newest —
 * a classic ring buffer implemented as a flat, pre-allocated array so
 * there is no per-frame heap allocation on the hot path.
 *
 * This class does no normalization and no ONNX I/O — it is a pure
 * data structure, unit-testable in isolation (window ordering, seeding,
 * eviction).
 */
class AINIMATOR_API FStateBuffer
{
public:
	FStateBuffer(
		int32 InContextFrames,
		int32 InBoneFrameWidth,
		int32 InGlobalFrameWidth);

	/** Fills every slot of the window with the same seed frame — used
	 *  to initialize a pawn standing still before any real history
	 *  exists. */
	void SeedWithFrame(
		const TArray<float>& SeedBoneFrame,
		const TArray<float>& SeedGlobalFrame);

	/** Appends one new (raw) frame, evicting the oldest. */
	void PushFrame(
		const TArray<float>& NewBoneFrame,
		const TArray<float>& NewGlobalFrame);

	/** Returns the full bone window flattened in chronological order
	 *  (oldest first), ready to feed FNormalizer frame-by-frame or to
	 *  be normalized frame-by-frame into an ONNX input buffer. Length
	 *  is ContextFrames * BoneFrameWidth. */
	const TArray<float>& GetBoneWindow() const { return BoneWindow; }

	/** Same as GetBoneWindow but for the global (root-local motion)
	 *  window. Length is ContextFrames * GlobalFrameWidth. */
	const TArray<float>& GetGlobalWindow() const { return GlobalWindow; }

	/** Returns the most recent raw bone frame (last ContextFrames-1
	 *  slot), needed by the runtime to add the raw bone delta on top
	 *  (state_{t+1} = state_t + rawDelta, per controller_rollout.py). */
	void GetLastBoneFrame(TArray<float>& OutFrame) const;

	int32 GetContextFrames() const { return ContextFrames; }
	int32 GetBoneFrameWidth() const { return BoneFrameWidth; }
	int32 GetGlobalFrameWidth() const { return GlobalFrameWidth; }

private:
	int32 ContextFrames = 0;
	int32 BoneFrameWidth = 0;
	int32 GlobalFrameWidth = 0;

	/** Flattened (ContextFrames, BoneFrameWidth) window, oldest-first. */
	TArray<float> BoneWindow;
	/** Flattened (ContextFrames, GlobalFrameWidth) window, oldest-first. */
	TArray<float> GlobalWindow;
};
