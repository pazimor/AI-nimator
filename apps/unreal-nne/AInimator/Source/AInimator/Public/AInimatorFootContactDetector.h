// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Re-derives per-frame foot-contact state from the pose (loss-only
 * signal — never predicted by the controller), following
 * `apps/spec/footlock_blending.md` §2 exactly.
 *
 * Base criterion (same as the data pipeline's
 * `controller_sequences.py::deriveFootContacts`): a foot is in contact
 * when, on the same frame, its world height is below HeightThreshold
 * **and** its planar (XZ) speed is below SpeedThreshold. Additive
 * engine-side hysteresis (anti-flicker, spec §2): entering contact
 * requires the raw criterion; leaving contact requires violating it
 * for ExitFrames consecutive frames **or** a height above
 * ReleaseHeightThreshold (1.5x the height threshold).
 *
 * Mirrors `apps/unity-sentis/.../PostProcess/FootContactDetector.cs`
 * bit-for-bit (Goal B vérité #7 parity). Pure data + math, no engine
 * subsystem dependency — unit-testable with synthetic trajectories.
 */
class AINIMATOR_API FFootContactDetector
{
public:
	/** World height below which a foot may be in contact (meters, Y axis). */
	static constexpr float DefaultHeightThreshold = 0.05f;
	/** Planar (XZ) speed below which a foot may be in contact (meters/frame). */
	static constexpr float DefaultSpeedThreshold = 0.01f;
	/** Consecutive violating frames required to release a lock. */
	static constexpr int32 DefaultExitFrames = 2;
	/** 1.5x the height threshold — an immediate release condition
	 *  regardless of frame count. */
	static constexpr float DefaultReleaseHeightMultiplier = 1.5f;

	/** Parameters
	 *  ----------
	 *  FootCount : number of tracked feet (2: left, right —
	 *      AInimatorSmpl22Skeleton::FootJointIndices). */
	explicit FFootContactDetector(
		int32 InFootCount,
		float InHeightThreshold = DefaultHeightThreshold,
		float InSpeedThreshold = DefaultSpeedThreshold,
		int32 InExitFrames = DefaultExitFrames,
		float InReleaseHeightMultiplier = DefaultReleaseHeightMultiplier);

	/** Whether foot FootIndex is currently considered locked (in contact). */
	bool IsLocked(int32 FootIndex) const { return IsLockedFlags[FootIndex]; }

	/** Update contact state for one foot from its current world
	 *  position; returns the (possibly hysteresis-held) contact state
	 *  for this frame. */
	bool Update(int32 FootIndex, const FVector& WorldPosition);

	/** Reset all tracked feet to the unlocked, no-history state. */
	void Reset();

	float GetHeightThreshold() const { return HeightThreshold; }
	float GetSpeedThreshold() const { return SpeedThreshold; }
	int32 GetExitFrames() const { return ExitFrames; }
	float GetReleaseHeightThreshold() const { return ReleaseHeightThreshold; }

private:
	float HeightThreshold;
	float SpeedThreshold;
	int32 ExitFrames;
	float ReleaseHeightThreshold;

	TArray<bool> IsLockedFlags;
	TArray<int32> ConsecutiveViolationFrames;
	TArray<FVector> PreviousPosition;
	TArray<bool> bHasPreviousPosition;
};
