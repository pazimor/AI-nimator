// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Debug utility measuring average planar foot displacement while a
 * foot is considered "in contact" — the metric
 * `apps/spec/footlock_blending.md` §5 calls "foot-sliding avant/après"
 * (before/after foot-lock IK). Not part of the runtime hot path: this
 * is a diagnostic tool for validating B4's acceptance criterion
 * ("réduction mesurable du foot-sliding"), typically driven from an
 * editor utility widget, a console command, or an Automation test —
 * never ticked automatically by `UAInimatorControllerRuntime`.
 *
 * Usage: call `AccumulateSample()` once per frame per foot with that
 * foot's current world position and whether the detector currently
 * considers it locked (in contact); call `ComputeAverageSlidingMeters()`
 * once the recording window is over (e.g. after a 10s `forward` walk,
 * spec §5) to get the mean planar (XZ) displacement per contact frame.
 * A lower value after enabling `FFootLockIK` is the acceptance
 * evidence; log both runs (before/after) in the plugin README.
 */
class AINIMATOR_API FFootSlidingMetric
{
public:
	/** Records one frame's sample for one foot. Only frames where
	 *  bIsInContact is true (both this frame and the previous sampled
	 *  frame for this foot) contribute displacement to the average —
	 *  matches "déplacement planaire moyen des pieds pendant leurs
	 *  frames de contact" (spec §5) exactly: sliding is only meaningful
	 *  while the foot is supposed to be planted. */
	void AccumulateSample(int32 FootIndex, const FVector& WorldPosition, bool bIsInContact);

	/** Mean planar (XZ) displacement per contact-to-contact frame pair,
	 *  in meters/frame, across every foot sampled so far. Returns 0 if
	 *  no contact-to-contact frame pair was ever recorded (e.g. an
	 *  empty or airborne-only rollout) — callers should treat that as
	 *  "no data", not as "zero sliding achieved". */
	float ComputeAverageSlidingMeters() const;

	/** Number of contact-to-contact frame pairs the average above is
	 *  computed from — report alongside the average so a near-zero
	 *  sample count is not mistaken for a strong result. */
	int32 GetSampleCount() const { return TotalContactFramePairs; }

	void Reset();

private:
	struct FFootTrackingState
	{
		bool bHasPreviousSample = false;
		bool bPreviousWasInContact = false;
		FVector PreviousPosition = FVector::ZeroVector;
	};

	TMap<int32, FFootTrackingState> PerFootState;
	double TotalPlanarDisplacement = 0.0;
	int32 TotalContactFramePairs = 0;
};
