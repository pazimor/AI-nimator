// Copyright AI-nimator.

#include "AInimatorFootLockIK.h"
#include "AInimatorTwoBoneIkSolver.h"

FFootLockIK::FFootLockIK(float InMaxCorrectionMeters, float InReleaseFadeSeconds)
	: MaxCorrectionMeters(InMaxCorrectionMeters)
	, ReleaseFadeSeconds(InReleaseFadeSeconds)
{
	check(MaxCorrectionMeters > 0.0f);
	check(ReleaseFadeSeconds >= 0.0f);
}

void FFootLockIK::Reset()
{
	for (FFootState& State : FeetState)
	{
		State = FFootState();
	}
}

void FFootLockIK::Solve(
	const FFootContactDetector& ContactDetector,
	const FLegChain& LeftChain,
	const FLegChain& RightChain,
	const FVector WorldHipPosition[2],
	const FVector WorldKneePosition[2],
	const FVector WorldAnklePosition[2],
	float DeltaSeconds,
	FVector OutCorrectedKneePosition[2],
	FVector OutCorrectedAnklePosition[2])
{
	const FLegChain* Chains[2] = {&LeftChain, &RightChain};
	for (int32 FootSlot = 0; FootSlot < 2; ++FootSlot)
	{
		SolveOneFoot(
			FootSlot,
			ContactDetector.IsLocked(FootSlot),
			*Chains[FootSlot],
			WorldHipPosition[FootSlot],
			WorldKneePosition[FootSlot],
			WorldAnklePosition[FootSlot],
			DeltaSeconds,
			OutCorrectedKneePosition[FootSlot],
			OutCorrectedAnklePosition[FootSlot]);
	}
}

void FFootLockIK::SolveOneFoot(
	int32 FootSlot,
	bool bIsLocked,
	const FLegChain& Chain,
	const FVector& WorldHip,
	const FVector& WorldKnee,
	const FVector& WorldAnkle,
	float DeltaSeconds,
	FVector& OutCorrectedKnee,
	FVector& OutCorrectedAnkle)
{
	FFootState& State = FeetState[FootSlot];

	// Default: no correction (pass the controller's own pose through).
	OutCorrectedKnee = WorldKnee;
	OutCorrectedAnkle = WorldAnkle;

	if (bIsLocked)
	{
		if (!State.bWasLocked)
		{
			// Step 1: capture the world position at contact entry.
			State.LockedPosition = WorldAnkle;
		}

		// Step 3: clamp — if the controller's own pose has already
		// moved the foot further than MaxCorrectionMeters from the
		// captured lock, release rather than stretch the leg.
		const float DisplacementFromLock =
			(State.LockedPosition - WorldAnkle).Size();
		if (DisplacementFromLock > MaxCorrectionMeters)
		{
			State.bWasLocked = false;
			State.FadeWeight = 0.0f;
			return;
		}

		// Step 2: two-bone IK toward the locked position, pole vector
		// derived from the current pose's own knee direction.
		const AInimatorTwoBoneIkSolver::FResult Result = AInimatorTwoBoneIkSolver::Solve(
			WorldHip, WorldKnee, WorldAnkle, State.LockedPosition,
			Chain.UpperLength, Chain.LowerLength);

		OutCorrectedKnee = Result.MidPosition;
		OutCorrectedAnkle = Result.EffectorPosition;
		State.bWasLocked = true;
		State.FadeWeight = 1.0f;
		return;
	}

	// Step 4: release fade — was locked last time this foot was
	// evaluated as locked, now released: lerp the correction to zero
	// over ReleaseFadeSeconds instead of snapping to the raw pose.
	if (State.bWasLocked || State.FadeWeight > 0.0f)
	{
		if (ReleaseFadeSeconds > 0.0f)
		{
			const float FadeStep = DeltaSeconds / ReleaseFadeSeconds;
			State.FadeWeight = FMath::Max(0.0f, State.FadeWeight - FadeStep);
		}
		else
		{
			State.FadeWeight = 0.0f;
		}

		if (State.FadeWeight > 0.0f)
		{
			const AInimatorTwoBoneIkSolver::FResult Result = AInimatorTwoBoneIkSolver::Solve(
				WorldHip, WorldKnee, WorldAnkle, State.LockedPosition,
				Chain.UpperLength, Chain.LowerLength);

			OutCorrectedKnee = FMath::Lerp(WorldKnee, Result.MidPosition, State.FadeWeight);
			OutCorrectedAnkle = FMath::Lerp(WorldAnkle, Result.EffectorPosition, State.FadeWeight);
		}
	}

	State.bWasLocked = false;
}
