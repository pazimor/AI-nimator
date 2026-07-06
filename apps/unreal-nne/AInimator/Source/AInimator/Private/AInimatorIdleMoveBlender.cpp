// Copyright AI-nimator.

#include "AInimatorIdleMoveBlender.h"

FIdleMoveBlender::FIdleMoveBlender(
	float InMoveSpeedThreshold,
	float InIdleDelaySeconds,
	float InCrossFadeSeconds)
	: MoveSpeedThreshold(InMoveSpeedThreshold)
	, IdleDelaySeconds(InIdleDelaySeconds)
	, CrossFadeSeconds(InCrossFadeSeconds)
{
	check(MoveSpeedThreshold >= 0.0f);
	check(IdleDelaySeconds >= 0.0f);
	check(CrossFadeSeconds >= 0.0f);
}

void FIdleMoveBlender::Reset()
{
	MoveWeight = 1.0f;
	bTargetIsIdle = false;
	SecondsBelowThreshold = 0.0f;
}

float FIdleMoveBlender::Tick(float RawVx, float RawVz, float DeltaSeconds)
{
	const float Speed = FMath::Sqrt(RawVx * RawVx + RawVz * RawVz);
	const bool bBelowThreshold = Speed <= MoveSpeedThreshold;

	if (bBelowThreshold)
	{
		SecondsBelowThreshold += DeltaSeconds;
		if (SecondsBelowThreshold >= IdleDelaySeconds)
		{
			bTargetIsIdle = true;
		}
	}
	else
	{
		SecondsBelowThreshold = 0.0f;
		bTargetIsIdle = false;
	}

	const float TargetWeight = bTargetIsIdle ? 0.0f : 1.0f;
	if (CrossFadeSeconds > 0.0f)
	{
		// Linear ramp toward TargetWeight covering the full [0,1] range
		// in exactly CrossFadeSeconds, clamped so it never overshoots —
		// equivalent to FMath::FInterpConstantTo(MoveWeight,
		// TargetWeight, DeltaSeconds, 1.0f / CrossFadeSeconds), written
		// out explicitly to avoid depending on that overload's exact
		// argument order (verify against the installed engine if this
		// helper's signature is preferred instead).
		const float MaxStep = DeltaSeconds / CrossFadeSeconds;
		const float Delta = FMath::Clamp(TargetWeight - MoveWeight, -MaxStep, MaxStep);
		MoveWeight = MoveWeight + Delta;
	}
	else
	{
		MoveWeight = TargetWeight;
	}

	return MoveWeight;
}
