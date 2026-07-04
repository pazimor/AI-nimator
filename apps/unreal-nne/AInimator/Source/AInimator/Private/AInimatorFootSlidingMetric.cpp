// Copyright AI-nimator.

#include "AInimatorFootSlidingMetric.h"

void FFootSlidingMetric::AccumulateSample(
	int32 FootIndex,
	const FVector& WorldPosition,
	bool bIsInContact)
{
	FFootTrackingState& State = PerFootState.FindOrAdd(FootIndex);

	if (State.bHasPreviousSample && State.bPreviousWasInContact && bIsInContact)
	{
		const float DeltaX = WorldPosition.X - State.PreviousPosition.X;
		const float DeltaZ = WorldPosition.Z - State.PreviousPosition.Z;
		const float PlanarDisplacement = FMath::Sqrt(DeltaX * DeltaX + DeltaZ * DeltaZ);
		TotalPlanarDisplacement += PlanarDisplacement;
		++TotalContactFramePairs;
	}

	State.PreviousPosition = WorldPosition;
	State.bPreviousWasInContact = bIsInContact;
	State.bHasPreviousSample = true;
}

float FFootSlidingMetric::ComputeAverageSlidingMeters() const
{
	if (TotalContactFramePairs == 0)
	{
		return 0.0f;
	}
	return static_cast<float>(TotalPlanarDisplacement / TotalContactFramePairs);
}

void FFootSlidingMetric::Reset()
{
	PerFootState.Reset();
	TotalPlanarDisplacement = 0.0;
	TotalContactFramePairs = 0;
}
