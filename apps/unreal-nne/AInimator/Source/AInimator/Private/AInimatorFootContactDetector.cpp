// Copyright AI-nimator.

#include "AInimatorFootContactDetector.h"

FFootContactDetector::FFootContactDetector(
	int32 InFootCount,
	float InHeightThreshold,
	float InSpeedThreshold,
	int32 InExitFrames,
	float InReleaseHeightMultiplier)
	: HeightThreshold(InHeightThreshold)
	, SpeedThreshold(InSpeedThreshold)
	, ExitFrames(InExitFrames)
	, ReleaseHeightThreshold(InHeightThreshold * InReleaseHeightMultiplier)
{
	check(InFootCount > 0);
	IsLockedFlags.Init(false, InFootCount);
	ConsecutiveViolationFrames.Init(0, InFootCount);
	PreviousPosition.Init(FVector::ZeroVector, InFootCount);
	bHasPreviousPosition.Init(false, InFootCount);
}

bool FFootContactDetector::Update(int32 FootIndex, const FVector& WorldPosition)
{
	const float Height = WorldPosition.Y;
	float Speed = 0.0f;
	if (bHasPreviousPosition[FootIndex])
	{
		const FVector& Previous = PreviousPosition[FootIndex];
		const float DeltaX = WorldPosition.X - Previous.X;
		const float DeltaZ = WorldPosition.Z - Previous.Z;
		Speed = FMath::Sqrt(DeltaX * DeltaX + DeltaZ * DeltaZ);
	}

	PreviousPosition[FootIndex] = WorldPosition;
	bHasPreviousPosition[FootIndex] = true;

	const bool bRawContact = Height < HeightThreshold && Speed < SpeedThreshold;

	if (!IsLockedFlags[FootIndex])
	{
		// Entering contact requires the raw criterion, no hysteresis delay.
		if (bRawContact)
		{
			IsLockedFlags[FootIndex] = true;
			ConsecutiveViolationFrames[FootIndex] = 0;
		}
		return IsLockedFlags[FootIndex];
	}

	// Currently locked: stay locked unless the raw criterion is
	// violated for ExitFrames consecutive frames, or the foot is
	// clearly airborne (height > ReleaseHeightThreshold), which
	// releases immediately regardless of the frame counter.
	if (Height > ReleaseHeightThreshold)
	{
		IsLockedFlags[FootIndex] = false;
		ConsecutiveViolationFrames[FootIndex] = 0;
		return false;
	}

	if (!bRawContact)
	{
		++ConsecutiveViolationFrames[FootIndex];
		if (ConsecutiveViolationFrames[FootIndex] >= ExitFrames)
		{
			IsLockedFlags[FootIndex] = false;
			ConsecutiveViolationFrames[FootIndex] = 0;
		}
	}
	else
	{
		ConsecutiveViolationFrames[FootIndex] = 0;
	}

	return IsLockedFlags[FootIndex];
}

void FFootContactDetector::Reset()
{
	for (int32 Index = 0; Index < IsLockedFlags.Num(); ++Index)
	{
		IsLockedFlags[Index] = false;
		ConsecutiveViolationFrames[Index] = 0;
		bHasPreviousPosition[Index] = false;
	}
}
