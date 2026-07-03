// Copyright AI-nimator.

#include "AInimatorNormalizer.h"
#include "AInimatorLog.h"
#include "AInimatorContractConstants.h"

FNormalizer::FNormalizer(
	const FAInimatorManifest& InManifest,
	const FAInimatorNormStats& InStats)
	: Manifest(InManifest)
	, Stats(InStats)
{
}

void FNormalizer::ApplyZNorm(
	const TArray<float>& Values,
	const TArray<float>& Mean,
	const TArray<float>& Std,
	TArray<float>& Out)
{
	check(Values.Num() == Mean.Num());
	check(Values.Num() == Std.Num());
	Out.SetNumUninitialized(Values.Num());
	for (int32 Index = 0; Index < Values.Num(); ++Index)
	{
		const float ClampedStd =
			FMath::Max(Std[Index], AInimatorContract::NormalizationEpsilon);
		Out[Index] = (Values[Index] - Mean[Index]) / ClampedStd;
	}
}

void FNormalizer::ApplyZDenorm(
	const TArray<float>& Values,
	const TArray<float>& Mean,
	const TArray<float>& Std,
	TArray<float>& Out)
{
	check(Values.Num() == Mean.Num());
	check(Values.Num() == Std.Num());
	Out.SetNumUninitialized(Values.Num());
	for (int32 Index = 0; Index < Values.Num(); ++Index)
	{
		Out[Index] = Values[Index] * Std[Index] + Mean[Index];
	}
}

void FNormalizer::NormalizeBoneFrame(
	const TArray<float>& RawBoneFrame,
	TArray<float>& OutNormalized) const
{
	ApplyZNorm(RawBoneFrame, Stats.BoneMean, Stats.BoneStd, OutNormalized);
}

void FNormalizer::NormalizeGlobalFrame(
	const TArray<float>& RawGlobalFrame,
	TArray<float>& OutNormalized) const
{
	ApplyZNorm(RawGlobalFrame, Stats.GlobalMean, Stats.GlobalStd, OutNormalized);
}

void FNormalizer::NormalizeControlVector(
	const TArray<float>& RawControl,
	TArray<float>& OutNormalized) const
{
	const int32 NumStatChannels = Stats.ControlStatChannels.Num();
	if (RawControl.Num() < NumStatChannels)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: control vector has %d entries but norm_stats ")
			TEXT("declares %d z-normed channels; refusing to normalize."),
			RawControl.Num(), NumStatChannels);
		OutNormalized = RawControl;
		return;
	}

	OutNormalized.SetNumUninitialized(RawControl.Num());
	for (int32 Index = 0; Index < NumStatChannels; ++Index)
	{
		const float ClampedStd = FMath::Max(
			Stats.ControlStd[Index], AInimatorContract::NormalizationEpsilon);
		OutNormalized[Index] =
			(RawControl[Index] - Stats.ControlMean[Index]) / ClampedStd;
	}
	// aim_x / aim_z (or any channel beyond the stat-covered ones):
	// unit-norm by construction, passed through unchanged.
	for (int32 Index = NumStatChannels; Index < RawControl.Num(); ++Index)
	{
		OutNormalized[Index] = RawControl[Index];
	}
}

void FNormalizer::DenormalizeBoneDelta(
	const TArray<float>& NormalizedDelta,
	TArray<float>& OutRawDelta) const
{
	ApplyZDenorm(
		NormalizedDelta, Stats.DeltaBoneMean, Stats.DeltaBoneStd, OutRawDelta);
}

void FNormalizer::DenormalizeGlobalDelta(
	const TArray<float>& NormalizedDelta,
	TArray<float>& OutRawDelta) const
{
	ApplyZDenorm(
		NormalizedDelta, Stats.DeltaGlobalMean, Stats.DeltaGlobalStd,
		OutRawDelta);
}
