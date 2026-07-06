// Copyright AI-nimator.

#include "AInimatorControlPreset.h"
#include "AInimatorLog.h"

bool UAInimatorControlPreset::BuildControlVector(
	const TArray<FString>& ControlLayout,
	TArray<float>& OutControlVector) const
{
	OutControlVector.SetNumZeroed(ControlLayout.Num());
	for (int32 Index = 0; Index < ControlLayout.Num(); ++Index)
	{
		const FString& ChannelName = ControlLayout[Index];
		if (ChannelName == TEXT("vx"))
		{
			OutControlVector[Index] = Vx;
		}
		else if (ChannelName == TEXT("vz"))
		{
			OutControlVector[Index] = Vz;
		}
		else if (ChannelName == TEXT("aim_x"))
		{
			if (!bHasAim)
			{
				UE_LOG(LogAInimator, Error,
					TEXT("AInimator: preset '%s' has no aim but the bundle's ")
					TEXT("control_layout requires aim_x."), *PresetName);
				return false;
			}
			OutControlVector[Index] = AimX;
		}
		else if (ChannelName == TEXT("aim_z"))
		{
			if (!bHasAim)
			{
				UE_LOG(LogAInimator, Error,
					TEXT("AInimator: preset '%s' has no aim but the bundle's ")
					TEXT("control_layout requires aim_z."), *PresetName);
				return false;
			}
			OutControlVector[Index] = AimZ;
		}
		else
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: unknown control_layout channel '%s' in ")
				TEXT("manifest — this plugin build does not recognise it."),
				*ChannelName);
			return false;
		}
	}
	return true;
}

void UAInimatorControlPreset::HydrateFromRaw(
	const FString& InName,
	float InVx,
	float InVz,
	bool bInHasAim,
	float InAimX,
	float InAimZ,
	const FString& InPrompt,
	const TArray<float>& InPromptEmb)
{
	PresetName = InName;
	Vx = InVx;
	Vz = InVz;
	bHasAim = bInHasAim;
	AimX = InAimX;
	AimZ = InAimZ;
	Prompt = InPrompt;
	bHasPromptEmb = InPromptEmb.Num() > 0;
	PromptEmb = InPromptEmb;
}
