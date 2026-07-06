// Copyright AI-nimator.

#include "AInimatorPoseableMeshApplier.h"
#include "AInimatorLog.h"
#include "Components/PoseableMeshComponent.h"

namespace AInimatorPoseableMeshApplier
{
	void ApplyFrame(
		UPoseableMeshComponent* PoseableMesh,
		const TArray<FName>& BoneNames,
		const TArray<FQuat>& BoneLocalRotations)
	{
		if (!PoseableMesh)
		{
			UE_LOG(LogAInimator, Warning,
				TEXT("AInimator: ApplyFrame called with a null PoseableMesh; ")
				TEXT("skipping this frame's retarget application."));
			return;
		}
		check(BoneNames.Num() == BoneLocalRotations.Num());

		for (int32 Index = 0; Index < BoneNames.Num(); ++Index)
		{
			const int32 BoneIndex = PoseableMesh->GetBoneIndex(BoneNames[Index]);
			if (BoneIndex == INDEX_NONE)
			{
				// Logged at Verbose (not Warning/Error): a RigMap authored
				// against a superset skeleton naturally has entries that
				// don't exist on every target mesh variant — this is
				// expected, not a contract violation, so it must not spam
				// the log every frame.
				UE_LOG(LogAInimator, Verbose,
					TEXT("AInimator: RigMap target bone '%s' not found on ")
					TEXT("PoseableMesh's skeleton; skipping."),
					*BoneNames[Index].ToString());
				continue;
			}
			PoseableMesh->SetBoneRotationByName(
				BoneNames[Index], BoneLocalRotations[Index].Rotator(), EBoneSpaces::LocalSpace);
		}
	}
}
