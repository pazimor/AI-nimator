// Copyright AI-nimator.

#include "AInimatorRigMap.h"
#include "AInimatorLog.h"

namespace
{
	/** SMPL bone names, in AInimatorSmpl22Skeleton order — used both as
	 *  the Details panel label (FAInimatorRigMapEntry::SmplBoneName) and
	 *  as the auto-map alias-list key. */
	const TCHAR* const SmplBoneLabels[AInimatorSmpl22Skeleton::NumBones] = {
		TEXT("pelvis"),
		TEXT("leftHip"), TEXT("rightHip"),
		TEXT("spine1"),
		TEXT("leftKnee"), TEXT("rightKnee"),
		TEXT("spine2"),
		TEXT("leftAnkle"), TEXT("rightAnkle"),
		TEXT("spine3"),
		TEXT("leftFoot"), TEXT("rightFoot"),
		TEXT("neck"),
		TEXT("leftCollar"), TEXT("rightCollar"),
		TEXT("head"),
		TEXT("leftShoulder"), TEXT("rightShoulder"),
		TEXT("leftElbow"), TEXT("rightElbow"),
		TEXT("leftWrist"), TEXT("rightWrist"),
	};

	/**
	 * Common bone-name aliases per SMPL bone, tried in order, UE5
	 * Mannequin (Manny/Quinn, both legacy UE4 and UE5 "root/pelvis/
	 * spine_0.../thigh_l/..." conventions) first, then generic
	 * Mixamo/Biped-style names. This is a best-effort authoring
	 * convenience (spec deliverable list) — NOT part of the canonical
	 * retargeting math (which only ever depends on `Entries`, however
	 * populated); a project with an unusual rig always falls back to
	 * hand-editing `Entries` in the Details panel.
	 */
	const TCHAR* const AliasesPerBone[AInimatorSmpl22Skeleton::NumBones][6] = {
		/* pelvis        */ {TEXT("pelvis"), TEXT("Pelvis"), TEXT("Hips"), TEXT("root"), nullptr, nullptr},
		/* leftHip       */ {TEXT("thigh_l"), TEXT("Thigh_L"), TEXT("LeftUpLeg"), TEXT("UpperLeg_L"), nullptr, nullptr},
		/* rightHip      */ {TEXT("thigh_r"), TEXT("Thigh_R"), TEXT("RightUpLeg"), TEXT("UpperLeg_R"), nullptr, nullptr},
		/* spine1        */ {TEXT("spine_01"), TEXT("Spine_01"), TEXT("Spine"), TEXT("Spine1"), nullptr, nullptr},
		/* leftKnee      */ {TEXT("calf_l"), TEXT("Calf_L"), TEXT("LeftLeg"), TEXT("LowerLeg_L"), nullptr, nullptr},
		/* rightKnee     */ {TEXT("calf_r"), TEXT("Calf_R"), TEXT("RightLeg"), TEXT("LowerLeg_R"), nullptr, nullptr},
		/* spine2        */ {TEXT("spine_02"), TEXT("Spine_02"), TEXT("Spine1"), TEXT("Spine2"), nullptr, nullptr},
		/* leftAnkle     */ {TEXT("foot_l"), TEXT("Foot_L"), TEXT("LeftFoot"), nullptr, nullptr, nullptr},
		/* rightAnkle    */ {TEXT("foot_r"), TEXT("Foot_R"), TEXT("RightFoot"), nullptr, nullptr, nullptr},
		/* spine3        */ {TEXT("spine_03"), TEXT("Spine_03"), TEXT("Spine2"), TEXT("Spine3"), nullptr, nullptr},
		/* leftFoot      */ {TEXT("ball_l"), TEXT("Ball_L"), TEXT("LeftToeBase"), TEXT("Toe_L"), nullptr, nullptr},
		/* rightFoot     */ {TEXT("ball_r"), TEXT("Ball_R"), TEXT("RightToeBase"), TEXT("Toe_R"), nullptr, nullptr},
		/* neck           */ {TEXT("neck_01"), TEXT("Neck_01"), TEXT("Neck"), nullptr, nullptr, nullptr},
		/* leftCollar    */ {TEXT("clavicle_l"), TEXT("Clavicle_L"), TEXT("LeftShoulder"), TEXT("Shoulder_L"), nullptr, nullptr},
		/* rightCollar   */ {TEXT("clavicle_r"), TEXT("Clavicle_R"), TEXT("RightShoulder"), TEXT("Shoulder_R"), nullptr, nullptr},
		/* head          */ {TEXT("head"), TEXT("Head"), nullptr, nullptr, nullptr, nullptr},
		/* leftShoulder  */ {TEXT("upperarm_l"), TEXT("UpperArm_L"), TEXT("LeftArm"), TEXT("Arm_L"), nullptr, nullptr},
		/* rightShoulder */ {TEXT("upperarm_r"), TEXT("UpperArm_R"), TEXT("RightArm"), TEXT("Arm_R"), nullptr, nullptr},
		/* leftElbow     */ {TEXT("lowerarm_l"), TEXT("LowerArm_L"), TEXT("LeftForeArm"), TEXT("ForeArm_L"), nullptr, nullptr},
		/* rightElbow    */ {TEXT("lowerarm_r"), TEXT("LowerArm_R"), TEXT("RightForeArm"), TEXT("ForeArm_R"), nullptr, nullptr},
		/* leftWrist     */ {TEXT("hand_l"), TEXT("Hand_L"), TEXT("LeftHand"), nullptr, nullptr, nullptr},
		/* rightWrist    */ {TEXT("hand_r"), TEXT("Hand_R"), TEXT("RightHand"), nullptr, nullptr, nullptr},
	};
}

UAInimatorRigMap::UAInimatorRigMap()
{
	EnsureCanonicalEntryCount();
}

#if WITH_EDITOR
void UAInimatorRigMap::PostLoad()
{
	Super::PostLoad();
	EnsureCanonicalEntryCount();
}

void UAInimatorRigMap::PostInitProperties()
{
	Super::PostInitProperties();
	EnsureCanonicalEntryCount();
}
#endif

void UAInimatorRigMap::EnsureCanonicalEntryCount()
{
	if (Entries.Num() == AInimatorSmpl22Skeleton::NumBones)
	{
		// Still make sure labels are correct (cheap, idempotent) in case
		// an older asset serialized empty/stale labels.
		for (int32 Index = 0; Index < AInimatorSmpl22Skeleton::NumBones; ++Index)
		{
			Entries[Index].SmplBoneName = SmplBoneLabels[Index];
		}
		return;
	}

	TArray<FAInimatorRigMapEntry> Rebuilt;
	Rebuilt.SetNum(AInimatorSmpl22Skeleton::NumBones);
	for (int32 Index = 0; Index < AInimatorSmpl22Skeleton::NumBones; ++Index)
	{
		Rebuilt[Index].SmplBoneName = SmplBoneLabels[Index];
		// Preserve any pre-existing mapping at the same index (best
		// effort — a resize should not normally happen post-authoring).
		if (Entries.IsValidIndex(Index))
		{
			Rebuilt[Index].TargetBoneName = Entries[Index].TargetBoneName;
		}
	}
	Entries = MoveTemp(Rebuilt);
}

FName UAInimatorRigMap::GetTargetBoneName(int32 SmplBoneIndex) const
{
	if (!Entries.IsValidIndex(SmplBoneIndex))
	{
		return NAME_None;
	}
	return Entries[SmplBoneIndex].TargetBoneName;
}

bool UAInimatorRigMap::IsMapped(int32 SmplBoneIndex) const
{
	return GetTargetBoneName(SmplBoneIndex) != NAME_None;
}

int32 UAInimatorRigMap::AutoMapByCommonNames(
	const TArray<FName>& CandidateBoneNames,
	bool bOverwriteExisting)
{
	EnsureCanonicalEntryCount();

	int32 NewlyMapped = 0;
	for (int32 SmplBoneIndex = 0; SmplBoneIndex < AInimatorSmpl22Skeleton::NumBones; ++SmplBoneIndex)
	{
		if (!bOverwriteExisting && Entries[SmplBoneIndex].TargetBoneName != NAME_None)
		{
			continue;
		}

		for (const TCHAR* Alias : AliasesPerBone[SmplBoneIndex])
		{
			if (!Alias)
			{
				break;
			}
			const FName AliasName(Alias);
			for (const FName& Candidate : CandidateBoneNames)
			{
				if (Candidate.IsEqual(AliasName, ENameCase::IgnoreCase))
				{
					Entries[SmplBoneIndex].TargetBoneName = Candidate;
					++NewlyMapped;
					break;
				}
			}
			if (Entries[SmplBoneIndex].TargetBoneName != NAME_None)
			{
				break;
			}
		}
	}

	UE_LOG(LogAInimator, Log,
		TEXT("AInimator: RigMap auto-map matched %d/%d SMPL bone(s) against ")
		TEXT("%d candidate rig bone(s)."),
		NewlyMapped, AInimatorSmpl22Skeleton::NumBones, CandidateBoneNames.Num());
	return NewlyMapped;
}
