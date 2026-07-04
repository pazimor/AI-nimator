// Copyright AI-nimator.

#include "AInimatorRigBinder.h"
#include "AInimatorLog.h"

bool FRigBinder::Calibrate(
	const UAInimatorRigMap& RigMap,
	TFunctionRef<FQuat(FName)> GetRigRestWorldRotation,
	TArray<FCalibratedBone>& OutCalibration)
{
	OutCalibration.Reset();

	if (!RigMap.IsMapped(AInimatorSmpl22Skeleton::Pelvis))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: RigMap has no mapping for the pelvis (bone 0) — ")
			TEXT("the retarget parent chain requires at least a mapped root. ")
			TEXT("Refusing to calibrate."));
		return false;
	}

	// SmplBoneIndex -> index into OutCalibration, or INDEX_NONE if that
	// SMPL bone is unmapped (spec §2.1 v1 rule: unmapped bones are
	// simply skipped, never substitute a nearby bone).
	int32 CalibrationIndexForSmplBone[AInimatorSmpl22Skeleton::NumBones];
	for (int32 Index = 0; Index < AInimatorSmpl22Skeleton::NumBones; ++Index)
	{
		CalibrationIndexForSmplBone[Index] = INDEX_NONE;
	}

	for (int32 SmplBoneIndex = 0; SmplBoneIndex < AInimatorSmpl22Skeleton::NumBones; ++SmplBoneIndex)
	{
		const FName TargetBoneName = RigMap.GetTargetBoneName(SmplBoneIndex);
		if (TargetBoneName == NAME_None)
		{
			continue; // spec §2.1: unmapped bone, rotation ignored.
		}

		FCalibratedBone Entry;
		Entry.SmplBoneIndex = SmplBoneIndex;
		Entry.TargetBoneName = TargetBoneName;
		Entry.RigRestWorldRotation = GetRigRestWorldRotation(TargetBoneName);

		// Walk up the SMPL hierarchy to find the nearest MAPPED ancestor
		// (spec §2.2 parenthetical: "parent = parent mappé le plus
		// proche dans le rig").
		int32 Ancestor = AInimatorSmpl22Skeleton::ParentIndices[SmplBoneIndex];
		while (Ancestor >= 0 && CalibrationIndexForSmplBone[Ancestor] == INDEX_NONE)
		{
			Ancestor = AInimatorSmpl22Skeleton::ParentIndices[Ancestor];
		}
		Entry.MappedParentCalibrationIndex =
			(Ancestor >= 0) ? CalibrationIndexForSmplBone[Ancestor] : INDEX_NONE;

		CalibrationIndexForSmplBone[SmplBoneIndex] = OutCalibration.Num();
		OutCalibration.Add(Entry);
	}

	if (OutCalibration.Num() == 0)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: RigMap has zero mapped bones — nothing to ")
			TEXT("calibrate."));
		return false;
	}

	return true;
}

FQuat FRigBinder::ComputeLocalRotation(
	const TArray<FQuat>& SmplWorldRotations,
	const TArray<FCalibratedBone>& Calibration,
	int32 CalibrationIndex,
	TArray<FQuat>& OutTargetWorldRotations)
{
	check(Calibration.IsValidIndex(CalibrationIndex));
	check(OutTargetWorldRotations.Num() == Calibration.Num());

	const FCalibratedBone& Entry = Calibration[CalibrationIndex];
	check(SmplWorldRotations.IsValidIndex(Entry.SmplBoneIndex));

	// worldRot_target(bone) = R_smpl_world(bone) * worldRot_rig_rest(bone)
	// (spec §2.2 formula, literal order — quaternion composition applies
	// the right-hand operand's frame first, matching the FMatrix
	// `ParentRotation * LocalRotation` convention already used by
	// AInimatorForwardKinematics for consistency across this codebase).
	const FQuat WorldRotTarget =
		SmplWorldRotations[Entry.SmplBoneIndex] * Entry.RigRestWorldRotation;
	OutTargetWorldRotations[CalibrationIndex] = WorldRotTarget;

	// localRot_target(bone) = worldRot_target(parent)^-1 * worldRot_target(bone)
	if (Entry.MappedParentCalibrationIndex == INDEX_NONE)
	{
		// No mapped ancestor: this bone is retargeted directly against
		// the rig's own root frame (identity parent), i.e. its local
		// rotation equals its world rotation.
		return WorldRotTarget;
	}

	check(Calibration.IsValidIndex(Entry.MappedParentCalibrationIndex));
	const FQuat& WorldRotTargetParent =
		OutTargetWorldRotations[Entry.MappedParentCalibrationIndex];
	return WorldRotTargetParent.Inverse() * WorldRotTarget;
}

void FRigBinder::RetargetFrame(
	const TArray<FQuat>& SmplWorldRotations,
	const TArray<FCalibratedBone>& Calibration,
	TArray<FQuat>& OutTargetWorldRotationsScratch,
	TArray<FName>& OutBoneNames,
	TArray<FQuat>& OutBoneLocalRotations)
{
	OutTargetWorldRotationsScratch.SetNumUninitialized(Calibration.Num());
	OutBoneNames.SetNumUninitialized(Calibration.Num());
	OutBoneLocalRotations.SetNumUninitialized(Calibration.Num());

	// Calibration is parent-before-child by construction (Calibrate
	// iterates SmplBoneIndex in ascending order, which is already
	// parent-before-child for AInimatorSmpl22Skeleton::ParentIndices) —
	// a single forward pass is sufficient for ComputeLocalRotation to
	// always find its parent's worldRot_target already written.
	for (int32 Index = 0; Index < Calibration.Num(); ++Index)
	{
		OutBoneNames[Index] = Calibration[Index].TargetBoneName;
		OutBoneLocalRotations[Index] = ComputeLocalRotation(
			SmplWorldRotations, Calibration, Index, OutTargetWorldRotationsScratch);
	}
}

float FRigBinder::ComputeRigScale(float RigRestPelvisHeightMeters)
{
	return RigRestPelvisHeightMeters / CanonicalSmplPelvisHeightMeters;
}

float FRigBinder::ComputeRigScaleFromCentimeters(float RigRestPelvisHeightCentimeters)
{
	// Unreal's native unit is centimeters; SMPL/the contract's
	// coord_system values are meters (spec §2.3 explicit reminder) —
	// convert before dividing.
	return ComputeRigScale(RigRestPelvisHeightCentimeters / 100.0f);
}
