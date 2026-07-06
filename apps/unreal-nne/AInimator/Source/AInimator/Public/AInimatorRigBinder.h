// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "AInimatorSmpl22Skeleton.h"
#include "AInimatorRigMap.h"

class USkeletalMeshComponent;
class UPoseableMeshComponent;

/**
 * SMPL-22 -> arbitrary-rig retargeting (`apps/spec/rig_binding.md` §2 —
 * the CANONICAL cross-engine-parity path). Pure math class (no UObject),
 * so its core (`ComputeLocalRotation`) is unit-testable against a
 * synthetic mini-rig without spinning up a World.
 *
 * Formula (spec §2.2, applied literally, no invention):
 * ```
 * worldRot_target(bone) = R_smpl_world(bone) * worldRot_rig_rest(bone)
 * localRot_target(bone) = worldRot_target(parent)^-1 * worldRot_target(bone)
 * ```
 * where `R_smpl_world` is the FK-composed SMPL world rotation
 * (`AInimatorForwardKinematics::ComputeJointPositions`'s
 * `OutGlobalRotations`, expressed root-local — i.e. pelvis-relative,
 * which is exactly what the spec's `R_smpl_world` means here: "monde"
 * in the SMPL root-local frame, not the engine's absolute world frame;
 * the engine-absolute placement of the whole rig is handled separately
 * by the actor transform / root motion, §2.3), `worldRot_rig_rest(bone)`
 * the rig's OWN rest-pose world orientation captured once at bind time
 * (`Calibrate`), and `parent` the nearest MAPPED ancestor in the rig
 * (spec §2.2 parenthetical).
 *
 * Bone translations are NEVER modified (spec §2.2: "les translations
 * d'os ne sont jamais modifiées") — only rotations are retargeted; the
 * rig's own proportions are preserved. Root motion (translation of the
 * actor) is handled entirely outside this class, scaled by RigScale
 * (spec §2.3), by whichever component owns the `USkeletalMeshComponent`'s
 * owning actor.
 */
class AINIMATOR_API FRigBinder
{
public:
	/**
	 * One calibration entry: for a mapped SMPL bone, its rig-rest-pose
	 * WORLD rotation (`O_self` in the spec's prose) and the SMPL bone
	 * index of its nearest MAPPED ancestor (INDEX_NONE if this bone
	 * itself is the mapped root, e.g. pelvis, or if no ancestor is
	 * mapped — treated as parented directly to the rig root in that
	 * case, since the retarget math needs *some* parent frame to divide
	 * out).
	 */
	struct FCalibratedBone
	{
		int32 SmplBoneIndex = INDEX_NONE;
		FName TargetBoneName = NAME_None;
		FQuat RigRestWorldRotation = FQuat::Identity;
		/** Index into this same Calibration array of the nearest mapped
		 *  ancestor, or INDEX_NONE. */
		int32 MappedParentCalibrationIndex = INDEX_NONE;
	};

	/**
	 * Calibrates against a target skeletal mesh's reference pose (spec
	 * §2.2 step 1-2): for every SMPL bone mapped by RigMap, records the
	 * rig's rest-pose bone WORLD rotation, and resolves the nearest
	 * mapped ancestor per the skeleton hierarchy.
	 *
	 * Parameters
	 * ----------
	 * RigMap : the 22-entry SMPL -> rig bone name map.
	 * GetRigRestWorldRotation : callback returning the given rig bone's
	 *     rest-pose WORLD rotation (as a quaternion) — kept as a
	 *     delegate rather than a hard `USkeletalMeshComponent&`
	 *     dependency so `Calibrate` (and hence this whole class) stays
	 *     testable with a synthetic mini-rig with no engine mesh asset
	 *     at all.
	 * OutCalibration : filled with one FCalibratedBone per SMPL bone
	 *     that RigMap maps (skips unmapped bones entirely, per spec
	 *     §2.1 v1 rule); order follows AInimatorSmpl22Skeleton bone
	 *     index order.
	 *
	 * Returns
	 * -------
	 * bool
	 *     False if RigMap has zero mapped bones (nothing to calibrate;
	 *     logged) or if the pelvis (bone 0) itself is unmapped (the
	 *     retarget parent chain needs at least a mapped root — logged).
	 */
	static bool Calibrate(
		const UAInimatorRigMap& RigMap,
		TFunctionRef<FQuat(FName)> GetRigRestWorldRotation,
		TArray<FCalibratedBone>& OutCalibration);

	/**
	 * Computes the rig-local rotation to apply to one calibrated bone
	 * this frame, given the SMPL world rotations already FK-composed
	 * for the whole skeleton this frame (spec §2.2 formula, applied
	 * literally).
	 *
	 * Parameters
	 * ----------
	 * SmplWorldRotations : per-SMPL-bone world rotation this frame,
	 *     indexed by AInimatorSmpl22Skeleton bone index (as produced by
	 *     AInimatorForwardKinematics::ComputeJointPositions's
	 *     OutGlobalRotations, converted to FQuat).
	 * Calibration : the array produced by Calibrate.
	 * CalibrationIndex : index into Calibration identifying which
	 *     calibrated bone to compute.
	 * OutTargetWorldRotations : scratch array, same length and order as
	 *     Calibration, that this function reads previously-computed
	 *     entries from (for `MappedParentCalibrationIndex`) and writes
	 *     this bone's own worldRot_target into — callers MUST process
	 *     Calibration in an order where every bone's mapped parent has
	 *     already been computed (Calibrate returns entries in SMPL bone
	 *     index order, which is already parent-before-child for the
	 *     skeleton's hierarchy, so a simple forward loop suffices — see
	 *     RetargetFrame).
	 *
	 * Returns
	 * -------
	 * FQuat
	 *     localRot_target(bone), ready to assign directly as the target
	 *     rig bone's LOCAL rotation.
	 */
	static FQuat ComputeLocalRotation(
		const TArray<FQuat>& SmplWorldRotations,
		const TArray<FCalibratedBone>& Calibration,
		int32 CalibrationIndex,
		TArray<FQuat>& OutTargetWorldRotations);

	/**
	 * Retargets one full frame: computes localRot_target for every
	 * calibrated bone (in order — Calibration is parent-before-child by
	 * construction) and writes (TargetBoneName, localRotation) pairs to
	 * OutBoneLocalRotations, ready to feed a UPoseableMeshComponent
	 * (`SetBoneRotationByName`, `EBoneSpaces::PelvisSpace` == world?
	 *  NO — see the class comment in AInimatorPoseableMeshApplier.h;
	 * this function only produces LOCAL rotations, application is that
	 * other class's job).
	 */
	static void RetargetFrame(
		const TArray<FQuat>& SmplWorldRotations,
		const TArray<FCalibratedBone>& Calibration,
		TArray<FQuat>& OutTargetWorldRotationsScratch,
		TArray<FName>& OutBoneNames,
		TArray<FQuat>& OutBoneLocalRotations);

	/**
	 * Root motion scale (spec §2.3): rig rest-pose pelvis height /
	 * canonical SMPL rest-pose pelvis height. Root local motion
	 * (dForward, dLateral, dHeight) must be multiplied by this factor
	 * before being applied to the actor transform; dYaw is dimensionless
	 * and is never scaled.
	 *
	 * Parameters
	 * ----------
	 * RigRestPelvisHeightMeters : the target rig's rest-pose pelvis
	 *     height above the ground, in METERS (see
	 *     `ComputeRigScaleFromCentimeters` for the common case where the
	 *     engine measurement is in Unreal's native centimeters).
	 *
	 * Returns
	 * -------
	 * float
	 *     RigRestPelvisHeightMeters / CanonicalSmplPelvisHeightMeters.
	 */
	static float ComputeRigScale(float RigRestPelvisHeightMeters);

	/**
	 * Convenience overload for the common case where the rig's rest
	 * pelvis height was measured directly from Unreal world-space
	 * transforms (centimeters, UE's native unit) — divides by 100
	 * first, per spec §2.3's explicit reminder ("attention aux unités
	 * UE en cm vs SMPL en m").
	 */
	static float ComputeRigScaleFromCentimeters(float RigRestPelvisHeightCentimeters);

	/** Canonical SMPL-22 rest-pose pelvis height (meters) referenced by
	 *  spec §2.3 ("~0.91 m sur le squelette canonique") — the
	 *  denominator of ComputeRigScale. */
	static constexpr float CanonicalSmplPelvisHeightMeters = 0.91f;
};
