// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "AInimatorSmpl22Skeleton.h"
#include "AInimatorRigMap.generated.h"

class USkeletalMesh;
class USkeletalMeshComponent;

/**
 * One SMPL-22 bone -> target-rig bone mapping row (`apps/spec/rig_binding.md`
 * §2.1). `TargetBoneName == NAME_None` means "not mapped" — v1 rule: an
 * unmapped SMPL bone's rotation is simply ignored (documented, spec §2.1 —
 * the 22 SMPL bones already cover standard humanoids so this should rarely
 * matter in practice).
 */
USTRUCT(BlueprintType)
struct AINIMATOR_API FAInimatorRigMapEntry
{
	GENERATED_BODY()

	/** Read-only label mirroring AInimatorSmpl22Skeleton's bone order —
	 *  purely informational in the Details panel (index is what actually
	 *  drives lookups; see UAInimatorRigMap::Entries being fixed-size 22
	 *  and always in SMPL22_BONE_ORDER order). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "AInimator|RigMap")
	FString SmplBoneName;

	/** Target rig bone this SMPL bone drives, or NAME_None if unmapped. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|RigMap")
	FName TargetBoneName = NAME_None;
};

/**
 * Explicit SMPL-22 -> arbitrary-rig bone map + bind-time calibration
 * (`apps/spec/rig_binding.md` §2 — the CANONICAL, cross-engine-parity
 * retargeting path; the native IK Retargeter path, §2.4, is optional and
 * NOT covered by parity).
 *
 * One `UAInimatorRigMap` asset targets one specific rig's bone-naming
 * convention (e.g. one per character skeleton or one per shared
 * convention such as the UE5 Mannequin). `Entries` is always exactly 22
 * elements, indexed by `AInimatorSmpl22Skeleton` bone index, in
 * `AInimatorSmpl22Skeleton::` order — never resized at runtime.
 *
 * This asset stores ONLY the name mapping (authoring-time, versioned).
 * The bind-time calibration (rest-pose world orientations, §2.2) is
 * computed fresh every time `FRigBinder::Calibrate` runs against an
 * actual `USkeletalMeshComponent` — it is NOT cached on this asset,
 * because the same RigMap can be reused across multiple
 * SkeletalMeshComponent instances (possibly with slightly different
 * ref poses) without invalidating anything here.
 */
UCLASS(BlueprintType)
class AINIMATOR_API UAInimatorRigMap : public UDataAsset
{
	GENERATED_BODY()

public:
	UAInimatorRigMap();

	/** Fixed-size (22) SMPL bone -> target rig bone name entries, in
	 *  AInimatorSmpl22Skeleton bone-index order. Edited in the Details
	 *  panel (each row's TargetBoneName is a plain FName — pick it to
	 *  match your SkeletalMesh's skeleton, e.g. "thigh_l" for the
	 *  UE5 Mannequin). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|RigMap")
	TArray<FAInimatorRigMapEntry> Entries;

	/** Returns Entries[SmplBoneIndex].TargetBoneName, or NAME_None if
	 *  SmplBoneIndex is out of range or unmapped. */
	UFUNCTION(BlueprintPure, Category = "AInimator|RigMap")
	FName GetTargetBoneName(int32 SmplBoneIndex) const;

	/** Whether SmplBoneIndex has a non-empty mapping. */
	UFUNCTION(BlueprintPure, Category = "AInimator|RigMap")
	bool IsMapped(int32 SmplBoneIndex) const;

	/**
	 * Best-effort auto-mapping by common bone-name conventions (editor
	 * authoring convenience, `apps/spec/rig_binding.md` deliverable list:
	 * "auto-mapping best-effort par noms d'os courants"). Tries, per SMPL
	 * bone, a short list of common aliases (UE5 Mannequin names first,
	 * then generic Mixamo/Biped-style names) against
	 * `CandidateBoneNames` (typically `SkeletalMesh->GetRefSkeleton()`'s
	 * bone names) and fills `Entries` for every SMPL bone that finds a
	 * case-insensitive match. Never touches an already-mapped entry
	 * unless `bOverwriteExisting` is true (so re-running auto-map after
	 * hand-tuning a few bones does not clobber the manual edits by
	 * default).
	 *
	 * Returns the number of newly-mapped entries.
	 */
	UFUNCTION(BlueprintCallable, Category = "AInimator|RigMap")
	int32 AutoMapByCommonNames(
		const TArray<FName>& CandidateBoneNames,
		bool bOverwriteExisting = false);

	/** Ensures Entries has exactly AInimatorSmpl22Skeleton::NumBones rows,
	 *  labelled SmplBoneName from the canonical bone order — call after
	 *  construction and whenever the asset is loaded, so a hand-edited
	 *  or serialized-before-this-field-existed asset self-repairs. */
	void EnsureCanonicalEntryCount();

#if WITH_EDITOR
	virtual void PostLoad() override;
	virtual void PostInitProperties() override;
#endif
};
