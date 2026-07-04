// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AInimatorRigMap.h"
#include "AInimatorRigBinder.h"
#include "AInimatorCharacterComponent.generated.h"

class UAInimatorControllerRuntime;
class UAInimatorPostProcessComponent;
class UAInimatorControlPreset;
class UPoseableMeshComponent;
class USkeletalMeshComponent;

/**
 * B3-bis "binding rig + pilotage par prompt" component
 * (`apps/spec/rig_binding.md`). Attaches an already-loaded
 * `UAInimatorControllerRuntime` (+ optionally its
 * `UAInimatorPostProcessComponent`) to a rigged character and:
 * - retargets the controller's SMPL-22 output onto the character's own
 *   skeleton every frame via `FRigBinder` + `UAInimatorRigMap`
 *   (`ApplyRigBinding`);
 * - applies scaled root motion to the owning actor (spec §2.3);
 * - exposes the prompt channel (`SetPrompt`/`SetPromptEmbedding`/
 *   `ClearPrompt`) with an embedding-space cross-fade (spec §3).
 *
 * This component is entirely OPTIONAL: the B2 capsule/pawn path keeps
 * working with no RigMap and no `UAInimatorCharacterComponent` at all
 * (spec §5 acceptance: "le chemin capsule (B1/B2) reste fonctionnel").
 * When `RigMap` is unset, `TickCharacter` still drives the prompt
 * cross-fade and root motion but skips retargeting/pose application
 * entirely (logged once, not every frame).
 *
 * Call order each frame (mirrors `UAInimatorPostProcessComponent`'s own
 * documented call order, one level further downstream):
 * ```
 * Runtime->Tick();
 * PostProcess->TickPostProcess(Dt, Vx, Vz);   // optional, B4
 * CharacterComponent->TickCharacter(Dt);       // this component
 * ```
 */
UCLASS(ClassGroup = (AInimator), meta = (BlueprintSpawnableComponent))
class AINIMATOR_API UAInimatorCharacterComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UAInimatorCharacterComponent();

	/** The runtime driving this character. Must already have a bundle
	 *  loaded before TickCharacter is called. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Character")
	TObjectPtr<UAInimatorControllerRuntime> Runtime;

	/** Optional B4 post-process (foot-lock IK); when set, this
	 *  component reads its corrected ankle/knee positions instead of
	 *  the raw FK pose for the legs it corrects. Foot-lock correction
	 *  is a WORLD-POSITION concept (spec footlock_blending.md §3), not
	 *  directly a rotation, so it currently only affects root motion
	 *  bookkeeping consistency, not the retarget math itself — the
	 *  retarget always follows the controller's raw rotation stream per
	 *  rig_binding.md §2.3 ("le foot-lock B4 opère avant retarget"). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Character")
	TObjectPtr<UAInimatorPostProcessComponent> PostProcess;

	/** The 22-entry SMPL -> rig bone name map for this character. Unset
	 *  = retargeting/pose application is skipped (capsule/pawn path
	 *  still works via Runtime alone). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Character|Rig")
	TObjectPtr<UAInimatorRigMap> RigMap;

	/** Target character mesh. Must already be posed in its reference
	 *  pose bind hierarchy (standard SkeletalMesh asset requirement) —
	 *  calibration reads its CURRENT bone world rotations at
	 *  CalibrateRig() time, so call CalibrateRig() before the mesh has
	 *  been posed by anything else this frame (typically once, at
	 *  BeginPlay, right after LoadBundle). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Character|Rig")
	TObjectPtr<UPoseableMeshComponent> PoseableMesh;

	/** Cross-fade duration for a prompt change, in the EMBEDDING space
	 *  (spec §3): "changement de prompt = cross-fade dans l'espace
	 *  d'embedding, lerp linéaire emb_old -> emb_new sur 0.3s (défaut,
	 *  exposé)". ⚠ HEURISTIC, NOT VALIDATED: the controller was never
	 *  trained on interpolated embeddings, so intermediate frames during
	 *  the fade sample an input the model has never seen — visual
	 *  quality during the transition must be validated in-engine before
	 *  shipping (spec §3 / §5). If transitions are visibly degraded, the
	 *  documented fallback (spec §3) is a hard prompt switch + a
	 *  POSE-space cross-fade instead, reusing
	 *  `UAInimatorPostProcessComponent`'s existing idle<->move blend
	 *  machinery (`FIdleMoveBlender`) as the mechanism — NOT implemented
	 *  here, since it requires Pazimor's visual validation call first
	 *  (open question, see session report).
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Character|Prompt")
	float PromptCrossFadeSeconds = 0.3f;

	/** Sets the active prompt from a preset's precomputed embedding
	 *  (spec §3 / inference_contract.md §4) and starts the cross-fade.
	 *  No-op (with a warning) if Preset has no PromptEmb. */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Character|Prompt")
	bool SetPrompt(UAInimatorControlPreset* Preset);

	/** Sets the active prompt from a raw embedding vector (length must
	 *  equal the bundle's prompt_emb_channels) and starts the
	 *  cross-fade. */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Character|Prompt")
	bool SetPromptEmbedding(const TArray<float>& PromptEmbedding);

	/** Reverts to the bundle's learned null embedding (never zeros,
	 *  inference_contract.md §4) and starts the cross-fade. */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Character|Prompt")
	bool ClearPrompt();

	/**
	 * Calibrates FRigBinder against PoseableMesh's CURRENT bone world
	 * rotations (spec §2.2 "au bind"). Call once after the mesh is in
	 * its reference/rest pose (typically right after BeginPlay, before
	 * the first TickCharacter). Also computes RigScale from the rig's
	 * rest pelvis height (spec §2.3).
	 *
	 * Returns false (logged) if RigMap/PoseableMesh are unset or
	 * calibration fails (e.g. pelvis unmapped) — TickCharacter then
	 * skips retargeting entirely until CalibrateRig succeeds.
	 */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Character|Rig")
	bool CalibrateRig();

	/**
	 * Runs one frame: advances the prompt cross-fade, retargets the
	 * controller's current pose onto PoseableMesh (if calibrated), and
	 * applies scaled root motion to the owning actor. Call AFTER
	 * `Runtime->Tick()` (and after `PostProcess->TickPostProcess()` if
	 * B4 post-processing is in use) — see class comment call order.
	 */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Character")
	void TickCharacter(float DeltaSeconds);

	/** Current root-motion scale factor (spec §2.3), 1.0 until
	 *  CalibrateRig() succeeds. */
	UFUNCTION(BlueprintPure, Category = "AInimator|Character|Rig")
	float GetRigScale() const { return RigScale; }

	/** Whether CalibrateRig() has succeeded and retargeting is active. */
	UFUNCTION(BlueprintPure, Category = "AInimator|Character|Rig")
	bool IsRigCalibrated() const { return bIsCalibrated; }

protected:
	virtual void BeginPlay() override;

private:
	/** Shared implementation for SetPrompt/SetPromptEmbedding/ClearPrompt:
	 *  captures the CURRENT (possibly mid-fade) embedding as the fade's
	 *  "old" endpoint, sets NewTargetEmb as the "new" endpoint, and
	 *  resets the fade timer to zero (spec §3: always a fresh 0.3s lerp
	 *  from wherever the embedding currently is, never a queued/second
	 *  fade — simplest literal reading of the spec, no invented queuing
	 *  behavior). */
	void StartPromptCrossFade(const TArray<float>& NewTargetEmb);

	/** Advances the embedding cross-fade by DeltaSeconds and pushes the
	 *  resulting lerp to Runtime->SetPromptEmbedding every frame while
	 *  active (spec §3's cross-fade is continuous, not just at the
	 *  start/end). */
	void TickPromptCrossFade(float DeltaSeconds);

	TArray<float> PromptEmbFadeFrom;
	TArray<float> PromptEmbFadeTo;
	TArray<float> PromptEmbFadeScratch;
	float PromptFadeElapsedSeconds = 0.0f;
	bool bPromptFadeActive = false;

	TArray<FRigBinder::FCalibratedBone> Calibration;
	bool bIsCalibrated = false;
	float RigScale = 1.0f;

	// Reused scratch buffers, no per-tick heap churn.
	TArray<FMatrix> ScratchGlobalRotationMatrices;
	TArray<FVector> ScratchRootLocalPositions;
	TArray<FQuat> ScratchSmplWorldRotations;
	TArray<FQuat> ScratchTargetWorldRotations;
	TArray<FName> ScratchBoneNames;
	TArray<FQuat> ScratchBoneLocalRotations;
};
