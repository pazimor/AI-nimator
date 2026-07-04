// Copyright AI-nimator.

#include "AInimatorCharacterComponent.h"
#include "AInimatorControllerRuntime.h"
#include "AInimatorPostProcessComponent.h"
#include "AInimatorControlPreset.h"
#include "AInimatorForwardKinematics.h"
#include "AInimatorSmpl22Skeleton.h"
#include "AInimatorPoseableMeshApplier.h"
#include "AInimatorLog.h"
#include "Components/PoseableMeshComponent.h"
#include "Engine/SkeletalMesh.h"
#include "GameFramework/Actor.h"

namespace
{
	/** Converts one bone's rotation6d (Gram-Schmidt columns b1,b2,b3, see
	 *  AInimatorForwardKinematics::SixDToRotationMatrix) FMatrix output
	 *  into an FQuat, matching the same column convention. */
	FQuat MatrixToQuat(const FMatrix& RotationMatrix)
	{
		return FQuat(RotationMatrix);
	}
}

UAInimatorCharacterComponent::UAInimatorCharacterComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UAInimatorCharacterComponent::BeginPlay()
{
	Super::BeginPlay();
}

bool UAInimatorCharacterComponent::CalibrateRig()
{
	bIsCalibrated = false;
	Calibration.Reset();

	if (!RigMap)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: CalibrateRig called with no RigMap set; ")
			TEXT("retargeting stays disabled (capsule/pawn path unaffected, ")
			TEXT("rig_binding.md §5)."));
		return false;
	}
	if (!PoseableMesh)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: CalibrateRig called with no PoseableMesh set."));
		return false;
	}
	USkeletalMesh* SkeletalMeshAsset = PoseableMesh->GetSkeletalMeshAsset();
	if (!SkeletalMeshAsset)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: CalibrateRig's PoseableMesh has no SkeletalMesh assigned."));
		return false;
	}

	// Capture each mapped rig bone's CURRENT world rotation as its rest
	// pose (spec §2.2 step 1-2: "Poser le rig dans sa pose de repos ...
	// calculer l'offset ... worldRot_rig_rest(bone)"). PoseableMesh is
	// expected to be in its reference pose at this point (class comment
	// contract) — this component does not force a reset-to-ref-pose
	// itself, since a project may want to calibrate against a
	// deliberately posed rest stance instead.
	auto GetRigRestWorldRotation = [this](FName BoneName) -> FQuat
	{
		const int32 BoneIndex = PoseableMesh->GetBoneIndex(BoneName);
		if (BoneIndex == INDEX_NONE)
		{
			UE_LOG(LogAInimator, Warning,
				TEXT("AInimator: RigMap target bone '%s' not found on the ")
				TEXT("PoseableMesh's skeleton during calibration."),
				*BoneName.ToString());
			return FQuat::Identity;
		}
		return PoseableMesh->GetBoneQuaternion(BoneName, EBoneSpaces::WorldSpace);
	};

	if (!FRigBinder::Calibrate(*RigMap, GetRigRestWorldRotation, Calibration))
	{
		// FRigBinder::Calibrate already logged the specific reason.
		return false;
	}

	// Root motion scale (spec §2.3): rig rest pelvis height / canonical
	// SMPL rest pelvis height. Pelvis is guaranteed mapped (Calibrate
	// requires it) — read the pelvis's rest WORLD position's height
	// (Z in Unreal's Z-up convention is the SkeletalMeshComponent's own
	// world; this measurement is purely about the RIG's own proportions,
	// independent of the contract's Y-up coord_system used for the
	// controller's OWN output space) above the mesh component's own
	// origin, in Unreal's native centimeters (ComputeRigScaleFromCentimeters
	// converts to meters).
	const FName PelvisTargetBone = RigMap->GetTargetBoneName(AInimatorSmpl22Skeleton::Pelvis);
	const FVector PelvisRestLocation =
		PoseableMesh->GetBoneLocationByName(PelvisTargetBone, EBoneSpaces::ComponentSpace);
	RigScale = FRigBinder::ComputeRigScaleFromCentimeters(FMath::Abs(PelvisRestLocation.Z));

	bIsCalibrated = true;
	UE_LOG(LogAInimator, Log,
		TEXT("AInimator: rig calibrated (%d bone(s) mapped, RigScale=%.4f)."),
		Calibration.Num(), RigScale);
	return true;
}

bool UAInimatorCharacterComponent::SetPrompt(UAInimatorControlPreset* Preset)
{
	if (!Preset)
	{
		UE_LOG(LogAInimator, Error, TEXT("AInimator: SetPrompt given a null preset."));
		return false;
	}
	if (!Preset->bHasPromptEmb)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: SetPrompt's preset '%s' has no prompt_emb; ")
			TEXT("ignoring (use ClearPrompt to explicitly revert to the null ")
			TEXT("embedding)."),
			*Preset->PresetName);
		return false;
	}
	StartPromptCrossFade(Preset->PromptEmb);
	return true;
}

bool UAInimatorCharacterComponent::SetPromptEmbedding(const TArray<float>& PromptEmbedding)
{
	if (PromptEmbedding.Num() == 0)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: SetPromptEmbedding given an empty vector; use ")
			TEXT("ClearPrompt to revert to the learned null embedding ")
			TEXT("explicitly."));
		return false;
	}
	StartPromptCrossFade(PromptEmbedding);
	return true;
}

bool UAInimatorCharacterComponent::ClearPrompt()
{
	if (!Runtime || !Runtime->IsLoaded())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: ClearPrompt called with no loaded Runtime."));
		return false;
	}
	// Never zeros (inference_contract.md §4) — the learned null
	// embedding, exactly like UAInimatorControllerRuntime::
	// SetPromptEmbedding's own empty-array convention.
	StartPromptCrossFade(TArray<float>());
	return true;
}

void UAInimatorCharacterComponent::StartPromptCrossFade(const TArray<float>& NewTargetEmb)
{
	if (!Runtime || !Runtime->IsLoaded())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: prompt change requested with no loaded Runtime."));
		return;
	}

	const int32 Channels = Runtime->GetManifest().PromptEmbChannels;
	if (Channels == 0)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: prompt change requested but this bundle has ")
			TEXT("prompt_emb_channels=0; ignoring."));
		return;
	}

	// Fade "from" wherever the embedding currently sits — the runtime's
	// own GetActivePromptEmbedding() already holds the correct value in
	// every case (an explicit embedding from a previous SetPrompt/
	// SetPromptEmbedding, or the learned null embedding seeded at
	// LoadBundle / left by a previous ClearPrompt). Spec §3 does not
	// describe queuing a second fade on top of an in-progress one, so
	// restarting the fade from the current instantaneous value (even if
	// a fade was already in flight) is the most literal reading.
	PromptEmbFadeFrom = Runtime->GetActivePromptEmbedding();

	// Resolve the concrete target vector by delegating to the runtime's
	// own validated setter (handles both the explicit-embedding and
	// empty-vector-means-null-embedding cases identically to
	// UAInimatorControllerRuntime::SetPromptEmbedding), then read back
	// exactly what it applied as the fade's "to" endpoint.
	if (!Runtime->SetPromptEmbedding(NewTargetEmb))
	{
		return; // Runtime already logged the specific validation error.
	}
	PromptEmbFadeTo = Runtime->GetActivePromptEmbedding();

	if (PromptEmbFadeFrom.Num() != Channels)
	{
		// First-ever prompt change this session before any frame ran:
		// GetActivePromptEmbedding() is already correctly sized (seeded
		// at LoadBundle), so this should not trigger in practice; kept
		// as a defensive fallback (never zeros — fades from the target
		// itself, i.e. an instantaneous switch) rather than crashing.
		PromptEmbFadeFrom = PromptEmbFadeTo;
	}

	PromptFadeElapsedSeconds = 0.0f;
	bPromptFadeActive = PromptCrossFadeSeconds > 0.0f;
	if (!bPromptFadeActive)
	{
		PromptEmbFadeScratch = PromptEmbFadeTo;
	}
}

void UAInimatorCharacterComponent::TickPromptCrossFade(float DeltaSeconds)
{
	if (!bPromptFadeActive || !Runtime || !Runtime->IsLoaded())
	{
		return;
	}

	PromptFadeElapsedSeconds += DeltaSeconds;
	const float Alpha = PromptCrossFadeSeconds > 0.0f
		? FMath::Clamp(PromptFadeElapsedSeconds / PromptCrossFadeSeconds, 0.0f, 1.0f)
		: 1.0f;

	const int32 Channels = PromptEmbFadeTo.Num();
	PromptEmbFadeScratch.SetNumUninitialized(Channels);
	for (int32 Index = 0; Index < Channels; ++Index)
	{
		// Spec §3: "lerp linéaire emb_old -> emb_new" — plain linear
		// interpolation in embedding space, no easing curve invented.
		PromptEmbFadeScratch[Index] =
			FMath::Lerp(PromptEmbFadeFrom[Index], PromptEmbFadeTo[Index], Alpha);
	}
	Runtime->SetPromptEmbedding(PromptEmbFadeScratch);

	if (Alpha >= 1.0f)
	{
		bPromptFadeActive = false;
	}
}

void UAInimatorCharacterComponent::TickCharacter(float DeltaSeconds)
{
	TickPromptCrossFade(DeltaSeconds);

	if (!Runtime || !Runtime->IsLoaded())
	{
		return;
	}

	if (AActor* Owner = GetOwner())
	{
		// Root motion, scaled by RigScale (spec §2.3) — applied to the
		// actor transform. dYaw is dimensionless, never scaled.
		const float Scale = bIsCalibrated ? RigScale : 1.0f;
		const FVector ScaledRootPosition = Runtime->GetWorldRootPosition() * Scale;
		Owner->SetActorLocation(ScaledRootPosition);
		Owner->SetActorRotation(FRotator(0.0f, FMath::RadiansToDegrees(Runtime->GetWorldYaw()), 0.0f));
	}

	if (!bIsCalibrated || !PoseableMesh)
	{
		return; // rig_binding.md §5: capsule/pawn path stays functional.
	}

	const FAInimatorManifest& Manifest = Runtime->GetManifest();
	const int32 NumBones = Manifest.NumBones;
	check(NumBones == AInimatorSmpl22Skeleton::NumBones);

	ScratchGlobalRotationMatrices.SetNum(NumBones);
	ScratchRootLocalPositions.SetNum(NumBones);
	AInimatorForwardKinematics::ComputeJointPositions(
		Runtime->GetLatestBoneFrame(), NumBones,
		ScratchGlobalRotationMatrices, ScratchRootLocalPositions);

	ScratchSmplWorldRotations.SetNumUninitialized(NumBones);
	for (int32 Bone = 0; Bone < NumBones; ++Bone)
	{
		ScratchSmplWorldRotations[Bone] = MatrixToQuat(ScratchGlobalRotationMatrices[Bone]);
	}

	FRigBinder::RetargetFrame(
		ScratchSmplWorldRotations,
		Calibration,
		ScratchTargetWorldRotations,
		ScratchBoneNames,
		ScratchBoneLocalRotations);

	AInimatorPoseableMeshApplier::ApplyFrame(
		PoseableMesh, ScratchBoneNames, ScratchBoneLocalRotations);
}
