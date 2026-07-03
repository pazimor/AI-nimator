// Copyright AI-nimator.

#include "AInimatorDemoPawn.h"
#include "AInimatorLog.h"
#include "AInimatorControlPreset.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "InputAction.h"
#include "InputMappingContext.h"
#include "GameFramework/PlayerController.h"

namespace
{
	const FString PresetNameIdle = TEXT("idle");
	const FString PresetNameForward = TEXT("forward");
	const FString PresetNameBackward = TEXT("backward");
	const FString PresetNameStrafeLeft = TEXT("strafe_left");
	const FString PresetNameStrafeRight = TEXT("strafe_right");
}

AAInimatorDemoPawn::AAInimatorDemoPawn()
{
	PrimaryActorTick.bCanEverTick = true;
}

void AAInimatorDemoPawn::BeginPlay()
{
	Super::BeginPlay();

	Runtime = NewObject<UAInimatorControllerRuntime>(this);
	if (BundleDirectory.IsEmpty())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimatorDemoPawn: BundleDirectory is empty; set it to a ")
			TEXT("bundle produced by `export_onnx bundle` (see plugin ")
			TEXT("README) before Play."));
		return;
	}
	if (!Runtime->LoadBundle(BundleDirectory))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimatorDemoPawn: failed to load bundle at '%s'; the ")
			TEXT("pawn will not move (see prior AInimator errors)."),
			*BundleDirectory);
		return;
	}
	SelectPresetByName(PresetNameIdle);

	if (APlayerController* PC = Cast<APlayerController>(GetController()))
	{
		if (UEnhancedInputLocalPlayerSubsystem* Subsystem =
			ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(
				PC->GetLocalPlayer()))
		{
			if (MappingContext)
			{
				Subsystem->AddMappingContext(MappingContext, 0);
			}
		}
	}
}

void AAInimatorDemoPawn::SetupPlayerInputComponent(
	UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	UEnhancedInputComponent* EnhancedInput =
		Cast<UEnhancedInputComponent>(PlayerInputComponent);
	if (!EnhancedInput)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimatorDemoPawn requires the Enhanced Input plugin."));
		return;
	}

	if (MoveForwardAction)
	{
		EnhancedInput->BindAction(MoveForwardAction, ETriggerEvent::Triggered,
			this, &AAInimatorDemoPawn::OnMoveForwardPressed);
		EnhancedInput->BindAction(MoveForwardAction, ETriggerEvent::Completed,
			this, &AAInimatorDemoPawn::OnMoveReleased);
	}
	if (MoveBackwardAction)
	{
		EnhancedInput->BindAction(MoveBackwardAction, ETriggerEvent::Triggered,
			this, &AAInimatorDemoPawn::OnMoveBackwardPressed);
		EnhancedInput->BindAction(MoveBackwardAction, ETriggerEvent::Completed,
			this, &AAInimatorDemoPawn::OnMoveReleased);
	}
	if (StrafeLeftAction)
	{
		EnhancedInput->BindAction(StrafeLeftAction, ETriggerEvent::Triggered,
			this, &AAInimatorDemoPawn::OnStrafeLeftPressed);
		EnhancedInput->BindAction(StrafeLeftAction, ETriggerEvent::Completed,
			this, &AAInimatorDemoPawn::OnMoveReleased);
	}
	if (StrafeRightAction)
	{
		EnhancedInput->BindAction(StrafeRightAction, ETriggerEvent::Triggered,
			this, &AAInimatorDemoPawn::OnStrafeRightPressed);
		EnhancedInput->BindAction(StrafeRightAction, ETriggerEvent::Completed,
			this, &AAInimatorDemoPawn::OnMoveReleased);
	}
}

void AAInimatorDemoPawn::OnMoveForwardPressed(const FInputActionValue&)
{
	SelectPresetByName(PresetNameForward);
}

void AAInimatorDemoPawn::OnMoveBackwardPressed(const FInputActionValue&)
{
	SelectPresetByName(PresetNameBackward);
}

void AAInimatorDemoPawn::OnStrafeLeftPressed(const FInputActionValue&)
{
	SelectPresetByName(PresetNameStrafeLeft);
}

void AAInimatorDemoPawn::OnStrafeRightPressed(const FInputActionValue&)
{
	SelectPresetByName(PresetNameStrafeRight);
}

void AAInimatorDemoPawn::OnMoveReleased(const FInputActionValue&)
{
	SelectPresetByName(PresetNameIdle);
}

void AAInimatorDemoPawn::SelectPresetByName(const FString& PresetName)
{
	if (!Runtime || !Runtime->IsLoaded())
	{
		return;
	}
	UAInimatorControlPreset* Preset = FindBundledPreset(PresetName);
	if (!Preset)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimatorDemoPawn: bundle has no preset named '%s'."),
			*PresetName);
		return;
	}
	Runtime->SetPreset(Preset);
}

UAInimatorControlPreset* AAInimatorDemoPawn::FindBundledPreset(
	const FString& PresetName) const
{
	if (!Runtime)
	{
		return nullptr;
	}
	for (UAInimatorControlPreset* Preset : Runtime->GetBundledPresets())
	{
		if (Preset && Preset->PresetName == PresetName)
		{
			return Preset;
		}
	}
	return nullptr;
}

void AAInimatorDemoPawn::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);

	if (!Runtime || !Runtime->IsLoaded())
	{
		return;
	}
	if (!Runtime->Tick())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimatorDemoPawn: runtime Tick() failed; stopping the ")
			TEXT("demo pawn's autoregression (see prior AInimator error)."));
		return;
	}

	// Apply the integrated world position/yaw to the actor transform.
	// Foot-lock IK / skeletal mesh binding are out of scope for B2
	// (B4); a real game binds LatestBoneFrame to a Skeleton/AnimGraph
	// instead of moving the whole actor root, but the actor transform
	// is enough to visually demonstrate locomotion end-to-end.
	SetActorLocation(Runtime->GetWorldRootPosition());
	SetActorRotation(FRotator(0.0, FMath::RadiansToDegrees(Runtime->GetWorldYaw()), 0.0));
}
