// Copyright AI-nimator.

#include "AInimatorActionComponent.h"
#include "AInimatorControllerRuntime.h"
#include "AInimatorControlPreset.h"
#include "AInimatorLog.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "Components/InputComponent.h"
#include "EnhancedInputComponent.h"
#include "InputAction.h"

UAInimatorActionComponent::UAInimatorActionComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

void UAInimatorActionComponent::BeginPlay()
{
	Super::BeginPlay();

	if (!Runtime)
	{
		Runtime = ResolveRuntime();
	}
	if (!Runtime)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: UAInimatorActionComponent on '%s' has no ")
			TEXT("Runtime assigned and none could be resolved from the ")
			TEXT("owning actor — bindings will be inert until Runtime is set."),
			*GetOwner()->GetName());
	}

	SetupInputBindings();
}

UAInimatorControllerRuntime* UAInimatorActionComponent::ResolveRuntime() const
{
	if (!GetOwner())
	{
		return nullptr;
	}
	// The simplest convention: a UAInimatorControllerRuntime UPROPERTY
	// named "Runtime" on the owning actor (as on AAInimatorDemoPawn).
	// Reflection-based lookup keeps this component decoupled from any
	// specific Pawn subclass — it works with the demo pawn or any
	// game-authored equivalent without a common interface.
	for (TFieldIterator<FObjectProperty> PropertyIt(GetOwner()->GetClass());
		PropertyIt; ++PropertyIt)
	{
		FObjectProperty* Property = *PropertyIt;
		if (Property->PropertyClass == UAInimatorControllerRuntime::StaticClass())
		{
			UObject* Value = Property->GetObjectPropertyValue_InContainer(GetOwner());
			if (UAInimatorControllerRuntime* Found = Cast<UAInimatorControllerRuntime>(Value))
			{
				return Found;
			}
		}
	}
	return nullptr;
}

void UAInimatorActionComponent::SetupInputBindings()
{
	APawn* OwningPawn = Cast<APawn>(GetOwner());
	if (!OwningPawn || !OwningPawn->IsPlayerControlled())
	{
		return;
	}
	APlayerController* PlayerController = Cast<APlayerController>(OwningPawn->GetController());
	if (!PlayerController)
	{
		return;
	}
	UEnhancedInputComponent* EnhancedInput =
		Cast<UEnhancedInputComponent>(PlayerController->InputComponent);
	if (!EnhancedInput)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: no UEnhancedInputComponent found on '%s' — ")
			TEXT("Action-based bindings will not fire; Key-based bindings ")
			TEXT("still work via Tick() polling."),
			*OwningPawn->GetName());
		return;
	}

	for (const FAInimatorActionBinding& Binding : Bindings)
	{
		if (!Binding.UsesEnhancedInput() || !Binding.IsValidBinding())
		{
			continue;
		}
		// Triggered covers both "pressed and held" (digital buttons)
		// and continuous analog values (e.g. an aim stick), matching
		// the ControlPreset's role as "whatever this input currently
		// wants applied" rather than a one-shot event.
		EnhancedInput->BindAction(
			Binding.Action,
			ETriggerEvent::Triggered,
			this,
			&UAInimatorActionComponent::OnEnhancedInputTriggered,
			Binding.Action);
	}
}

void UAInimatorActionComponent::OnEnhancedInputTriggered(
	const FInputActionValue& Value,
	UInputAction* SourceAction)
{
	UAInimatorControlPreset* Preset = ResolvePresetForAction(SourceAction);
	if (Preset)
	{
		ActivatePreset(Preset);
	}
}

void UAInimatorActionComponent::TickComponent(
	float DeltaTime,
	ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	PollKeyBindings();
}

void UAInimatorActionComponent::PollKeyBindings()
{
	APawn* OwningPawn = Cast<APawn>(GetOwner());
	if (!OwningPawn || !OwningPawn->IsPlayerControlled())
	{
		return;
	}
	APlayerController* PlayerController = Cast<APlayerController>(OwningPawn->GetController());
	if (!PlayerController)
	{
		return;
	}

	bool bAnyActive = false;
	for (const FAInimatorActionBinding& Binding : Bindings)
	{
		if (Binding.UsesEnhancedInput() || !Binding.IsValidBinding())
		{
			// Action-based bindings are handled by Enhanced Input
			// callbacks only, never polled here (resolution order,
			// see class header comment).
			continue;
		}
		if (PlayerController->IsInputKeyDown(Binding.Key))
		{
			ActivatePreset(Binding.Preset);
			bAnyActive = true;
			break;
		}
	}

	if (!bAnyActive && bAnyKeyBindingActiveLastTick && IdlePreset)
	{
		ActivatePreset(IdlePreset);
	}
	bAnyKeyBindingActiveLastTick = bAnyActive;
}

bool UAInimatorActionComponent::ActivatePreset(UAInimatorControlPreset* Preset)
{
	if (!Runtime)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: ActivatePreset called with no Runtime assigned."));
		return false;
	}
	if (!Preset)
	{
		UE_LOG(LogAInimator, Error, TEXT("AInimator: ActivatePreset given null preset."));
		return false;
	}
	return Runtime->SetPreset(Preset);
}

bool UAInimatorActionComponent::ActivateTextCommand(const FString& Command)
{
	if (!Runtime)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: ActivateTextCommand called with no Runtime assigned."));
		return false;
	}
	return Runtime->SetTextCommand(Command);
}

UAInimatorControlPreset* UAInimatorActionComponent::ResolvePresetForAction(
	UInputAction* InAction) const
{
	if (!InAction)
	{
		return nullptr;
	}
	for (const FAInimatorActionBinding& Binding : Bindings)
	{
		if (Binding.Action == InAction && Binding.Preset)
		{
			return Binding.Preset;
		}
	}
	return nullptr;
}

UAInimatorControlPreset* UAInimatorActionComponent::ResolvePresetForKey(FKey InKey) const
{
	for (const FAInimatorActionBinding& Binding : Bindings)
	{
		if (!Binding.UsesEnhancedInput() && Binding.Key == InKey && Binding.Preset)
		{
			return Binding.Preset;
		}
	}
	return nullptr;
}
