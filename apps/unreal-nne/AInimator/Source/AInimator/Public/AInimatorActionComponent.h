// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AInimatorActionBinding.h"
#include "AInimatorActionComponent.generated.h"

class UAInimatorControllerRuntime;
class UAInimatorControlPreset;
class UInputComponent;
class UEnhancedInputComponent;
class UInputMappingContext;
struct FInputActionValue;

/**
 * Authoring component for phase B3 (ROADMAP_PLUGINS.md §4): maps
 * `[Input] -> [ControlPreset]` bindings, editable in the Details panel
 * with zero C++/Blueprint scripting, and drives the actor's
 * `UAInimatorControllerRuntime` accordingly every frame.
 *
 * This component does NOT own the runtime's lifecycle (loading a
 * bundle, ticking inference) — it only *selects* which preset is
 * active on an already-loaded `UAInimatorControllerRuntime`, mirroring
 * the split already established by `AAInimatorDemoPawn` in B2 but
 * generalized to an arbitrary, data-driven list of bindings instead of
 * four hardcoded WASD actions.
 *
 * Resolution order per binding (see `IsValidBinding()` /
 * `UsesEnhancedInput()` on `FAInimatorActionBinding`):
 * 1. If `Action` (a `UInputAction`) is set, bind through the Enhanced
 *    Input subsystem (requires a `UInputMappingContext` with this
 *    action mapped, added by the owning Pawn/PlayerController as
 *    usual — this component does not add the mapping context itself,
 *    it only binds callbacks for the actions it is given).
 * 2. Otherwise, if `Key` is a valid `FKey`, poll it directly every
 *    Tick via the legacy `UInputComponent` (no Enhanced Input Action
 *    asset required at all — useful for a five-minute prototype).
 *
 * When no bound input is currently active, the component applies
 * `IdlePreset` (falls back to doing nothing if unset, so a game with
 * its own idle handling is not forced to opt in).
 */
UCLASS(ClassGroup = (AInimator), meta = (BlueprintSpawnableComponent))
class AINIMATOR_API UAInimatorActionComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UAInimatorActionComponent();

	/** The runtime this component drives. Left unset, the component
	 *  looks for a sibling `UAInimatorControllerRuntime`-owning
	 *  component/actor at BeginPlay via `ResolveRuntime()` — but the
	 *  simplest and most explicit setup is to assign this directly. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Action")
	TObjectPtr<UAInimatorControllerRuntime> Runtime;

	/** The editable list of `[Input] -> [Preset]` bindings — the
	 *  authoring surface this whole component exists for. Edited via
	 *  the plain array in the default Details panel, or via the richer
	 *  `FAInimatorActionComponentDetails` customization (editor
	 *  module) when available. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Action")
	TArray<FAInimatorActionBinding> Bindings;

	/** Preset applied when no binding is currently active. Optional:
	 *  leave unset to let the game keep whatever preset was last
	 *  applied (or drive control directly itself). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Action")
	TObjectPtr<UAInimatorControlPreset> IdlePreset;

	/**
	 * Applies Preset to Runtime immediately, independent of any
	 * binding — used by the legacy-key polling path in Tick() and
	 * exposed to Blueprint for manual/gameplay-driven activation
	 * (e.g. a cutscene forcing a specific preset).
	 */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Action")
	bool ActivatePreset(UAInimatorControlPreset* Preset);

	/** Finds the first binding whose Action matches InAction; returns
	 *  nullptr if none. Exposed for unit tests and Blueprint
	 *  introspection — pure lookup, no side effect. */
	UFUNCTION(BlueprintPure, Category = "AInimator|Action")
	UAInimatorControlPreset* ResolvePresetForAction(UInputAction* InAction) const;

	/** Finds the first binding whose Key matches InKey and which has no
	 *  Action set (Action-based bindings never fall back to Key
	 *  polling — see class comment resolution order). */
	UFUNCTION(BlueprintPure, Category = "AInimator|Action")
	UAInimatorControlPreset* ResolvePresetForKey(FKey InKey) const;

protected:
	virtual void BeginPlay() override;
	virtual void TickComponent(
		float DeltaTime,
		ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;
	virtual void SetupInputBindings();

private:
	/** Enhanced Input callback: applies ResolvePresetForAction's result
	 *  for the action that triggered this callback. */
	void OnEnhancedInputTriggered(const FInputActionValue& Value, UInputAction* SourceAction);

	/** Polls every Key-based (non-Action) binding every Tick and
	 *  applies the first one whose key is currently down; falls back
	 *  to IdlePreset if none are. Action-based bindings are handled by
	 *  Enhanced Input callbacks instead, not here. */
	void PollKeyBindings();

	UAInimatorControllerRuntime* ResolveRuntime() const;

	bool bAnyKeyBindingActiveLastTick = false;
};
