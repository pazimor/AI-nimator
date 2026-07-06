// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorActionComponent.h"
#include "AInimatorControlPreset.h"
#include "InputAction.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorActionComponentResolvePresetForActionTest,
	"AInimator.ActionComponent.ResolvePresetForActionFindsMatchingBinding",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorActionComponentResolvePresetForActionTest::RunTest(const FString& Parameters)
{
	UAInimatorActionComponent* Component =
		NewObject<UAInimatorActionComponent>();
	UInputAction* ActionA = NewObject<UInputAction>();
	UInputAction* ActionB = NewObject<UInputAction>();
	UAInimatorControlPreset* PresetA = NewObject<UAInimatorControlPreset>();
	PresetA->PresetName = TEXT("forward");

	FAInimatorActionBinding BindingA;
	BindingA.Action = ActionA;
	BindingA.Preset = PresetA;
	Component->Bindings.Add(BindingA);

	TestEqual(TEXT("Resolves the bound action"),
		Component->ResolvePresetForAction(ActionA), PresetA);
	TestNull(TEXT("Unbound action resolves to nullptr"),
		Component->ResolvePresetForAction(ActionB));
	TestNull(TEXT("Null action resolves to nullptr"),
		Component->ResolvePresetForAction(nullptr));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorActionComponentResolvePresetForKeyTest,
	"AInimator.ActionComponent.ResolvePresetForKeyIgnoresActionBoundRows",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorActionComponentResolvePresetForKeyTest::RunTest(const FString& Parameters)
{
	UAInimatorActionComponent* Component =
		NewObject<UAInimatorActionComponent>();
	UAInimatorControlPreset* KeyPreset = NewObject<UAInimatorControlPreset>();
	KeyPreset->PresetName = TEXT("strafe_left");
	UAInimatorControlPreset* ActionPreset = NewObject<UAInimatorControlPreset>();
	ActionPreset->PresetName = TEXT("forward");
	UInputAction* Action = NewObject<UInputAction>();

	FAInimatorActionBinding KeyBinding;
	KeyBinding.Key = EKeys::Q;
	KeyBinding.Preset = KeyPreset;
	Component->Bindings.Add(KeyBinding);

	// This row also targets EKeys::Q via a legacy Key field, but since
	// it ALSO has an Action set it must never be returned by
	// ResolvePresetForKey (Action-based bindings never fall back to
	// key polling — see class header resolution order).
	FAInimatorActionBinding ActionBindingWithSameKey;
	ActionBindingWithSameKey.Action = Action;
	ActionBindingWithSameKey.Key = EKeys::Q;
	ActionBindingWithSameKey.Preset = ActionPreset;
	Component->Bindings.Add(ActionBindingWithSameKey);

	TestEqual(TEXT("Resolves the pure-key binding, not the action one"),
		Component->ResolvePresetForKey(EKeys::Q), KeyPreset);
	TestNull(TEXT("Unbound key resolves to nullptr"),
		Component->ResolvePresetForKey(EKeys::Z));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorActionBindingValidityTest,
	"AInimator.ActionComponent.BindingValidityRequiresPresetAndInputSource",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorActionBindingValidityTest::RunTest(const FString& Parameters)
{
	UAInimatorControlPreset* Preset = NewObject<UAInimatorControlPreset>();

	FAInimatorActionBinding Empty;
	TestFalse(TEXT("No preset, no input -> invalid"), Empty.IsValidBinding());

	FAInimatorActionBinding PresetOnly;
	PresetOnly.Preset = Preset;
	TestFalse(TEXT("Preset without Action/Key -> invalid"), PresetOnly.IsValidBinding());

	FAInimatorActionBinding KeyAndPreset;
	KeyAndPreset.Preset = Preset;
	KeyAndPreset.Key = EKeys::W;
	TestTrue(TEXT("Preset + Key -> valid"), KeyAndPreset.IsValidBinding());
	TestFalse(TEXT("Key-only binding does not use Enhanced Input"),
		KeyAndPreset.UsesEnhancedInput());

	FAInimatorActionBinding ActionAndPreset;
	ActionAndPreset.Preset = Preset;
	ActionAndPreset.Action = NewObject<UInputAction>();
	TestTrue(TEXT("Preset + Action -> valid"), ActionAndPreset.IsValidBinding());
	TestTrue(TEXT("Action binding uses Enhanced Input"),
		ActionAndPreset.UsesEnhancedInput());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorActionComponentActivatePresetRequiresRuntimeTest,
	"AInimator.ActionComponent.ActivatePresetFailsWithoutRuntime",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorActionComponentActivatePresetRequiresRuntimeTest::RunTest(const FString& Parameters)
{
	UAInimatorActionComponent* Component =
		NewObject<UAInimatorActionComponent>();
	UAInimatorControlPreset* Preset = NewObject<UAInimatorControlPreset>();

	TestFalse(TEXT("No Runtime assigned -> ActivatePreset fails"),
		Component->ActivatePreset(Preset));
	TestFalse(TEXT("Null preset -> ActivatePreset fails even with no Runtime"),
		Component->ActivatePreset(nullptr));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
