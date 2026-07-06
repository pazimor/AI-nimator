// Copyright AI-nimator.

#include "AInimatorControlPresetFactory.h"
#include "AInimatorControlPreset.h"

UAInimatorControlPresetFactory::UAInimatorControlPresetFactory()
{
	bCreateNew = true;
	bEditAfterNew = true;
	SupportedClass = UAInimatorControlPreset::StaticClass();
}

UObject* UAInimatorControlPresetFactory::FactoryCreateNew(
	UClass* InClass,
	UObject* InParent,
	FName InName,
	EObjectFlags Flags,
	UObject* Context,
	FFeedbackContext* Warn)
{
	UAInimatorControlPreset* NewPreset =
		NewObject<UAInimatorControlPreset>(InParent, InClass, InName, Flags);
	if (NewPreset)
	{
		// Default the asset's PresetName to the asset's own name so a
		// freshly-created preset already satisfies
		// control_preset.schema.json's snake_case "name" requirement
		// when the user picks a conventional asset name.
		NewPreset->PresetName = InName.ToString().ToLower();
	}
	return NewPreset;
}
