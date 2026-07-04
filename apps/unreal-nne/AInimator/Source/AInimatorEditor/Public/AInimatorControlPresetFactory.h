// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Factories/Factory.h"
#include "AInimatorControlPresetFactory.generated.h"

/**
 * Editor asset factory for `UAInimatorControlPreset` — lets a
 * developer right-click in the Content Browser (Miscellaneous ->
 * AInimator Control Preset) and get a new, empty preset asset without
 * writing any code (ROADMAP_PLUGINS.md §4 B3 acceptance: "presets
 * sauvegardés comme assets versionnés").
 *
 * Bundle-provided presets (`presets/*.json`) still load through
 * `FPresetLoader` as transient instances at runtime — this factory is
 * only for hand-authored, project-versioned presets a game team wants
 * to keep as first-class `.uasset` content, editable and diffable in
 * source control.
 */
UCLASS()
class AINIMATOREDITOR_API UAInimatorControlPresetFactory : public UFactory
{
	GENERATED_BODY()

public:
	UAInimatorControlPresetFactory();

	virtual UObject* FactoryCreateNew(
		UClass* InClass,
		UObject* InParent,
		FName InName,
		EObjectFlags Flags,
		UObject* Context,
		FFeedbackContext* Warn) override;
};
