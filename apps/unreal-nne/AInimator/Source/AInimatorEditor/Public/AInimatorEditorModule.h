// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * Editor-only module for the AInimator plugin.
 *
 * Registers `FAInimatorActionComponentDetails` (the B3 Details panel
 * customization for `UAInimatorActionComponent`: editable binding
 * list, "Create Preset" / "Import Presets From Bundle..." buttons) and
 * `UAInimatorControlPresetFactory` (Content Browser asset creation).
 * Kept strictly separate from the runtime module so it never ships in a
 * cooked/packaged build (ROADMAP_PLUGINS.md architecture note).
 */
class FAInimatorEditorModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	/** Name registered with FPropertyEditorModule — unregistered
	 *  symmetrically in ShutdownModule(). */
	static FName ActionComponentClassName;
};
