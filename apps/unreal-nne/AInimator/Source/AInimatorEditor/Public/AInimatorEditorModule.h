// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * Editor-only module for the AInimator plugin.
 *
 * Registers `FAInimatorActionComponentDetails` (the B3 Details panel
 * customization for `UAInimatorActionComponent`: editable binding
 * list, "Create Preset" / "Import Presets From Bundle..." buttons),
 * `FAInimatorRigMapDetails` (B3-bis: "Auto-Map From Skeletal Mesh..."
 * button for `UAInimatorRigMap`, `apps/spec/rig_binding.md`),
 * `FAInimatorControlPresetDetails` (B3-bis: "Import Prompt Embedding
 * (JSON)..." button, consumes `ainimator.cli.encode_prompt` output), and
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

	/** Class layout name for FAInimatorRigMapDetails (B3-bis,
	 *  rig_binding.md), unregistered symmetrically in ShutdownModule(). */
	static FName RigMapClassName;

	/** Class layout name for FAInimatorControlPresetDetails (B3-bis
	 *  "Import Prompt Embedding (JSON)..." button), unregistered
	 *  symmetrically in ShutdownModule(). */
	static FName ControlPresetClassName;
};
