// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * Editor-only module for the AInimator plugin.
 *
 * Reserved for authoring niceties (asset thumbnails, Details panel
 * customization for UAInimatorControlPreset, an "import bundle" button).
 * Kept strictly separate from the runtime module so it never ships in a
 * cooked/packaged build (ROADMAP_PLUGINS.md architecture note).
 */
class FAInimatorEditorModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
