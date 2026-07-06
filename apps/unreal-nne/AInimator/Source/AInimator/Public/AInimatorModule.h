// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * Runtime module for the AInimator controller plugin.
 *
 * Owns no persistent state itself; it exists so the plugin can hook
 * NNE runtime availability checks at startup (StartupModule) and log
 * a single, clear diagnostic if the expected NNE runtime is missing —
 * fail-fast is a hard rule of the inference contract (ROADMAP_PLUGINS.md
 * §2 vérité #4), so we surface runtime capability problems as early as
 * possible rather than at the first inference call.
 */
class FAInimatorModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
