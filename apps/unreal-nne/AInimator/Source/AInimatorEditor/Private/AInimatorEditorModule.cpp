// Copyright AI-nimator.

#include "AInimatorEditorModule.h"
#include "AInimatorLog.h"

void FAInimatorEditorModule::StartupModule()
{
	UE_LOG(LogAInimator, Log, TEXT("AInimatorEditor module started."));
}

void FAInimatorEditorModule::ShutdownModule()
{
	UE_LOG(LogAInimator, Log, TEXT("AInimatorEditor module shut down."));
}

IMPLEMENT_MODULE(FAInimatorEditorModule, AInimatorEditor)
