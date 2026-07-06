// Copyright AI-nimator.

#include "AInimatorModule.h"
#include "AInimatorLog.h"

void FAInimatorModule::StartupModule()
{
	UE_LOG(LogAInimator, Log,
		TEXT("AInimator runtime module started. Controller bundles are ")
		TEXT("loaded on demand via FBundleLoader; no model is defined by ")
		TEXT("this plugin."));
}

void FAInimatorModule::ShutdownModule()
{
	UE_LOG(LogAInimator, Log, TEXT("AInimator runtime module shut down."));
}

IMPLEMENT_MODULE(FAInimatorModule, AInimator)
