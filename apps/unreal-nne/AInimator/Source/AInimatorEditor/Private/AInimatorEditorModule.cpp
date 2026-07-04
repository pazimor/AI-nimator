// Copyright AI-nimator.

#include "AInimatorEditorModule.h"
#include "AInimatorLog.h"
#include "AInimatorActionComponent.h"
#include "AInimatorActionComponentDetails.h"
#include "PropertyEditorModule.h"
#include "Modules/ModuleManager.h"

FName FAInimatorEditorModule::ActionComponentClassName =
	UAInimatorActionComponent::StaticClass()->GetFName();

void FAInimatorEditorModule::StartupModule()
{
	FPropertyEditorModule& PropertyModule =
		FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
	PropertyModule.RegisterCustomClassLayout(
		ActionComponentClassName,
		FOnGetDetailCustomizationInstance::CreateStatic(
			&FAInimatorActionComponentDetails::MakeInstance));

	UE_LOG(LogAInimator, Log, TEXT("AInimatorEditor module started."));
}

void FAInimatorEditorModule::ShutdownModule()
{
	if (FModuleManager::Get().IsModuleLoaded("PropertyEditor"))
	{
		FPropertyEditorModule& PropertyModule =
			FModuleManager::GetModuleChecked<FPropertyEditorModule>("PropertyEditor");
		PropertyModule.UnregisterCustomClassLayout(ActionComponentClassName);
	}

	UE_LOG(LogAInimator, Log, TEXT("AInimatorEditor module shut down."));
}

IMPLEMENT_MODULE(FAInimatorEditorModule, AInimatorEditor)
