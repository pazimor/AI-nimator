// Copyright AI-nimator.

#include "AInimatorEditorModule.h"
#include "AInimatorLog.h"
#include "AInimatorActionComponent.h"
#include "AInimatorActionComponentDetails.h"
#include "AInimatorRigMap.h"
#include "AInimatorRigMapDetails.h"
#include "AInimatorControlPreset.h"
#include "AInimatorControlPresetDetails.h"
#include "PropertyEditorModule.h"
#include "Modules/ModuleManager.h"

FName FAInimatorEditorModule::ActionComponentClassName =
	UAInimatorActionComponent::StaticClass()->GetFName();
FName FAInimatorEditorModule::RigMapClassName =
	UAInimatorRigMap::StaticClass()->GetFName();
FName FAInimatorEditorModule::ControlPresetClassName =
	UAInimatorControlPreset::StaticClass()->GetFName();

void FAInimatorEditorModule::StartupModule()
{
	FPropertyEditorModule& PropertyModule =
		FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
	PropertyModule.RegisterCustomClassLayout(
		ActionComponentClassName,
		FOnGetDetailCustomizationInstance::CreateStatic(
			&FAInimatorActionComponentDetails::MakeInstance));
	PropertyModule.RegisterCustomClassLayout(
		RigMapClassName,
		FOnGetDetailCustomizationInstance::CreateStatic(
			&FAInimatorRigMapDetails::MakeInstance));
	PropertyModule.RegisterCustomClassLayout(
		ControlPresetClassName,
		FOnGetDetailCustomizationInstance::CreateStatic(
			&FAInimatorControlPresetDetails::MakeInstance));

	UE_LOG(LogAInimator, Log, TEXT("AInimatorEditor module started."));
}

void FAInimatorEditorModule::ShutdownModule()
{
	if (FModuleManager::Get().IsModuleLoaded("PropertyEditor"))
	{
		FPropertyEditorModule& PropertyModule =
			FModuleManager::GetModuleChecked<FPropertyEditorModule>("PropertyEditor");
		PropertyModule.UnregisterCustomClassLayout(ActionComponentClassName);
		PropertyModule.UnregisterCustomClassLayout(RigMapClassName);
		PropertyModule.UnregisterCustomClassLayout(ControlPresetClassName);
	}

	UE_LOG(LogAInimator, Log, TEXT("AInimatorEditor module shut down."));
}

IMPLEMENT_MODULE(FAInimatorEditorModule, AInimatorEditor)
