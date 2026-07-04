// Copyright AI-nimator.

#include "AInimatorActionComponentDetails.h"
#include "AInimatorActionComponent.h"
#include "AInimatorControlPreset.h"
#include "AInimatorControlPresetFactory.h"
#include "AInimatorPresetLoader.h"
#include "AInimatorLog.h"

#include "DetailLayoutBuilder.h"
#include "DetailCategoryBuilder.h"
#include "DetailWidgetRow.h"
#include "AssetToolsModule.h"
#include "IAssetTools.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Framework/Application/SlateApplication.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Text/STextBlock.h"
#include "PackageTools.h"
#include "Misc/PackageName.h"

#define LOCTEXT_NAMESPACE "AInimatorActionComponentDetails"

namespace
{
	/** Content Browser folder new/imported presets land in by default —
	 *  a project can freely move the assets afterwards; this only
	 *  picks a sensible starting point (mirrors the Unity editor's
	 *  DefaultPresetFolder for authoring parity). */
	const FString DefaultPresetPackagePath = TEXT("/Game/AInimatorPresets");

	/** Creates one new UAInimatorControlPreset asset named DesiredName
	 *  (uniquified if it collides) under DefaultPresetPackagePath via
	 *  UAInimatorControlPresetFactory, returning the new asset or
	 *  nullptr on failure (logged). */
	UAInimatorControlPreset* CreatePresetAsset(const FString& DesiredName)
	{
		FAssetToolsModule& AssetToolsModule =
			FModuleManager::LoadModuleChecked<FAssetToolsModule>("AssetTools");
		IAssetTools& AssetTools = AssetToolsModule.Get();

		FString PackageName;
		FString AssetName;
		AssetTools.CreateUniqueAssetName(
			DefaultPresetPackagePath / DesiredName, TEXT(""), PackageName, AssetName);

		UAInimatorControlPresetFactory* Factory = NewObject<UAInimatorControlPresetFactory>();
		UObject* NewAsset = AssetTools.CreateAsset(
			AssetName,
			FPackageName::GetLongPackagePath(PackageName),
			UAInimatorControlPreset::StaticClass(),
			Factory);

		return Cast<UAInimatorControlPreset>(NewAsset);
	}
}

TSharedRef<IDetailCustomization> FAInimatorActionComponentDetails::MakeInstance()
{
	return MakeShared<FAInimatorActionComponentDetails>();
}

void FAInimatorActionComponentDetails::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
	CachedDetailBuilder = &DetailBuilder;

	TArray<TWeakObjectPtr<UObject>> Objects;
	DetailBuilder.GetObjectsBeingCustomized(Objects);
	if (Objects.Num() == 1)
	{
		EditedComponent = Cast<UAInimatorActionComponent>(Objects[0].Get());
	}

	IDetailCategoryBuilder& ActionCategory =
		DetailBuilder.EditCategory(TEXT("AInimator|Action"));

	// The Bindings array itself keeps the engine's default array UI
	// (each row already exposes Action/Key/Preset/DisplayName pickers
	// via UPROPERTY — no custom row widget needed to satisfy "editable
	// list of bindings"). These two rows only add the authoring
	// shortcuts the B3 acceptance criteria call for.
	ActionCategory.AddCustomRow(LOCTEXT("CreatePresetRowFilter", "Create Preset"))
		.WholeRowContent()
		[
			SNew(SButton)
			.Text(LOCTEXT("CreatePresetButton", "+ Create New Preset"))
			.ToolTipText(LOCTEXT("CreatePresetTooltip",
				"Creates a new UAInimatorControlPreset asset and appends a new binding row pointing at it."))
			.OnClicked(this, &FAInimatorActionComponentDetails::OnCreatePresetClicked)
		];

	ActionCategory.AddCustomRow(LOCTEXT("ImportPresetsRowFilter", "Import Presets From Bundle"))
		.WholeRowContent()
		[
			SNew(SButton)
			.Text(LOCTEXT("ImportPresetsButton", "Import Presets From Bundle..."))
			.ToolTipText(LOCTEXT("ImportPresetsTooltip",
				"Loads every presets/*.json in a chosen bundle directory (FPresetLoader) and saves each as a versioned project asset."))
			.OnClicked(this, &FAInimatorActionComponentDetails::OnImportFromBundleClicked)
		];

	// B6 (apps/spec/text_to_control.md): the "TextCommand" FString
	// property already gets a default editable text row from its
	// UPROPERTY (Category "AInimator|Action|Text") — this only adds the
	// resolve-and-apply button next to it, for a one-click PIE test.
	IDetailCategoryBuilder& TextCategory =
		DetailBuilder.EditCategory(TEXT("AInimator|Action|Text"));
	TextCategory.AddCustomRow(LOCTEXT("TestTextCommandRowFilter", "Test Text Command"))
		.WholeRowContent()
		[
			SNew(SButton)
			.Text(LOCTEXT("TestTextCommandButton", "Test Text Command (PIE)"))
			.ToolTipText(LOCTEXT("TestTextCommandTooltip",
				"Resolves the TextCommand field via FTextToControlResolver and applies it to Runtime "
				"(UAInimatorActionComponent::ActivateTextCommand) — only meaningful while the game is "
				"running with a loaded bundle. B6 is NOT the prompt channel: this only writes the "
				"low-level control vector."))
			.OnClicked(this, &FAInimatorActionComponentDetails::OnTestTextCommandClicked)
		];
}

FReply FAInimatorActionComponentDetails::OnCreatePresetClicked()
{
	UAInimatorControlPreset* NewPreset = CreatePresetAsset(TEXT("NewControlPreset"));
	if (!NewPreset)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: failed to create a new ControlPreset asset."));
		return FReply::Handled();
	}

	if (UAInimatorActionComponent* Component = EditedComponent.Get())
	{
		Component->Modify();
		FAInimatorActionBinding NewBinding;
		NewBinding.Preset = NewPreset;
		NewBinding.DisplayName = NewPreset->GetName();
		Component->Bindings.Add(NewBinding);
		if (CachedDetailBuilder)
		{
			CachedDetailBuilder->ForceRefreshDetails();
		}
	}
	return FReply::Handled();
}

FReply FAInimatorActionComponentDetails::OnImportFromBundleClicked()
{
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (!DesktopPlatform)
	{
		return FReply::Handled();
	}

	const void* ParentWindowHandle =
		FSlateApplication::Get().GetActiveTopLevelWindow().IsValid()
			? FSlateApplication::Get().GetActiveTopLevelWindow()->GetNativeWindow()->GetOSWindowHandle()
			: nullptr;

	FString BundleDirectory;
	const bool bPicked = DesktopPlatform->OpenDirectoryDialog(
		ParentWindowHandle,
		LOCTEXT("PickBundleDirTitle", "Select controller bundle directory").ToString(),
		TEXT(""),
		BundleDirectory);
	if (!bPicked || BundleDirectory.IsEmpty())
	{
		return FReply::Handled();
	}

	const FString PresetsDirectory = BundleDirectory / TEXT("presets");
	TArray<UAInimatorControlPreset*> TransientPresets;
	const int32 LoadedCount = FPresetLoader::LoadPresetsFromDirectory(
		PresetsDirectory, GetTransientPackage(), TransientPresets);

	int32 SavedCount = 0;
	for (UAInimatorControlPreset* Transient : TransientPresets)
	{
		UAInimatorControlPreset* SavedAsset = CreatePresetAsset(Transient->PresetName);
		if (!SavedAsset)
		{
			continue;
		}
		SavedAsset->HydrateFromRaw(
			Transient->PresetName,
			Transient->Vx,
			Transient->Vz,
			Transient->bHasAim,
			Transient->AimX,
			Transient->AimZ,
			Transient->Prompt,
			Transient->PromptEmb);
		++SavedCount;
	}

	UE_LOG(LogAInimator, Log,
		TEXT("AInimator: imported %d/%d preset(s) from '%s' into '%s'."),
		SavedCount, LoadedCount, *PresetsDirectory, *DefaultPresetPackagePath);
	return FReply::Handled();
}

FReply FAInimatorActionComponentDetails::OnTestTextCommandClicked()
{
	if (UAInimatorActionComponent* Component = EditedComponent.Get())
	{
		if (!Component->ActivateTextCommand(Component->TextCommand))
		{
			UE_LOG(LogAInimator, Warning,
				TEXT("AInimator: 'Test Text Command' could not resolve/apply '%s' ")
				TEXT("(see preceding log line for the reason — unresolved command, ")
				TEXT("no Runtime, or Runtime not yet loaded; only meaningful in PIE)."),
				*Component->TextCommand);
		}
	}
	return FReply::Handled();
}

#undef LOCTEXT_NAMESPACE
