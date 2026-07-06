// Copyright AI-nimator.

#include "AInimatorControlPresetDetails.h"
#include "AInimatorControlPreset.h"
#include "AInimatorLog.h"

#include "DetailLayoutBuilder.h"
#include "DetailCategoryBuilder.h"
#include "DetailWidgetRow.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Framework/Application/SlateApplication.h"
#include "Widgets/Input/SButton.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"

#define LOCTEXT_NAMESPACE "AInimatorControlPresetDetails"

TSharedRef<IDetailCustomization> FAInimatorControlPresetDetails::MakeInstance()
{
	return MakeShared<FAInimatorControlPresetDetails>();
}

void FAInimatorControlPresetDetails::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
	CachedDetailBuilder = &DetailBuilder;

	TArray<TWeakObjectPtr<UObject>> Objects;
	DetailBuilder.GetObjectsBeingCustomized(Objects);
	if (Objects.Num() == 1)
	{
		EditedPreset = Cast<UAInimatorControlPreset>(Objects[0].Get());
	}

	IDetailCategoryBuilder& PresetCategory =
		DetailBuilder.EditCategory(TEXT("AInimator|Preset"));

	PresetCategory.AddCustomRow(LOCTEXT("ImportPromptEmbRowFilter", "Import Prompt Embedding"))
		.WholeRowContent()
		[
			SNew(SButton)
			.Text(LOCTEXT("ImportPromptEmbButton", "Import Prompt Embedding (JSON)..."))
			.ToolTipText(LOCTEXT("ImportPromptEmbTooltip",
				"Loads a JSON file produced by 'python -m ainimator.cli.encode_prompt' "
				"({\"prompt\": str, \"prompt_emb\": [float,...]}) and fills this preset's "
				"Prompt/PromptEmb fields from it."))
			.OnClicked(this, &FAInimatorControlPresetDetails::OnImportPromptEmbeddingClicked)
		];
}

FReply FAInimatorControlPresetDetails::OnImportPromptEmbeddingClicked()
{
	UAInimatorControlPreset* Preset = EditedPreset.Get();
	if (!Preset)
	{
		return FReply::Handled();
	}

	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (!DesktopPlatform)
	{
		return FReply::Handled();
	}

	const void* ParentWindowHandle =
		FSlateApplication::Get().GetActiveTopLevelWindow().IsValid()
			? FSlateApplication::Get().GetActiveTopLevelWindow()->GetNativeWindow()->GetOSWindowHandle()
			: nullptr;

	TArray<FString> OutFiles;
	const bool bPicked = DesktopPlatform->OpenFileDialog(
		ParentWindowHandle,
		LOCTEXT("PickPromptEmbJsonTitle", "Select an encode_prompt JSON file").ToString(),
		TEXT(""),
		TEXT(""),
		TEXT("JSON (*.json)|*.json"),
		0,
		OutFiles);
	if (!bPicked || OutFiles.Num() == 0)
	{
		return FReply::Handled();
	}

	const FString& FilePath = OutFiles[0];
	FString FileText;
	if (!FFileHelper::LoadFileToString(FileText, *FilePath))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: failed to read '%s'."), *FilePath);
		return FReply::Handled();
	}

	TSharedPtr<FJsonObject> Root;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileText);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: '%s' is not valid JSON."), *FilePath);
		return FReply::Handled();
	}

	FString Prompt;
	Root->TryGetStringField(TEXT("prompt"), Prompt);

	const TArray<TSharedPtr<FJsonValue>>* PromptEmbArray = nullptr;
	if (!Root->TryGetArrayField(TEXT("prompt_emb"), PromptEmbArray))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: '%s' has no 'prompt_emb' array field — expected the ")
			TEXT("output of 'python -m ainimator.cli.encode_prompt'."),
			*FilePath);
		return FReply::Handled();
	}

	TArray<float> PromptEmb;
	PromptEmb.Reserve(PromptEmbArray->Num());
	for (const TSharedPtr<FJsonValue>& Value : *PromptEmbArray)
	{
		double Number = 0.0;
		Value->TryGetNumber(Number);
		PromptEmb.Add(static_cast<float>(Number));
	}

	Preset->Modify();
	Preset->HydrateFromRaw(
		Preset->PresetName,
		Preset->Vx,
		Preset->Vz,
		Preset->bHasAim,
		Preset->AimX,
		Preset->AimZ,
		Prompt,
		PromptEmb);

	UE_LOG(LogAInimator, Log,
		TEXT("AInimator: imported prompt_emb (dim=%d) from '%s' into preset '%s'."),
		PromptEmb.Num(), *FilePath, *Preset->PresetName);

	if (CachedDetailBuilder)
	{
		CachedDetailBuilder->ForceRefreshDetails();
	}
	return FReply::Handled();
}

#undef LOCTEXT_NAMESPACE
