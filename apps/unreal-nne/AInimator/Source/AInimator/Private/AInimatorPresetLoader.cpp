// Copyright AI-nimator.

#include "AInimatorPresetLoader.h"
#include "AInimatorControlPreset.h"
#include "AInimatorLog.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFileManager.h"
#include "GenericPlatform/GenericPlatformFile.h"
#include "HAL/FileManager.h"

int32 FPresetLoader::LoadPresetsFromDirectory(
	const FString& PresetsDirectory,
	UObject* Outer,
	TArray<UAInimatorControlPreset*>& OutPresets)
{
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
	if (!PlatformFile.DirectoryExists(*PresetsDirectory))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: presets directory '%s' does not exist."),
			*PresetsDirectory);
		return 0;
	}

	TArray<FString> JsonFiles;
	IFileManager::Get().FindFiles(JsonFiles, *PresetsDirectory, TEXT("json"));

	int32 LoadedCount = 0;
	for (const FString& FileName : JsonFiles)
	{
		const FString FilePath = FPaths::Combine(PresetsDirectory, FileName);
		UAInimatorControlPreset* Preset =
			NewObject<UAInimatorControlPreset>(Outer);
		if (LoadPresetFile(FilePath, Preset))
		{
			OutPresets.Add(Preset);
			++LoadedCount;
		}
		else
		{
			UE_LOG(LogAInimator, Warning,
				TEXT("AInimator: skipped invalid preset file '%s'."),
				*FilePath);
		}
	}
	return LoadedCount;
}

bool FPresetLoader::LoadPresetFile(
	const FString& FilePath,
	UAInimatorControlPreset* OutPreset)
{
	FString FileText;
	if (!FFileHelper::LoadFileToString(FileText, *FilePath))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: failed to read preset file '%s'."), *FilePath);
		return false;
	}

	TSharedPtr<FJsonObject> Root;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileText);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: '%s' is not valid JSON."), *FilePath);
		return false;
	}

	FString Name;
	const TSharedPtr<FJsonObject>* ControlObject = nullptr;
	if (!Root->TryGetStringField(TEXT("name"), Name) ||
		!Root->TryGetObjectField(TEXT("control"), ControlObject))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: preset '%s' missing required 'name' or ")
			TEXT("'control' field."), *FilePath);
		return false;
	}

	double Vx = 0.0;
	double Vz = 0.0;
	if (!(*ControlObject)->TryGetNumberField(TEXT("vx"), Vx) ||
		!(*ControlObject)->TryGetNumberField(TEXT("vz"), Vz))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: preset '%s' control section missing vx/vz."),
			*FilePath);
		return false;
	}

	double AimX = 0.0;
	double AimZ = 0.0;
	const bool bHasAimX = (*ControlObject)->TryGetNumberField(TEXT("aim_x"), AimX);
	const bool bHasAimZ = (*ControlObject)->TryGetNumberField(TEXT("aim_z"), AimZ);
	const bool bHasAim = bHasAimX && bHasAimZ;
	if (bHasAimX != bHasAimZ)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: preset '%s' has only one of aim_x/aim_z — ")
			TEXT("both or neither are required."), *FilePath);
		return false;
	}

	FString Prompt;
	Root->TryGetStringField(TEXT("prompt"), Prompt);

	TArray<float> PromptEmb;
	const TArray<TSharedPtr<FJsonValue>>* PromptEmbArray = nullptr;
	if (Root->TryGetArrayField(TEXT("prompt_emb"), PromptEmbArray))
	{
		PromptEmb.Reserve(PromptEmbArray->Num());
		for (const TSharedPtr<FJsonValue>& Value : *PromptEmbArray)
		{
			double Number = 0.0;
			Value->TryGetNumber(Number);
			PromptEmb.Add(static_cast<float>(Number));
		}
	}

	OutPreset->HydrateFromRaw(
		Name,
		static_cast<float>(Vx),
		static_cast<float>(Vz),
		bHasAim,
		static_cast<float>(AimX),
		static_cast<float>(AimZ),
		Prompt,
		PromptEmb);
	return true;
}
