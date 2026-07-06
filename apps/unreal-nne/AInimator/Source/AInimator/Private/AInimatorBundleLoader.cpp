// Copyright AI-nimator.

#include "AInimatorBundleLoader.h"
#include "AInimatorLog.h"
#include "AInimatorContractConstants.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

namespace
{
	/** Reads a whole file into a JSON object, logging on any failure. */
	bool ReadJsonObject(const FString& FilePath, TSharedPtr<FJsonObject>& OutObject)
	{
		FString FileText;
		if (!FFileHelper::LoadFileToString(FileText, *FilePath))
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: failed to read bundle file '%s'."),
				*FilePath);
			return false;
		}

		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileText);
		if (!FJsonSerializer::Deserialize(Reader, OutObject) || !OutObject.IsValid())
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: '%s' is not valid JSON."), *FilePath);
			return false;
		}
		return true;
	}

	/** Flattens a nested JSON number array into Out, appending values in
	 *  the array's natural (already row-major) traversal order. Used for
	 *  the state/delta bone and global stat blocks, whose JSON shape is
	 *  deeply nested (e.g. [[[[f,f,f,f,f,f], ...] ]]) purely because the
	 *  Python buffers keep broadcasting dims — flattening recovers the
	 *  same values FNormalizer indexes by (bone, channel) or (channel). */
	void FlattenNumberArray(const TArray<TSharedPtr<FJsonValue>>& Values,
		TArray<float>& Out)
	{
		for (const TSharedPtr<FJsonValue>& Value : Values)
		{
			const TArray<TSharedPtr<FJsonValue>>* Nested = nullptr;
			if (Value->TryGetArray(Nested))
			{
				FlattenNumberArray(*Nested, Out);
			}
			else
			{
				double Number = 0.0;
				Value->TryGetNumber(Number);
				Out.Add(static_cast<float>(Number));
			}
		}
	}

	/** Flattens a named field of Root that is expected to be a nested
	 *  JSON number array. Returns false (and logs) if the field is
	 *  missing or not an array — fail-fast on malformed norm_stats. */
	bool FlattenField(const TSharedPtr<FJsonObject>& Root, const FString& FieldName,
		TArray<float>& Out, const FString& ContextForLog)
	{
		const TArray<TSharedPtr<FJsonValue>>* Array = nullptr;
		if (!Root->TryGetArrayField(FieldName, Array))
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: norm_stats.json missing array field '%s' ")
				TEXT("(%s)."), *FieldName, *ContextForLog);
			return false;
		}
		FlattenNumberArray(*Array, Out);
		return true;
	}
}

bool FBundleLoader::LoadFromDirectory(
	const FString& BundleDirectory,
	FAInimatorManifest& OutManifest,
	FAInimatorNormStats& OutNormStats)
{
	const FString ManifestPath =
		FPaths::Combine(BundleDirectory, TEXT("manifest.json"));
	const FString NormStatsPath =
		FPaths::Combine(BundleDirectory, TEXT("norm_stats.json"));

	if (!LoadManifest(ManifestPath, OutManifest))
	{
		return false;
	}
	if (!ValidateManifest(OutManifest))
	{
		return false;
	}
	if (!LoadNormStats(NormStatsPath, OutManifest, OutNormStats))
	{
		return false;
	}

	UE_LOG(LogAInimator, Log,
		TEXT("AInimator: bundle at '%s' loaded (contract %s, ")
		TEXT("context_frames=%d, control_channels=%d, ")
		TEXT("prompt_emb_channels=%d, phase_channels=%d)."),
		*BundleDirectory, *OutManifest.BundleVersion,
		OutManifest.ContextFrames, OutManifest.ControlChannels,
		OutManifest.PromptEmbChannels, OutManifest.PhaseChannels);
	return true;
}

bool FBundleLoader::LoadManifest(
	const FString& ManifestPath,
	FAInimatorManifest& OutManifest)
{
	TSharedPtr<FJsonObject> Root;
	if (!ReadJsonObject(ManifestPath, Root))
	{
		return false;
	}

	bool bOk = true;
	bOk &= Root->TryGetStringField(
		TEXT("bundle_version"), OutManifest.BundleVersion);
	bOk &= Root->TryGetNumberField(
		TEXT("state_channels"), OutManifest.StateChannels);
	bOk &= Root->TryGetNumberField(TEXT("num_bones"), OutManifest.NumBones);
	bOk &= Root->TryGetNumberField(
		TEXT("rotation_channels_per_bone"), OutManifest.RotationChannelsPerBone);
	bOk &= Root->TryGetNumberField(
		TEXT("root_local_motion_channels"), OutManifest.RootLocalMotionChannels);
	bOk &= Root->TryGetNumberField(
		TEXT("control_channels"), OutManifest.ControlChannels);
	bOk &= Root->TryGetNumberField(
		TEXT("phase_channels"), OutManifest.PhaseChannels);
	bOk &= Root->TryGetNumberField(
		TEXT("prompt_emb_channels"), OutManifest.PromptEmbChannels);
	bOk &= Root->TryGetNumberField(
		TEXT("context_frames"), OutManifest.ContextFrames);
	bOk &= Root->TryGetStringField(
		TEXT("output_layout"), OutManifest.OutputLayout);
	bOk &= Root->TryGetStringField(
		TEXT("coord_system"), OutManifest.CoordSystem);
	bOk &= Root->TryGetStringField(
		TEXT("normalization_note"), OutManifest.NormalizationNote);

	TArray<FString> ControlLayout;
	if (Root->TryGetStringArrayField(TEXT("control_layout"), ControlLayout))
	{
		OutManifest.ControlLayout = ControlLayout;
	}
	else
	{
		bOk = false;
	}

	TArray<FString> ReservedGroups;
	Root->TryGetStringArrayField(TEXT("reserved_input_groups"), ReservedGroups);
	OutManifest.ReservedInputGroups = ReservedGroups;

	if (!bOk)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest.json at '%s' is missing one or more ")
			TEXT("required fields (see manifest.schema.json)."), *ManifestPath);
		return false;
	}

	// Optional B7 text_encoder section (text_encoding.md §1) — absent
	// on A7.0-style bundles, malformed-if-present is fail-fast.
	const TSharedPtr<FJsonObject>* TextEncoderObject = nullptr;
	if (Root->TryGetObjectField(TEXT("text_encoder"), TextEncoderObject))
	{
		if (!LoadTextEncoderSection(*TextEncoderObject, OutManifest.TextEncoder))
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: manifest.json at '%s' has a malformed ")
				TEXT("text_encoder section (text_encoding.md §1)."), *ManifestPath);
			return false;
		}
	}

	OutManifest.bIsFullyParsed = true;
	return true;
}

bool FBundleLoader::LoadTextEncoderSection(
	const TSharedPtr<FJsonObject>& Section,
	FAInimatorTextEncoderManifest& OutTextEncoder)
{
	bool bOk = true;
	bOk &= Section->TryGetStringField(TEXT("file"), OutTextEncoder.File);
	bOk &= Section->TryGetStringField(TEXT("pooling"), OutTextEncoder.Pooling);
	bOk &= Section->TryGetNumberField(
		TEXT("embedding_channels"), OutTextEncoder.EmbeddingChannels);

	const TSharedPtr<FJsonObject>* Tokenizer = nullptr;
	if (!Section->TryGetObjectField(TEXT("tokenizer"), Tokenizer))
	{
		return false;
	}
	bOk &= (*Tokenizer)->TryGetStringField(
		TEXT("type"), OutTextEncoder.TokenizerType);
	bOk &= (*Tokenizer)->TryGetStringField(
		TEXT("vocab"), OutTextEncoder.TokenizerVocab);
	bOk &= (*Tokenizer)->TryGetStringField(
		TEXT("merges"), OutTextEncoder.TokenizerMerges);
	bOk &= (*Tokenizer)->TryGetNumberField(
		TEXT("max_length"), OutTextEncoder.MaxLength);
	bOk &= (*Tokenizer)->TryGetNumberField(
		TEXT("bos_id"), OutTextEncoder.BosId);
	bOk &= (*Tokenizer)->TryGetNumberField(
		TEXT("eos_id"), OutTextEncoder.EosId);
	bOk &= (*Tokenizer)->TryGetNumberField(
		TEXT("pad_id"), OutTextEncoder.PadId);

	OutTextEncoder.bIsPresent = bOk;
	return bOk;
}

bool FBundleLoader::TryParseVersionMajor(const FString& Version, int32& OutMajor)
{
	FString MajorPart;
	FString MinorPart;
	if (!Version.Split(TEXT("."), &MajorPart, &MinorPart))
	{
		return false;
	}
	// Strip an optional leading letter, e.g. "A7" -> "7".
	FString Digits = MajorPart;
	while (Digits.Len() > 0 && !FChar::IsDigit(Digits[0]))
	{
		Digits.RightChopInline(1);
	}
	if (Digits.IsEmpty() || !Digits.IsNumeric())
	{
		return false;
	}
	OutMajor = FCString::Atoi(*Digits);
	return true;
}

bool FBundleLoader::ValidateManifest(const FAInimatorManifest& Manifest)
{
	using namespace AInimatorContract;

	int32 Major = 0;
	if (!TryParseVersionMajor(Manifest.BundleVersion, Major))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: malformed bundle_version '%s' — expected ")
			TEXT("e.g. 'A7.0'. Refusing to load (fail-fast)."),
			*Manifest.BundleVersion);
		return false;
	}
	if (Major != SupportedBundleVersionMajor)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: bundle_version '%s' (major %d) is not ")
			TEXT("compatible with this plugin build (expects major %d). ")
			TEXT("Refusing to load an incompatible contract (fail-fast, ")
			TEXT("ROADMAP_PLUGINS.md vérité #4)."),
			*Manifest.BundleVersion, Major, SupportedBundleVersionMajor);
		return false;
	}
	if (Manifest.StateChannels != StateChannels)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest state_channels=%d, expected %d."),
			Manifest.StateChannels, StateChannels);
		return false;
	}
	if (Manifest.NumBones != NumBones)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest num_bones=%d, expected %d."),
			Manifest.NumBones, NumBones);
		return false;
	}
	if (Manifest.RotationChannelsPerBone != RotationChannelsPerBone)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest rotation_channels_per_bone=%d, ")
			TEXT("expected %d."),
			Manifest.RotationChannelsPerBone, RotationChannelsPerBone);
		return false;
	}
	if (Manifest.RootLocalMotionChannels != RootLocalMotionChannels)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest root_local_motion_channels=%d, ")
			TEXT("expected %d."),
			Manifest.RootLocalMotionChannels, RootLocalMotionChannels);
		return false;
	}
	if (Manifest.ControlChannels != 2 && Manifest.ControlChannels != 4)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest control_channels=%d, expected 2 or 4."),
			Manifest.ControlChannels);
		return false;
	}
	if (Manifest.ControlLayout.Num() != Manifest.ControlChannels)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: control_layout has %d entries but ")
			TEXT("control_channels=%d."),
			Manifest.ControlLayout.Num(), Manifest.ControlChannels);
		return false;
	}
	if (Manifest.PhaseChannels != 0 && Manifest.PhaseChannels != PhaseChannelsWhenActive)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest phase_channels=%d, expected 0 or %d."),
			Manifest.PhaseChannels, PhaseChannelsWhenActive);
		return false;
	}
	if (Manifest.PromptEmbChannels < 0)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest prompt_emb_channels=%d is negative."),
			Manifest.PromptEmbChannels);
		return false;
	}
	if (Manifest.ContextFrames < 1)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest context_frames=%d must be >= 1."),
			Manifest.ContextFrames);
		return false;
	}
	if (Manifest.OutputLayout != TEXT("bone_delta|global_delta"))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest output_layout='%s', expected ")
			TEXT("'bone_delta|global_delta'."), *Manifest.OutputLayout);
		return false;
	}
	if (Manifest.CoordSystem != TEXT("Y-up right-handed"))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: manifest coord_system='%s', expected ")
			TEXT("'Y-up right-handed'."), *Manifest.CoordSystem);
		return false;
	}
	if (Manifest.HasTextEncoder())
	{
		const FAInimatorTextEncoderManifest& TextEncoder = Manifest.TextEncoder;
		if (Manifest.PromptEmbChannels <= 0)
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: text_encoder section present but ")
				TEXT("prompt_emb_channels == 0 — an encoder cannot condition ")
				TEXT("a promptless controller."));
			return false;
		}
		if (TextEncoder.TokenizerType != TEXT("clip-bpe"))
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: unsupported text_encoder.tokenizer.type '%s' ")
				TEXT("(this plugin build implements 'clip-bpe' only)."),
				*TextEncoder.TokenizerType);
			return false;
		}
		if (TextEncoder.Pooling != TEXT("masked_mean"))
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: unsupported text_encoder.pooling '%s' ")
				TEXT("(pooling is baked into the graph; 'masked_mean' expected)."),
				*TextEncoder.Pooling);
			return false;
		}
		if (TextEncoder.EmbeddingChannels != Manifest.PromptEmbChannels)
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: text_encoder.embedding_channels=%d must equal ")
				TEXT("prompt_emb_channels=%d."),
				TextEncoder.EmbeddingChannels, Manifest.PromptEmbChannels);
			return false;
		}
		if (TextEncoder.MaxLength < 2)
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: text_encoder.tokenizer.max_length=%d must be >= 2."),
				TextEncoder.MaxLength);
			return false;
		}
	}
	return true;
}

bool FBundleLoader::LoadNormStats(
	const FString& NormStatsPath,
	const FAInimatorManifest& Manifest,
	FAInimatorNormStats& OutNormStats)
{
	TSharedPtr<FJsonObject> Root;
	if (!ReadJsonObject(NormStatsPath, Root))
	{
		return false;
	}

	const TSharedPtr<FJsonObject>* StateObject = nullptr;
	const TSharedPtr<FJsonObject>* DeltaObject = nullptr;
	const TSharedPtr<FJsonObject>* ControlObject = nullptr;
	if (!Root->TryGetObjectField(TEXT("state"), StateObject) ||
		!Root->TryGetObjectField(TEXT("delta"), DeltaObject) ||
		!Root->TryGetObjectField(TEXT("control"), ControlObject))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: norm_stats.json at '%s' is missing one of ")
			TEXT("the required top-level sections (state/delta/control)."),
			*NormStatsPath);
		return false;
	}

	bool bOk = true;
	bOk &= FlattenField(*StateObject, TEXT("bone_mean"),
		OutNormStats.BoneMean, TEXT("state.bone_mean"));
	bOk &= FlattenField(*StateObject, TEXT("bone_std"),
		OutNormStats.BoneStd, TEXT("state.bone_std"));
	bOk &= FlattenField(*StateObject, TEXT("global_mean"),
		OutNormStats.GlobalMean, TEXT("state.global_mean"));
	bOk &= FlattenField(*StateObject, TEXT("global_std"),
		OutNormStats.GlobalStd, TEXT("state.global_std"));

	bOk &= FlattenField(*DeltaObject, TEXT("bone_mean"),
		OutNormStats.DeltaBoneMean, TEXT("delta.bone_mean"));
	bOk &= FlattenField(*DeltaObject, TEXT("bone_std"),
		OutNormStats.DeltaBoneStd, TEXT("delta.bone_std"));
	bOk &= FlattenField(*DeltaObject, TEXT("global_mean"),
		OutNormStats.DeltaGlobalMean, TEXT("delta.global_mean"));
	bOk &= FlattenField(*DeltaObject, TEXT("global_std"),
		OutNormStats.DeltaGlobalStd, TEXT("delta.global_std"));

	bOk &= FlattenField(*ControlObject, TEXT("mean"),
		OutNormStats.ControlMean, TEXT("control.mean"));
	bOk &= FlattenField(*ControlObject, TEXT("std"),
		OutNormStats.ControlStd, TEXT("control.std"));

	TArray<FString> ControlChannelNames;
	if ((*ControlObject)->TryGetStringArrayField(TEXT("channels"), ControlChannelNames))
	{
		OutNormStats.ControlStatChannels = ControlChannelNames;
	}
	else
	{
		bOk = false;
	}

	if (!bOk)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: norm_stats.json at '%s' has a malformed ")
			TEXT("state/delta/control section."), *NormStatsPath);
		return false;
	}

	if (Manifest.PromptEmbChannels > 0)
	{
		const TSharedPtr<FJsonObject>* PromptObject = nullptr;
		if (!Root->TryGetObjectField(TEXT("prompt"), PromptObject))
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: manifest declares prompt_emb_channels=%d ")
				TEXT("but norm_stats.json has no 'prompt' section — the ")
				TEXT("learned null embedding is required, zeros are not a ")
				TEXT("valid substitute (inference_contract.md §4)."),
				Manifest.PromptEmbChannels);
			return false;
		}
		if (!FlattenField(*PromptObject, TEXT("null_emb"),
			OutNormStats.PromptNullEmb, TEXT("prompt.null_emb")))
		{
			return false;
		}
		if (OutNormStats.PromptNullEmb.Num() != Manifest.PromptEmbChannels)
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: prompt.null_emb has %d values, expected ")
				TEXT("prompt_emb_channels=%d."),
				OutNormStats.PromptNullEmb.Num(), Manifest.PromptEmbChannels);
			return false;
		}
	}

	// Cross-check flattened sizes against the manifest dims: this is the
	// second half of the fail-fast validation gate (the first half,
	// ValidateManifest, only checks the manifest's own internal
	// consistency; this checks norm_stats.json against it).
	const int32 ExpectedBoneCount = Manifest.NumBones * Manifest.RotationChannelsPerBone;
	if (OutNormStats.BoneMean.Num() != ExpectedBoneCount ||
		OutNormStats.BoneStd.Num() != ExpectedBoneCount)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: state bone_mean/bone_std have %d/%d values, ")
			TEXT("expected %d (num_bones * rotation_channels_per_bone)."),
			OutNormStats.BoneMean.Num(), OutNormStats.BoneStd.Num(),
			ExpectedBoneCount);
		return false;
	}
	if (OutNormStats.DeltaBoneMean.Num() != ExpectedBoneCount ||
		OutNormStats.DeltaBoneStd.Num() != ExpectedBoneCount)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: delta bone_mean/bone_std have %d/%d values, ")
			TEXT("expected %d."),
			OutNormStats.DeltaBoneMean.Num(), OutNormStats.DeltaBoneStd.Num(),
			ExpectedBoneCount);
		return false;
	}
	const int32 ExpectedGlobalCount = Manifest.RootLocalMotionChannels;
	if (OutNormStats.GlobalMean.Num() != ExpectedGlobalCount ||
		OutNormStats.GlobalStd.Num() != ExpectedGlobalCount ||
		OutNormStats.DeltaGlobalMean.Num() != ExpectedGlobalCount ||
		OutNormStats.DeltaGlobalStd.Num() != ExpectedGlobalCount)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: global mean/std arrays do not all have the ")
			TEXT("expected %d (root_local_motion_channels) entries."),
			ExpectedGlobalCount);
		return false;
	}
	// Control stats cover vx,vz always; aim_x/aim_z are unit-norm and
	// carry no stats even when control_channels==4 (inference_contract
	// §3.2) — so the stat arrays are sized by ControlStatChannels, not
	// blindly by manifest.ControlChannels.
	const int32 ExpectedControlStatCount = OutNormStats.ControlStatChannels.Num();
	if (OutNormStats.ControlMean.Num() != ExpectedControlStatCount ||
		OutNormStats.ControlStd.Num() != ExpectedControlStatCount)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: control mean/std have %d/%d values but ")
			TEXT("control.channels lists %d names."),
			OutNormStats.ControlMean.Num(), OutNormStats.ControlStd.Num(),
			ExpectedControlStatCount);
		return false;
	}

	OutNormStats.bIsFullyParsed = true;
	return true;
}
