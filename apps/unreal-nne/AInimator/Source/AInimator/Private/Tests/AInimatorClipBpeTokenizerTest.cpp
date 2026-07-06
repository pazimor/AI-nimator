// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorClipBpeTokenizer.h"
#include "AInimatorTextEncodingParity.h"

#include "Dom/JsonObject.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

#if WITH_DEV_AUTOMATION_TESTS

// Parity suite for FClipBpeTokenizer (Goal B phase B7): the canonical
// cases are the embedded verbatim copy of
// apps/spec/text_encoding_parity.json — same prompts, same ids as the
// Python reference (clip_bpe_reference.py) and the Unity EditMode
// suite (text_encoding.md §4).
//
// The vocabulary (vocab.json / merges.txt, ~1.5 MB) is a build
// artifact delivered inside the bundle — never committed. The suite
// loads it from the delivered Content/AInimatorBundle/tokenizer/ and
// self-skips (logged, not failed) when the delivered bundle predates
// A7.1: run `make plugin-unreal … ENCODER_ARTIFACT=…` first.

namespace AInimatorClipBpeTokenizerTestHelpers
{
	bool ResolveTokenizerFiles(FString& OutVocabJson, FString& OutMergesText)
	{
		const TSharedPtr<IPlugin> Plugin =
			IPluginManager::Get().FindPlugin(TEXT("AInimator"));
		if (!Plugin.IsValid())
		{
			return false;
		}
		const FString TokenizerDir = FPaths::Combine(
			Plugin->GetContentDir(), TEXT("AInimatorBundle"), TEXT("tokenizer"));
		return FFileHelper::LoadFileToString(
				OutVocabJson, *FPaths::Combine(TokenizerDir, TEXT("vocab.json"))) &&
			FFileHelper::LoadFileToString(
				OutMergesText, *FPaths::Combine(TokenizerDir, TEXT("merges.txt")));
	}

	TSharedPtr<FJsonObject> ParseParityFile()
	{
		TSharedPtr<FJsonObject> Root;
		const TSharedRef<TJsonReader<>> Reader =
			TJsonReaderFactory<>::Create(GAInimatorTextEncodingParityJson);
		FJsonSerializer::Deserialize(Reader, Root);
		return Root;
	}
} // namespace AInimatorClipBpeTokenizerTestHelpers

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorClipBpeParityTest,
	"AInimator.TextEncoding.TokenizerParityWithPythonReference",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorClipBpeParityTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorClipBpeTokenizerTestHelpers;

	const TSharedPtr<FJsonObject> Parity = ParseParityFile();
	if (!TestTrue(TEXT("embedded parity JSON parses"), Parity.IsValid()))
	{
		return false;
	}

	FString VocabJson;
	FString MergesText;
	if (!ResolveTokenizerFiles(VocabJson, MergesText))
	{
		AddInfo(TEXT(
			"Delivered bundle ships no tokenizer/ (A7.0 bundle) — deliver an "
			"A7.1 bundle (make plugin-unreal … ENCODER_ARTIFACT="
			"output/clip_text_artifact) to run this suite. Skipping."));
		return true;
	}

	const int32 MaxLength = Parity->GetIntegerField(TEXT("max_length"));
	FClipBpeTokenizer Tokenizer;
	if (!TestTrue(TEXT("tokenizer Init succeeds"), Tokenizer.Init(
			VocabJson, MergesText, MaxLength,
			Parity->GetIntegerField(TEXT("bos_id")),
			Parity->GetIntegerField(TEXT("eos_id")),
			Parity->GetIntegerField(TEXT("pad_id")))))
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
	if (!TestTrue(TEXT("parity JSON has cases"),
			Parity->TryGetArrayField(TEXT("cases"), Cases) && Cases->Num() > 0))
	{
		return false;
	}

	TArray<int64> InputIds;
	TArray<float> AttentionMask;
	for (const TSharedPtr<FJsonValue>& CaseValue : *Cases)
	{
		const TSharedPtr<FJsonObject> Case = CaseValue->AsObject();
		const FString Prompt = Case->GetStringField(TEXT("prompt"));
		TestTrue(FString::Printf(TEXT("'%s' encodes"), *Prompt),
			Tokenizer.Encode(Prompt, InputIds, AttentionMask));

		const TArray<TSharedPtr<FJsonValue>>& ExpectedIds =
			Case->GetArrayField(TEXT("input_ids"));
		const TArray<TSharedPtr<FJsonValue>>& ExpectedMask =
			Case->GetArrayField(TEXT("attention_mask"));
		TestEqual(FString::Printf(TEXT("'%s' id count"), *Prompt),
			InputIds.Num(), ExpectedIds.Num());
		for (int32 Index = 0; Index < InputIds.Num(); ++Index)
		{
			TestEqual(
				FString::Printf(TEXT("'%s' input_ids[%d]"), *Prompt, Index),
				static_cast<int32>(InputIds[Index]),
				static_cast<int32>(ExpectedIds[Index]->AsNumber()));
			TestEqual(
				FString::Printf(TEXT("'%s' attention_mask[%d]"), *Prompt, Index),
				static_cast<int32>(AttentionMask[Index]),
				static_cast<int32>(ExpectedMask[Index]->AsNumber()));
		}
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorClipBpeFailFastTest,
	"AInimator.TextEncoding.TokenizerFailsFastOnMalformedAssets",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorClipBpeFailFastTest::RunTest(const FString& Parameters)
{
	FClipBpeTokenizer Tokenizer;

	// Each rejection logs a specific LogAInimator error (fail-fast,
	// vérité #4) — declared expected so the suite verifies the loud
	// failure instead of failing on it.
	AddExpectedError(
		TEXT("not a non-empty JSON object"),
		EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("array vocab rejected"), Tokenizer.Init(
		TEXT("[]"), TEXT("a b\n"), 32, 49406, 49407, 49407));

	AddExpectedError(
		TEXT("no merge rules"),
		EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("empty merges rejected"), Tokenizer.Init(
		TEXT("{\"a\": 1}"), TEXT(""), 32, 49406, 49407, 49407));

	AddExpectedError(
		TEXT("max_length must be >= 2"),
		EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("max_length 1 rejected"), Tokenizer.Init(
		TEXT("{\"a\": 1}"), TEXT("a b\n"), 1, 49406, 49407, 49407));

	// Encode before Init fails loudly, not silently.
	TArray<int64> Ids;
	TArray<float> Mask;
	AddExpectedError(
		TEXT("Encode called before a successful Init"),
		EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("Encode before Init rejected"),
		Tokenizer.Encode(TEXT("a person dances"), Ids, Mask));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
