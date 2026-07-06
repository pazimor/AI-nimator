// Copyright AI-nimator.

#include "AInimatorPromptTextEncoder.h"

#include "AInimatorLog.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

// Same NNE targeting caveat as AInimatorControllerRuntime.cpp (UE 5.3+,
// CPU runtime for determinism/parity — text_encoding.md §1).
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNETypes.h"
#include "NNEModelData.h"
#include "UObject/WeakInterfacePtr.h"

bool FAInimatorPromptTextEncoder::Init(
	const FString& BundleDirectory,
	const FAInimatorTextEncoderManifest& Section)
{
	bInitialized = false;
	if (!Section.bIsPresent)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder: manifest has no text_encoder section — ")
			TEXT("this bundle predates A7.1 or was exported without ")
			TEXT("--encoder-artifact (text_encoding.md §1)."));
		return false;
	}

	FString VocabJson;
	const FString VocabPath =
		FPaths::Combine(BundleDirectory, Section.TokenizerVocab);
	if (!FFileHelper::LoadFileToString(VocabJson, *VocabPath))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder: failed to read '%s'."), *VocabPath);
		return false;
	}

	FString MergesText;
	const FString MergesPath =
		FPaths::Combine(BundleDirectory, Section.TokenizerMerges);
	if (!FFileHelper::LoadFileToString(MergesText, *MergesPath))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder: failed to read '%s'."), *MergesPath);
		return false;
	}

	if (!Tokenizer.Init(
			VocabJson, MergesText, Section.MaxLength,
			Section.BosId, Section.EosId, Section.PadId))
	{
		return false; // Tokenizer already logged the specific reason.
	}

	TArray<uint8> OnnxBytes;
	const FString OnnxPath = FPaths::Combine(BundleDirectory, Section.File);
	if (!FFileHelper::LoadFileToArray(OnnxBytes, *OnnxPath))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder: failed to read '%s'."), *OnnxPath);
		return false;
	}

	// Same runtime-side UNNEModelData init caveat as
	// UAInimatorControllerRuntime::InitializeModel — VERIFY against the
	// installed NNE version.
	ModelData = TStrongObjectPtr<UNNEModelData>(
		NewObject<UNNEModelData>(GetTransientPackage()));
	if (!ModelData.IsValid() || !ModelData->Init(TEXT("onnx"), OnnxBytes))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder: UNNEModelData failed to init from '%s'."),
			*OnnxPath);
		return false;
	}

	TWeakInterfacePtr<INNERuntimeCPU> Runtime =
		UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
	if (!Runtime.IsValid())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder: no NNE CPU runtime available (expected ")
			TEXT("'NNERuntimeORTCpu' or equivalent)."));
		return false;
	}

	TSharedPtr<UE::NNE::IModelCPU> Model =
		Runtime->CreateModelCPU(ModelData.Get());
	if (!Model.IsValid())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder: CreateModelCPU failed for '%s'."),
			*OnnxPath);
		return false;
	}

	ModelInstance = Model->CreateModelInstanceCPU();
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder: CreateModelInstanceCPU failed for '%s'."),
			*OnnxPath);
		return false;
	}

	EmbeddingChannels = Section.EmbeddingChannels;
	bInitialized = true;
	return true;
}

bool FAInimatorPromptTextEncoder::EncodePrompt(
	const FString& Text,
	TArray<float>& OutEmbedding)
{
	if (!bInitialized)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder::EncodePrompt called before a ")
			TEXT("successful Init()."));
		return false;
	}

	if (!Tokenizer.Encode(Text, ScratchInputIds, ScratchAttentionMask))
	{
		return false; // Tokenizer already logged the specific reason.
	}

	// Bind and run one encode (contract: input_ids int64 (1, T),
	// attention_mask float (1, T) -> prompt_emb float (1, D) —
	// text_encoding.md §1). Binding order matches the graph's declared
	// input order, same convention as the controller forward.
	using namespace UE::NNE;

	TArray<FTensorBindingCPU> Inputs;
	FTensorBindingCPU InputIdsBinding;
	InputIdsBinding.Data = ScratchInputIds.GetData();
	InputIdsBinding.SizeInBytes = ScratchInputIds.Num() * sizeof(int64);
	Inputs.Add(InputIdsBinding);

	FTensorBindingCPU AttentionMaskBinding;
	AttentionMaskBinding.Data = ScratchAttentionMask.GetData();
	AttentionMaskBinding.SizeInBytes =
		ScratchAttentionMask.Num() * sizeof(float);
	Inputs.Add(AttentionMaskBinding);

	OutEmbedding.SetNumUninitialized(EmbeddingChannels);
	TArray<FTensorBindingCPU> Outputs;
	FTensorBindingCPU PromptEmbBinding;
	PromptEmbBinding.Data = OutEmbedding.GetData();
	PromptEmbBinding.SizeInBytes = OutEmbedding.Num() * sizeof(float);
	Outputs.Add(PromptEmbBinding);

	const int32 RunStatus = ModelInstance->RunSync(Inputs, Outputs);
	if (RunStatus != 0)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("PromptTextEncoder: NNE RunSync failed with status %d."),
			RunStatus);
		return false;
	}
	return true;
}
