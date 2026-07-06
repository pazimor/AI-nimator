// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "UObject/StrongObjectPtr.h"
#include "AInimatorClipBpeTokenizer.h"
#include "AInimatorManifest.h"

namespace UE::NNE
{
	class IModelInstanceCPU;
}
class UNNEModelData;

/**
 * Owns the NNE model instance for the bundle's `text_encoder.onnx`
 * (Goal B phase B7, `apps/spec/text_encoding.md` §1/§3) and runs one
 * encode pass per prompt change:
 * `input_ids + attention_mask → prompt_emb`.
 *
 * The masked-mean pooling is baked INTO the ONNX graph — this class
 * never pools. Encoding happens at action scale (a prompt change),
 * never per frame, so a synchronous RunSync is acceptable here
 * (unlike the controller hot path).
 *
 * NNE API note: same caveat as UAInimatorControllerRuntime — written
 * against the UE 5.3+ NNE CPU surface; VERIFY exact type/method names
 * against the installed engine before compiling. The ONNX `input_ids`
 * input is int64; the binding uploads an int64 buffer directly.
 */
class AINIMATOR_API FAInimatorPromptTextEncoder
{
public:
	/**
	 * Load `text_encoder.onnx` + `tokenizer/` from BundleDirectory as
	 * described by the manifest section. Returns false (with a
	 * specific UE_LOG(LogAInimator, Error, ...)) on any missing file,
	 * malformed vocabulary, or NNE init failure — never a silent
	 * partial init (vérité #4).
	 */
	bool Init(
		const FString& BundleDirectory,
		const FAInimatorTextEncoderManifest& Section);

	bool IsInitialized() const { return bInitialized; }
	int32 GetEmbeddingChannels() const { return EmbeddingChannels; }
	const FClipBpeTokenizer& GetTokenizer() const { return Tokenizer; }

	/**
	 * Tokenize `Text` and run one encoder forward; `OutEmbedding` is
	 * resized to `GetEmbeddingChannels()`. Returns false on an
	 * uninitialized encoder or an NNE failure (logged).
	 */
	bool EncodePrompt(const FString& Text, TArray<float>& OutEmbedding);

private:
	FClipBpeTokenizer Tokenizer;
	int32 EmbeddingChannels = 0;
	bool bInitialized = false;

	/** Keeps the runtime-initialized model data alive without a
	 *  UPROPERTY (this is not a UObject). */
	TStrongObjectPtr<UNNEModelData> ModelData;
	TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;

	// Reused per encode — a prompt change allocates nothing after the
	// first call.
	TArray<int64> ScratchInputIds;
	TArray<float> ScratchAttentionMask;
};
