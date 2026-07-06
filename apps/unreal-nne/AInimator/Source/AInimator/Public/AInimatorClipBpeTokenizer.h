// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Pure C++ CLIP byte-level BPE tokenizer (Goal B phase B7,
 * `apps/spec/text_encoding.md` §2): free text → fixed-length
 * `input_ids` + `attention_mask` for the bundle's `text_encoder.onnx`.
 *
 * Value-for-value mirror of `apps/spec/clip_bpe_reference.py` (itself
 * locked to HuggingFace CLIPTokenizerFast) and of the Unity C#
 * `ClipBpeTokenizer`; parity is verified by the Automation suite
 * against the shared `text_encoding_parity.json` cases.
 *
 * The vocabulary (`vocab.json`) and ranked merges (`merges.txt`) are
 * the verbatim HuggingFace files shipped in the bundle's `tokenizer/`
 * directory — never edited by hand.
 *
 * Normative simplification (spec §2.1): the "letter" class is `[a-z]`
 * (after ASCII lowercasing) plus any codepoint above U+007F —
 * identical to Unicode `\p{L}` on the FR/EN prompt domain.
 *
 * Engine-free (Core only, no UObject / file I/O) so it stays unit
 * testable; `Init` parses ~49k vocab entries — build once per bundle,
 * at load time, never per frame.
 */
class AINIMATOR_API FClipBpeTokenizer
{
public:
	/**
	 * Parse the vocabulary + merges. Returns false (with a specific
	 * UE_LOG(LogAInimator, Error, ...)) on malformed input — the
	 * tokenizer is unusable then (fail-fast, vérité #4).
	 *
	 * Parameters mirror the manifest `text_encoder.tokenizer` section:
	 * verbatim vocab.json / merges.txt content, fixed token budget and
	 * special-token ids (bos/eos/pad, pad == eos for CLIP).
	 */
	bool Init(
		const FString& VocabJson,
		const FString& MergesText,
		int32 InMaxLength,
		int32 InBosId,
		int32 InEosId,
		int32 InPadId);

	bool IsInitialized() const { return bInitialized; }
	int32 GetMaxLength() const { return MaxLength; }
	int32 GetBosId() const { return BosId; }
	int32 GetEosId() const { return EosId; }
	int32 GetPadId() const { return PadId; }

	/**
	 * Encode `Text` into `OutInputIds` / `OutAttentionMask`, both
	 * resized to `MaxLength` (spec §2.5: BPE stream truncated to
	 * max_length − 2, wrapped in bos/eos, padded with the pad id;
	 * mask 1.0 on real tokens, 0.0 on padding). `OutInputIds` is
	 * int64 to match the ONNX `input_ids` tensor type directly.
	 *
	 * Returns false only when the tokenizer is uninitialized or a BPE
	 * piece is missing from the vocabulary (vocab/merges mismatch —
	 * logged, never silent).
	 */
	bool Encode(
		const FString& Text,
		TArray<int64>& OutInputIds,
		TArray<float>& OutAttentionMask) const;

private:
	/** Lowercase ASCII + collapse ASCII whitespace runs (spec §2.1). */
	static FString CleanText(const FString& Text);

	/** Pre-tokenization: contraction > letter run > digit > other (§2.2). */
	static void SplitWords(const FString& Text, TArray<FString>& OutWords);

	/** Byte-level BPE of one word: UTF-8 bytes → mapped chars, `</w>`
	 *  on the last char, then ranked merges (§2.3/§2.4). */
	void ApplyBpe(const FString& Word, TArray<FString>& OutPieces) const;

	bool ParseVocab(const FString& VocabJson);
	bool ParseMerges(const FString& MergesText);
	void BuildByteEncoder();

	/** Merge-rank lookup key: "left right" (a space never occurs
	 *  inside a BPE piece, so the concatenation is unambiguous). */
	static FString MergeKey(const FString& Left, const FString& Right);

	TMap<FString, int32> Vocab;
	TMap<FString, int32> MergeRanks;
	TCHAR ByteEncoder[256] = {0};

	int32 MaxLength = 0;
	int32 BosId = 0;
	int32 EosId = 0;
	int32 PadId = 0;
	bool bInitialized = false;
};
