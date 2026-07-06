// Copyright AI-nimator.

#include "AInimatorClipBpeTokenizer.h"

#include "AInimatorLog.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
	const FString EndOfWordSuffix = TEXT("</w>");

	// Contraction suffixes split off before word tokenization (CLIP
	// regex alternation order — checked before letter runs, spec §2.2).
	const TCHAR* GContractions[] = {
		TEXT("'s"), TEXT("'t"), TEXT("'re"), TEXT("'ve"),
		TEXT("'m"), TEXT("'ll"), TEXT("'d")};

	bool IsAsciiSpace(TCHAR Char)
	{
		return Char == TEXT(' ') || Char == TEXT('\t') ||
			Char == TEXT('\n') || Char == TEXT('\r') ||
			Char == TEXT('\v') || Char == TEXT('\f');
	}

	bool IsLetter(TCHAR Char)
	{
		return (Char >= TEXT('a') && Char <= TEXT('z')) || Char > 0x7F;
	}

	bool IsDigit(TCHAR Char)
	{
		return Char >= TEXT('0') && Char <= TEXT('9');
	}

	bool IsOther(TCHAR Char)
	{
		return !IsAsciiSpace(Char) && !IsLetter(Char) && !IsDigit(Char);
	}

	/** Length of the contraction starting at Index, or 0. */
	int32 MatchContraction(const FString& Text, int32 Index)
	{
		for (const TCHAR* Contraction : GContractions)
		{
			const int32 Length = FCString::Strlen(Contraction);
			if (Index + Length <= Text.Len() &&
				FCString::Strncmp(*Text + Index, Contraction, Length) == 0)
			{
				return Length;
			}
		}
		return 0;
	}

	int32 ConsumeRun(
		const FString& Text,
		int32 Start,
		TArray<FString>& OutWords,
		bool (*Predicate)(TCHAR))
	{
		int32 End = Start;
		while (End < Text.Len() && Predicate(Text[End]))
		{
			// A contraction suffix terminates a run (spec §2.2 — the
			// CLIP regex alternation matches contractions first).
			if (End > Start && MatchContraction(Text, End) > 0)
			{
				break;
			}
			++End;
		}
		OutWords.Add(Text.Mid(Start, End - Start));
		return End;
	}
} // namespace

bool FClipBpeTokenizer::Init(
	const FString& VocabJson,
	const FString& MergesText,
	int32 InMaxLength,
	int32 InBosId,
	int32 InEosId,
	int32 InPadId)
{
	bInitialized = false;
	if (InMaxLength < 2)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("ClipBpeTokenizer: max_length must be >= 2 (bos + eos); got %d."),
			InMaxLength);
		return false;
	}

	MaxLength = InMaxLength;
	BosId = InBosId;
	EosId = InEosId;
	PadId = InPadId;
	if (!ParseVocab(VocabJson) || !ParseMerges(MergesText))
	{
		return false;
	}

	BuildByteEncoder();
	bInitialized = true;
	return true;
}

bool FClipBpeTokenizer::Encode(
	const FString& Text,
	TArray<int64>& OutInputIds,
	TArray<float>& OutAttentionMask) const
{
	if (!bInitialized)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("ClipBpeTokenizer::Encode called before a successful Init()."));
		return false;
	}

	TArray<FString> Words;
	SplitWords(CleanText(Text), Words);

	TArray<int32> Tokens;
	TArray<FString> Pieces;
	for (const FString& Word : Words)
	{
		Pieces.Reset();
		ApplyBpe(Word, Pieces);
		for (const FString& Piece : Pieces)
		{
			const int32* Id = Vocab.Find(Piece);
			if (Id == nullptr)
			{
				UE_LOG(LogAInimator, Error,
					TEXT("ClipBpeTokenizer: BPE piece '%s' missing from vocab.json — ")
					TEXT("vocab/merges mismatch."), *Piece);
				return false;
			}
			Tokens.Add(*Id);
		}
	}

	// Spec §2.5: truncate to max_length - 2, wrap in bos/eos, pad with
	// the pad id; mask 1.0 on real tokens.
	const int32 BodyCount = FMath::Min(Tokens.Num(), MaxLength - 2);
	OutInputIds.SetNumUninitialized(MaxLength);
	OutAttentionMask.SetNumUninitialized(MaxLength);
	OutInputIds[0] = BosId;
	for (int32 Index = 0; Index < BodyCount; ++Index)
	{
		OutInputIds[Index + 1] = Tokens[Index];
	}
	OutInputIds[BodyCount + 1] = EosId;
	const int32 RealCount = BodyCount + 2;
	for (int32 Index = RealCount; Index < MaxLength; ++Index)
	{
		OutInputIds[Index] = PadId;
	}
	for (int32 Index = 0; Index < MaxLength; ++Index)
	{
		OutAttentionMask[Index] = Index < RealCount ? 1.0f : 0.0f;
	}
	return true;
}

FString FClipBpeTokenizer::CleanText(const FString& Text)
{
	FString Cleaned;
	Cleaned.Reserve(Text.Len());
	bool bPendingSpace = false;
	for (TCHAR Raw : Text)
	{
		const TCHAR Char =
			(Raw >= TEXT('A') && Raw <= TEXT('Z')) ? Raw + 32 : Raw;
		if (IsAsciiSpace(Char))
		{
			bPendingSpace = Cleaned.Len() > 0;
			continue;
		}
		if (bPendingSpace)
		{
			Cleaned.AppendChar(TEXT(' '));
			bPendingSpace = false;
		}
		Cleaned.AppendChar(Char);
	}
	return Cleaned;
}

void FClipBpeTokenizer::SplitWords(const FString& Text, TArray<FString>& OutWords)
{
	int32 Index = 0;
	while (Index < Text.Len())
	{
		const TCHAR Char = Text[Index];
		if (IsAsciiSpace(Char))
		{
			++Index;
			continue;
		}

		const int32 ContractionLength = MatchContraction(Text, Index);
		if (ContractionLength > 0)
		{
			OutWords.Add(Text.Mid(Index, ContractionLength));
			Index += ContractionLength;
		}
		else if (IsLetter(Char))
		{
			Index = ConsumeRun(Text, Index, OutWords, &IsLetter);
		}
		else if (IsDigit(Char))
		{
			OutWords.Add(Text.Mid(Index, 1));
			++Index;
		}
		else
		{
			Index = ConsumeRun(Text, Index, OutWords, &IsOther);
		}
	}
}

void FClipBpeTokenizer::ApplyBpe(const FString& Word, TArray<FString>& OutPieces) const
{
	const FTCHARToUTF8 Utf8(*Word);
	const int32 ByteCount = Utf8.Length();
	if (ByteCount == 0)
	{
		return;
	}

	OutPieces.Reserve(ByteCount);
	const uint8* Bytes = reinterpret_cast<const uint8*>(Utf8.Get());
	for (int32 Index = 0; Index < ByteCount; ++Index)
	{
		OutPieces.Add(FString::Chr(ByteEncoder[Bytes[Index]]));
	}

	// CLIP specific: the final character carries </w> BEFORE merges.
	OutPieces.Last() += EndOfWordSuffix;

	while (OutPieces.Num() > 1)
	{
		int32 BestRank = MAX_int32;
		int32 BestIndex = INDEX_NONE;
		for (int32 Index = 0; Index < OutPieces.Num() - 1; ++Index)
		{
			const int32* Rank =
				MergeRanks.Find(MergeKey(OutPieces[Index], OutPieces[Index + 1]));
			if (Rank != nullptr && *Rank < BestRank)
			{
				BestRank = *Rank;
				BestIndex = Index;
			}
		}
		if (BestIndex == INDEX_NONE)
		{
			break;
		}

		// Merge every adjacent (Left, Right) occurrence, left to right.
		const FString Left = OutPieces[BestIndex];
		const FString Right = OutPieces[BestIndex + 1];
		int32 Write = 0;
		int32 Read = 0;
		while (Read < OutPieces.Num())
		{
			if (Read + 1 < OutPieces.Num() &&
				OutPieces[Read] == Left && OutPieces[Read + 1] == Right)
			{
				OutPieces[Write++] = Left + Right;
				Read += 2;
			}
			else
			{
				OutPieces[Write++] = MoveTemp(OutPieces[Read]);
				++Read;
			}
		}
		OutPieces.SetNum(Write);
	}
}

bool FClipBpeTokenizer::ParseVocab(const FString& VocabJson)
{
	TSharedPtr<FJsonObject> Root;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(VocabJson);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid() ||
		Root->Values.Num() == 0)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("ClipBpeTokenizer: tokenizer vocab.json is not a non-empty JSON object."));
		return false;
	}

	Vocab.Empty(Root->Values.Num());
	for (const TPair<FString, TSharedPtr<FJsonValue>>& Entry : Root->Values)
	{
		double Id = 0.0;
		if (!Entry.Value.IsValid() || !Entry.Value->TryGetNumber(Id))
		{
			UE_LOG(LogAInimator, Error,
				TEXT("ClipBpeTokenizer: vocab.json entry '%s' has a non-numeric id."),
				*Entry.Key);
			return false;
		}
		Vocab.Add(Entry.Key, static_cast<int32>(Id));
	}
	return true;
}

bool FClipBpeTokenizer::ParseMerges(const FString& MergesText)
{
	MergeRanks.Empty();
	TArray<FString> Lines;
	MergesText.ParseIntoArrayLines(Lines, /*bCullEmpty=*/true);
	for (const FString& RawLine : Lines)
	{
		FString Line = RawLine;
		Line.TrimEndInline();
		if (Line.IsEmpty() || Line.StartsWith(TEXT("#")))
		{
			continue;
		}

		int32 Space = INDEX_NONE;
		if (!Line.FindChar(TEXT(' '), Space) || Space <= 0 ||
			Line.Find(TEXT(" "), ESearchCase::CaseSensitive,
				ESearchDir::FromStart, Space + 1) != INDEX_NONE)
		{
			UE_LOG(LogAInimator, Error,
				TEXT("ClipBpeTokenizer: malformed merges.txt line '%s'."), *Line);
			return false;
		}
		MergeRanks.Add(Line, MergeRanks.Num());
	}

	if (MergeRanks.Num() == 0)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("ClipBpeTokenizer: merges.txt contains no merge rules."));
		return false;
	}
	return true;
}

void FClipBpeTokenizer::BuildByteEncoder()
{
	// GPT-2/CLIP byte -> printable-unicode bijection (spec §2.3):
	// printable Latin-1 bytes map to themselves, the remaining 68
	// bytes map to U+0100.. so every byte has a map-safe character.
	int32 Offset = 0;
	for (int32 Byte = 0; Byte < 256; ++Byte)
	{
		const bool bPrintable =
			(Byte >= TEXT('!') && Byte <= TEXT('~')) ||
			(Byte >= 0xA1 && Byte <= 0xAC) ||
			(Byte >= 0xAE && Byte <= 0xFF);
		ByteEncoder[Byte] = bPrintable
			? static_cast<TCHAR>(Byte)
			: static_cast<TCHAR>(256 + Offset++);
	}
}

FString FClipBpeTokenizer::MergeKey(const FString& Left, const FString& Right)
{
	return Left + TEXT(" ") + Right;
}
