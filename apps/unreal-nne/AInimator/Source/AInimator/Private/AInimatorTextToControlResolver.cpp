// Copyright AI-nimator.

#include "AInimatorTextToControlResolver.h"
#include "AInimatorTextToControlTable.h"
#include "AInimatorLog.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
	/** Parses GAInimatorTextToControlTableJson once and caches the
	 *  result — the table is a static compiled-in constant, never
	 *  changes at runtime, so re-parsing every Resolve() call would be
	 *  pure per-call waste (no correctness reason to avoid a static
	 *  local here: this is a data mapper, not the hot per-tick
	 *  inference path covered by the "reused scratch buffers" NNE
	 *  guidance). */
	TSharedPtr<FJsonObject> GetTable()
	{
		static TSharedPtr<FJsonObject> CachedTable = [] {
			TSharedPtr<FJsonObject> Parsed;
			const FString JsonString(GAInimatorTextToControlTableJson);
			TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
			if (!FJsonSerializer::Deserialize(Reader, Parsed) || !Parsed.IsValid())
			{
				UE_LOG(LogAInimator, Error,
					TEXT("AInimator: failed to parse the embedded B6 text-to-control ")
					TEXT("table (AInimatorTextToControlTable.h) — this is a plugin bug, ")
					TEXT("not a user error."));
				return TSharedPtr<FJsonObject>();
			}
			return Parsed;
		}();
		return CachedTable;
	}

	/** (x, z) direction pair read from the "directions" JSON object. */
	bool TryGetDirection(const TSharedPtr<FJsonObject>& Directions, const FString& Token, float& OutX, float& OutZ)
	{
		const TArray<TSharedPtr<FJsonValue>>* Pair = nullptr;
		if (!Directions.IsValid() || !Directions->TryGetArrayField(Token, Pair) || !Pair || Pair->Num() != 2)
		{
			return false;
		}
		OutX = static_cast<float>((*Pair)[0]->AsNumber());
		OutZ = static_cast<float>((*Pair)[1]->AsNumber());
		return true;
	}

	bool TryGetSpeed(const TSharedPtr<FJsonObject>& Speeds, const FString& Token, float& OutSpeed)
	{
		double Value = 0.0;
		if (!Speeds.IsValid() || !Speeds->TryGetNumberField(Token, Value))
		{
			return false;
		}
		OutSpeed = static_cast<float>(Value);
		return true;
	}
}

FString FTextToControlResolver::StripAccents(const FString& Lowered)
{
	// Explicit FR common-accent replacement table (see header comment
	// for the documented divergence from Python's NFKD approach).
	// text_to_control.md §2: "e→e, e→e, a→a, c→c…".
	FString Result;
	Result.Reserve(Lowered.Len());
	for (const TCHAR Ch : Lowered)
	{
		switch (Ch)
		{
			case TEXT('é'): // e-acute (e)
			case TEXT('è'): // e-grave (e)
			case TEXT('ê'): // e-circumflex (e)
			case TEXT('ë'): // e-diaeresis (e)
				Result.AppendChar(TEXT('e'));
				break;
			case TEXT('à'): // a-grave (a)
			case TEXT('â'): // a-circumflex (a)
				Result.AppendChar(TEXT('a'));
				break;
			case TEXT('ç'): // c-cedilla (c)
				Result.AppendChar(TEXT('c'));
				break;
			case TEXT('î'): // i-circumflex (i)
			case TEXT('ï'): // i-diaeresis (i)
				Result.AppendChar(TEXT('i'));
				break;
			case TEXT('ô'): // o-circumflex (o)
				Result.AppendChar(TEXT('o'));
				break;
			case TEXT('û'): // u-circumflex (u)
			case TEXT('ù'): // u-grave (u)
			case TEXT('ü'): // u-diaeresis (u)
				Result.AppendChar(TEXT('u'));
				break;
			default:
				Result.AppendChar(Ch);
				break;
		}
	}
	return Result;
}

void FTextToControlResolver::NormalizeAndTokenize(const FString& Text, TArray<FString>& OutTokens)
{
	OutTokens.Reset();
	const FString Lowered = Text.ToLower();
	const FString Ascii = StripAccents(Lowered);

	// Tokenize on runs of ASCII letters — mirrors the Python reference's
	// `re.compile(r"[a-z]+")` findall exactly (text_to_control.md §2:
	// "tokeniser sur tout caractere non alphabetique").
	FString CurrentToken;
	for (const TCHAR Ch : Ascii)
	{
		const bool bIsAsciiLetter = (Ch >= TEXT('a') && Ch <= TEXT('z'));
		if (bIsAsciiLetter)
		{
			CurrentToken.AppendChar(Ch);
		}
		else if (!CurrentToken.IsEmpty())
		{
			OutTokens.Add(CurrentToken);
			CurrentToken.Reset();
		}
	}
	if (!CurrentToken.IsEmpty())
	{
		OutTokens.Add(CurrentToken);
	}
}

TOptional<FAInimatorResolvedControl> FTextToControlResolver::Resolve(const FString& Text)
{
	const TSharedPtr<FJsonObject> Table = GetTable();
	if (!Table.IsValid())
	{
		return TOptional<FAInimatorResolvedControl>(); // GetTable() already logged.
	}

	TArray<FString> Tokens;
	NormalizeAndTokenize(Text, Tokens);

	const TSharedPtr<FJsonObject>* DirectionsPtr = nullptr;
	const TSharedPtr<FJsonObject>* SpeedsPtr = nullptr;
	const TSharedPtr<FJsonObject>* DefaultsPtr = nullptr;
	Table->TryGetObjectField(TEXT("directions"), DirectionsPtr);
	Table->TryGetObjectField(TEXT("speeds"), SpeedsPtr);
	Table->TryGetObjectField(TEXT("defaults"), DefaultsPtr);
	const TSharedPtr<FJsonObject> Directions = DirectionsPtr ? *DirectionsPtr : nullptr;
	const TSharedPtr<FJsonObject> Speeds = SpeedsPtr ? *SpeedsPtr : nullptr;
	const TSharedPtr<FJsonObject> Defaults = DefaultsPtr ? *DefaultsPtr : nullptr;

	// --- Step 2: directions (sum, then unit-normalize; found-but-zero = ambiguous). ---
	bool bDirectionFound = false;
	bool bAmbiguous = false;
	float SumX = 0.0f;
	float SumZ = 0.0f;
	for (const FString& Token : Tokens)
	{
		float DirX = 0.0f;
		float DirZ = 0.0f;
		if (TryGetDirection(Directions, Token, DirX, DirZ))
		{
			bDirectionFound = true;
			SumX += DirX;
			SumZ += DirZ;
		}
	}

	TOptional<FVector2D> Direction;
	if (bDirectionFound)
	{
		const float Norm = FMath::Sqrt(SumX * SumX + SumZ * SumZ);
		if (Norm < 1e-9f)
		{
			bAmbiguous = true; // e.g. "gauche droite": cancels out.
		}
		else
		{
			Direction = FVector2D(SumX / Norm, SumZ / Norm);
		}
	}

	// --- Step 3: speed (max; any zero/"stop"-family keyword wins outright). ---
	bool bSpeedFound = false;
	bool bStopWins = false;
	float BestSpeed = 0.0f;
	for (const FString& Token : Tokens)
	{
		float Value = 0.0f;
		if (TryGetSpeed(Speeds, Token, Value))
		{
			bSpeedFound = true;
			if (Value == 0.0f)
			{
				bStopWins = true;
				BestSpeed = 0.0f;
				break;
			}
			BestSpeed = bSpeedFound ? FMath::Max(BestSpeed, Value) : Value;
		}
	}

	// --- Step 6: resolution failure (no tokens recognized, or ambiguous). ---
	if (bAmbiguous)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: text command '%s' is ambiguous (direction words ")
			TEXT("cancel out) — keeping current control (text_to_control.md §2)."),
			*Text);
		return TOptional<FAInimatorResolvedControl>();
	}
	if (!bDirectionFound && !bSpeedFound)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: text command '%s' matched no known direction or ")
			TEXT("speed keyword — keeping current control (no silent fallback, ")
			TEXT("text_to_control.md §2)."),
			*Text);
		return TOptional<FAInimatorResolvedControl>();
	}

	// --- Step 4: defaults. ---
	float Speed = BestSpeed;
	if (!bSpeedFound)
	{
		double DefaultSpeed = 0.033;
		if (Defaults.IsValid())
		{
			Defaults->TryGetNumberField(TEXT("speed_when_direction_only"), DefaultSpeed);
		}
		Speed = static_cast<float>(DefaultSpeed);
	}

	FVector2D ResolvedDirection = Direction.Get(FVector2D(0.0f, 1.0f));
	if (!Direction.IsSet())
	{
		const TArray<TSharedPtr<FJsonValue>>* DefaultDir = nullptr;
		if (Defaults.IsValid() && Defaults->TryGetArrayField(TEXT("direction_when_speed_only"), DefaultDir)
			&& DefaultDir && DefaultDir->Num() == 2)
		{
			ResolvedDirection = FVector2D(
				static_cast<float>((*DefaultDir)[0]->AsNumber()),
				static_cast<float>((*DefaultDir)[1]->AsNumber()));
		}
	}

	// --- Step 5: output. ---
	FAInimatorResolvedControl Result;
	if (bStopWins || Speed == 0.0f)
	{
		Result.Vx = 0.0f;
		Result.Vz = 0.0f;
		Result.AimX = 0.0f;
		Result.AimZ = 1.0f;
		return Result;
	}

	Result.Vx = ResolvedDirection.X * Speed;
	Result.Vz = ResolvedDirection.Y * Speed;
	Result.AimX = ResolvedDirection.X;
	Result.AimZ = ResolvedDirection.Y;
	return Result;
}
