// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * B6 "free text -> raw control vector" mapper
 * (`apps/spec/text_to_control.md`, canonical table
 * `apps/spec/text_to_control.json`, reference implementation
 * `apps/spec/text_to_control_reference.py`). Pure, engine-free class —
 * testable with plain `FString`s, no `UObject`/world dependency.
 *
 * ⚠ B6 is NOT the prompt channel (rig_binding.md §3 / DETERMINIST §2.4):
 * this only resolves a phrase to the LOW-LEVEL control vector
 * `(vx, vz[, aim_x, aim_z])`. The presets/direct-vector control path
 * remains the default entry point; this mapper is an optional ergonomic
 * shortcut, never a required one (ROADMAP_PLUGINS.md §4 B6 / §6).
 *
 * Algorithm (mirrors text_to_control.md §2 / text_to_control_reference.py
 * value-for-value, same normative order):
 *   1. Normalize: lowercase, strip common French accents, tokenize on
 *      runs of ASCII letters.
 *   2. Directions: sum every recognized direction keyword's (x, z);
 *      non-zero sum -> unit-normalize; zero sum WITH direction words
 *      present (e.g. "gauche droite") -> ambiguous -> resolution fails.
 *   3. Speed: max of every recognized speed keyword; any zero-speed
 *      ("stop" family) keyword always wins outright.
 *   4. Defaults: direction-only -> `speed_when_direction_only` (0.033);
 *      speed-only -> `direction_when_speed_only` ((0,1), i.e. forward);
 *      zero speed -> control (0,0), direction ignored.
 *   5. Output: raw `control = direction * speed` (vx, vz), meters/frame
 *      (the engine still z-normalizes exactly like a preset value); if
 *      the bundle has 4 control channels, `aim = direction` (or (0,1)
 *      when speed is zero).
 *   6. No recognized token at all, or ambiguity (step 2) -> resolution
 *      FAILS: caller keeps its current control, `Resolve` logs the
 *      reason via `LogAInimator` and returns an unset `TOptional`. Never
 *      a silent fallback to some movement.
 */
struct AINIMATOR_API FAInimatorResolvedControl
{
	/** Raw desired velocity, meters/frame, root-local frame. */
	float Vx = 0.0f;
	float Vz = 0.0f;

	/** Unit aim direction (movement direction; forward when idle). */
	float AimX = 0.0f;
	float AimZ = 0.0f;
};

class AINIMATOR_API FTextToControlResolver
{
public:
	/**
	 * Resolves a free-text command (French or English, e.g. "cours vers
	 * la gauche") into a raw control vector.
	 *
	 * Returns an unset TOptional when nothing was recognized, or when
	 * direction keywords are present but cancel out (ambiguous) — see
	 * class comment step 6. Logs the specific reason via LogAInimator in
	 * both failure cases (fail LOUD, never silent).
	 */
	static TOptional<FAInimatorResolvedControl> Resolve(const FString& Text);

private:
	/** Lowercase, accent-strip (FR common diacritics), tokenize on runs
	 *  of ASCII letters — mirrors _normalizeText() in the Python
	 *  reference. See StripAccents()'s own comment for the one
	 *  documented divergence from Python's NFKD+ascii-encode approach. */
	static void NormalizeAndTokenize(const FString& Text, TArray<FString>& OutTokens);

	/** Replaces common French accented letters with their ASCII
	 *  equivalent (e.g. e-acute/e-grave/e-circumflex/e-diaeresis -> 'e').
	 *
	 *  DOCUMENTED DIVERGENCE from the Python reference: Python uses
	 *  Unicode NFKD decomposition + ASCII-encode-with-ignore, which
	 *  strips ANY combining diacritic on ANY base letter (covers all of
	 *  Unicode, e.g. Spanish n-tilde, German u-umlaut, etc.) and also
	 *  drops any remaining non-ASCII character outright. This resolver
	 *  instead uses an explicit replacement table for the common French
	 *  accented letters listed in text_to_control.md §2
	 *  (e/e/e/e -> e, a/a -> a, c -> c, i/i -> i, o -> o, u/u/u -> u).
	 *  For every keyword actually present in text_to_control.json and
	 *  every parity test case, the two approaches produce IDENTICAL
	 *  tokens — the divergence only matters for languages/diacritics
	 *  outside this explicit table, which the canonical table has no
	 *  keywords for anyway. If a caller feeds e.g. Spanish text, an
	 *  unmapped accented character is left as-is here (Python would
	 *  drop it); tokenization on ASCII-letter runs then still splits
	 *  around it identically to Python dropping it, so the token
	 *  boundaries end up the same in every case observed in the parity
	 *  suite. Flagged here for engine validation, not assumed safe
	 *  beyond FR/EN.
	 */
	static FString StripAccents(const FString& Lowered);
};
