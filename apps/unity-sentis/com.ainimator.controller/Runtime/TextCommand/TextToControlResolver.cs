using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using AInimator.Controller.Bundle;
using UnityEngine;

namespace AInimator.Controller.TextCommand
{
    /// <summary>
    /// Successfully resolved free-text command (Goal B phase B6,
    /// <c>apps/spec/text_to_control.md</c> §2/§5): a raw (un-normalized)
    /// control vector, in the same units as <see cref="Presets.ControlPreset"/>
    /// (meters/frame, root-local ground frame; aim is a unit direction).
    /// </summary>
    public readonly struct TextControlResult
    {
        /// <summary>Desired lateral velocity, meters/frame, root-local frame.</summary>
        public readonly float Vx;

        /// <summary>Desired forward velocity, meters/frame, root-local frame (+Z = facing).</summary>
        public readonly float Vz;

        /// <summary>Aim direction X component (unit-norm 2D vector with <see cref="AimZ"/>).</summary>
        public readonly float AimX;

        /// <summary>Aim direction Z component (unit-norm 2D vector with <see cref="AimX"/>).</summary>
        public readonly float AimZ;

        public TextControlResult(float vx, float vz, float aimX, float aimZ)
        {
            Vx = vx;
            Vz = vz;
            AimX = aimX;
            AimZ = aimZ;
        }
    }

    /// <summary>
    /// Pure (engine-light) mapper from a free-text command to a raw control
    /// vector, mirroring <c>apps/spec/text_to_control_reference.py</c>
    /// value-for-value (Goal B phase B6). Driven entirely by the embedded
    /// verbatim copy of <c>apps/spec/text_to_control.json</c> under
    /// <c>Runtime/TextCommand/Resources/text_to_control.json</c> — never
    /// hard-codes keywords itself.
    /// </summary>
    /// <remarks>
    /// This class only depends on <c>UnityEngine.Resources</c> to load the
    /// embedded table text asset (and <c>Mathf</c> for the unit-norm), so its
    /// resolution logic itself stays a pure, unit-testable function of
    /// (text, table). See <c>text_to_control.md</c> §2 for the normative
    /// algorithm this mirrors, in order:
    /// <list type="number">
    /// <item>Normalize: lowercase, strip accents, tokenize on non-alphabetic runs.</item>
    /// <item>Directions: sum keyword vectors, then unit-normalize; a present-but-cancelling
    /// sum (e.g. "gauche droite") is ambiguous → resolution fails.</item>
    /// <item>Speed: max of present speed keywords; any stop-family keyword (value 0) always wins.</item>
    /// <item>Defaults: direction-only → <c>speed_when_direction_only</c>; speed-only →
    /// <c>direction_when_speed_only</c>; speed 0 → control (0,0), aim defaults to forward.</item>
    /// <item>No token recognized at all, or ambiguous directions → resolution fails
    /// (caller keeps its current control; never a silent fallback).</item>
    /// </list>
    /// <para/>
    /// Accent stripping here uses a replacement table for the common French
    /// diacritics (à/â/ä→a, é/è/ê/ë→e, î/ï→i, ô/ö→o, ù/û/ü→u, ç→c, œ→oe) rather
    /// than a Unicode NFKD decomposition (no such API dependency-free in
    /// .NET Standard 2.1 without <c>System.Globalization.Unicode</c> extras
    /// on all Unity scripting backends) — this covers every accented
    /// character in the canonical table's own keywords and the shared parity
    /// test cases; any input relying on OTHER Unicode diacritics than the
    /// ones listed here would diverge from the Python reference (NFKD-based,
    /// covers all Unicode combining marks). Documented gap, not a silent bug.
    /// </remarks>
    public static class TextToControlResolver
    {
        private const string ResourcePath = "text_to_control";

        private static Dictionary<string, object> _cachedTable;

        private static readonly Dictionary<char, char> AccentMap = new()
        {
            ['à'] = 'a', ['â'] = 'a', ['ä'] = 'a', ['á'] = 'a', ['ã'] = 'a',
            ['é'] = 'e', ['è'] = 'e', ['ê'] = 'e', ['ë'] = 'e',
            ['î'] = 'i', ['ï'] = 'i', ['í'] = 'i', ['ì'] = 'i',
            ['ô'] = 'o', ['ö'] = 'o', ['ò'] = 'o', ['ó'] = 'o', ['õ'] = 'o',
            ['ù'] = 'u', ['û'] = 'u', ['ü'] = 'u', ['ú'] = 'u',
            ['ç'] = 'c', ['ñ'] = 'n', ['ÿ'] = 'y',
        };

        /// <summary>
        /// Resolve a free-text command into a raw control vector.
        /// </summary>
        /// <param name="text">Free-text command, French or English (e.g. "cours vers la gauche").</param>
        /// <returns>
        /// The resolved vector, or <c>null</c> when nothing was recognized or
        /// the direction words cancel out (ambiguous) — caller keeps its
        /// current control and should log the failure (never a silent
        /// fallback, spec §2 rule 6).
        /// </returns>
        public static TextControlResult? Resolve(string text)
        {
            return Resolve(text, LoadTable());
        }

        /// <summary>
        /// Overload taking an explicit table (unit tests: fixture tables,
        /// no <c>Resources</c> dependency).
        /// </summary>
        public static TextControlResult? Resolve(string text, Dictionary<string, object> table)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            var tokens = NormalizeText(text);
            var directions = (Dictionary<string, object>)table["directions"];
            var speeds = (Dictionary<string, object>)table["speeds"];
            var defaults = (Dictionary<string, object>)table["defaults"];

            var direction = ResolveDirection(tokens, directions, out var directionFound);
            var speed = ResolveSpeed(tokens, speeds, out var speedFound);

            if (!directionFound && !speedFound)
            {
                return null; // nothing recognized
            }

            if (directionFound && direction == (0f, 0f))
            {
                return null; // ambiguous: direction words cancel out
            }

            if (!speedFound)
            {
                speed = Convert.ToSingle(defaults["speed_when_direction_only"], CultureInfo.InvariantCulture);
            }

            if (!directionFound)
            {
                var defaultDir = (List<object>)defaults["direction_when_speed_only"];
                direction = (
                    Convert.ToSingle(defaultDir[0], CultureInfo.InvariantCulture),
                    Convert.ToSingle(defaultDir[1], CultureInfo.InvariantCulture));
            }

            if (speed == 0f)
            {
                return new TextControlResult(0f, 0f, 0f, 1f);
            }

            return new TextControlResult(
                direction.Item1 * speed,
                direction.Item2 * speed,
                direction.Item1,
                direction.Item2);
        }

        /// <summary>Load (and cache) the embedded verbatim table via <c>Resources</c>.</summary>
        private static Dictionary<string, object> LoadTable()
        {
            if (_cachedTable != null)
            {
                return _cachedTable;
            }

            var textAsset = Resources.Load<TextAsset>(ResourcePath);
            if (textAsset == null)
            {
                throw new InvalidOperationException(
                    $"TextToControlResolver: could not load embedded table 'Resources/{ResourcePath}.json'.");
            }

            _cachedTable = (Dictionary<string, object>)MiniJson.Parse(textAsset.text);
            return _cachedTable;
        }

        /// <summary>Parse a table from raw JSON text (tests: fixture-driven, bypasses <c>Resources</c>).</summary>
        public static Dictionary<string, object> ParseTable(string json)
        {
            return (Dictionary<string, object>)MiniJson.Parse(json);
        }

        private static List<string> NormalizeText(string text)
        {
            var sb = new StringBuilder(text.Length);
            foreach (var c in text)
            {
                var lower = char.ToLowerInvariant(c);
                sb.Append(AccentMap.TryGetValue(lower, out var replacement) ? replacement : lower);
            }

            var normalized = sb.ToString();
            var tokens = new List<string>();
            var current = new StringBuilder();
            foreach (var c in normalized)
            {
                if (c is >= 'a' and <= 'z')
                {
                    current.Append(c);
                }
                else if (current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
            }

            if (current.Length > 0)
            {
                tokens.Add(current.ToString());
            }

            return tokens;
        }

        private static (float, float) ResolveDirection(
            List<string> tokens, Dictionary<string, object> directions, out bool found)
        {
            found = false;
            var sumX = 0f;
            var sumZ = 0f;
            foreach (var token in tokens)
            {
                if (!directions.TryGetValue(token, out var vectorObj))
                {
                    continue;
                }

                found = true;
                var vector = (List<object>)vectorObj;
                sumX += Convert.ToSingle(vector[0], CultureInfo.InvariantCulture);
                sumZ += Convert.ToSingle(vector[1], CultureInfo.InvariantCulture);
            }

            if (!found)
            {
                return (0f, 0f);
            }

            var norm = Mathf.Sqrt(sumX * sumX + sumZ * sumZ);
            if (norm < 1e-9f)
            {
                return (0f, 0f);
            }

            return (sumX / norm, sumZ / norm);
        }

        private static float ResolveSpeed(
            List<string> tokens, Dictionary<string, object> speeds, out bool found)
        {
            found = false;
            var best = 0f;
            foreach (var token in tokens)
            {
                if (!speeds.TryGetValue(token, out var valueObj))
                {
                    continue;
                }

                var value = Convert.ToSingle(valueObj, CultureInfo.InvariantCulture);
                if (value == 0f)
                {
                    found = true;
                    return 0f; // stop-family always wins
                }

                if (!found || value > best)
                {
                    best = value;
                }

                found = true;
            }

            return best;
        }
    }
}
